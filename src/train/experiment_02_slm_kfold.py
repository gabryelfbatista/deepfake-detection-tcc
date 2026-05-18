"""
Experiment 02 — Qwen3-VL-2B-Instruct with QLoRA + Stratified K-Fold (SID_Set).
Multi-GPU via HuggingFace Accelerate (DDP) + bf16 + 4-bit NF4 quantization.

Run with:
    accelerate launch --num_processes 2 src/train/experiment_02_slm_kfold.py
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import csv
import gc
from datetime import datetime

import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from transformers import Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

from data.dataset import load_sidset, stratified_sample, NUM_CLASSES, LABEL_NAMES
from utils import set_seed, gpu_info, TEXT2LABEL

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "slm_config_experiment_02_kfold.yaml"

PROMPT_TEMPLATE = (
    "Analise esta imagem e classifique como REAL, SYNTHETIC ou TAMPERED.\n"
    "REAL: fotografia autêntica.\n"
    "SYNTHETIC: imagem gerada inteiramente por IA (ex: FLUX).\n"
    "TAMPERED: imagem real com regiões modificadas por inpainting.\n"
    "Responda apenas com uma palavra: REAL, SYNTHETIC ou TAMPERED."
)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def log(msg: str):
    if is_main_process():
        print(msg, flush=True)


class SIDSetSLM(Dataset):
    """Formats SID_Set examples as multimodal conversations for Qwen-VL training."""

    def __init__(self, hf_split, processor, indices, max_length=512):
        self.split = hf_split
        self.processor = processor
        self.indices = list(indices)
        self.max_length = max_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.split[self.indices[idx]]
        image = item["image"]
        image = image.convert("RGB") if hasattr(image, "convert") else Image.open(image).convert("RGB")
        label_text = LABEL_NAMES[item["label"]]

        user_turn = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEMPLATE},
            ],
        }
        messages_full   = [user_turn, {"role": "assistant", "content": label_text}]
        messages_prompt = [user_turn]

        text_full   = self.processor.apply_chat_template(messages_full,   tokenize=False, add_generation_prompt=False)
        text_prompt = self.processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages_full)

        # Length of the prompt portion (no padding) — everything before this is masked from the loss.
        prompt_only = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        prompt_len = prompt_only["input_ids"].shape[1]

        inputs = self.processor(
            text=[text_full],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Response-only loss: mask prompt tokens and padding with -100 so the model is only
        # scored on the answer tokens (REAL / SYNTHETIC / TAMPERED + EOS).
        labels = inputs["input_ids"].clone()
        labels[:prompt_len] = -100
        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        return inputs


def qwen_vl_collator(features):
    """
    Qwen-VL returns variable-length `pixel_values` per image (one row per visual patch).
    The default collator tries to stack them and fails. We:
      - stack text tensors (already padded to max_length in the Dataset),
      - concatenate `pixel_values` along dim 0,
      - stack `image_grid_thw` normally (one [3] vector per image).
    """
    batch = {}
    keys = features[0].keys()
    for k in keys:
        vals = [f[k] for f in features]
        if k == "pixel_values":
            batch[k] = torch.cat(vals, dim=0)
        elif k == "image_grid_thw":
            batch[k] = torch.cat(vals, dim=0) if vals[0].dim() == 2 else torch.stack(vals)
        else:
            batch[k] = torch.stack(vals)
    return batch


def build_model(cfg):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["quant_type"],
        bnb_4bit_use_double_quant=cfg["quantization"]["double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # For DDP + QLoRA, pin the base model to the local rank rather than using "auto".
    rank = local_rank()
    device_map = {"": rank} if torch.cuda.is_available() else None

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model"]["id"],
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"]
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    if is_main_process():
        model.print_trainable_parameters()
    return model


@torch.inference_mode()
def evaluate_fold(model, processor, hf_split, eval_indices):
    """Generation-based F1 evaluation. Runs only on rank 0."""
    model.eval()
    y_true, y_pred = [], []
    device = next(model.parameters()).device

    n = len(eval_indices)
    log(f"  [eval] generating predictions for {n:,} examples...")

    for i, idx in enumerate(eval_indices):
        if i % 200 == 0 and i > 0:
            log(f"  [eval] {i:,}/{n:,}")

        item = hf_split[int(idx)]
        image = item["image"]
        image = image.convert("RGB") if hasattr(image, "convert") else Image.open(image).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_TEMPLATE},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(device)

        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        generated = processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().upper()

        pred = TEXT2LABEL.get(generated, -1)
        if pred == -1:
            for k, v in TEXT2LABEL.items():
                if k in generated:
                    pred = v
                    break
            if pred == -1:
                pred = 0

        y_true.append(item["label"])
        y_pred.append(pred)

    return y_true, y_pred


def train_fold(fold_idx, train_indices, val_indices, hf_train, processor, cfg, output_dir):
    fold_dir = output_dir / f"fold_{fold_idx + 1}"
    if is_main_process():
        fold_dir.mkdir(parents=True, exist_ok=True)

    t = cfg["training"]
    max_len = t.get("max_seq_length", 512)

    train_ds = SIDSetSLM(hf_train, processor, train_indices, max_length=max_len)
    val_ds   = SIDSetSLM(hf_train, processor, val_indices,   max_length=max_len)

    log(f"\n[Fold {fold_idx+1}] {len(train_ds):,} train | {len(val_ds):,} val")

    model = build_model(cfg)

    args = TrainingArguments(
        output_dir=str(fold_dir),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accumulation"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        optim=t["optimizer"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        save_strategy="no",            # we save the LoRA adapter manually at the end
        eval_strategy="no",            # loss-based eval is uninformative for F1; we do generation eval after training
        logging_steps=50,
        report_to=cfg["output"].get("report_to", "none"),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=qwen_vl_collator,
    )

    log(f"[Fold {fold_idx+1}] starting training...")
    trainer.train()
    log(f"[Fold {fold_idx+1}] training finished.")

    # Save LoRA adapter (rank-0 only). Trainer.save_model handles this safely.
    if cfg["output"].get("save_adapters", True):
        adapter_dir = fold_dir / "lora_adapter"
        trainer.save_model(str(adapter_dir))
        if is_main_process():
            processor.save_pretrained(str(adapter_dir))
            log(f"[Fold {fold_idx+1}] adapter saved to {adapter_dir}")

    # Generation-based eval on a stratified subset of this fold's validation indices.
    # Only rank 0 runs eval — the trained adapter is identical across ranks.
    fold_result = None
    if is_main_process():
        eval_cap = cfg["dataset"].get("eval_max_per_fold")
        if eval_cap and eval_cap < len(val_indices):
            val_labels = np.array(hf_train["label"])[val_indices]
            per_class = eval_cap // NUM_CLASSES
            rng = np.random.default_rng(cfg["kfold"]["seed"] + fold_idx)
            sampled = []
            for cls in range(NUM_CLASSES):
                cls_idxs = val_indices[val_labels == cls]
                rng.shuffle(cls_idxs)
                sampled.extend(cls_idxs[:per_class].tolist())
            eval_indices = sampled
        else:
            eval_indices = list(val_indices)

        eval_model = trainer.model
        y_true, y_pred = evaluate_fold(eval_model, processor, hf_train, eval_indices)

        f1_macro = float(f1_score(y_true, y_pred, average="macro"))
        acc      = float(accuracy_score(y_true, y_pred))
        per_cls  = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])

        log(
            f"[Fold {fold_idx+1}] eval acc={acc*100:.2f}% | "
            f"F1-Macro={f1_macro*100:.2f}% | "
            f"REAL={per_cls[0]*100:.1f} SYN={per_cls[1]*100:.1f} TAM={per_cls[2]*100:.1f}"
        )

        fold_result = {
            "fold"        : fold_idx + 1,
            "best_f1"     : f1_macro,
            "accuracy"    : acc,
            "f1_real"     : float(per_cls[0]),
            "f1_synthetic": float(per_cls[1]),
            "f1_tampered" : float(per_cls[2]),
            "n_eval"      : len(eval_indices),
        }

    # Cleanup before the next fold
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return fold_result


def save_kfold_results(fold_metrics, output_dir, cfg):
    f1_scores = [m["best_f1"] for m in fold_metrics]
    accs      = [m["accuracy"] for m in fold_metrics]
    mean_f1, std_f1 = float(np.mean(f1_scores)), float(np.std(f1_scores))
    mean_acc        = float(np.mean(accs))

    print(f"\n{'='*55}")
    print(f"SLM K-Fold Results ({len(fold_metrics)} folds) — Qwen3-VL-2B + QLoRA")
    print(f"Per-fold F1-Macro: {[f'{v*100:.2f}%' for v in f1_scores]}")
    print(f"Mean F1-Macro : {mean_f1*100:.2f}%  ±  {std_f1*100:.2f}%")
    print(f"Mean Accuracy : {mean_acc*100:.2f}%")
    print(f"{'='*55}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregate = {
        "timestamp": ts,
        "model"    : cfg["model"]["id"],
        "n_splits" : cfg["kfold"]["n_splits"],
        "mean_f1"  : mean_f1,
        "std_f1"   : std_f1,
        "mean_acc" : mean_acc,
        "per_fold" : fold_metrics,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    csv_path = Path("experiments/results_summary.csv")
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "model", "accuracy", "f1_macro",
            "f1_real", "f1_synthetic", "f1_tampered"
        ])
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp"   : ts,
            "model"       : f"slm_qwen3vl2b_qlora_kfold_{cfg['kfold']['n_splits']}fold",
            "accuracy"    : f"{mean_acc*100:.2f}",
            "f1_macro"    : f"{mean_f1*100:.2f}±{std_f1*100:.2f}",
            "f1_real"     : f"{np.mean([m['f1_real']     for m in fold_metrics])*100:.2f}",
            "f1_synthetic": f"{np.mean([m['f1_synthetic'] for m in fold_metrics])*100:.2f}",
            "f1_tampered" : f"{np.mean([m['f1_tampered']  for m in fold_metrics])*100:.2f}",
        })

    print(f"\nResults saved to: {out}")
    print(f"Global CSV updated: {csv_path}")


def train(cfg):
    set_seed(cfg["kfold"]["seed"])

    log("\n" + "="*60)
    log("TRAINING — Multimodal SLM K-Fold (Qwen3-VL-2B + QLoRA)")
    log("="*60)
    log(f"[setup] GPU: {gpu_info()}")
    log(f"[setup] Local rank: {local_rank()} | World size: {os.environ.get('WORLD_SIZE', 1)}")
    log(f"[setup] Output dir: {cfg['output']['dir']}")

    min_px = cfg["model"].get("min_pixels", 3136)
    max_px = cfg["model"].get("max_pixels", 200704)
    log(f"\n[setup] Loading processor ({cfg['model']['id']}) — pixels: {min_px}–{max_px}...")
    processor = AutoProcessor.from_pretrained(
        cfg["model"]["id"],
        min_pixels=min_px,
        max_pixels=max_px,
    )
    # SFT requires right-padding so the answer span sits at a known offset from `prompt_len`.
    processor.tokenizer.padding_side = "right"

    log("\n[setup] Loading dataset...")
    sidset = load_sidset(cfg["dataset"].get("local_path", "dataset/sid_set"))
    hf_train = sidset["train"]

    max_train = cfg["dataset"].get("max_train")
    if max_train:
        pool_indices = stratified_sample(hf_train, max_train, seed=cfg["kfold"]["seed"])
    else:
        pool_indices = list(range(len(hf_train)))

    all_labels   = np.array(hf_train["label"])[pool_indices]
    pool_indices = np.array(pool_indices)

    kf = StratifiedKFold(
        n_splits=cfg["kfold"]["n_splits"],
        shuffle=cfg["kfold"]["shuffle"],
        random_state=cfg["kfold"]["seed"],
    )

    output_dir = Path(cfg["output"]["dir"])
    fold_metrics = []

    for fold_idx, (train_pos, val_pos) in enumerate(kf.split(pool_indices, all_labels)):
        train_indices = pool_indices[train_pos]
        val_indices   = pool_indices[val_pos]

        result = train_fold(
            fold_idx, train_indices, val_indices, hf_train, processor, cfg, output_dir
        )
        if result is not None:
            fold_metrics.append(result)

    if is_main_process() and fold_metrics:
        save_kfold_results(fold_metrics, output_dir, cfg)


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
