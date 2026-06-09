"""
Experiment 03 — Qwen3-VL-2B-Instruct with QLoRA + Stratified K-Fold (SID_Set).
Single-GPU (1x RTX 4090) + bf16 + 4-bit NF4 quantization.

Pareado com experiment_03_liquid_vl.py: mesmos folds (seed 42), mesmo batch
efetivo (64) e logging por fold em TensorBoard.

Run with (single GPU):
    uv run python src/train/experiment_03_qwen_vl.py
Or, for multi-GPU via accelerate (DDP):
    accelerate launch --num_processes 2 src/train/experiment_03_qwen_vl.py
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc

import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from PIL import Image

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from transformers import Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

from data.dataset import load_sidset, stratified_sample, LABEL_NAMES
from utils import set_seed, gpu_info

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "slm_config_experiment_03_qwen.yaml"

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
        logging_dir=str(fold_dir / "runs"),
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

    # Cleanup before the next fold
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def train(cfg, start_fold: int = 1):
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

    if start_fold > 1:
        log(f"[setup] Resuming from fold {start_fold} (skipping folds 1–{start_fold - 1})")

    for fold_idx, (train_pos, val_pos) in enumerate(kf.split(pool_indices, all_labels)):
        if fold_idx + 1 < start_fold:
            continue

        train_indices = pool_indices[train_pos]
        val_indices   = pool_indices[val_pos]

        train_fold(fold_idx, train_indices, val_indices, hf_train, processor, cfg, output_dir)

    log("\nAll folds trained. Run evaluate_qwen_kfold.py for inference-based evaluation.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-fold", type=int, default=1,
                        help="Resume from this fold number (1-indexed). Skips earlier folds.")
    args = parser.parse_args()
    cfg = load_config()
    train(cfg, start_fold=args.start_fold)
