"""
Experiment 01 — Qwen2-VL-2B with QLoRA for multimodal deepfake detection.
References: Hu et al. (2021) LoRA; Dettmers et al. (2023) QLoRA.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reduce CUDA memory fragmentation (necessary for 8GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml
import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

from data.dataset import load_sidset, stratified_sample, LABEL_NAMES
from utils import set_seed, gpu_info

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "slm_config_experiment_01.yaml"

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


class SIDSetSLM(Dataset):
    """Dataset wrapper that formats examples as multimodal conversations for Qwen2-VL."""

    def __init__(self, hf_split, processor, indices=None, max_length=256):
        self.split = hf_split
        self.processor = processor
        self.indices = indices if indices is not None else list(range(len(hf_split)))
        self.max_length = max_length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.split[self.indices[idx]]
        image = item["image"].convert("RGB") if hasattr(item["image"], "convert") else Image.open(item["image"]).convert("RGB")
        label_text = LABEL_NAMES[item["label"]]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_TEMPLATE},
                ],
            },
            {"role": "assistant", "content": label_text},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            truncation=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


def build_model(cfg):
    print("[model] Configuring QLoRA quantization (4-bit NF4)...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["quant_type"],
        bnb_4bit_use_double_quant=cfg["quantization"]["double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"[model] Loading {cfg['model']['id']}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg["model"]["id"],
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("[model] Base model loaded.")

    print(f"[model] Applying LoRA (r={cfg['lora']['r']}, alpha={cfg['lora']['alpha']})...")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def train(cfg):
    set_seed(42)
    print("\n" + "="*60)
    print("TRAINING — Multimodal SLM (Qwen2-VL-2B + QLoRA)")
    print("="*60)
    print(f"[setup] GPU: {gpu_info()}")
    print(f"[setup] Output dir: {cfg['output']['dir']}")

    min_px = cfg["model"].get("min_pixels", 3136)
    max_px = cfg["model"].get("max_pixels", 200704)
    print(f"\n[setup] Loading processor ({cfg['model']['id']}) — pixels: {min_px}–{max_px}...")
    processor = AutoProcessor.from_pretrained(
        cfg["model"]["id"],
        min_pixels=min_px,
        max_pixels=max_px,
    )
    print("[setup] Processor loaded.")

    print("\n[setup] Loading dataset...")
    sidset = load_sidset(cfg["dataset"].get("local_path", "dataset/sid_set"))

    print("\n[setup] Preparing splits...")
    train_indices = (
        stratified_sample(sidset["train"], cfg["dataset"]["max_train"])
        if cfg["dataset"]["max_train"] else None
    )
    val_indices = (
        stratified_sample(sidset["validation"], cfg["dataset"]["max_val"])
        if cfg["dataset"]["max_val"] else None
    )

    max_len = cfg["training"].get("max_seq_length", 256)
    train_ds = SIDSetSLM(sidset["train"], processor, train_indices, max_length=max_len)
    val_ds   = SIDSetSLM(sidset["validation"], processor, val_indices, max_length=max_len)
    print(f"[setup] Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    print("\n[setup] Building model...")
    model = build_model(cfg)

    t = cfg["training"]
    steps_per_epoch = len(train_ds) // (t["batch_size"] * t["grad_accumulation"])
    print(f"\n[setup] Epochs: {t['epochs']} | Steps per epoch (approx): {steps_per_epoch:,}")
    print(f"[setup] LR: {t['learning_rate']} | Scheduler: {t['lr_scheduler']}")

    args = TrainingArguments(
        output_dir=cfg["output"]["dir"],
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
        save_steps=cfg["output"]["save_steps"],
        eval_steps=cfg["output"]["eval_steps"],
        save_total_limit=cfg["output"]["save_total_limit"],
        eval_strategy="steps",
        load_best_model_at_end=True,
        report_to=cfg["output"]["report_to"],
        logging_steps=50,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"\n{'='*60}")
    print(f"STARTING TRAINING — {len(train_ds):,} train / {len(val_ds):,} val")
    print(f"{'='*60}\n")
    trainer.train()
    print("\n[train] Training finished.")

    lora_dir = Path(cfg["output"]["dir"]) / "lora_adapter"
    print(f"[train] Saving LoRA adapter to '{lora_dir}'...")
    model.save_pretrained(lora_dir)
    processor.save_pretrained(lora_dir)
    print(f"[train] Adapter saved.")
    return model, processor


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
