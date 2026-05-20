"""
Standalone K-Fold evaluation for experiment_02 (Qwen3-VL-2B + QLoRA).

Reconstructs the same K-Fold splits used during training (same seed/config)
and evaluates each saved LoRA adapter on its full validation split.
Run on a single GPU — no DDP, no NCCL barriers.

    uv run python src/inference/evaluate_slm_kfold.py
    uv run python src/inference/evaluate_slm_kfold.py --folds 1 2   # specific folds
    uv run python src/inference/evaluate_slm_kfold.py --adapter-dir experiments/slm_kfold_02
"""
import os
import sys
import argparse
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
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from data.dataset import load_sidset, stratified_sample, NUM_CLASSES, LABEL_NAMES
from utils import set_seed, TEXT2LABEL

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


def load_model_for_eval(cfg, adapter_dir: Path):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["quant_type"],
        bnb_4bit_use_double_quant=cfg["quantization"]["double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model"]["id"],
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    min_px = cfg["model"].get("min_pixels", 3136)
    max_px = cfg["model"].get("max_pixels", 200704)
    processor = AutoProcessor.from_pretrained(
        cfg["model"]["id"], min_pixels=min_px, max_pixels=max_px
    )
    processor.tokenizer.padding_side = "left"
    return model, processor


def _decode_generated(text: str) -> int:
    text = text.strip().upper()
    if text in TEXT2LABEL:
        return TEXT2LABEL[text]
    for k, v in TEXT2LABEL.items():
        if k in text:
            return v
    return 0


@torch.inference_mode()
def eval_fold(model, processor, hf_split, val_indices, fold_idx, batch_size=8):
    device = next(model.parameters()).device
    n = len(val_indices)
    print(f"\n[Fold {fold_idx}] evaluating {n:,} examples (batch_size={batch_size})...", flush=True)

    y_true, y_pred = [], []

    for batch_start in range(0, n, batch_size):
        batch_indices = val_indices[batch_start : batch_start + batch_size]
        if batch_start % (batch_size * 50) == 0 and batch_start > 0:
            print(f"  [eval] {batch_start:,}/{n:,} ({100*batch_start/n:.1f}%)", flush=True)

        texts, all_image_inputs, labels = [], [], []
        for idx in batch_indices:
            item = hf_split[int(idx)]
            image = item["image"]
            image = image.convert("RGB") if hasattr(image, "convert") else Image.open(image).convert("RGB")
            labels.append(item["label"])

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPT_TEMPLATE},
                    ],
                }
            ]
            texts.append(
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            img_inputs, _ = process_vision_info(messages)
            all_image_inputs.extend(img_inputs)

        inputs = processor(
            text=texts,
            images=all_image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        prompt_len = inputs["input_ids"].shape[1]
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)

        for j, seq in enumerate(out):
            generated = processor.decode(seq[prompt_len:], skip_special_tokens=True)
            y_pred.append(_decode_generated(generated))
        y_true.extend(labels)

    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    acc      = float(accuracy_score(y_true, y_pred))
    per_cls  = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])

    print(
        f"[Fold {fold_idx}] acc={acc*100:.2f}% | F1-Macro={f1_macro*100:.2f}% | "
        f"REAL={per_cls[0]*100:.1f} SYN={per_cls[1]*100:.1f} TAM={per_cls[2]*100:.1f}",
        flush=True,
    )
    return {
        "fold"        : fold_idx,
        "best_f1"     : f1_macro,
        "accuracy"    : acc,
        "f1_real"     : float(per_cls[0]),
        "f1_synthetic": float(per_cls[1]),
        "f1_tampered" : float(per_cls[2]),
        "n_eval"      : n,
    }


def save_results(fold_metrics, output_dir: Path, cfg):
    f1_scores = [m["best_f1"] for m in fold_metrics]
    accs      = [m["accuracy"] for m in fold_metrics]
    mean_f1, std_f1 = float(np.mean(f1_scores)), float(np.std(f1_scores))
    mean_acc        = float(np.mean(accs))

    print(f"\n{'='*55}")
    print(f"K-Fold Eval Results ({len(fold_metrics)} folds) — Qwen3-VL-2B + QLoRA")
    print(f"Per-fold F1-Macro: {[f'{v*100:.2f}%' for v in f1_scores]}")
    print(f"Mean F1-Macro : {mean_f1*100:.2f}%  ±  {std_f1*100:.2f}%")
    print(f"Mean Accuracy : {mean_acc*100:.2f}%")
    print(f"{'='*55}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregate = {
        "timestamp": ts,
        "model"    : cfg["model"]["id"],
        "n_splits" : len(fold_metrics),
        "mean_f1"  : mean_f1,
        "std_f1"   : std_f1,
        "mean_acc" : mean_acc,
        "per_fold" : fold_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "aggregate_metrics.json", "w") as f:
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

    print(f"\nResults saved to: {output_dir}")
    print(f"Global CSV updated: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Root dir with fold_N/lora_adapter/ subdirs (default: output.dir from config)",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Which fold numbers to evaluate (1-indexed). Default: all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size per GPU step (default: 8). Increase if VRAM allows.",
    )
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["kfold"]["seed"])

    adapter_root = Path(args.adapter_dir) if args.adapter_dir else Path(cfg["output"]["dir"])
    print(f"[setup] Adapter root: {adapter_root}")

    min_px = cfg["model"].get("min_pixels", 3136)
    max_px = cfg["model"].get("max_pixels", 200704)
    print(f"[setup] Loading dataset...")
    sidset   = load_sidset(cfg["dataset"].get("local_path", "dataset/sid_set"))
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
    folds = list(kf.split(pool_indices, all_labels))
    target_folds = args.folds if args.folds else list(range(1, cfg["kfold"]["n_splits"] + 1))

    fold_metrics = []

    for fold_num in target_folds:
        fold_idx   = fold_num - 1
        _, val_pos = folds[fold_idx]
        val_indices = pool_indices[val_pos].tolist()

        adapter_dir = adapter_root / f"fold_{fold_num}" / "lora_adapter"
        if not adapter_dir.exists():
            print(f"[Fold {fold_num}] adapter not found at {adapter_dir}, skipping.")
            continue

        print(f"\n[Fold {fold_num}] Loading adapter from {adapter_dir}...")
        model, processor = load_model_for_eval(cfg, adapter_dir)

        result = eval_fold(model, processor, hf_train, val_indices, fold_num, batch_size=args.batch_size)
        fold_metrics.append(result)

        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    if fold_metrics:
        save_results(fold_metrics, adapter_root, cfg)


if __name__ == "__main__":
    main()
