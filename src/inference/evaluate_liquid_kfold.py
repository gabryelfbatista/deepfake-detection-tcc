"""
Standalone K-Fold evaluation for experiment_03 (LFM2.5-VL-450M + QLoRA).

Reconstructs the same K-Fold splits used during training (same seed/config)
and evaluates each saved LoRA adapter on its held-out validation split.
Run on a single GPU — no DDP, no NCCL barriers.

    uv run python src/inference/evaluate_liquid_kfold.py
    uv run python src/inference/evaluate_liquid_kfold.py --folds 1 2   # specific folds
    uv run python src/inference/evaluate_liquid_kfold.py --adapter-dir experiments/liquid_kfold_03
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

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

from data.dataset import load_sidset, stratified_sample
from utils import set_seed, TEXT2LABEL

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "slm_config_experiment_03_liquid.yaml"

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
        llm_int8_skip_modules=["model.vision_tower", "lm_head"],
    )
    base = AutoModelForImageTextToText.from_pretrained(
        cfg["model"]["id"],
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    processor = AutoProcessor.from_pretrained(str(adapter_dir))
    return model, processor


@torch.inference_mode()
def eval_fold(model, processor, hf_split, val_indices, fold_idx):
    device = next(model.parameters()).device
    n = len(val_indices)
    print(f"\n[Fold {fold_idx}] evaluating {n:,} examples...", flush=True)

    y_true, y_pred = [], []

    for i, idx in enumerate(val_indices):
        if i % 500 == 0 and i > 0:
            print(f"  [eval] {i:,}/{n:,} ({100*i/n:.1f}%)", flush=True)

        item = hf_split[int(idx)]
        image = item["image"]
        image = image.convert("RGB") if hasattr(image, "convert") else Image.open(image).convert("RGB")
        image.thumbnail((512, 512), Image.LANCZOS)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_TEMPLATE},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
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


def checkpoint_path(adapter_root: Path, fold_num: int) -> Path:
    return adapter_root / f"fold_{fold_num}_eval_checkpoint.json"


def save_fold_checkpoint(result: dict, adapter_root: Path):
    path = checkpoint_path(adapter_root, result["fold"])
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [checkpoint] Fold {result['fold']} saved to {path}", flush=True)


def load_fold_checkpoint(fold_num: int, adapter_root: Path):
    path = checkpoint_path(adapter_root, fold_num)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"[Fold {fold_num}] Checkpoint found — skipping eval (loaded from {path})", flush=True)
        return data
    return None


def save_results(fold_metrics, output_dir: Path, cfg):
    f1_scores = [m["best_f1"] for m in fold_metrics]
    accs      = [m["accuracy"] for m in fold_metrics]
    mean_f1, std_f1 = float(np.mean(f1_scores)), float(np.std(f1_scores))
    mean_acc        = float(np.mean(accs))

    print(f"\n{'='*60}")
    print(f"K-Fold Eval Results ({len(fold_metrics)} folds) — LFM2.5-VL-450M + QLoRA")
    print(f"Per-fold F1-Macro: {[f'{v*100:.2f}%' for v in f1_scores]}")
    print(f"Mean F1-Macro : {mean_f1*100:.2f}%  ±  {std_f1*100:.2f}%")
    print(f"Mean Accuracy : {mean_acc*100:.2f}%")
    print(f"{'='*60}")

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
            "model"       : f"slm_lfm25vl450m_qlora_kfold_{cfg['kfold']['n_splits']}fold",
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
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["kfold"]["seed"])

    adapter_root = Path(args.adapter_dir) if args.adapter_dir else Path(cfg["output"]["dir"])
    print(f"[setup] Adapter root: {adapter_root}")

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
        cached = load_fold_checkpoint(fold_num, adapter_root)
        if cached is not None:
            fold_metrics.append(cached)
            continue

        fold_idx   = fold_num - 1
        _, val_pos = folds[fold_idx]
        val_indices = pool_indices[val_pos].tolist()

        adapter_dir = adapter_root / f"fold_{fold_num}" / "lora_adapter"
        if not adapter_dir.exists():
            print(f"[Fold {fold_num}] adapter not found at {adapter_dir}, skipping.")
            continue

        print(f"\n[Fold {fold_num}] Loading adapter from {adapter_dir}...")
        model, processor = load_model_for_eval(cfg, adapter_dir)

        result = eval_fold(model, processor, hf_train, val_indices, fold_num)
        save_fold_checkpoint(result, adapter_root)
        fold_metrics.append(result)

        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    if fold_metrics:
        save_results(fold_metrics, adapter_root, cfg)


if __name__ == "__main__":
    main()
