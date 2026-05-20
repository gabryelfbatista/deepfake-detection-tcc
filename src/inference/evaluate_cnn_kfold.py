"""
Standalone K-Fold evaluation for experiment_02 (EfficientNet-B0).

Reconstructs the same K-Fold splits used during training (same seed/config)
and evaluates each saved checkpoint on its validation split.

    uv run python src/inference/evaluate_cnn_kfold.py
    uv run python src/inference/evaluate_cnn_kfold.py --folds 1 2
    uv run python src/inference/evaluate_cnn_kfold.py --checkpoint-dir experiments/cnn_kfold_02
"""
import sys
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from data.dataset import load_sidset, stratified_sample
from utils import set_seed
from train.experiment_02_cnn_kfold import load_config, build_model, build_transforms, SIDSetCNN

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "cnn_config_experiment_02_kfold.yaml"


@torch.inference_mode()
def eval_fold(model, hf_split, val_indices, cfg, fold_num):
    device = next(model.parameters()).device
    n = len(val_indices)
    print(f"\n[Fold {fold_num}] evaluating {n:,} examples...", flush=True)

    val_ds = SIDSetCNN(hf_split, build_transforms(cfg, train=False), list(val_indices))
    loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    y_true, y_pred = [], []
    for imgs, labels in loader:
        preds = model(imgs.to(device)).argmax(1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())

    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    acc      = float(accuracy_score(y_true, y_pred))
    per_cls  = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])

    print(
        f"[Fold {fold_num}] acc={acc*100:.2f}% | F1-Macro={f1_macro*100:.2f}% | "
        f"REAL={per_cls[0]*100:.1f} SYN={per_cls[1]*100:.1f} TAM={per_cls[2]*100:.1f}",
        flush=True,
    )
    return {
        "fold"        : fold_num,
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
    print(f"K-Fold Eval Results ({len(fold_metrics)} folds) — EfficientNet-B0")
    print(f"Per-fold F1-Macro: {[f'{v*100:.2f}%' for v in f1_scores]}")
    print(f"Mean F1-Macro : {mean_f1*100:.2f}%  ±  {std_f1*100:.2f}%")
    print(f"Mean Accuracy : {mean_acc*100:.2f}%")
    print(f"{'='*55}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregate = {
        "timestamp": ts,
        "model"    : cfg["model"]["backbone"],
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
            "model"       : f"cnn_kfold_{cfg['kfold']['n_splits']}fold",
            "accuracy"    : f"{mean_acc*100:.2f}",
            "f1_macro"    : f"{mean_f1*100:.2f}±{std_f1*100:.2f}",
            "f1_real"     : f"{np.mean([m['f1_real']      for m in fold_metrics])*100:.2f}",
            "f1_synthetic": f"{np.mean([m['f1_synthetic'] for m in fold_metrics])*100:.2f}",
            "f1_tampered" : f"{np.mean([m['f1_tampered']  for m in fold_metrics])*100:.2f}",
        })

    print(f"\nResults saved to: {output_dir}")
    print(f"Global CSV updated: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Root dir with fold_N/best_model.pth subdirs (default: output.dir from config)",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(cfg["output"]["dir"])
    print(f"[setup] Checkpoint root: {checkpoint_root}")
    print(f"[setup] Device: {device}")

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
        checkpoint = checkpoint_root / f"fold_{fold_num}" / "best_model.pth"
        if not checkpoint.exists():
            print(f"[Fold {fold_num}] checkpoint not found at {checkpoint}, skipping.")
            continue

        _, val_pos  = folds[fold_num - 1]
        val_indices = pool_indices[val_pos].tolist()

        model = build_model(cfg)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval().to(device)
        print(f"\n[Fold {fold_num}] Loaded checkpoint from {checkpoint}")

        result = eval_fold(model, hf_train, val_indices, cfg, fold_num)
        fold_metrics.append(result)

        del model

    if fold_metrics:
        save_results(fold_metrics, checkpoint_root, cfg)


if __name__ == "__main__":
    main()
