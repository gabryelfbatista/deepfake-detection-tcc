"""
Shared utilities for both the SLM and the CNN baseline.
"""
import json
import csv
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

LABEL_NAMES  = ["REAL", "SYNTHETIC", "TAMPERED"]
LABEL2TEXT   = {0: "REAL", 1: "SYNTHETIC", 2: "TAMPERED"}
TEXT2LABEL   = {"REAL": 0, "SYNTHETIC": 1, "TAMPERED": 2}


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_info() -> dict:
    """Return GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available"    : True,
        "name"         : torch.cuda.get_device_name(0),
        "vram_total"   : f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB",
        "vram_reserved": f"{torch.cuda.memory_reserved(0)/1e9:.1f} GB reserved",
    }


def compute_metrics(y_true, y_pred) -> dict:
    """Return accuracy, F1-Macro, per-class report, and confusion matrix."""
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)
    cm     = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*55}")
    print(f"Global Accuracy : {acc*100:.2f}%")
    print(f"F1-Score Macro  : {f1*100:.2f}%")
    print()
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    print("Confusion Matrix:")
    print(f"{'':12}", " ".join(f"{c:>10}" for c in LABEL_NAMES))
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name:12}", " ".join(f"{cm[i,j]:>10}" for j in range(3)))

    return {
        "accuracy"   : acc,
        "f1_macro"   : f1,
        "report"     : report,
        "conf_matrix": cm.tolist(),
    }


def save_results(metrics: dict, output_dir: str, name: str):
    """
    Save results as JSON and append a row to the cross-experiment comparison CSV.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(out / f"metrics_{ts}.json", "w") as f:
        json.dump({
            "name"     : name,
            "timestamp": ts,
            "metrics"  : metrics,
        }, f, indent=2, default=str)

    csv_path = Path("experiments/results_summary.csv")
    exists   = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "model", "accuracy", "f1_macro",
            "f1_real", "f1_synthetic", "f1_tampered"
        ])
        if not exists:
            writer.writeheader()

        rep = metrics.get("report", {})
        writer.writerow({
            "timestamp"    : ts,
            "model"        : name,
            "accuracy"     : f"{metrics['accuracy']*100:.2f}",
            "f1_macro"     : f"{metrics['f1_macro']*100:.2f}",
            "f1_real"      : f"{rep.get('REAL',{}).get('f1-score',0)*100:.2f}",
            "f1_synthetic" : f"{rep.get('SYNTHETIC',{}).get('f1-score',0)*100:.2f}",
            "f1_tampered"  : f"{rep.get('TAMPERED',{}).get('f1-score',0)*100:.2f}",
        })

    print(f"\nResults saved to: {out}")
    print(f"Global CSV updated: {csv_path}")
