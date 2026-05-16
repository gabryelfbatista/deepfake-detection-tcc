"""
Inference — evaluate EfficientNet-B0 checkpoint (experiment_01) on SID_Set.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data.dataset import load_sidset, stratified_sample
from utils import compute_metrics, save_results, set_seed
from train.experiment_01_cnn_baseline import (
    load_config,
    build_model,
    build_transforms,
    SIDSetCNN,
)


@torch.inference_mode()
def evaluate(cfg):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = Path(cfg["output"]["dir"]) / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval().to(device)
    print(f"Checkpoint loaded: {checkpoint}")

    sidset = load_sidset(cfg["dataset"].get("local_path", "dataset/sid_set"))
    val_indices = (
        stratified_sample(sidset["validation"], cfg["dataset"]["max_val"])
        if cfg["dataset"]["max_val"] else None
    )
    val_ds = SIDSetCNN(sidset["validation"], build_transforms(cfg, train=False), val_indices)
    loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], num_workers=4)

    y_true, y_pred = [], []
    for imgs, labels in loader:
        preds = model(imgs.to(device)).argmax(1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())

    metrics = compute_metrics(y_true, y_pred)
    save_results(metrics, cfg["output"]["dir"], "CNN_EfficientNetB0_exp01")
    return metrics


if __name__ == "__main__":
    cfg = load_config()
    evaluate(cfg)
