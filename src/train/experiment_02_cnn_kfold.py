"""
Experiment 02 — EfficientNet-B0 with Stratified K-Fold (SID_Set).
Multi-GPU via HuggingFace Accelerate (DDP) + bf16 mixed precision.

Run with:
    accelerate launch --num_processes 2 src/train/experiment_02_cnn_kfold.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import csv
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

from data.dataset import load_sidset, stratified_sample, NUM_CLASSES, LABEL_NAMES
from utils import set_seed, gpu_info

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "cnn_config_experiment_02_kfold.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_transforms(cfg, train=True):
    img_size = cfg["training"]["img_size"]
    aug = cfg.get("augmentation", {})

    if train:
        ops = [transforms.Resize((img_size, img_size))]
        if aug.get("horizontal_flip"):
            ops.append(transforms.RandomHorizontalFlip())
        if aug.get("rotation"):
            ops.append(transforms.RandomRotation(aug["rotation"]))
        if aug.get("color_jitter"):
            cj = aug["color_jitter"]
            ops.append(transforms.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
            ))
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        ops = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    return transforms.Compose(ops)


class SIDSetCNN(Dataset):
    def __init__(self, hf_split, transform, indices):
        self.split = hf_split
        self.transform = transform
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.split[self.indices[idx]]
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        return self.transform(image), item["label"]


def build_model(cfg):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg["model"]["pretrained"] else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def train_fold(fold_idx, train_indices, val_indices, hf_train, cfg, accelerator, output_dir):
    """Train one fold and return (best_val_f1, y_true, y_pred) — only valid on main process."""
    t = cfg["training"]
    fold_dir = output_dir / f"fold_{fold_idx + 1}"
    if accelerator.is_main_process:
        fold_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SIDSetCNN(hf_train, build_transforms(cfg, train=True),  list(train_indices))
    val_ds   = SIDSetCNN(hf_train, build_transforms(cfg, train=False), list(val_indices))

    # pin_memory and device placement handled by accelerate
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,  num_workers=8)
    val_loader   = DataLoader(val_ds,   batch_size=t["batch_size"], shuffle=False, num_workers=8)

    model     = build_model(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    best_f1 = 0.0
    best_y_true, best_y_pred = [], []
    n_batches = len(train_loader)
    log_every = max(1, n_batches // 10)

    writer = SummaryWriter(log_dir=str(fold_dir / "runs")) if accelerator.is_main_process else None

    accelerator.print(f"\n[Fold {fold_idx+1}] {len(train_ds)} train | {len(val_ds)} val — {n_batches} batches/epoch/GPU")

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        loss_total, correct, total = 0.0, 0, 0

        for step, (imgs, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            accelerator.backward(loss)
            optimizer.step()

            loss_total += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

            if step % log_every == 0 or step == n_batches:
                accelerator.print(
                    f"  Fold {fold_idx+1} Epoch {epoch:02d} [{step:5d}/{n_batches}] "
                    f"loss={loss_total/total:.4f}  acc={correct/total*100:.1f}%"
                )

        scheduler.step()

        # Gather predictions across both GPUs before computing metrics
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds = model(imgs).argmax(1)
                preds_all, labels_all = accelerator.gather_for_metrics((preds, labels))
                y_pred.extend(preds_all.cpu().tolist())
                y_true.extend(labels_all.cpu().tolist())

        val_acc = accuracy_score(y_true, y_pred)
        val_f1  = f1_score(y_true, y_pred, average="macro")
        train_acc = correct / total
        mean_loss = loss_total / total

        accelerator.print(
            f"  Fold {fold_idx+1} Epoch {epoch:02d}/{t['epochs']} | "
            f"loss={mean_loss:.4f} | train_acc={train_acc*100:.1f}% | "
            f"val_acc={val_acc*100:.1f}% | val_f1={val_f1*100:.1f}%"
        )

        if writer:
            writer.add_scalar("Loss/train", mean_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)
            writer.add_scalar("F1-Macro/val", val_f1, epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_y_true, best_y_pred = y_true, y_pred
            # Unwrap DDP wrapper before saving state dict
            unwrapped = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                torch.save(unwrapped.state_dict(), fold_dir / "best_model.pth")
            accelerator.print(f"  -> Best model saved (F1={best_f1*100:.1f}%)")

    accelerator.print(f"\n[Fold {fold_idx+1}] Best val F1: {best_f1*100:.1f}%")

    if writer:
        writer.close()

    # Free GPU memory before next fold
    accelerator.free_memory()

    return best_f1, best_y_true, best_y_pred


def save_kfold_results(fold_metrics, output_dir, cfg):
    f1_scores = [m["best_f1"] for m in fold_metrics]
    accs      = [m["accuracy"] for m in fold_metrics]
    mean_f1, std_f1 = float(np.mean(f1_scores)), float(np.std(f1_scores))
    mean_acc        = float(np.mean(accs))
    mean_f1_real    = float(np.mean([m["f1_real"]      for m in fold_metrics]))
    mean_f1_syn     = float(np.mean([m["f1_synthetic"] for m in fold_metrics]))
    mean_f1_tam     = float(np.mean([m["f1_tampered"]  for m in fold_metrics]))

    print(f"\n{'='*55}")
    print(f"K-Fold Results ({len(fold_metrics)} folds)")
    print(f"Per-fold F1-Macro: {[f'{v*100:.1f}%' for v in f1_scores]}")
    print(f"Mean F1-Macro : {mean_f1*100:.2f}%  ±  {std_f1*100:.2f}%")
    print(f"Mean Accuracy : {mean_acc*100:.2f}%")
    print(f"{'='*55}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregate = {
        "timestamp": ts,
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
            "model"       : f"cnn_kfold_{cfg['kfold']['n_splits']}fold",
            "accuracy"    : f"{mean_acc*100:.2f}",
            "f1_macro"    : f"{mean_f1*100:.2f}±{std_f1*100:.2f}",
            "f1_real"     : f"{mean_f1_real*100:.2f}",
            "f1_synthetic": f"{mean_f1_syn*100:.2f}",
            "f1_tampered" : f"{mean_f1_tam*100:.2f}",
        })

    print(f"\nResults saved to: {out}")
    print(f"Global CSV updated: {csv_path}")
    return mean_f1, std_f1


def train(cfg):
    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(cfg["kfold"]["seed"])

    accelerator.print("\nGPU:", gpu_info())
    accelerator.print(f"Processes: {accelerator.num_processes} | Device: {accelerator.device} | bf16: on")

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

    output_dir   = Path(cfg["output"]["dir"])
    fold_metrics = []

    for fold_idx, (train_pos, val_pos) in enumerate(kf.split(pool_indices, all_labels)):
        train_indices = pool_indices[train_pos]
        val_indices   = pool_indices[val_pos]

        best_f1, y_true, y_pred = train_fold(
            fold_idx, train_indices, val_indices, hf_train, cfg, accelerator, output_dir
        )

        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
        fold_metrics.append({
            "fold"        : fold_idx + 1,
            "best_f1"     : best_f1,
            "accuracy"    : float(accuracy_score(y_true, y_pred)),
            "f1_real"     : float(per_class_f1[0]),
            "f1_synthetic": float(per_class_f1[1]),
            "f1_tampered" : float(per_class_f1[2]),
        })

    if accelerator.is_main_process:
        save_kfold_results(fold_metrics, output_dir, cfg)


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
