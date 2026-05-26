"""
Experiment 01 — EfficientNet-B0 baseline (SID_Set).
Reference: Wang et al. (2020) — CNN-Generated Images are Surprisingly Easy to Spot.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from torch.utils.tensorboard import SummaryWriter

from data.dataset import load_sidset, stratified_sample, NUM_CLASSES
from utils import set_seed, gpu_info

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "cnn_config_experiment_01.yaml"


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
    def __init__(self, hf_split, transform, indices=None):
        self.split = hf_split
        self.transform = transform
        self.indices = indices if indices is not None else list(range(len(hf_split)))

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


def train(cfg):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nGPU:", gpu_info())
    print(f"Device: {device}")

    sidset = load_sidset(cfg["dataset"].get("local_path", "dataset/sid_set"))

    train_indices = (
        stratified_sample(sidset["train"], cfg["dataset"]["max_train"])
        if cfg["dataset"]["max_train"] else None
    )
    val_indices = (
        stratified_sample(sidset["validation"], cfg["dataset"]["max_val"])
        if cfg["dataset"]["max_val"] else None
    )

    train_ds = SIDSetCNN(sidset["train"], build_transforms(cfg, train=True), train_indices)
    val_ds   = SIDSetCNN(sidset["validation"], build_transforms(cfg, train=False), val_indices)

    t = cfg["training"]
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=t["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=output_dir / "runs")
    best_f1 = 0.0
    n_batches = len(train_loader)
    log_every = max(1, n_batches // 10)
    print(f"\nStarting training — {len(train_ds)} train | {len(val_ds)} validation")
    print(f"Batches per epoch: {n_batches} | log every {log_every} batches\n")

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        loss_total, correct, total = 0.0, 0, 0

        for step, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_total += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

            if step % log_every == 0 or step == n_batches:
                print(f"  Epoch {epoch:02d} [{step:5d}/{n_batches}] "
                      f"loss={loss_total/total:.4f}  acc={correct/total*100:.1f}%")

        scheduler.step()
        train_acc = correct / total
        mean_loss = loss_total / total

        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.tolist())

        from sklearn.metrics import f1_score, accuracy_score
        val_acc = accuracy_score(y_true, y_pred)
        val_f1  = f1_score(y_true, y_pred, average="macro")

        print(f"Epoch {epoch:02d}/{t['epochs']} | loss={mean_loss:.4f} | "
              f"train_acc={train_acc*100:.1f}% | val_acc={val_acc*100:.1f}% | val_f1={val_f1*100:.1f}%")

        writer.add_scalar("Loss/train", mean_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("F1-Macro/val", val_f1, epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"  -> Best model saved (F1={best_f1*100:.1f}%)")

    writer.close()
    print(f"\nTraining finished. Best val F1: {best_f1*100:.1f}%")
    return model


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
