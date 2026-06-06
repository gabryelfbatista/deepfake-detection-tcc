"""
Diagnostic: dump RAW model outputs of the Liquid (LFM2.5-VL-450M + QLoRA) model.

Goal: figure out WHY f1_synthetic == 0.0 in the K-Fold eval — is the model truly
confusing SYNTHETIC with another class, or is it a decoding/parsing artifact
(TEXT2LABEL fallback defaulting unmatched outputs to class 0 = REAL)?

Loads one fold's adapter, runs inference on a balanced sample of the held-out
split, and prints, per true class:
  - the distribution of RAW generated strings (no .upper(), no parsing)
  - the parsed prediction (same logic as evaluate_liquid_kfold.py)
  - a small confusion table (true label -> predicted label)

Saves the per-example log to JSON for inspection.

    uv run python src/inference/diagnose_liquid_outputs.py
    uv run python src/inference/diagnose_liquid_outputs.py --fold 1 --per-class 80
    uv run python src/inference/diagnose_liquid_outputs.py --adapter-dir experiments/liquid_kfold_03
"""
import os
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from data.dataset import load_sidset, stratified_sample
from utils import set_seed, TEXT2LABEL, LABEL_NAMES
from inference.evaluate_liquid_kfold import (
    load_config,
    load_model_for_eval,
    PROMPT_TEMPLATE,
)


def parse_prediction(generated_upper: str) -> int:
    """Same parsing logic as evaluate_liquid_kfold.eval_fold."""
    pred = TEXT2LABEL.get(generated_upper, -1)
    if pred == -1:
        for k, v in TEXT2LABEL.items():
            if k in generated_upper:
                pred = v
                break
        if pred == -1:
            pred = 0  # fallback -> REAL
    return pred


@torch.inference_mode()
def run(model, processor, hf_split, indices):
    device = next(model.parameters()).device
    records = []
    for i, idx in enumerate(indices):
        item = hf_split[int(idx)]
        image = item["image"]
        image = image.convert("RGB") if hasattr(image, "convert") else Image.open(image).convert("RGB")
        image.thumbnail((512, 512), Image.LANCZOS)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEMPLATE},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, tokenize=True,
        ).to(device)

        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        raw = processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        generated_upper = raw.strip().upper()
        pred = parse_prediction(generated_upper)

        records.append({
            "true_label"     : int(item["label"]),
            "true_name"      : LABEL_NAMES[int(item["label"])],
            "raw_output"     : raw,                 # exactly what the model emitted
            "raw_output_repr": repr(raw),           # reveals whitespace / empties
            "parsed_pred"    : pred,
            "parsed_name"    : LABEL_NAMES[pred],
            "exact_match"    : generated_upper in TEXT2LABEL,
        })
        if (i + 1) % 25 == 0:
            print(f"  [diag] {i+1}/{len(indices)}", flush=True)
    return records


def report(records):
    print(f"\n{'='*64}\nRAW OUTPUT DISTRIBUTION PER TRUE CLASS\n{'='*64}")
    by_true = defaultdict(list)
    for r in records:
        by_true[r["true_name"]].append(r)

    for cls in LABEL_NAMES:
        rs = by_true.get(cls, [])
        if not rs:
            continue
        print(f"\n--- TRUE = {cls}  (n={len(rs)}) ---")
        raw_counts = Counter(r["raw_output_repr"] for r in rs)
        for raw_repr, c in raw_counts.most_common():
            print(f"   {c:4d}x  raw={raw_repr}")
        n_exact = sum(r["exact_match"] for r in rs)
        print(f"   exact keyword match: {n_exact}/{len(rs)}")

    print(f"\n{'='*64}\nCONFUSION (true -> parsed pred)\n{'='*64}")
    header = "true\\pred  " + "  ".join(f"{n:>10}" for n in LABEL_NAMES)
    print(header)
    for t in range(3):
        row = [sum(1 for r in records if r["true_label"] == t and r["parsed_pred"] == p) for p in range(3)]
        print(f"{LABEL_NAMES[t]:>9}  " + "  ".join(f"{v:>10}" for v in row))

    n_fallback = sum(1 for r in records if not r["exact_match"])
    print(f"\nOutputs needing fallback/substring (not an exact keyword): "
          f"{n_fallback}/{len(records)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", default="experiments/liquid_kfold_03",
                    help="Root dir with fold_N/lora_adapter/ subdirs")
    ap.add_argument("--fold", type=int, default=1, help="Which fold adapter to probe")
    ap.add_argument("--per-class", type=int, default=80,
                    help="How many held-out examples to sample per class")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["kfold"]["seed"])

    adapter_root = Path(args.adapter_dir)
    print(f"[setup] Adapter root: {adapter_root} | fold {args.fold} | {args.per_class}/class")
    print("[setup] Loading dataset...")
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
    _, val_pos = folds[args.fold - 1]
    val_indices = pool_indices[val_pos]
    val_labels  = np.array(hf_train["label"])[val_indices]

    # balanced subsample of the held-out split
    rng = np.random.default_rng(cfg["kfold"]["seed"])
    sample = []
    for c in range(3):
        cls_idx = val_indices[val_labels == c]
        take = min(args.per_class, len(cls_idx))
        sample.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    rng.shuffle(sample)
    print(f"[setup] Sampled {len(sample)} examples from held-out fold {args.fold}.")

    adapter_dir = adapter_root / f"fold_{args.fold}" / "lora_adapter"
    if not adapter_dir.exists():
        print(f"[error] adapter not found at {adapter_dir}")
        sys.exit(1)
    print(f"[setup] Loading adapter from {adapter_dir}...")
    model, processor = load_model_for_eval(cfg, adapter_dir)
    model.eval()

    records = run(model, processor, hf_train, sample)
    report(records)

    out_path = adapter_root / f"fold_{args.fold}_raw_output_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n[done] Per-example log saved to: {out_path}")


if __name__ == "__main__":
    main()
