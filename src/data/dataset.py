"""
Shared dataset module — SID_Set (saberzl/SID_Set).
Used by both the multimodal SLM and the CNN baseline.
"""
import numpy as np
from pathlib import Path
from datasets import load_dataset, load_from_disk

DATASET_ID    = "saberzl/SID_Set"
LABEL_NAMES   = ["REAL", "SYNTHETIC", "TAMPERED"]
NUM_CLASSES   = 3
DEFAULT_LOCAL = "dataset/sid_set"

# Project root (three levels above src/data/dataset.py)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _resolve_path(local_dir: str) -> Path:
    """Resolve local_dir relative to the project root if not absolute."""
    p = Path(local_dir)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def load_sidset(local_dir: str = DEFAULT_LOCAL):
    """
    Load SID_Set from disk if already downloaded; otherwise download from
    HuggingFace and save under local_dir for future runs.
    """
    path = _resolve_path(local_dir)
    dataset_file = path / "dataset_dict.json"

    if path.is_dir() and dataset_file.exists():
        print(f"[dataset] Loading from disk: '{path}'")
        sidset = load_from_disk(str(path))
    else:
        print(f"[dataset] '{path}' not found — downloading {DATASET_ID} from HuggingFace...")
        sidset = load_dataset(DATASET_ID)
        path.mkdir(parents=True, exist_ok=True)
        print(f"[dataset] Saving to '{path}'...")
        sidset.save_to_disk(str(path))
        print(f"[dataset] Dataset saved.")

    print(f"[dataset] Train: {len(sidset['train']):,} | Val: {len(sidset['validation']):,}")
    return sidset


def stratified_sample(hf_split, n_total: int, seed: int = 42) -> list:
    """
    Return indices via stratified sampling (equal proportion per class).
    Important for keeping F1-Macro reliable on subsets.
    """
    print(f"[dataset] Sampling {n_total:,} stratified examples (seed={seed})...")
    np.random.seed(seed)
    all_labels = np.array(hf_split["label"])
    per_class = {k: np.where(all_labels == k)[0].tolist() for k in range(NUM_CLASSES)}

    n_per = n_total // NUM_CLASSES
    indices = []
    for cls_idx, idxs in per_class.items():
        np.random.shuffle(idxs)
        indices.extend(idxs[:n_per])
        print(f"[dataset]   {LABEL_NAMES[cls_idx]}: {len(idxs):,} available → {n_per:,} selected")

    np.random.shuffle(indices)
    print(f"[dataset] Total sampled: {len(indices):,}")
    return indices


def split_stats(hf_split, name: str):
    """Print the class distribution of a split."""
    labels = [hf_split[i]["label"] for i in range(min(1000, len(hf_split)))]
    print(f"\n[{name}] distribution (sample of {len(labels)}):")
    for i, class_name in enumerate(LABEL_NAMES):
        n = labels.count(i)
        print(f"  {class_name}: {n} ({n/len(labels)*100:.1f}%)")
