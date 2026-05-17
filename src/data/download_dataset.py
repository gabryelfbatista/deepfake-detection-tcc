"""
Download SID_Set from HuggingFace and save to disk.
Run once before any training:
    uv run python src/data/download_dataset.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datasets import load_dataset
from data.dataset import DATASET_ID, DEFAULT_LOCAL, _resolve_path


def download(local_dir: str = DEFAULT_LOCAL):
    path = _resolve_path(local_dir)
    dataset_file = path / "dataset_dict.json"

    if path.is_dir() and dataset_file.exists():
        print(f"[download] Dataset already exists at '{path}'. Nothing to do.")
        return

    print(f"[download] Downloading {DATASET_ID} from HuggingFace...")
    sidset = load_dataset(DATASET_ID)
    path.mkdir(parents=True, exist_ok=True)
    print(f"[download] Saving to '{path}'...")
    sidset.save_to_disk(str(path))
    print(f"[download] Done. Train: {len(sidset['train']):,} | Val: {len(sidset['validation']):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", default=DEFAULT_LOCAL, help="Where to save the dataset")
    args = parser.parse_args()
    download(args.local_path)
