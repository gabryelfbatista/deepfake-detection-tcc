"""
Zero-shot (NO fine-tuning) evaluation of Qwen3-VL-2B-Instruct on SID_Set.

Companion to evaluate_liquid_zeroshot.py — same ablation, the medium-size SLM.
Loads the *base* Qwen3-VL-2B (4-bit NF4, no LoRA adapter) and runs the exact same
generative classification protocol used by evaluate_slm_kfold.py: same
PROMPT_TEMPLATE, same min/max pixels, same TEXT2LABEL mapping with fallback. The
point is to measure what the model does *before* task adaptation, isolating the
gain attributable to QLoRA fine-tuning.

Besides the usual metrics, it reports the **valid-output rate**: how often the
base model emitted exactly one of {REAL, SYNTHETIC, TAMPERED} without needing the
substring/prefix fallback or the default-to-REAL escape.

Single GPU — no DDP. Run on the VM (or locally; 2B in 4-bit fits in ~8GB):

    uv run python src/inference/evaluate_qwen_zeroshot.py
    uv run python src/inference/evaluate_qwen_zeroshot.py --n-samples 3000
    uv run python src/inference/evaluate_qwen_zeroshot.py --split train --n-samples 1500
"""
import os
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml
import torch
from PIL import Image

from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from data.dataset import load_sidset, stratified_sample
from utils import compute_metrics, save_results, save_predictions, set_seed, TEXT2LABEL

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


def load_base_model(cfg):
    """Load the base Qwen3-VL-2B (4-bit NF4) WITHOUT any LoRA adapter."""
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["quant_type"],
        bnb_4bit_use_double_quant=cfg["quantization"]["double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model"]["id"],
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()

    min_px = cfg["model"].get("min_pixels", 3136)
    max_px = cfg["model"].get("max_pixels", 200704)
    processor = AutoProcessor.from_pretrained(
        cfg["model"]["id"], min_pixels=min_px, max_pixels=max_px
    )
    processor.tokenizer.padding_side = "left"
    print(f"[zero-shot] Base model loaded (NO LoRA adapter): {cfg['model']['id']}", flush=True)
    return model, processor


def classify_output(raw: str):
    """
    Map a raw generation to (label, kind) where kind is one of:
      'exact'   — generated text is exactly one keyword (valid output);
      'fuzzy'   — matched only via substring/prefix fallback;
      'default' — no match at all, defaulted to REAL (0).
    """
    generated = raw.strip().upper()

    if generated in TEXT2LABEL:
        return TEXT2LABEL[generated], "exact"

    for k, v in TEXT2LABEL.items():
        if k in generated or (generated and k.startswith(generated)):
            return v, "fuzzy"

    return 0, "default"


def _checkpoint_path(output_dir: str) -> Path:
    return Path(output_dir) / "checkpoint_zero_shot.json"


def _run_signature(cfg, n_samples: int, split_name: str) -> dict:
    """Identifies a run; a checkpoint is only resumed if its signature matches."""
    return {
        "model_id" : cfg["model"]["id"],
        "split"    : split_name,
        "n_samples": n_samples,
        "seed"     : cfg["kfold"]["seed"],
    }


def load_checkpoint(output_dir: str, signature: dict):
    """Return (y_true, y_pred, raw_outputs, kinds) to resume, or None to start fresh."""
    path = _checkpoint_path(output_dir)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            ckpt = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[checkpoint] could not read {path} ({e}); starting fresh.", flush=True)
        return None

    if ckpt.get("signature") != signature:
        print(
            f"[checkpoint] found {path} but its parameters differ from this run "
            f"(saved={ckpt.get('signature')}, now={signature}); ignoring it and starting fresh.",
            flush=True,
        )
        return None

    y_true = ckpt["y_true"]
    print(f"[checkpoint] resuming from {path}: {len(y_true):,} examples already done.", flush=True)
    return y_true, ckpt["y_pred"], ckpt["raw_outputs"], ckpt["kinds"]


def save_checkpoint(output_dir, signature, y_true, y_pred, raw_outputs, kinds):
    """Atomically persist progress so an interrupted run can resume."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(output_dir)
    tmp = path.with_suffix(".json.tmp")

    payload = {
        "signature"  : signature,
        "done"       : len(y_pred),
        "y_true"     : [int(t) for t in y_true],
        "y_pred"     : [int(p) for p in y_pred],
        "raw_outputs": list(raw_outputs),
        "kinds"      : kinds,
    }
    with open(tmp, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)  # atomic rename
    print(f"[checkpoint] saved progress: {len(y_pred):,} examples -> {path}", flush=True)


@torch.inference_mode()
def evaluate(cfg, n_samples: int, split_name: str, output_dir: str, checkpoint_every: int = 5000):
    set_seed(cfg["kfold"]["seed"])
    model, processor = load_base_model(cfg)
    device = next(model.parameters()).device

    sidset = load_sidset(cfg["dataset"].get("local_path", "dataset/sid_set"))
    split = sidset[split_name]

    indices = (
        stratified_sample(split, n_samples, seed=cfg["kfold"]["seed"])
        if n_samples else list(range(len(split)))
    )

    signature = _run_signature(cfg, n_samples, split_name)
    resumed = load_checkpoint(output_dir, signature)
    if resumed is not None:
        y_true, y_pred, raw_outputs, kinds = resumed
    else:
        y_true, y_pred, raw_outputs = [], [], []
        kinds = {"exact": 0, "fuzzy": 0, "default": 0}

    start = len(y_pred)  # resume point (deterministic indices => safe to skip the done prefix)
    print(
        f"\n[zero-shot] Running inference on {len(indices):,} '{split_name}' examples "
        f"(starting at {start:,})...",
        flush=True,
    )

    for i in range(start, len(indices)):
        idx = indices[i]
        if i % 500 == 0 and i > start:
            print(f"[zero-shot]   {i:,}/{len(indices):,} ({100*i/len(indices):.1f}%)", flush=True)

        item = split[int(idx)]
        image = item["image"]
        image = image.convert("RGB") if hasattr(image, "convert") else Image.open(image).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_TEMPLATE},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
        ).to(device)

        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        raw = processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        pred, kind = classify_output(raw)
        kinds[kind] += 1

        y_true.append(item["label"])
        y_pred.append(pred)
        raw_outputs.append(raw)

        if checkpoint_every and len(y_pred) % checkpoint_every == 0:
            save_checkpoint(output_dir, signature, y_true, y_pred, raw_outputs, kinds)

    n = len(indices)
    valid_rate = kinds["exact"] / n if n else 0.0

    print(f"\n{'='*55}")
    print("Output validity (zero-shot, base model):")
    print(f"  exact keyword  : {kinds['exact']:,}/{n:,} ({100*kinds['exact']/n:.1f}%)")
    print(f"  fuzzy fallback : {kinds['fuzzy']:,}/{n:,} ({100*kinds['fuzzy']/n:.1f}%)")
    print(f"  defaulted REAL : {kinds['default']:,}/{n:,} ({100*kinds['default']/n:.1f}%)")
    print(f"  => valid-output rate: {100*valid_rate:.1f}%  (fine-tuned model: 100%)")
    print(f"{'='*55}")

    metrics = compute_metrics(y_true, y_pred)
    # Persist the validity breakdown alongside the standard metrics.
    metrics["valid_output_rate"] = valid_rate
    metrics["output_breakdown"] = kinds
    metrics["n_eval"] = n
    metrics["split"] = split_name

    save_predictions(output_dir, "zero_shot", y_true, y_pred, raw_outputs)
    save_results(metrics, output_dir, "SLM_QWEN3VL2B_ZEROSHOT_no_finetune")

    # Run finished cleanly: drop the checkpoint so the next run starts fresh.
    ckpt = _checkpoint_path(output_dir)
    if ckpt.exists():
        ckpt.unlink()
        print(f"[checkpoint] run complete; removed {ckpt}", flush=True)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-samples", type=int, default=1500,
        help="Stratified sample size (balanced across the 3 classes). 0 = full split.",
    )
    parser.add_argument(
        "--split", default="validation", choices=["validation", "train"],
        help="Which SID_Set split to sample from (default: validation).",
    )
    parser.add_argument(
        "--output-dir", default="experiments/qwen_zeroshot",
        help="Where to write predictions/metrics (kept separate from the fine-tuned run).",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=5000,
        help="Save resumable progress every N examples (0 disables). "
             "A matching interrupted run auto-resumes from the latest checkpoint.",
    )
    args = parser.parse_args()

    cfg = load_config()
    evaluate(
        cfg,
        n_samples=args.n_samples,
        split_name=args.split,
        output_dir=args.output_dir,
        checkpoint_every=args.checkpoint_every,
    )
