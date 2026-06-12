"""
Zero-shot (NO fine-tuning) evaluation of LFM2.5-VL-450M on SID_Set.

Ablation for the thesis: loads the *base* LFM2.5-VL-450M (4-bit NF4, no LoRA
adapter) and runs the exact same generative classification protocol used by
evaluate_liquid_kfold.py — same PROMPT_TEMPLATE, same 512px thumbnail, same
TEXT2LABEL mapping with fallback. The point is to measure what the model does
*before* any task adaptation, so the gain attributable to QLoRA fine-tuning can
be isolated (format conditioning + readout of the frozen visual features).

Besides the usual metrics, it reports the **valid-output rate**: how often the
base model emitted exactly one of {REAL, SYNTHETIC, TAMPERED} without needing the
substring/prefix fallback or the default-to-REAL escape. The fine-tuned model
hits 100% valid (no fallback); the contrast is the headline of this ablation.

Single GPU — no DDP. Run on the VM:

    uv run python src/inference/evaluate_liquid_zeroshot.py
    uv run python src/inference/evaluate_liquid_zeroshot.py --n-samples 3000
    uv run python src/inference/evaluate_liquid_zeroshot.py --split train --n-samples 1500
"""
import os
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from data.dataset import load_sidset, stratified_sample
from utils import compute_metrics, save_results, save_predictions, set_seed, TEXT2LABEL
from train.experiment_03_liquid_vl import load_config, PROMPT_TEMPLATE


def load_base_model(cfg):
    """Load the base LFM2.5-VL-450M (4-bit NF4) WITHOUT any LoRA adapter."""
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["quant_type"],
        bnb_4bit_use_double_quant=cfg["quantization"]["double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["model.vision_tower", "lm_head"],
    )
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model"]["id"],
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(cfg["model"]["id"])
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


@torch.inference_mode()
def evaluate(cfg, n_samples: int, split_name: str, output_dir: str):
    set_seed(cfg["kfold"]["seed"])
    model, processor = load_base_model(cfg)
    device = next(model.parameters()).device

    sidset = load_sidset(cfg["dataset"].get("local_path", "dataset/sid_set"))
    split = sidset[split_name]

    indices = (
        stratified_sample(split, n_samples, seed=cfg["kfold"]["seed"])
        if n_samples else list(range(len(split)))
    )

    print(f"\n[zero-shot] Running inference on {len(indices):,} '{split_name}' examples...", flush=True)
    y_true, y_pred, raw_outputs = [], [], []
    kinds = {"exact": 0, "fuzzy": 0, "default": 0}

    for i, idx in enumerate(indices):
        if i % 500 == 0 and i > 0:
            print(f"[zero-shot]   {i:,}/{len(indices):,} ({100*i/len(indices):.1f}%)", flush=True)

        item = split[int(idx)]
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

        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        raw = processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        pred, kind = classify_output(raw)
        kinds[kind] += 1

        y_true.append(item["label"])
        y_pred.append(pred)
        raw_outputs.append(raw)

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
    save_results(metrics, output_dir, "SLM_LFM25VL_ZEROSHOT_no_finetune")
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
        "--output-dir", default="experiments/liquid_zeroshot",
        help="Where to write predictions/metrics (kept separate from the fine-tuned run).",
    )
    args = parser.parse_args()

    cfg = load_config()
    evaluate(cfg, n_samples=args.n_samples, split_name=args.split, output_dir=args.output_dir)
