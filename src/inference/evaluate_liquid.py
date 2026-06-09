"""
Inference — evaluate LFM2.5-VL-450M LoRA adapter (experiment_03) on SID_Set.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

from data.dataset import load_sidset, stratified_sample
from utils import compute_metrics, save_results, save_predictions, set_seed, TEXT2LABEL
from train.experiment_03_liquid_vl import load_config, PROMPT_TEMPLATE


def load_model(cfg):
    lora_dir = Path(cfg["output"]["dir"]) / "lora_adapter"
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {lora_dir}")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["quant_type"],
        bnb_4bit_use_double_quant=cfg["quantization"]["double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        cfg["model"]["id"],
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, lora_dir)
    processor = AutoProcessor.from_pretrained(lora_dir)
    print(f"[inference] LoRA adapter loaded: {lora_dir}")
    return model, processor


@torch.inference_mode()
def evaluate(cfg):
    set_seed(42)
    model, processor = load_model(cfg)
    model.eval()

    sidset = load_sidset(cfg["dataset"].get("local_path", "dataset/sid_set"))
    val_split = sidset["validation"]

    val_indices = (
        stratified_sample(val_split, cfg["dataset"]["max_val"])
        if cfg["dataset"]["max_val"] else list(range(len(val_split)))
    )

    print(f"\n[eval] Running inference on {len(val_indices):,} examples...")
    y_true, y_pred, raw_outputs = [], [], []

    for i, idx in enumerate(val_indices):
        if i % 500 == 0:
            print(f"[eval]   {i:,}/{len(val_indices):,} ({100*i/len(val_indices):.1f}%)")

        item = val_split[idx]
        image = item["image"].convert("RGB") if hasattr(item["image"], "convert") else Image.open(item["image"]).convert("RGB")

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
        ).to(model.device)

        out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        raw = processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        generated = raw.strip().upper()

        pred_label = TEXT2LABEL.get(generated, -1)
        if pred_label == -1:
            # substring OR prefix match (handles truncated tokens like "SYNTHET")
            for k in TEXT2LABEL:
                if k in generated or (generated and k.startswith(generated)):
                    pred_label = TEXT2LABEL[k]
                    break
            if pred_label == -1:
                pred_label = 0

        y_true.append(item["label"])
        y_pred.append(pred_label)
        raw_outputs.append(raw)

    metrics = compute_metrics(y_true, y_pred)
    save_predictions(cfg["output"]["dir"], "val", y_true, y_pred, raw_outputs)
    save_results(metrics, cfg["output"]["dir"], "SLM_LFM25VL_QLoRA_exp03")
    return metrics


if __name__ == "__main__":
    cfg = load_config()
    evaluate(cfg)
