"""Run one VLM across the three Phase-2 product samples.

The script is intentionally independent from the FastAPI generation queue. It
loads one VLM per process, produces a strict product/v1 JSON record, and exits
so the next VLM can reuse the same 24 GB card without keeping two language
models resident at the same time.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/phase2-comparison.json"))
parser.add_argument("--model", required=True, choices=("qwen3_vl", "internvl3_5_8b"))
parser.add_argument("--model-path", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("runs/phase2/vlm"))
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--sample-manifest", type=Path, default=None)
args = parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def token_set(value: Any) -> set[str]:
    if isinstance(value, list):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value or "")
    return {item for item in re.findall(r"[\w]+", text.lower()) if len(item) > 1}


def f1(predicted: Any, expected: Any) -> float:
    pred, truth = token_set(predicted), token_set(expected)
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    overlap = len(pred & truth)
    precision, recall = overlap / len(pred), overlap / len(truth)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, "no_json_object"
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"
    return value if isinstance(value, dict) else None, None


def schema_and_metrics(prediction: dict[str, Any] | None, ground_truth: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "product_name", "category", "visual_description",
        "usage_method", "core_selling_points", "target_audience", "usage_scene",
        "pain_point_gain_point", "ingredients_material", "spec_size", "attributes",
        "ocr_text", "ocr_confidence",
    }
    valid = bool(prediction) and required.issubset(prediction)
    attrs = prediction.get("attributes", {}) if isinstance(prediction, dict) else {}
    if not isinstance(attrs, dict):
        attrs = {}
        valid = False
    valid = valid and prediction.get("schema_version") == "product/v1"
    valid = valid and isinstance(prediction.get("core_selling_points"), list)
    valid = valid and isinstance(prediction.get("ocr_text"), list)
    valid = valid and isinstance(attrs.get("colors"), list)
    # A newly supplied product image has no reviewed attribute labels yet.
    # Keep the schema result, but never manufacture an "accuracy" number.
    if ground_truth.get("evaluation_label_source") == "unavailable":
        return {
            "json_schema_pass": bool(valid),
            "attribute_recognition": None,
            "ocr_accuracy": None,
            "ocr_protocol": "unavailable_manual_labels_required",
        }
    fields = ("category", "form_factor", "colors", "material", "finish", "closure", "container_count")
    scores = []
    for field in fields:
        pred_value = attrs.get(field) if field not in {"category"} else prediction.get(field)
        scores.append(f1(pred_value, ground_truth.get(field)))
    expected_ocr = ground_truth.get("ocr_text", [])
    predicted_ocr = prediction.get("ocr_text", []) if isinstance(prediction, dict) else []
    return {
        "json_schema_pass": bool(valid),
        "attribute_recognition": round(sum(scores) / len(scores), 4),
        "ocr_accuracy": round(f1(predicted_ocr, expected_ocr), 4),
        "ocr_protocol": "negative_text_control" if not expected_ocr else "token_f1",
    }


def model_device(model: Any, fallback: str) -> Any:
    import torch

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device(fallback)


def load_model_and_processor(model_id: str, model_path: Path, device: str):
    import torch
    from transformers import AutoProcessor

    if model_id == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True
        ).to(device).eval()
    else:
        from transformers import AutoModelForImageTextToText

        # InternVL3.5-HF is BF16 on the Hub. FP16 is used here because RTX 3090
        # has no native BF16 tensor-core path and FP16 leaves more headroom.
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            max_memory={0: "22GiB", "cpu": "80GiB"},
        ).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    return model, processor


def infer(model: Any, processor: Any, image_path: Path, prompt: str, device: str) -> tuple[str, int]:
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    input_device = model_device(model, device)
    moved = {key: value.to(input_device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**moved, max_new_tokens=700, do_sample=False, num_beams=1)
    input_ids = moved.get("input_ids")
    prompt_tokens = input_ids.shape[1] if input_ids is not None else 0
    answer = processor.batch_decode(generated[:, prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer, prompt_tokens


config = load_json(args.config)
prompt = config["prompts"]["vlm"]
samples = load_json(args.sample_manifest)["samples"] if args.sample_manifest else config["samples"]
args.output.mkdir(parents=True, exist_ok=True)
started_at = datetime.now(timezone.utc).isoformat()
records: list[dict[str, Any]] = []

try:
    import torch

    model, processor = load_model_and_processor(args.model, args.model_path.resolve(strict=True), args.device)
    for sample in samples:
        image_path = Path(sample["source"]).resolve()
        item_started = time.perf_counter()
        if not image_path.is_file():
            records.append({"sample_id": sample["id"], "status": "failed", "error": f"missing_input:{image_path}"})
            continue
        try:
            torch.cuda.reset_peak_memory_stats(args.device)
            raw_text, _ = infer(model, processor, image_path, prompt, args.device)
            prediction, parse_error = extract_json(raw_text)
            metrics = schema_and_metrics(prediction, sample["ground_truth"])
            records.append({
                "sample_id": sample["id"],
                "status": "succeeded" if parse_error is None else "failed",
                "model": args.model,
                "model_path": str(args.model_path),
                "input": str(image_path),
                "raw_text": raw_text,
                "prediction": prediction,
                "parse_error": parse_error,
                "metrics": metrics,
                "latency_ms": int((time.perf_counter() - item_started) * 1000),
                "peak_vram_mb": int(torch.cuda.max_memory_allocated(args.device) / 1024 / 1024),
            })
        except Exception as exc:  # deployment/runtime errors are recorded per sample
            records.append({
                "sample_id": sample["id"], "status": "failed", "model": args.model,
                "error": repr(exc), "latency_ms": int((time.perf_counter() - item_started) * 1000),
            })
finally:
    summary = {
        "schema_version": "metacut.phase2_vlm/v1",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_path": str(args.model_path),
        "records": records,
        "success_rate": round(sum(item.get("status") == "succeeded" for item in records) / max(1, len(records)), 4),
    }
    target = args.output / f"{args.model}.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
