"""Run an optional real VLM review over event evidence frames.

This is deliberately a post-processing command: it consumes an existing
``run-v01-evaluation.py`` directory, so adding a reviewer never regenerates
media.  The deterministic visual-rule score remains the primary score; the
VLM output is attached to each event as auditable semantic evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, "no_json_object"
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"
    return value if isinstance(value, dict) else None, None


def model_device(model: Any, fallback: str) -> Any:
    import torch

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device(fallback)


def load_model(model_id: str, model_path: Path, device: str):
    import torch
    from transformers import AutoProcessor

    if model_id == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True,
        ).to(device).eval()
    else:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.float16, local_files_only=True,
            low_cpu_mem_usage=True, trust_remote_code=True, device_map="auto",
            max_memory={0: "22GiB", "cpu": "80GiB"},
        ).eval()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    return model, processor


def review_one(model: Any, processor: Any, images: list[Any], event: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    prompt = (
        "You are an independent storyboard reviewer. Inspect the ordered evidence frames "
        "for exactly one event in a product advertisement. Return JSON only with keys "
        "event_present (boolean), observed_start_s (number or null), observed_end_s "
        "(number or null), confidence (0 to 1), evidence_frame_ids (array of integers), "
        "failure_reason (string or null). Do not infer an event that is not visually "
        "supported. Do not invent product attributes. Target event: "
        f"{json.dumps(event, ensure_ascii=False)}"
    )
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    input_device = model_device(model, device)
    moved = {key: value.to(input_device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**moved, max_new_tokens=220, do_sample=False, num_beams=1)
    prompt_tokens = moved["input_ids"].shape[1] if "input_ids" in moved else 0
    answer = processor.batch_decode(
        generated[:, prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    prediction, error = extract_json(answer)
    if prediction is None:
        return {"status": "failed", "parse_error": error, "raw_text": answer}
    event_present = prediction.get("event_present")
    if not isinstance(event_present, bool):
        event_present = str(event_present).strip().lower() in {"true", "yes", "1"}
    try:
        confidence = max(0.0, min(1.0, float(prediction.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    frame_ids = prediction.get("evidence_frame_ids", [])
    if not isinstance(frame_ids, list):
        frame_ids = []
    return {
        "status": "succeeded", "event_present": event_present,
        "observed_start_s": prediction.get("observed_start_s"),
        "observed_end_s": prediction.get("observed_end_s"),
        "confidence": round(confidence, 4),
        "evidence_frame_ids": [item for item in frame_ids if isinstance(item, int)],
        "failure_reason": prediction.get("failure_reason"), "raw_text": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen3_vl", "internvl3_5_8b"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    from PIL import Image
    import torch

    run_dir = args.run_dir.resolve(strict=True)
    storyboard = read_json(run_dir / "storyboard/storyboard.json")
    metric_paths = sorted((run_dir / "metrics").glob("*.json"))
    metric_paths = [path for path in metric_paths if path.name != "summary.json"]
    if args.max_videos is not None:
        metric_paths = metric_paths[:max(0, args.max_videos)]
    model, processor = load_model(args.model, args.model_path.resolve(strict=True), args.device)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for metric_path in metric_paths:
        metric = read_json(metric_path)
        video_id = metric_path.stem
        event_dir = run_dir / "evidence" / video_id / "storyboard"
        event_rows: list[dict[str, Any]] = []
        for event in storyboard.get("events", []):
            event_id = str(event.get("id", "event"))
            frame_paths = sorted(event_dir.glob(f"{event_id}_*.png"))
            images = [Image.open(path).convert("RGB") for path in frame_paths]
            if not images:
                event_rows.append({"status": "failed", "parse_error": "missing_event_evidence", "event_id": event_id})
                continue
            torch.cuda.reset_peak_memory_stats(args.device)
            item_started = time.perf_counter()
            result = review_one(model, processor, images, event, args.device)
            result.update({
                "event_id": event_id,
                "target_window_s": [event.get("start_s"), event.get("end_s")],
                "evidence_paths": [str(path) for path in frame_paths],
                "latency_ms": int((time.perf_counter() - item_started) * 1000),
                "peak_vram_mb": int(torch.cuda.max_memory_allocated(args.device) / 1024 / 1024),
            })
            event_rows.append(result)
        rows.append({"video_id": video_id, "events": event_rows})

        # Attach, but do not replace, the deterministic event-level score.
        storyboard_result = metric.get("storyboard", {})
        by_id = {str(item.get("event_id")): item for item in event_rows}
        for event_row in storyboard_result.get("events", []):
            review = by_id.get(str(event_row.get("id")))
            if review is not None:
                event_row["vlm_review"] = review
        storyboard_result["vlm_reviewer"] = args.model
        metric["storyboard"] = storyboard_result
        write_json(metric_path, metric)

    result = {
        "schema_version": "storyboard_vlm_review/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "model_path": str(args.model_path.resolve()),
        "run_dir": str(run_dir), "videos": rows,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
    output = run_dir / "storyboard" / "vlm_review" / f"{args.model}.json"
    write_json(output, result)
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    manifest.setdefault("storyboard_vlm_reviews", []).append({
        "model": args.model, "path": str(output), "status": "succeeded",
    })
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "succeeded", "output": str(output), "videos": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
