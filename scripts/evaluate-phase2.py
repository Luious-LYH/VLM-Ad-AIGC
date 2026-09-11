"""Evaluate Phase-2 media with reproducible, lightweight metrics.

The evaluator never fabricates missing metrics: DINO/CLIP are reported as
unavailable when their local snapshots are not supplied. Video stability and
storyboard adherence are explicitly labelled heuristics, not human preference
scores.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


parser = argparse.ArgumentParser()
parser.add_argument("--phase2-root", type=Path, default=Path("public/uploads/phase2"))
parser.add_argument("--run-root", type=Path, default=Path("runs/phase2"))
parser.add_argument("--dino-path", type=Path, default=Path("models/dinov2-small"))
parser.add_argument("--clip-path", type=Path, default=Path("/public/lyh/projects/VQA/shared_cache/model_downloads/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"))
args = parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def masked_image(image: Any) -> Any:
    from PIL import Image, ImageDraw

    image = image.convert("RGB")
    # A deterministic center ellipse suppresses the studio background while
    # retaining the complete centered product family in our 3 sample images.
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((int(image.width * 0.08), int(image.height * 0.04), int(image.width * 0.92), int(image.height * 0.98)), fill=255)
    background = Image.new("RGB", image.size, (0, 0, 0))
    return Image.composite(image, background, mask)


def cosine(left: Any, right: Any) -> float:
    import torch

    left = left.float().flatten()
    right = right.float().flatten()
    return float(torch.nn.functional.cosine_similarity(left[None], right[None]).item())


def decode_video(path: Path) -> tuple[list[Any], dict[str, Any]]:
    import av

    container = av.open(str(path))
    stream = container.streams.video[0]
    frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    rate = float(stream.average_rate) if stream.average_rate else 0.0
    return frames, {"frame_count": len(frames), "fps": rate, "width": stream.width, "height": stream.height}


def resolve_video_path(sample_dir: Path, record: dict[str, Any]) -> Path:
    """Prefer a legacy local final.mp4, then use the shared video artifact.

    Phase 2 used to copy the same video into every VLM combination folder.
    New runs keep one canonical video in ``_shared/videos`` and store its
    absolute path in ``record.video.output``.
    """
    local_path = sample_dir / "final.mp4"
    if local_path.is_file():
        return local_path
    shared_path = record.get("video", {}).get("output")
    if isinstance(shared_path, str) and shared_path:
        candidate = Path(shared_path)
        if candidate.is_file():
            return candidate
    return local_path


def video_heuristics(frames: list[Any], probe: dict[str, Any], requested_frames: int) -> dict[str, Any]:
    import numpy as np

    if len(frames) < 2:
        return {"stability": None, "storyboard_adherence": None, "error": "fewer_than_two_frames"}
    arrays = [np.asarray(frame.resize((96, 128)), dtype=np.float32) / 255.0 for frame in frames]
    diffs = [float(np.abs(right - left).mean()) for left, right in zip(arrays, arrays[1:])]
    mean_diff = statistics.mean(diffs)
    jitter = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    stability = max(0.0, min(1.0, 1.0 - jitter * 25.0))
    motion = max(0.0, min(1.0, mean_diff / 0.045))
    duration_ok = 1.0 if abs(len(frames) - requested_frames) <= max(3, requested_frames * 0.05) else 0.0
    storyboard = 0.45 * motion + 0.35 * stability + 0.20 * duration_ok
    return {
        "stability": round(stability, 4),
        "storyboard_adherence": round(storyboard, 4),
        "method": "heuristic: temporal smoothness + nonzero motion + requested frame-count check",
        "mean_frame_delta": round(mean_diff, 6),
        "frame_delta_jitter": round(jitter, 6),
        "motion_presence": round(motion, 4),
        "duration_ok": bool(duration_ok),
        "probe": probe,
    }


class EmbeddingModels:
    def __init__(self, dino_path: Path, clip_path: Path):
        self.dino = None
        self.dino_processor = None
        self.clip = None
        self.clip_processor = None
        if dino_path.is_dir():
            try:
                from transformers import AutoImageProcessor, AutoModel

                self.dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
                self.dino = AutoModel.from_pretrained(dino_path, local_files_only=True).eval()
            except Exception as exc:
                print(f"DINO unavailable: {exc}")
        if clip_path.is_dir():
            try:
                from transformers import CLIPModel, CLIPProcessor

                self.clip_processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
                self.clip = CLIPModel.from_pretrained(clip_path, local_files_only=True).eval()
            except Exception as exc:
                print(f"CLIP unavailable: {exc}")

    def dino_embedding(self, image: Any):
        if self.dino is None:
            return None
        import torch

        inputs = self.dino_processor(images=masked_image(image), return_tensors="pt")
        with torch.inference_mode():
            outputs = self.dino(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output[0].cpu()
        return outputs.last_hidden_state[:, 0][0].cpu()

    def clip_similarity(self, image: Any, text: str) -> float | None:
        if self.clip is None:
            return None
        import torch

        inputs = self.clip_processor(text=[text], images=[image.convert("RGB")], return_tensors="pt", padding=True)
        with torch.inference_mode():
            outputs = self.clip(**inputs)
            image_features = outputs.image_embeds[0]
            text_features = outputs.text_embeds[0]
        return round(cosine(image_features, text_features), 4)


def main() -> None:
    models = EmbeddingModels(args.dino_path.resolve(), args.clip_path.resolve())
    records: list[dict[str, Any]] = []
    for combo_dir in sorted(path for path in args.phase2_root.iterdir() if path.is_dir() and not path.name.startswith("_")):
        combo = load_json(combo_dir / "combo.json") if (combo_dir / "combo.json").is_file() else {"combo_id": combo_dir.name}
        for sample_dir in sorted(combo_dir.glob("sample-*")):
            record_path = sample_dir / "record.json"
            if not record_path.is_file():
                continue
            record = load_json(record_path)
            video_path = resolve_video_path(sample_dir, record)
            if not video_path.is_file():
                metrics = {"status": "failed", "error": "missing_final_video"}
                (sample_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                records.append({"combo_id": combo_dir.name, "sample_id": record.get("sample_id"), **metrics})
                continue
            input_path = sample_dir / "input.png"
            keyframe_path = sample_dir / "keyframe.png"
            try:
                frames, probe = decode_video(video_path)
                requested = int(record.get("spec", {}).get("video", {}).get("num_frames", 97))
                metrics = {"status": "succeeded", "video": video_heuristics(frames, probe, requested)}
                if input_path.is_file() and keyframe_path.is_file():
                    from PIL import Image

                    input_image = Image.open(input_path).convert("RGB")
                    keyframe_image = Image.open(keyframe_path).convert("RGB")
                    sampled = [frames[0], frames[len(frames) // 2], frames[-1]] if frames else []
                    if models.dino is not None:
                        anchor = models.dino_embedding(input_image)
                        key_emb = models.dino_embedding(keyframe_image)
                        video_embs = [models.dino_embedding(frame) for frame in sampled]
                        similarities = [cosine(anchor, emb) for emb in video_embs if emb is not None]
                        metrics["masked_dino_product_similarity"] = round(statistics.mean(similarities), 4) if similarities else None
                        metrics["masked_dino_keyframe_similarity"] = round(cosine(anchor, key_emb), 4) if key_emb is not None else None
                    else:
                        metrics["masked_dino_product_similarity"] = None
                        metrics["masked_dino_keyframe_similarity"] = None
                    gt = record.get("input", {}).get("ground_truth", {})
                    caption = "premium " + str(gt.get("category", "product")) + ", " + str(gt.get("form_factor", "hero product")) + ", colors " + ", ".join(gt.get("colors", []))
                    metrics["clip_text_image_match"] = models.clip_similarity(keyframe_image, caption)
                    metrics["clip_text_video_match"] = round(statistics.mean([models.clip_similarity(frame, caption) or 0.0 for frame in sampled]), 4) if sampled and models.clip is not None else None
                else:
                    metrics["masked_dino_product_similarity"] = None
                    metrics["masked_dino_keyframe_similarity"] = None
                    metrics["clip_text_image_match"] = None
                    metrics["clip_text_video_match"] = None
            except Exception as exc:
                metrics = {"status": "failed", "error": repr(exc)}
            (sample_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append({"combo_id": combo_dir.name, "sample_id": record.get("sample_id"), **metrics})
    summary = {"schema_version": "metacut.phase2_metrics/v1", "evaluated_at": datetime.now(timezone.utc).isoformat(), "records": records}
    (args.run_root / "phase2-metrics.json").parent.mkdir(parents=True, exist_ok=True)
    (args.run_root / "phase2-metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.run_root / "phase2-metrics.json")
    print(json.dumps({"samples": len(records), "successful": sum(r.get("status") == "succeeded" for r in records), "dino": models.dino is not None, "clip": models.clip is not None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
