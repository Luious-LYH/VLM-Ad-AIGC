"""Run the VLM-Ad-AIGC v0.1 evidence pipeline on existing media.

This command intentionally separates expensive generation from evaluation. It
reuses the four curated sample-02 videos, computes real product masks and
tracking masks, runs local DINOv2/CLIP when the snapshots are available, and
writes a complete run directory containing product/intent/storyboard JSON,
mask overlays, evidence frames, metrics and a report.  The same command can
later be pointed at freshly generated keyframes/videos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/phase2-comparison.json"
DEFAULT_EVAL_CONFIG = PROJECT_ROOT / "configs/v0.1/evaluation.json"
DEFAULT_SAMPLE_ROOT = PROJECT_ROOT / "samples/example"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs/v0.1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        return (root / path).resolve()
    return path.resolve()


def hardware_snapshot() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        return {"nvidia_smi": output.splitlines()}
    except Exception as exc:
        return {"nvidia_smi": [], "error": repr(exc)}


def default_storyboard(duration: float = 4.0417) -> dict[str, Any]:
    return {
        "schema_version": "storyboard/v1",
        "duration_seconds": duration,
        "events": [
            {
                "id": "event_01", "start_s": 0.0, "end_s": 0.8,
                "camera_motion": "macro_push_in", "subject_action": "product_static",
                "lighting_action": "soft_highlight", "composition": "centered hero product",
                "expected_visual_evidence": ["product_scale_increases", "product_center_stable"],
                "forbidden_changes": ["duplicate_product", "shape_mutation"],
            },
            {
                "id": "event_02", "start_s": 0.8, "end_s": 2.1,
                "camera_motion": "orbit", "subject_action": "product_static",
                "lighting_action": "controlled_rim_light", "composition": "stable horizon",
                "expected_visual_evidence": ["continuous_horizontal_parallax", "product_center_stable"],
                "forbidden_changes": ["hard_cut", "product_disappearance"],
            },
            {
                "id": "event_03", "start_s": 2.1, "end_s": 3.2,
                "camera_motion": "lighting_sweep", "subject_action": "product_static",
                "lighting_action": "warm_caustic_sweep", "composition": "close hero detail",
                "expected_visual_evidence": ["foreground_brightness_changes"],
                "forbidden_changes": ["label_flicker", "material_morph"],
            },
            {
                "id": "event_04", "start_s": 3.2, "end_s": duration,
                "camera_motion": "hero_hold", "subject_action": "product_static",
                "lighting_action": "steady_soft_light", "composition": "final centered hero",
                "expected_visual_evidence": ["late_motion_settles", "product_remains_visible"],
                "forbidden_changes": ["scene_cut", "new_object"],
            },
        ],
    }


def load_video(path: Path) -> tuple[list[Image.Image], dict[str, Any]]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for video evaluation; install it in the server runtime") from exc
    container = av.open(str(path))
    stream = container.streams.video[0]
    frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    probe = {"frame_count": len(frames), "fps": fps, "width": stream.width, "height": stream.height}
    container.close()
    return frames, probe


def legacy_mask(image: Image.Image) -> np.ndarray:
    from app.evaluation.segmentation import ellipse_mask

    return ellipse_mask((image.width, image.height))


class EmbeddingModels:
    """Load local DINOv2 and CLIP snapshots lazily and keep failure explicit."""

    def __init__(self, dino_path: Path, clip_path: Path):
        self.dino_path = dino_path
        self.clip_path = clip_path
        self.dino = None
        self.dino_processor = None
        self.clip = None
        self.clip_processor = None
        self.errors: dict[str, str] = {}
        if dino_path.is_dir():
            try:
                from transformers import AutoImageProcessor, AutoModel

                self.dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
                self.dino = AutoModel.from_pretrained(dino_path, local_files_only=True).eval().to("cpu")
            except Exception as exc:
                self.errors["dino"] = repr(exc)
        else:
            self.errors["dino"] = f"missing_snapshot:{dino_path}"
        if clip_path.is_dir():
            try:
                from transformers import CLIPModel, CLIPProcessor

                self.clip_processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
                self.clip = CLIPModel.from_pretrained(clip_path, local_files_only=True).eval().to("cpu")
            except Exception as exc:
                self.errors["clip"] = repr(exc)
        else:
            self.errors["clip"] = f"missing_snapshot:{clip_path}"

    @staticmethod
    def _crop(image: Image.Image, mask: np.ndarray | None) -> Image.Image:
        from app.evaluation.segmentation import masked_crop

        return masked_crop(image, mask)

    def dino_embedding(self, image: Image.Image, mask: np.ndarray | None) -> Any | None:
        if self.dino is None:
            return None
        import torch

        inputs = self.dino_processor(images=self._crop(image, mask), return_tensors="pt")
        with torch.inference_mode():
            outputs = self.dino(**inputs)
        vector = outputs.pooler_output[0] if getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state[:, 0][0]
        return torch.nn.functional.normalize(vector.float(), dim=0).cpu()

    def clip_text_embedding(self, text: str) -> Any | None:
        if self.clip is None:
            return None
        import torch

        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        with torch.inference_mode():
            return torch.nn.functional.normalize(self.clip.get_text_features(**inputs)[0].float(), dim=0).cpu()

    def clip_image_embedding(self, image: Image.Image, mask: np.ndarray | None) -> Any | None:
        if self.clip is None:
            return None
        import torch

        inputs = self.clip_processor(images=[self._crop(image, mask)], return_tensors="pt")
        with torch.inference_mode():
            return torch.nn.functional.normalize(self.clip.get_image_features(**inputs)[0].float(), dim=0).cpu()

    @staticmethod
    def cosine(left: Any | None, right: Any | None) -> float | None:
        if left is None or right is None:
            return None
        import torch

        return float(torch.nn.functional.cosine_similarity(left[None], right[None]).item())


def dino_stats(anchor: Any | None, vectors: list[Any | None]) -> dict[str, Any]:
    values = [EmbeddingModels.cosine(anchor, vector) for vector in vectors]
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"status": "unavailable", "values": [], "reason": "dino_snapshot_or_mask_missing"}
    ordered = sorted(values)
    p10_index = min(len(ordered) - 1, max(0, int(np.floor((len(ordered) - 1) * 0.10))))
    return {
        "status": "succeeded", "values": [round(v, 5) for v in values],
        "mean": round(statistics.mean(values), 5), "min": round(min(values), 5),
        "p10": round(ordered[p10_index], 5), "std": round(statistics.pstdev(values), 5) if len(values) > 1 else 0.0,
    }


def temporal_metrics(frames: list[Image.Image], masks: list[np.ndarray | None], probe: dict[str, Any]) -> dict[str, Any]:
    if len(frames) < 2:
        return {"status": "unavailable", "reason": "fewer_than_two_frames", "probe": probe}
    arrays = [np.asarray(frame.resize((96, 128)), dtype=np.float32) / 255.0 for frame in frames]
    deltas = np.asarray([float(np.abs(right - left).mean()) for left, right in zip(arrays, arrays[1:])])
    jitter = float(np.std(deltas)) if len(deltas) > 1 else 0.0
    mean_delta = float(np.mean(deltas))
    # These are generic quality diagnostics, not official VBench values.
    stability = max(0.0, min(1.0, 1.0 - jitter * 25.0))
    smoothness = max(0.0, min(1.0, 1.0 - jitter * 30.0))
    dynamic = max(0.0, min(1.0, mean_delta / 0.045))
    mask_areas = [float(mask.mean()) for mask in masks if mask is not None]
    return {
        "status": "succeeded", "mean_frame_delta": round(mean_delta, 6),
        "frame_delta_jitter": round(jitter, 6), "temporal_stability": round(stability, 4),
        "motion_smoothness": round(smoothness, 4), "dynamic_degree": round(dynamic, 4),
        "mask_area_mean": round(float(np.mean(mask_areas)), 5) if mask_areas else None,
        "mask_area_std": round(float(np.std(mask_areas)), 5) if len(mask_areas) > 1 else 0.0,
        "probe": probe,
        "method": "uniform sampled frame deltas; diagnostic heuristic, not official VBench",
    }


def run_vlm_product(config_path: Path, model: str, model_path: Path, input_path: Path, out_dir: Path) -> dict[str, Any]:
    """Reuse the tested strict product/v1 runner and materialize its output."""
    manifest = out_dir / "vlm-sample-manifest.json"
    write_json(manifest, {"samples": [{
        "id": "perfume_flacon", "label": "Example perfume flacon", "source": str(input_path),
        "ground_truth": {"category": "perfume", "form_factor": "single rectangular glass flacon",
                          "colors": ["amber", "champagne gold"], "material": "glass and metal",
                          "finish": "glossy glass with brushed metal", "closure": "fitted cap",
                          "container_count": 1, "ocr_text": []},
    }]})
    target = out_dir / "vlm"
    command = [sys.executable, str(PROJECT_ROOT / "scripts/run-phase2-vlm.py"),
               "--config", str(config_path), "--model", model, "--model-path", str(model_path),
               "--output", str(target), "--device", os.environ.get("AIGC_VLM_DEVICE", "cuda:0"),
               "--sample-manifest", str(manifest)]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True)
    record_path = target / f"{model}.json"
    if completed.returncode != 0 or not record_path.is_file():
        return {"status": "failed", "model": model, "error": (completed.stderr or completed.stdout)[-4000:],
                "latency_ms": int((time.perf_counter() - started) * 1000)}
    summary = read_json(record_path)
    record = summary.get("records", [{}])[0]
    product = record.get("prediction")
    write_json(out_dir / "product.json", {
        "schema_version": "product/v1", "status": record.get("status", "failed"),
        "model": model, "model_path": str(model_path), "source_record": str(record_path),
        "prediction": product, "metrics": record.get("metrics"), "raw_text": record.get("raw_text"),
        "latency_ms": record.get("latency_ms"), "peak_vram_mb": record.get("peak_vram_mb"),
    })
    return record


def save_evidence(frames: list[Image.Image], masks: list[np.ndarray | None], indices: list[int], out_dir: Path) -> list[str]:
    from app.evaluation.segmentation import overlay_image

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for rank, (index, mask) in enumerate(zip(indices, masks)):
        frame_path = out_dir / f"frame-{index:04d}.png"
        overlay_path = out_dir / f"frame-{index:04d}.overlay.png"
        frames[rank].save(frame_path)
        overlay_image(frames[rank], mask).save(overlay_path)
        paths.extend([str(frame_path), str(overlay_path)])
    return paths


def markdown_report(manifest: dict[str, Any], metrics: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# VLM-Ad-AIGC v0.1 实验报告", "",
        f"Run: `{manifest['run_id']}`  ·  created: `{manifest['created_at']}`", "",
        "本报告复用已有四个唯一视频，先验证真实商品分割、视频跟踪、商品一致性和事件级 Storyboard；没有重新生成视频。",
        "", "## 运行清单", "",
        f"- 输入: `{manifest['input']['path']}`", f"- 输入 SHA256: `{manifest['input']['sha256']}`",
        f"- Prompt hash: `{manifest['prompt']['sha256']}`", f"- 硬件: `{manifest['hardware'].get('nvidia_smi', [])}`",
        "", "## 视频结果", "",
        "| 组合 | 分割/跟踪 | 商品一致性 mean/min/p10/std | 输入-关键帧 | CLIP-I | CLIP-V | Storyboard | 稳定性 | 证据 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        product = item.get("product_consistency", {})
        dino = product.get("video", {})
        temporal = item.get("temporal", {})
        lines.append(
            f"| {item['video_id']} | {item['tracking'].get('status')} / {item['tracking'].get('product_appearance_rate')} | "
            f"{dino.get('mean', 'unavailable')} / {dino.get('min', 'unavailable')} / {dino.get('p10', 'unavailable')} / {dino.get('std', 'unavailable')} | "
            f"{product.get('keyframe', {}).get('mean', 'unavailable')} | {item.get('text_match', {}).get('image', 'unavailable')} | "
            f"{item.get('text_match', {}).get('video', 'unavailable')} | {item.get('storyboard', {}).get('score', 'unavailable')} | "
            f"{temporal.get('temporal_stability', 'unavailable')} | `{item.get('evidence_dir')}` |"
        )
    lines += ["", "## 解释", "", "- `商品一致性` 只对真实 mask 的商品 crop 计算 DINOv2，不使用旧中心椭圆作为默认值。",
              "- `Storyboard` 是逐事件视觉规则分数，并保留事件窗口、证据帧和违规项；不是人工审美评分，也不冒充官方 VBench。",
              "- SAM3.1/GroundingDINO+SAM2 若未部署，manifest 会保留显式回退原因；回退 mask 不是 SAM 结果。", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_SAMPLE_ROOT / "input.png")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--keyframe-root", type=Path, default=DEFAULT_SAMPLE_ROOT / "keyframes")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_SAMPLE_ROOT / "videos")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dino-path", type=Path, default=None)
    parser.add_argument("--clip-path", type=Path, default=None)
    parser.add_argument("--vlm-model", choices=("none", "qwen3_vl", "internvl3_5_8b"), default="none")
    parser.add_argument("--vlm-path", type=Path, default=None)
    parser.add_argument("--max-video-samples", type=int, default=None)
    args = parser.parse_args()

    # The server run uses /public/lyh/projects/CV_projects/models; local users
    # can override these paths without editing the code.
    model_root = Path(os.environ.get("AIGC_MODEL_ROOT", PROJECT_ROOT / "models"))
    dino_path = (args.dino_path or model_root / "dinov2-small").resolve()
    clip_path = (args.clip_path or model_root / "clip-vit-base-patch32").resolve()
    config = read_json(args.config.resolve())
    evaluation_config = read_json(args.evaluation_config.resolve()) if args.evaluation_config.resolve().is_file() else {}
    prompt = args.prompt or config["prompts"]["video"]
    input_path = resolve_path(args.input)
    keyframe_root = resolve_path(args.keyframe_root)
    video_root = resolve_path(args.video_root)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    from app.evaluation.segmentation import SegmentResult, masked_crop, save_mask_bundle, segment_product, track_video_masks
    from app.evaluation.storyboard import evaluate_storyboard

    input_image = Image.open(input_path).convert("RGBA")
    input_segment = segment_product(input_image)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_copy = input_dir / input_path.name
    shutil.copy2(input_path, input_copy)
    input_mask_meta = save_mask_bundle(input_image, input_segment, run_dir / "masks" / "input", "input")

    product_record: dict[str, Any] = {"status": "unavailable", "reason": "vlm_not_requested"}
    if args.vlm_model != "none":
        if args.vlm_path is None:
            raise ValueError("--vlm-path is required when --vlm-model is selected")
        product_record = run_vlm_product(args.config.resolve(), args.vlm_model, args.vlm_path.resolve(), input_path, run_dir / "understanding")
    if args.vlm_model == "none":
        write_json(run_dir / "understanding" / "product.json", product_record)
    intent = {
        "schema_version": "intent/v1", "user_prompt": args.prompt or "",
        "generation_prompt": prompt, "rewrite_enabled": False,
        "rewrite_model": None, "prompt_version": "v0.1-canonical",
        "prompt_sha256": prompt_hash(prompt),
        "constraints": {"no_new_brand": True, "no_extra_container": True, "preserve_identity": True},
    }
    write_json(run_dir / "understanding" / "intent.json", intent)
    storyboard = default_storyboard(float(config.get("video_spec", {}).get("duration_seconds", 4.0417)))
    configured_storyboard = evaluation_config.get("storyboard", {})
    if isinstance(configured_storyboard.get("events"), list) and configured_storyboard["events"]:
        storyboard["events"] = configured_storyboard["events"]
    storyboard["prompt_sha256"] = prompt_hash(prompt)
    write_json(run_dir / "storyboard" / "storyboard.json", storyboard)

    embedder = EmbeddingModels(dino_path, clip_path)
    input_dino = embedder.dino_embedding(input_image, input_segment.mask)
    caption = "premium perfume, single rectangular amber glass flacon, champagne gold cap"
    text_vector = embedder.clip_text_embedding(caption)
    metrics: list[dict[str, Any]] = []
    media_manifest: list[dict[str, Any]] = []
    keyframe_paths = sorted(keyframe_root.glob("*.png"))
    video_paths = sorted(video_root.glob("*.mp4"))
    keyframe_by_prefix = {path.stem.split("__")[0]: path for path in keyframe_paths}
    for video_path in video_paths:
        video_id = video_path.stem
        image_model = "sdxl_ip_adapter" if video_id.startswith("sdxl_ip_adapter") else "flux2_klein"
        keyframe_path = keyframe_by_prefix.get(image_model)
        if keyframe_path is None:
            continue
        keyframe_image = Image.open(keyframe_path).convert("RGBA")
        keyframe_seg = segment_product(keyframe_image)
        keyframe_dir = run_dir / "masks" / "keyframes" / image_model
        keyframe_meta = save_mask_bundle(keyframe_image, keyframe_seg, keyframe_dir, "keyframe")
        video_frames, probe = load_video(video_path)
        sample_limit = args.max_video_samples or int(evaluation_config.get("video_sampling", {}).get("max_samples", 16))
        sampled_frames, sampled_masks, tracking = track_video_masks(video_frames, max_samples=sample_limit)
        evidence_dir = run_dir / "evidence" / video_id
        evidence_dir.mkdir(parents=True, exist_ok=True)
        source_indices = np.linspace(0, len(video_frames) - 1, len(sampled_frames)).round().astype(int).tolist() if video_frames else []
        evidence_paths: list[str] = []
        for rank, (frame, mask, source_index) in enumerate(zip(sampled_frames, sampled_masks, source_indices)):
            frame_path = evidence_dir / f"frame-{source_index:04d}.png"
            overlay_path = evidence_dir / f"frame-{source_index:04d}.overlay.png"
            frame.save(frame_path)
            from app.evaluation.segmentation import overlay_image

            overlay_image(frame, mask).save(overlay_path)
            mask_dir = run_dir / "masks" / "videos" / video_id
            tracked_result = SegmentResult(
                mask=mask, backend="video_track",
                status="succeeded" if mask is not None else "segmentation_failed",
                confidence=None, bbox=None,
                area_fraction=float(mask.mean()) if mask is not None else None,
                fallback_reason="reused_previous_mask" if tracking.get("tracking_interruptions") else None,
                attempted_backends=["component_track_fallback"],
            )
            save_mask_bundle(frame, tracked_result, mask_dir, f"frame-{source_index:04d}")
            evidence_paths.extend([str(frame_path), str(overlay_path)])
        keyframe_dino = embedder.dino_embedding(keyframe_image, keyframe_seg.mask)
        video_dino = [embedder.dino_embedding(frame, mask) for frame, mask in zip(sampled_frames, sampled_masks)]
        legacy_anchor = embedder.dino_embedding(input_image, legacy_mask(input_image))
        legacy_vectors = [embedder.dino_embedding(frame, legacy_mask(frame)) for frame in sampled_frames]
        product_consistency = {
            "status": "succeeded" if input_segment.mask is not None and keyframe_seg.mask is not None and any(mask is not None for mask in sampled_masks) else "unavailable",
            "reference_mask_backend": input_segment.backend,
            "keyframe_mask_backend": keyframe_seg.backend,
            "video": dino_stats(input_dino, video_dino),
            "keyframe": dino_stats(input_dino, [keyframe_dino]),
            "legacy_ellipse": dino_stats(legacy_anchor, legacy_vectors),
            "tracking_interruptions": tracking.get("tracking_interruptions"),
            "sample_count": tracking.get("sample_count"),
        }
        image_clip = embedder.cosine(text_vector, embedder.clip_image_embedding(keyframe_image, keyframe_seg.mask))
        video_clip_values = [embedder.cosine(text_vector, embedder.clip_image_embedding(frame, mask)) for frame, mask in zip(sampled_frames, sampled_masks)]
        video_clip_values = [float(value) for value in video_clip_values if value is not None]
        sample_times = [float(index) / max(float(probe.get("fps", 24.0)), 1.0) for index in source_indices]
        storyboard_result = evaluate_storyboard(
            sampled_frames, sampled_masks, probe.get("fps", 24.0), storyboard["events"],
            evidence_dir / "storyboard", frame_times=sample_times,
        )
        temporal = temporal_metrics(sampled_frames, sampled_masks, {**probe, "sampled_frame_count": len(sampled_frames)})
        item = {
            "schema_version": "evaluation/v1", "video_id": video_id, "video_path": str(video_path),
            "video_sha256": sha256(video_path), "keyframe_path": str(keyframe_path),
            "tracking": tracking, "segmentation": {"input": input_mask_meta, "keyframe": keyframe_meta},
            "product_consistency": product_consistency,
            "text_match": {"caption": caption, "image": round(image_clip, 5) if image_clip is not None else None,
                            "video": round(statistics.mean(video_clip_values), 5) if video_clip_values else None,
                            "clip_status": "succeeded" if embedder.clip is not None else "unavailable"},
            "storyboard": storyboard_result, "temporal": temporal,
            "engineering": {"probe": probe, "vlm_latency_ms": product_record.get("latency_ms"),
                            "image_latency_ms": None, "video_latency_ms": None, "peak_vram_mb": None,
                            "failure_rate": round(1.0 - tracking.get("product_appearance_rate", 0.0), 4)},
            "evidence_dir": str(evidence_dir), "evidence_paths": evidence_paths,
        }
        write_json(run_dir / "metrics" / f"{video_id}.json", item)
        metrics.append(item)
        media_manifest.append({"video_id": video_id, "video": str(video_path), "keyframe": str(keyframe_path), "mask_dir": str(run_dir / "masks" / "videos" / video_id)})

    manifest = {
        "schema_version": "run_manifest/v1", "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
        "input": {"path": str(input_path), "copied_path": str(input_copy), "sha256": sha256(input_path), "mask": input_mask_meta},
        "prompt": {"user_prompt": args.prompt or "", "generation_prompt": prompt, "sha256": prompt_hash(prompt), "rewrite_enabled": False},
        "models": {"vlm": args.vlm_model, "dino": str(dino_path), "clip": str(clip_path),
                   "image": ["sdxl_ip_adapter", "flux2_klein"], "video": ["ltxv_2b", "wan22_ti2v_5b"]},
        "config": str(args.config.resolve()), "evaluation_config": str(args.evaluation_config.resolve()), "seed": config.get("seed"), "hardware": hardware_snapshot(),
        "embedding_model_errors": embedder.errors, "segmentation_policy": input_segment.metadata(),
        "media": media_manifest, "metrics_count": len(metrics),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "metrics" / "summary.json", {"schema_version": "evaluation/v1", "run_id": run_id, "records": metrics, "embedding_model_errors": embedder.errors})
    markdown_report(manifest, metrics, run_dir / "reports" / "report.md")
    print(json.dumps({"status": "succeeded", "run_dir": str(run_dir), "videos": len(metrics), "dino": embedder.dino is not None, "clip": embedder.clip is not None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Allow ``python scripts/run-v01-evaluation.py`` from the repository root
    # without installing the FastAPI service as a package.
    sys.path.insert(0, str(PROJECT_ROOT / "services" / "aigc-service"))
    main()
