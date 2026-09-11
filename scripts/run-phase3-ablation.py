"""Generate a reproducible Phase 3 before/after comparison.

The migrated repository ships a curated FLUX.2-klein + LTXV example as the
baseline.  The optional improved request keeps its input, seed and output
specification while asking the HunyuanVideo-1.5 worker for an explicit camera
storyboard and identity/temporal candidate ranking.  No curated sample is
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD = PROJECT_ROOT / "samples/example/records/record_flux.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "public/uploads/phase3/sample-02"
DEFAULT_RUN = PROJECT_ROOT / "runs/phase3/sample-02"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_media_path(value: Any) -> Path | None:
    """Resolve a record path against this checkout without trusting old roots."""
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
        if candidate.is_file():
            return candidate
    return None


def request(base: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    with urlopen(Request(f"{base.rstrip('/')}{path}", data=body, headers=headers), timeout=timeout) as response:
        return json.load(response)


def wait_job(base: str, job_id: str, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = request(base, f"/v1/jobs/{job_id}", timeout=30)
        if job.get("status") in {"succeeded", "failed"}:
            return job
        time.sleep(2)
    raise TimeoutError(f"job timed out: {job_id}")


def url_to_path(url: str) -> Path:
    if not url.startswith("/uploads/"):
        raise ValueError(f"unexpected local worker URL: {url}")
    public_root = (PROJECT_ROOT / "public").resolve()
    path = (public_root / url.removeprefix("/uploads/")).resolve()
    # The service URL is /uploads/local-aigc/...; restore the public/uploads
    # prefix explicitly and reject path traversal.
    path = (public_root / "uploads" / url.removeprefix("/uploads/")).resolve()
    if public_root not in path.parents:
        raise ValueError(f"worker URL escaped public root: {url}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_video(video_path: Path, reference_path: Path, dino_path: Path) -> dict[str, Any]:
    """Use the same lightweight, product-masked metrics as Phase 2."""
    import av
    import numpy as np
    from PIL import Image, ImageDraw

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    width, height = stream.width, stream.height
    frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    container.close()
    arrays = [np.asarray(frame.resize((96, 128)), dtype=np.float32) / 255.0 for frame in frames]
    diffs = [float(np.abs(right - left).mean()) for left, right in zip(arrays, arrays[1:])]
    mean_diff = statistics.mean(diffs) if diffs else 0.0
    jitter = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    stability = max(0.0, min(1.0, 1.0 - jitter * 25.0))
    motion = max(0.0, min(1.0, mean_diff / 0.045))
    storyboard = 0.45 * motion + 0.35 * stability + 0.20 * (1.0 if len(frames) == 97 else 0.0)

    dino_similarity = None
    if dino_path.is_dir() and frames:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
            model = AutoModel.from_pretrained(dino_path, local_files_only=True).eval().to("cpu")

            def mask_product(image: Image.Image) -> Image.Image:
                image = image.convert("RGB")
                mask = Image.new("L", image.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((int(image.width * 0.08), int(image.height * 0.04), int(image.width * 0.92), int(image.height * 0.98)), fill=255)
                return Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask)

            def embed(image: Image.Image):
                inputs = processor(images=mask_product(image), return_tensors="pt")
                with torch.inference_mode():
                    output = model(**inputs)
                vector = output.pooler_output[0] if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0][0]
                return torch.nn.functional.normalize(vector.float(), dim=0)

            anchor = embed(Image.open(reference_path).convert("RGB"))
            selected = [frames[0], frames[len(frames) // 2], frames[-1]]
            dino_similarity = statistics.mean(float(torch.dot(anchor, embed(frame)).item()) for frame in selected)
        except Exception as exc:
            dino_similarity = None
            dino_error = repr(exc)
        else:
            dino_error = None
    else:
        dino_error = "missing_dino_or_frames"
    return {
        "frame_count": len(frames),
        "fps": fps,
        "width": width,
        "height": height,
        "mean_frame_delta": round(mean_diff, 6),
        "frame_delta_jitter": round(jitter, 6),
        "temporal_stability": round(stability, 4),
        "motion_presence": round(motion, 4),
        "storyboard_adherence_heuristic": round(storyboard, 4),
        "masked_dino_product_similarity": round(float(dino_similarity), 4) if dino_similarity is not None else None,
        "dino_error": dino_error,
        "metric_note": "DINO compares a centered product mask in the reference keyframe with the candidate first/middle/last frames; it is an identity proxy, not a human quality score.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--baseline-record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dino-path", type=Path, default=PROJECT_ROOT / "models/dinov2-small")
    parser.add_argument("--request-timeout", type=int, default=2700)
    parser.add_argument("--identity-weight", type=float, default=0.55)
    parser.add_argument("--evaluate-only", action="store_true", help="Recompute metrics for existing Phase 3 videos without launching Hunyuan")
    args = parser.parse_args()

    baseline_record = read_json(args.baseline_record)
    baseline_path = resolve_media_path((baseline_record.get("video") or {}).get("output"))
    baseline_path = baseline_path or resolve_media_path(args.baseline_record.parent / "final.mp4")
    reference_path = PROJECT_ROOT / "samples/example/keyframes/flux2_klein.png"
    if baseline_path is None or not baseline_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError(f"baseline/reference missing: {baseline_path}, {reference_path}")
    if args.evaluate_only:
        baseline_out = args.output_root / "baseline.mp4"
        improved_out = args.output_root / "improved.mp4"
        if not baseline_out.is_file() or not improved_out.is_file():
            raise FileNotFoundError(f"Phase 3 outputs missing: {baseline_out}, {improved_out}")
        comparison_path = args.output_root / "comparison.json"
        comparison = read_json(comparison_path) if comparison_path.is_file() else {"schema_version": "metacut.phase3_ablation/v1"}
        baseline_metrics = evaluate_video(baseline_out, reference_path, args.dino_path)
        improved_metrics = evaluate_video(improved_out, reference_path, args.dino_path)
        comparison.setdefault("baseline", {}).update({"metrics": baseline_metrics, "output": str(baseline_out), "sha256": sha256(baseline_out)})
        comparison.setdefault("improved", {}).update({"metrics": improved_metrics, "output": str(improved_out), "sha256": sha256(improved_out)})
        comparison["delta"] = {
            key: improved_metrics.get(key) - baseline_metrics.get(key)
            for key in ("temporal_stability", "motion_presence", "storyboard_adherence_heuristic", "masked_dino_product_similarity")
            if isinstance(improved_metrics.get(key), (int, float)) and isinstance(baseline_metrics.get(key), (int, float))
        }
        comparison["metrics_recomputed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(comparison_path, comparison)
        write_json(args.run_root / "comparison.json", comparison)
        print(json.dumps({"status": "metrics_recomputed", "comparison": str(comparison_path), "delta": comparison["delta"]}, ensure_ascii=False, indent=2))
        return
    video_request = baseline_record["video"]["request"]
    improved_prompt = str(video_request["prompt"]) + (
        "\nPhase 3 refinement: make the camera motion visibly readable but restrained. "
        "Use one continuous camera path with a 10 percent macro push-in from 0.0-1.0s, "
        "a smooth 20-degree clockwise orbit from 1.0-2.7s, a single warm caustic sweep "
        "across the glass from 2.7-3.4s, then a steady 0.6s hero hold. Keep the bottle's "
        "center, footprint, cap, liquid level and base locked to the same identity; move the "
        "camera and light, never the product geometry. The first and last frames should be "
        "clean hero compositions with a stable horizon and continuous background."
    )
    payload = {
        "model": "hunyuanvideo_15_i2v",
        "prompt": improved_prompt,
        "negative_prompt": video_request.get("negative_prompt"),
        "input_image": str(reference_path.resolve()),
        "width": int(video_request["width"]),
        "height": int(video_request["height"]),
        "num_frames": int(video_request["num_frames"]),
        "fps": int(video_request["fps"]),
        "seed": int(video_request["seed"]),
        "candidate_ranking": "identity_temporal",
        "identity_weight": args.identity_weight,
    }

    print(json.dumps({"phase": "3", "sample": "sample-02", "payload": payload}, ensure_ascii=False, indent=2))
    submission = request(args.base_url, "/v1/video", payload)
    improved_job = wait_job(args.base_url, submission["id"], args.request_timeout)
    if improved_job.get("status") != "succeeded":
        raise RuntimeError(json.dumps(improved_job, ensure_ascii=False))
    result = improved_job.get("result") or {}
    improved_worker_path = url_to_path(str(result["video_url"]))
    if not improved_worker_path.is_file():
        raise FileNotFoundError(improved_worker_path)

    args.output_root.mkdir(parents=True, exist_ok=True)
    baseline_out = args.output_root / "baseline.mp4"
    improved_out = args.output_root / "improved.mp4"
    shutil.copy2(baseline_path, baseline_out)
    shutil.copy2(improved_worker_path, improved_out)
    baseline_metrics = evaluate_video(baseline_out, reference_path, args.dino_path)
    improved_metrics = evaluate_video(improved_out, reference_path, args.dino_path)
    comparison = {
        "schema_version": "metacut.phase3_ablation/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": "perfume_flacon",
        "sample_label": "example / perfume flacon",
        "baseline": {
            "source": str(baseline_path),
            "output": str(baseline_out),
            "sha256": sha256(baseline_out),
            "phase2_record": str(args.baseline_record),
            "metrics": baseline_metrics,
        },
        "improved": {
            "output": str(improved_out),
            "sha256": sha256(improved_out),
            "request": payload,
            "service_result": result,
            "metrics": improved_metrics,
        },
        "delta": {
            key: (improved_metrics.get(key) - baseline_metrics.get(key))
            for key in ("temporal_stability", "motion_presence", "storyboard_adherence_heuristic", "masked_dino_product_similarity")
            if isinstance(improved_metrics.get(key), (int, float)) and isinstance(baseline_metrics.get(key), (int, float))
        },
        "acceptance_note": "The improved clip is a candidate method change; final visual quality should be judged by side-by-side playback, with metrics used as supporting evidence.",
    }
    write_json(args.output_root / "comparison.json", comparison)
    write_json(args.output_root / "README.json", {
        "baseline": "baseline.mp4 (Phase 2.5 unchanged)",
        "improved": "improved.mp4 (Phase 3 explicit camera prompt + DINO/temporal candidate ranking)",
        "comparison": "comparison.json",
    })
    write_json(args.run_root / "comparison.json", comparison)
    (args.run_root / "improved-job.json").parent.mkdir(parents=True, exist_ok=True)
    write_json(args.run_root / "improved-job.json", improved_job)
    print(json.dumps({"status": "succeeded", "baseline": str(baseline_out), "improved": str(improved_out), "comparison": str(args.output_root / "comparison.json"), "service_candidate_ranking": result.get("candidate_ranking"), "selected_candidate": result.get("selected_candidate")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
