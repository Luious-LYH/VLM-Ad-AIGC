"""Run and record MetaCut's real local P0 chain on the GPU host.

This intentionally uses the production HTTP contract rather than importing
model classes. It is therefore both a demo runner and an integration smoke:
product image -> Qwen3-VL product/v1 -> SDXL/IP-Adapter keyframe -> LTX I2V.
The script never treats a `fallback_reason` as evidence that the requested
experimental model actually ran.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


parser = argparse.ArgumentParser()
parser.add_argument("--asset", type=Path, required=True, help="Absolute product image below AIGC_ASSET_ROOTS")
parser.add_argument("--base-url", default="http://127.0.0.1:8100")
parser.add_argument("--output", type=Path, default=Path("runs/phase1-demo"))
parser.add_argument("--image-model", default="sdxl_ip_adapter", choices=("sdxl_ip_adapter", "flux2_klein"))
parser.add_argument("--video-model", default="ltxv_2b", choices=("ltxv_2b", "wan22_ti2v_5b"))
parser.add_argument(
    "--request-timeout",
    type=int,
    default=300,
    help="HTTP timeout for cold-start model requests (seconds)",
)
args = parser.parse_args()

asset = args.asset.resolve(strict=True)
base = args.base_url.rstrip("/")
started_at = datetime.now(timezone.utc).isoformat()
uploads_root = next((parent for parent in (asset, *asset.parents) if parent.name == "uploads"), None)
if uploads_root is None:
    parser.error("--asset must be inside a directory named uploads so generated URLs can be resolved")
public_root = uploads_root.parent


def request(path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(f"{base}{path}", data=body, headers={"Content-Type": "application/json"} if body else {})
    with urlopen(req, timeout=args.request_timeout) as response:  # noqa: S310 -- explicit local gateway URL
        return json.load(response)


def wait_job(job_id: str, timeout_seconds: int = 1800) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = request(f"/v1/jobs/{job_id}")
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(2)
    raise TimeoutError(f"local AIGC job timed out: {job_id}")


def output_url_to_path(url: str) -> str:
    if not url.startswith("/uploads/"):
        raise ValueError(f"expected worker-local upload URL, got {url!r}")
    return str((public_root / url.removeprefix("/")).resolve())


record: dict = {
    "schema_version": "metacut.phase1_demo/v1",
    "started_at": started_at,
    "input_product_image": str(asset),
    "requested_models": {"image": args.image_model, "video": args.video_model},
}
try:
    record["health"] = request("/health")
    parsed = request("/v1/product/parse", {
        "product_description": "",
        "product_image_urls": [str(asset)],
        "product_video_urls": [],
    })
    record["product_parse"] = parsed
    keyframe_submission = request("/v1/keyframes", {
        "model": args.image_model,
        "prompt": "Premium commercial hero shot of the reference insulated tumbler on a clean warm-gray studio table. Preserve the product silhouette, matte white body, and dark teal silicone lid.",
        "negative_prompt": "text, logo, watermark, duplicate product, distorted lid, blur",
        "reference_images": [str(asset)],
        "width": 768,
        "height": 1024,
        "seed": 20260906,
        "num_candidates": 1,
    })
    record["keyframe_submission"] = keyframe_submission
    keyframe = wait_job(keyframe_submission["id"])
    record["keyframe_job"] = keyframe
    if keyframe["status"] != "succeeded":
        raise RuntimeError(keyframe.get("error") or "keyframe generation failed")
    image_result = keyframe["result"]
    keyframe_path = output_url_to_path(image_result["selected_image"])
    video_submission = request("/v1/video", {
        "model": args.video_model,
        "prompt": "The insulated tumbler performs a slow stable three-quarter commercial turn on the studio table. Preserve its white matte body and dark teal lid. Smooth camera orbit, soft premium lighting, no text.",
        "input_image": keyframe_path,
        "width": 576,
        "height": 768,
        "num_frames": 49,
        "fps": 24,
        "seed": 20260907,
    })
    record["video_submission"] = video_submission
    video = wait_job(video_submission["id"])
    record["video_job"] = video
    if video["status"] != "succeeded":
        raise RuntimeError(video.get("error") or "video generation failed")
    record["status"] = "fallback" if (
        image_result.get("fallback_reason") or video["result"].get("fallback_reason")
    ) else "succeeded"
except (HTTPError, OSError, RuntimeError, TimeoutError, ValueError) as error:
    record["status"] = "failed"
    record["error"] = repr(error)

args.output.mkdir(parents=True, exist_ok=True)
target = args.output / f"phase1-{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(target)
print(json.dumps(record, ensure_ascii=False, indent=2))
raise SystemExit(0 if record["status"] == "succeeded" else 1)
