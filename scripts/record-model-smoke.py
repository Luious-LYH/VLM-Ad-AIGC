"""Write a reproducible model smoke-test record without committing media."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://127.0.0.1:8100")
parser.add_argument("--kind", required=True, choices=("image", "video"))
parser.add_argument("--model", required=True)
parser.add_argument("--asset", help="Absolute product/keyframe image path; required for video")
parser.add_argument("--output", type=Path, default=Path("runs/model-smoke"))
parser.add_argument(
    "--timeout-seconds",
    type=int,
    default=1800,
    help="Maximum worker time after submission (LTX cold start can exceed five minutes)",
)
args = parser.parse_args()


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True
        ).strip()
    except Exception as error:  # pragma: no cover - environment dependent
        return f"unavailable: {error}"


if args.kind == "image":
    path = "/v1/keyframes"
    payload = {
        "model": args.model,
        "prompt": "A premium commercial hero shot of the reference product on a clean studio table, product identity preserved",
        "negative_prompt": "distorted logo, duplicate product, watermark, blur",
        "reference_images": [args.asset] if args.asset else [],
        "width": 768,
        "height": 1024,
        "seed": 20260906,
        "num_candidates": 1,
    }
else:
    if not args.asset:
        parser.error("--asset is required for video smoke testing")
    path = "/v1/video"
    payload = {
        "model": args.model,
        "prompt": "The product makes a slow, stable commercial turn on a studio table; preserve identity and logo",
        "input_image": args.asset,
        "width": 576,
        "height": 768,
        "num_frames": 49,
        "fps": 24,
        "seed": 20260906,
    }

request = Request(
    f"{args.base_url.rstrip('/')}{path}",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
record = {
    "schema_version": "metacut.model_smoke/v1",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "kind": args.kind,
    "model": args.model,
    "hardware": nvidia_smi(),
    "python": sys.version,
    "platform": platform.platform(),
    "request": payload,
}
try:
    with urlopen(request, timeout=30) as response:  # noqa: S310 -- explicit local base URL
        record["submission"] = json.load(response)
        job_id = record["submission"].get("id")
        if not isinstance(job_id, str):
            raise RuntimeError(f"worker did not return a job ID: {record['submission']!r}")
        deadline = time.monotonic() + args.timeout_seconds
        while time.monotonic() < deadline:
            with urlopen(f"{args.base_url.rstrip('/')}/v1/jobs/{job_id}", timeout=20) as poll:  # noqa: S310
                job = json.load(poll)
            if job.get("status") in {"succeeded", "failed"}:
                record["job"] = job
                result = job.get("result") if isinstance(job.get("result"), dict) else {}
                record["worker_status"] = job["status"]
                # A fallback proves operational resilience, not the requested
                # experimental model. Keep it distinct from a real smoke pass.
                record["status"] = "fallback" if result.get("fallback_reason") else job["status"]
                record["actual_model_id"] = result.get("model_id")
                record["fallback_reason"] = result.get("fallback_reason")
                break
            time.sleep(2)
        else:
            record["status"] = "timed_out"
except HTTPError as error:
    record["status"] = "failed"
    record["http_status"] = error.code
    record["error"] = error.read().decode(errors="replace")
except Exception as error:  # pragma: no cover - environment dependent
    record["status"] = "failed"
    record["error"] = repr(error)

args.output.mkdir(parents=True, exist_ok=True)
target = args.output / f"{args.kind}-{args.model}-{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(target)
print(json.dumps(record, ensure_ascii=False, indent=2))
