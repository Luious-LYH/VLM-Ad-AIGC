"""Dependency-free remote smoke test for the local AIGC gateway."""

from __future__ import annotations

import json
import sys
import time
from urllib.request import Request, urlopen


BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8100"


def request(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urlopen(req, timeout=20) as response:  # noqa: S310 -- test address is explicit
        return json.load(response)


health = request("/health")
assert health["status"] == "ok", health
submitted = request("/v1/keyframes", {
    "model": "sdxl_ip_adapter",
    "prompt": "studio product hero shot",
    "seed": 7,
    "num_candidates": 2,
})
assert submitted["status"] == "queued", submitted

deadline = time.monotonic() + 15
while time.monotonic() < deadline:
    job = request(f"/v1/jobs/{submitted['id']}")
    if job["status"] == "succeeded":
        break
    if job["status"] == "failed":
        raise RuntimeError(job["error"])
    time.sleep(0.1)
else:
    raise TimeoutError(submitted["id"])

assert job["result"]["selected_image"], job
video = request("/v1/video", {
    "model": "ltxv_2b",
    "prompt": "stable product turntable",
    "input_image": "https://assets.example.test/keyframe.png",
    "width": 576,
    "height": 1024,
    "num_frames": 49,
    "fps": 24,
    "seed": 8,
})
deadline = time.monotonic() + 15
while time.monotonic() < deadline:
    video_job = request(f"/v1/jobs/{video['id']}")
    if video_job["status"] == "succeeded":
        break
    if video_job["status"] == "failed":
        raise RuntimeError(video_job["error"])
    time.sleep(0.1)
else:
    raise TimeoutError(video["id"])
assert video_job["result"]["video_url"].endswith(".mp4"), video_job
print(json.dumps({"health": health, "image_job": job, "video_job": video_job}, ensure_ascii=False))
