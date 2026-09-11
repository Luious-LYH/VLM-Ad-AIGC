"""Generate the six SDXL-keyframe video variants on an isolated GPU.

The main matrix runner handles the canonical manifest.  This companion worker
lets the SDXL branch run in parallel with the long Wan jobs while preserving
the same request payloads and record format.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from urllib.request import Request, urlopen


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/phase2-comparison.json"))
parser.add_argument("--base-url", default="http://127.0.0.1:8102")
parser.add_argument("--output-root", type=Path, default=Path("public/uploads/phase2"))
parser.add_argument("--run-root", type=Path, default=Path("runs/phase2"))
parser.add_argument("--timeout", type=int, default=2400)
args = parser.parse_args()


def call(path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(args.base_url.rstrip("/") + path, data=body, headers={"Content-Type": "application/json"} if body else {})
    with urlopen(req, timeout=60) as response:
        return json.load(response)


def wait_job(job_id: str) -> dict:
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        job = call(f"/v1/jobs/{job_id}")
        if job.get("status") in {"succeeded", "failed"}:
            return job
        time.sleep(2)
    raise TimeoutError(job_id)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


config = json.loads(args.config.read_text(encoding="utf-8"))
shared_inputs = args.output_root / "_shared" / "inputs"
shared_keyframes = args.output_root / "_shared" / "keyframes"
shared_videos = args.output_root / "_shared" / "videos"
records_dir = args.run_root / "videos"
shared_videos.mkdir(parents=True, exist_ok=True)
records_dir.mkdir(parents=True, exist_ok=True)

for video_model in config["matrix"]["video"]:
    for index, sample in enumerate(config["samples"]):
        key = f"sdxl_ip_adapter__{video_model['id']}__{sample['id']}"
        target = shared_videos / f"{key}.mp4"
        record_path = records_dir / f"{key}.json"
        if target.is_file() and record_path.is_file():
            try:
                if json.loads(record_path.read_text(encoding="utf-8")).get("status") == "succeeded":
                    print(f"skip {key}", flush=True)
                    continue
            except Exception:
                pass
        image_path = shared_keyframes / f"sdxl_ip_adapter__{sample['id']}.png"
        if not image_path.is_file():
            record = {"status": "failed", "error": f"missing keyframe: {image_path}", "model": video_model["id"], "image_model": "sdxl_ip_adapter", "sample_id": sample["id"]}
            write_json(record_path, record)
            print(f"failed {key}: missing keyframe", flush=True)
            continue
        prompt = config["prompts"]["video"] + "\nProduct-specific micro-action: " + sample["micro_action"] + "."
        payload = {
            "model": video_model["id"],
            "prompt": prompt,
            "negative_prompt": config["prompts"]["negative"],
            "input_image": str(image_path.resolve()),
            "width": config["video_spec"]["width"],
            "height": config["video_spec"]["height"],
            "num_frames": config["video_spec"]["num_frames"],
            "fps": config["video_spec"]["fps"],
            "seed": config["seed"] + index,
        }
        try:
            submission = call("/v1/video", payload)
            job = wait_job(submission["id"])
        except Exception as exc:
            record = {"status": "failed", "error": repr(exc), "model": video_model["id"], "image_model": "sdxl_ip_adapter", "sample_id": sample["id"]}
            write_json(record_path, record)
            print(f"failed {key}: {exc}", flush=True)
            continue
        result = job.get("result") or {}
        url = result.get("video_url")
        if job.get("status") != "succeeded" or not url:
            record = {"status": "failed", "job": job, "model": video_model["id"], "image_model": "sdxl_ip_adapter", "sample_id": sample["id"]}
        else:
            worker_path = Path("public") / url.removeprefix("/")
            if not worker_path.is_file():
                record = {"status": "failed", "job": job, "error": f"missing worker artifact: {worker_path}", "model": video_model["id"], "image_model": "sdxl_ip_adapter", "sample_id": sample["id"]}
            else:
                shutil.copy2(worker_path, target)
                record = {"status": "succeeded", "model": video_model["id"], "image_model": "sdxl_ip_adapter", "sample_id": sample["id"], "input": str(image_path.resolve()), "output": str(target.resolve()), "request": payload, "result": result}
                try:
                    worker_path.unlink()
                    worker_path.parent.rmdir()
                except OSError:
                    pass
        write_json(record_path, record)
        print(f"{record['status']} {key}", flush=True)

print("sdxl videos complete", flush=True)
