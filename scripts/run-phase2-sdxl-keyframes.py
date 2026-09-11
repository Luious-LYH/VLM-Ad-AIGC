"""Generate the SDXL/IP-Adapter keyframes for Phase 2 while video jobs run.

This small worker is intentionally independent from the full matrix runner so
the SDXL image branch can use the otherwise idle image GPU in parallel with
LTX/Wan video generation.  It writes the same image records consumed by
``run-phase2-comparison.py``.
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
parser.add_argument("--base-url", default="http://127.0.0.1:8101")
parser.add_argument("--output-root", type=Path, default=Path("public/uploads/phase2"))
parser.add_argument("--run-root", type=Path, default=Path("runs/phase2"))
parser.add_argument("--timeout", type=int, default=2400)
args = parser.parse_args()


def call(path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(
        args.base_url.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
    )
    with urlopen(req, timeout=60) as response:  # local-only worker URL
        return json.load(response)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def wait_job(job_id: str) -> dict:
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        job = call(f"/v1/jobs/{job_id}")
        if job.get("status") in {"succeeded", "failed"}:
            return job
        time.sleep(2)
    raise TimeoutError(job_id)


config = json.loads(args.config.read_text(encoding="utf-8"))
shared_inputs = args.output_root / "_shared" / "inputs"
shared_keyframes = args.output_root / "_shared" / "keyframes"
records_dir = args.run_root / "images"
shared_keyframes.mkdir(parents=True, exist_ok=True)
records_dir.mkdir(parents=True, exist_ok=True)

for index, sample in enumerate(config["samples"]):
    key = f"sdxl_ip_adapter__{sample['id']}"
    target = shared_keyframes / f"{key}.png"
    record_path = records_dir / f"{key}.json"
    if target.is_file() and record_path.is_file():
        print(f"skip {key}", flush=True)
        continue
    prompt = config["prompts"]["image"] + "\nReference product class: " + sample["ground_truth"]["form_factor"] + ". Preserve the reference composition and product family."
    payload = {
        "model": "sdxl_ip_adapter",
        "prompt": prompt,
        "negative_prompt": config["prompts"]["negative"],
        "reference_images": [str((shared_inputs / f"{sample['id']}.png").resolve())],
        "width": config["image_spec"]["width"],
        "height": config["image_spec"]["height"],
        "num_candidates": 1,
        "seed": config["seed"] + index,
    }
    try:
        submission = call("/v1/keyframes", payload)
        job = wait_job(submission["id"])
    except Exception as exc:
        record = {"status": "failed", "error": repr(exc), "model": "sdxl_ip_adapter", "sample_id": sample["id"]}
        write_json(record_path, record)
        print(f"failed {key}: {exc}", flush=True)
        continue
    result = job.get("result") or {}
    selected = result.get("selected_image")
    if job.get("status") != "succeeded" or not selected:
        record = {"status": "failed", "job": job, "model": "sdxl_ip_adapter", "sample_id": sample["id"]}
    else:
        worker_path = Path("public") / selected.removeprefix("/")
        if not worker_path.is_file():
            record = {"status": "failed", "job": job, "error": f"missing worker artifact: {worker_path}", "model": "sdxl_ip_adapter", "sample_id": sample["id"]}
        else:
            shutil.copy2(worker_path, target)
            record = {
                "status": "succeeded", "model": "sdxl_ip_adapter", "sample_id": sample["id"],
                "input": str((shared_inputs / f"{sample['id']}.png").resolve()), "output": str(target.resolve()),
                "request": {**payload, "steps": config["image_spec"]["steps"]}, "result": result,
            }
            try:
                worker_path.unlink()
                worker_path.parent.rmdir()
            except OSError:
                pass
    write_json(record_path, record)
    print(f"{record['status']} {key}", flush=True)

print("sdxl keyframes complete", flush=True)
