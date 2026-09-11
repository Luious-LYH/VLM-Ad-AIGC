"""Run the complete Phase-2/2.5 model comparison.

The expensive image/video artifacts are generated once per (image, video,
sample) pair and copied into one human-readable folder per matrix combination.
The VLM
dimension is evaluated independently with the same product images and prompt;
this keeps the comparison fair instead of letting a different VLM silently
change the generation prompt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/phase2-comparison.json"))
parser.add_argument("--base-url", default="http://127.0.0.1:8100")
parser.add_argument("--output-root", type=Path, default=Path("public/uploads/phase2"))
parser.add_argument("--run-root", type=Path, default=Path("runs/phase2"))
parser.add_argument("--qwen-path", type=Path, default=Path("models/Qwen3-VL-4B-Instruct-v3"))
parser.add_argument("--internvl-path", type=Path, default=Path("models/InternVL3_5-8B-HF"))
parser.add_argument("--request-timeout", type=int, default=2400)
parser.add_argument("--skip-vlm", action="store_true")
parser.add_argument("--skip-input-generation", action="store_true")
parser.add_argument("--skip-existing", action="store_true")
parser.add_argument("--prepare-only", action="store_true", help="Only materialize the three stable input images")
args = parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def request(base: str, path: str, payload: dict | None = None, timeout: int = 60) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(f"{base.rstrip('/')}{path}", data=body, headers={"Content-Type": "application/json"} if body else {})
    with urlopen(req, timeout=timeout) as response:  # noqa: S310 - local worker URL is explicit
        return json.load(response)


def wait_job(base: str, job_id: str, timeout: int) -> dict:
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
    public_root = Path("public").resolve()
    path = (public_root / url.removeprefix("/")).resolve()
    if public_root not in path.parents:
        raise ValueError(f"worker URL escaped public root: {url}")
    return path


def copy_media(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def cleanup_worker_artifact(path: Path) -> None:
    """Remove only the just-created worker artifact, never stable phase2 files."""
    try:
        path.unlink(missing_ok=True)
        parent = path.parent
        if parent.name.startswith("ltxv-run_"):
            parent.rmdir()
    except OSError:
        pass


def run_vlm(config: dict, model_id: str, model_path: Path, manifest_path: Path, output_dir: Path) -> Path:
    target = output_dir / f"{model_id}.json"
    if target.is_file() and args.skip_existing:
        return target
    command = [
        sys.executable, "scripts/run-phase2-vlm.py",
        "--config", str(args.config), "--model", model_id,
        "--model-path", str(model_path), "--output", str(output_dir), "--device", "cuda:0",
        "--sample-manifest", str(manifest_path),
    ]
    env = dict(**__import__("os").environ)
    # GPU 2 is free during VLM evaluation; the API keeps Qwen on GPU 0 and the
    # image pipeline on GPU 1. Masking makes InternVL's auto device map stable.
    env["CUDA_VISIBLE_DEVICES"] = "2"
    subprocess.run(command, check=True, env=env)
    if not target.is_file():
        raise RuntimeError(f"VLM runner did not produce {target}")
    return target


def submit_generation(base: str, kind: str, payload: dict) -> dict:
    endpoint = "/v1/keyframes" if kind == "image" else "/v1/video"
    submission = request(base, endpoint, payload, timeout=60)
    return wait_job(base, submission["id"], args.request_timeout)


config_document = read_json(args.config)
if config_document.get("base_config"):
    base_path = (args.config.parent / str(config_document["base_config"])).resolve()
    config = read_json(base_path)
    extra_video = config_document.get("append_video")
    if extra_video:
        config["matrix"]["video"] = [*config["matrix"]["video"], extra_video]
else:
    config = config_document
args.output_root.mkdir(parents=True, exist_ok=True)
args.run_root.mkdir(parents=True, exist_ok=True)
shared_root = args.output_root / "_shared"
shared_inputs = shared_root / "inputs"
shared_keyframes = shared_root / "keyframes"
shared_videos = shared_root / "videos"
for directory in (shared_inputs, shared_keyframes, shared_videos):
    directory.mkdir(parents=True, exist_ok=True)

run_manifest: dict = {
    "schema_version": "metacut.phase2/v1",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "config": str(args.config),
    "seed": config["seed"],
    "image_spec": config["image_spec"],
    "video_spec": config["video_spec"],
    "fairness": "All image/video models receive the same resolved input image, literal prompt, seed and output specification. VLMs are compared on the product JSON task and do not rewrite the generation prompt.",
    "samples": {}, "vlm": {}, "images": {}, "videos": {}, "combinations": {},
}

# 1. Resolve three stable input images. Two are generated once by FLUX when
# absent; the existing cosmetic set is reused as the first sample.
for sample in config["samples"]:
    target = shared_inputs / f"{sample['id']}.png"
    source = Path(sample["source"])
    # Configs may use the browser-visible `/public/uploads/...` shorthand;
    # resolve it against the project checkout when running on the GPU host.
    if not source.is_file() and str(source).startswith("/public/"):
        source = Path("public") / str(source).removeprefix("/public/")
    if target.is_file():
        pass
    elif source.is_file():
        copy_media(source, target)
    elif sample.get("generate_prompt") and not args.skip_input_generation:
        input_seed = config["seed"] + 1000 + config["samples"].index(sample)
        job = submit_generation(args.base_url, "image", {
            "model": "flux2_klein", "prompt": sample["generate_prompt"],
            "negative_prompt": config["prompts"]["negative"],
            "reference_images": [], "width": config["image_spec"]["width"],
            "height": config["image_spec"]["height"], "num_candidates": 1,
            "seed": input_seed,
        })
        if job.get("status") != "succeeded":
            raise RuntimeError(f"input generation failed for {sample['id']}: {job}")
        worker_path = url_to_path(job["result"]["selected_image"])
        copy_media(worker_path, target)
        cleanup_worker_artifact(worker_path)
    else:
        raise FileNotFoundError(f"missing sample input {source}; run without --skip-input-generation")
    run_manifest["samples"][sample["id"]] = {"input": str(target.resolve()), "label": sample["label"], "ground_truth": sample["ground_truth"]}

sample_manifest = args.run_root / "sample-manifest.json"
write_json(sample_manifest, {"samples": [{**sample, "source": str((shared_inputs / f"{sample['id']}.png").resolve())} for sample in config["samples"]]})

if args.prepare_only:
    run_manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(args.run_root / "phase2-inputs-manifest.json", run_manifest)
    print(args.run_root / "phase2-inputs-manifest.json")
    raise SystemExit(0)

# 2. VLM product JSON comparison. Each VLM is one process, so its model is
# released before the next one starts on the shared 3090.
if not args.skip_vlm:
    for model in config["matrix"]["vlm"]:
        model_path = args.qwen_path if model["id"] == "qwen3_vl" else args.internvl_path
        result_path = run_vlm(config, model["id"], model_path, sample_manifest, args.run_root / "vlm")
        run_manifest["vlm"][model["id"]] = read_json(result_path)
else:
    for model in config["matrix"]["vlm"]:
        result_path = args.run_root / "vlm" / f"{model['id']}.json"
        run_manifest["vlm"][model["id"]] = read_json(result_path) if result_path.is_file() else {"status": "skipped"}

# 3. Generate one keyframe per (image model, sample), then one video per
# (image model, video model, sample). These are reused in both VLM branches.
for image_model in config["matrix"]["image"]:
    for index, sample in enumerate(config["samples"]):
        key = f"{image_model['id']}__{sample['id']}"
        target = shared_keyframes / f"{key}.png"
        record_path = args.run_root / "images" / f"{key}.json"
        if target.is_file() and args.skip_existing and record_path.is_file():
            run_manifest["images"][key] = read_json(record_path)
            continue
        prompt = config["prompts"]["image"] + "\nReference product class: " + sample["ground_truth"]["form_factor"] + ". Preserve the reference composition and product family."
        job = submit_generation(args.base_url, "image", {
            "model": image_model["id"], "prompt": prompt,
            "negative_prompt": config["prompts"]["negative"],
            "reference_images": [str((shared_inputs / f"{sample['id']}.png").resolve())],
            "width": config["image_spec"]["width"], "height": config["image_spec"]["height"],
            "num_candidates": config["image_spec"]["num_candidates"], "seed": config["seed"] + index,
        })
        result = job.get("result") or {}
        if job.get("status") != "succeeded" or not result.get("selected_image"):
            record = {"status": "failed", "job": job, "model": image_model["id"], "sample_id": sample["id"]}
        else:
            worker_path = url_to_path(result["selected_image"])
            copy_media(worker_path, target)
            record = {"status": "succeeded", "model": image_model["id"], "sample_id": sample["id"], "input": str((shared_inputs / f"{sample['id']}.png").resolve()), "output": str(target.resolve()), "request": {"prompt": prompt, "negative_prompt": config["prompts"]["negative"], "seed": config["seed"] + index, **config["image_spec"]}, "result": result}
            cleanup_worker_artifact(worker_path)
        write_json(record_path, record)
        run_manifest["images"][key] = record

for image_model in config["matrix"]["image"]:
    for video_model in config["matrix"]["video"]:
        for index, sample in enumerate(config["samples"]):
            key = f"{image_model['id']}__{video_model['id']}__{sample['id']}"
            target = shared_videos / f"{key}.mp4"
            record_path = args.run_root / "videos" / f"{key}.json"
            image_key = f"{image_model['id']}__{sample['id']}"
            image_path = shared_keyframes / f"{image_key}.png"
            if target.is_file() and args.skip_existing and record_path.is_file():
                run_manifest["videos"][key] = read_json(record_path)
                continue
            prompt = config["prompts"]["video"] + "\nProduct-specific micro-action: " + sample["micro_action"] + "."
            job = submit_generation(args.base_url, "video", {
                "model": video_model["id"], "prompt": prompt,
                "negative_prompt": config["prompts"]["negative"], "input_image": str(image_path.resolve()),
                "width": config["video_spec"]["width"], "height": config["video_spec"]["height"],
                "num_frames": config["video_spec"]["num_frames"], "fps": config["video_spec"]["fps"],
                "seed": config["seed"] + index,
            })
            result = job.get("result") or {}
            if job.get("status") != "succeeded" or not result.get("video_url"):
                record = {"status": "failed", "job": job, "model": video_model["id"], "image_model": image_model["id"], "sample_id": sample["id"]}
            else:
                worker_path = url_to_path(result["video_url"])
                copy_media(worker_path, target)
                record = {"status": "succeeded", "model": video_model["id"], "image_model": image_model["id"], "sample_id": sample["id"], "input": str(image_path.resolve()), "output": str(target.resolve()), "request": {"prompt": prompt, "negative_prompt": config["prompts"]["negative"], "seed": config["seed"] + index, **config["video_spec"]}, "result": result}
                cleanup_worker_artifact(worker_path)
            write_json(record_path, record)
            run_manifest["videos"][key] = record

# 4. Materialize one human-readable combination folder for every matrix
# combination (8 folders in Phase 2, 12 after the Phase 2.5 Hunyuan append).
# Input images and keyframes remain local to each combination for review, but
# videos stay in shared_videos: VLMs do not participate in video generation,
# so copying the same MP4 into each VLM branch only wastes storage and creates
# misleading VLM-prefixed artifact names.
for vlm in config["matrix"]["vlm"]:
    for image_model in config["matrix"]["image"]:
        for video_model in config["matrix"]["video"]:
            combo_id = f"{vlm['id']}__{image_model['id']}__{video_model['id']}"
            combo_dir = args.output_root / combo_id
            combo_record = {"schema_version": "metacut.phase2_combination/v1", "combo_id": combo_id, "vlm": vlm, "image_model": image_model, "video_model": video_model, "samples": []}
            for index, sample in enumerate(config["samples"], start=1):
                sample_dir = combo_dir / f"sample-{index:02d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                image_key = f"{image_model['id']}__{sample['id']}"
                video_key = f"{image_model['id']}__{video_model['id']}__{sample['id']}"
                copy_media(shared_inputs / f"{sample['id']}.png", sample_dir / "input.png")
                if (shared_keyframes / f"{image_key}.png").is_file():
                    copy_media(shared_keyframes / f"{image_key}.png", sample_dir / "keyframe.png")
                vlm_record = run_manifest["vlm"].get(vlm["id"], {})
                vlm_item = next((item for item in vlm_record.get("records", []) if item.get("sample_id") == sample["id"]), {"status": "missing"})
                image_record = run_manifest["images"].get(image_key, {"status": "missing"})
                video_record = run_manifest["videos"].get(video_key, {"status": "missing"})
                record = {"sample_id": sample["id"], "input": {"path": "input.png", "ground_truth": sample["ground_truth"]}, "prompts": {"image": image_record.get("request", {}).get("prompt"), "video": video_record.get("request", {}).get("prompt"), "negative": config["prompts"]["negative"]}, "spec": {"seed": config["seed"] + index - 1, "image": config["image_spec"], "video": config["video_spec"]}, "vlm": vlm_item, "image": image_record, "video": video_record, "status": "succeeded" if video_record.get("status") == "succeeded" and image_record.get("status") == "succeeded" else "failed"}
                write_json(sample_dir / "record.json", record)
                combo_record["samples"].append({"sample_id": sample["id"], "path": f"sample-{index:02d}", "status": record["status"]})
            write_json(combo_dir / "combo.json", combo_record)
            run_manifest["combinations"][combo_id] = {"path": str(combo_dir.resolve()), "samples": combo_record["samples"]}

run_manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
write_json(args.run_root / "phase2-manifest.json", run_manifest)
write_json(args.output_root / "MANIFEST.json", run_manifest)
print(args.run_root / "phase2-manifest.json")
print(json.dumps({"combinations": len(run_manifest["combinations"]), "images": len(run_manifest["images"]), "videos": len(run_manifest["videos"]), "vlm": list(run_manifest["vlm"])}, ensure_ascii=False))
