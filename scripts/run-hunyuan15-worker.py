"""One isolated HunyuanVideo-1.5 I2V candidate.

The parent service masks this process to one physical GPU and launches two
copies concurrently. Keeping the model runtime here prevents Diffusers 0.40
and its newer Hugging Face Hub API from leaking into the VLM worker process.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


def hub_compat() -> None:
    """Allow Diffusers 0.40 to coexist with the host's older HF Hub package."""
    import huggingface_hub
    from huggingface_hub.utils import LocalEntryNotFoundError
    import huggingface_hub.errors as hub_errors

    if not hasattr(huggingface_hub, "get_cached_repo_tree"):
        def get_cached_repo_tree(*_args, **_kwargs):
            raise LocalEntryNotFoundError("cached repository tree is unavailable")

        huggingface_hub.get_cached_repo_tree = get_cached_repo_tree
    if not hasattr(hub_errors, "CachedRepoTreeNotFoundError"):
        hub_errors.CachedRepoTreeNotFoundError = type("CachedRepoTreeNotFoundError", (Exception,), {})


def stability_score(frames: list[np.ndarray]) -> float:
    """Reward smooth, non-flickering motion without pretending it is quality GT."""
    if len(frames) < 2:
        return 0.0
    sampled = []
    for frame in frames[::4]:
        image = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("L")
        image.thumbnail((64, 64))
        sampled.append(np.asarray(image, dtype=np.float32) / 255.0)
    diffs = [float(np.mean(np.abs(curr - prev))) for prev, curr in zip(sampled, sampled[1:])]
    if not diffs:
        return 0.0
    motion = float(np.mean(diffs))
    return max(0.0, 1.0 - abs(motion - 0.10) / 0.10)


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Normalize Diffusers' float [0, 1] frames without producing a black MP4.

    HunyuanVideo15 returns NumPy frames in the same convention as most
    Diffusers video pipelines: floating point RGB values in [0, 1].  Casting
    those values directly to uint8 would collapse almost the entire image to
    0/1.  The explicit conversion also keeps this worker independent from the
    exact imageio backend version used by the host runtime.
    """
    array = np.asarray(frame)
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.5 else 1.0
        array = np.clip(array * scale, 0.0, 255.0)
    return array.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--attention-slicing",
        default="max",
        choices=("auto", "max", "none"),
        help="Attention slicing mode. max keeps 97-frame 480p I2V within 24GB GPUs.",
    )
    parser.add_argument(
        "--offload-mode",
        default="auto",
        choices=("auto", "model", "group", "sequential"),
        help="CPU offload strategy; auto uses block-level group offload for clips over 64 frames.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES maps this to the requested card.
    # PyTorch 2.5 on the server rejects the string form here after CUDA
    # masking; the no-argument form resets the current visible device.
    torch.cuda.reset_peak_memory_stats()
    hub_compat()
    from diffusers import HunyuanVideo15ImageToVideoPipeline
    from diffusers.utils import export_to_video

    image = Image.open(args.input_image).convert("RGB").resize((480, 640), Image.Resampling.LANCZOS)
    pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16, local_files_only=True,
    )
    offload_mode = args.offload_mode
    if offload_mode == "auto":
        offload_mode = "group" if args.num_frames > 64 else "model"
    if offload_mode == "group" and hasattr(pipe, "enable_group_offload"):
        # Block-level offload keeps transformer activations bounded for 97
        # frames on a 24GB card. One block per group is slower but robust and
        # still lets the two independent Hunyuan candidates run in parallel.
        pipe.enable_group_offload(
            onload_device=torch.device(device),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
            exclude_modules=["vae"],
        )
        # The Hunyuan VAE performs a 3-D convolution on the conditioning
        # image before the transformer hooks run. Keep this one tiled module
        # resident so Accelerate does not feed CUDA tensors to CPU weights.
        pipe.vae.to(device=device, dtype=torch.bfloat16)
    elif offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        pipe.enable_model_cpu_offload(device=device)
    # A 97-frame clip can otherwise exceed 24GB during the transformer
    # attention pass even though the model weights are CPU-offloaded. Diffusers
    # slices attention heads without changing the denoising trajectory; this is
    # the memory/speed trade-off needed for a reproducible 4-second output on
    # a single RTX 3090. ``none`` remains available for profiling.
    if args.attention_slicing != "none" and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(args.attention_slicing)
    # Tiled VAE decoding keeps 480p/97-frame clips within a 24GB card and is
    # the optimization recommended by the official Diffusers example.
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        image=image, prompt=args.prompt, negative_prompt=args.negative_prompt,
        num_frames=args.num_frames, num_inference_steps=args.steps,
        generator=generator, output_type="np",
    )
    frames = result.frames[0]
    resized: list[np.ndarray] = []
    for frame in frames:
        image = Image.fromarray(to_uint8(frame), mode="RGB")
        resized.append(np.asarray(image.resize((args.width, args.height), Image.Resampling.LANCZOS)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(resized, str(args.output), fps=args.fps)
    peak = int(torch.cuda.max_memory_allocated(device) / 1024 / 1024)
    record = {
        "video_path": str(args.output.resolve()), "seed": args.seed,
        "frames": len(resized), "fps": args.fps, "width": args.width,
        "height": args.height, "steps": args.steps,
        "attention_slicing": args.attention_slicing,
        "offload_mode": offload_mode,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "peak_vram_mb": peak, "temporal_stability": stability_score(resized),
    }
    args.result_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
