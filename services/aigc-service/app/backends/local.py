"""Lazy, local-only adapters for the MetaCut model portfolio.

No ML package is imported until a particular model is selected.  This matters
on the offline host: an unavailable experimental backend is reported as an
actionable job failure while SDXL/LTXV can remain a usable default path.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from ..schemas import ProductV1Output, RunMetadata, VideoAnalysisV1Output
from .base import AigcBackend


class BackendUnavailable(RuntimeError):
    """A selected model cannot be loaded in the current offline environment."""


def _json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("VLM output did not contain a JSON object")
    return json.loads(text[start : end + 1])


class LocalBackend(AigcBackend):
    name = "local"

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self._loaded: dict[str, Any] = {}
        self.cfg.output_root.mkdir(parents=True, exist_ok=True)

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "ready": True,
            "local_only": True,
            "operations": [
                "reference.analyze", "product.parse", "script.stitch",
                "image.generate", "video.generate",
            ],
            "models": {
                "vlm": {"id": self.cfg.vlm_model, "revision": self.cfg.vlm_revision, "loaded": "vlm" in self._loaded},
                "image": {
                    "sdxl_ip_adapter": {"id": self.cfg.image_model, "path": self.cfg.sdxl_path, "loaded": "sdxl" in self._loaded},
                    "flux2_klein": {"id": self.cfg.flux2_model, "path": self.cfg.flux2_path, "loaded": "flux2" in self._loaded},
                },
                "video": {
                    "ltxv_2b": {"id": self.cfg.video_model, "path": self.cfg.ltxv_path, "loaded": "ltxv" in self._loaded},
                    "wan22_ti2v_5b": {"id": self.cfg.wan22_model, "path": self.cfg.wan22_path, "loaded": "wan22" in self._loaded},
                    "hunyuanvideo_15_i2v": {
                        "id": self.cfg.hunyuan15_model,
                        "path": self.cfg.hunyuan15_path,
                        "loaded": False,
                        "execution": "dual-gpu-candidate-parallel",
                        "gpus": list(self.cfg.hunyuan15_gpus),
                    },
                },
            },
        }

    def _run(self, model_id: str, revision: str, seed: int | None, started: float) -> RunMetadata:
        return RunMetadata(
            run_id=f"run_{uuid.uuid4().hex[:20]}",
            backend=self.name,
            model_id=model_id,
            model_revision=revision,
            seed=seed,
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
            deterministic=False,
        )

    def generation_run(self, kind: str, payload: dict[str, Any]) -> RunMetadata:
        model = str(payload.get("model") or "")
        if kind == "image_generation":
            model_id = self.cfg.image_model if model == "sdxl_ip_adapter" else self.cfg.flux2_model
        else:
            model_id = {
                "ltxv_2b": self.cfg.video_model,
                "wan22_ti2v_5b": self.cfg.wan22_model,
                "hunyuanvideo_15_i2v": self.cfg.hunyuan15_model,
            }.get(model, self.cfg.video_model)
        return self._run(model_id, "local-pinned-at-deploy", payload.get("seed"), time.perf_counter())

    def _torch(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on deployment
            raise BackendUnavailable("PyTorch is not installed in this worker environment") from exc
        if not torch.cuda.is_available():
            raise BackendUnavailable("CUDA is not available to the local model worker")
        return torch

    def _prepare_diffusers_hub_compat(self) -> None:
        """Bridge the small Hub API gap used by Diffusers 0.40 in our VLM env.

        The host keeps Hugging Face Hub 0.36 because Transformers/Qwen3-VL pins
        Hub below 1.0, while Diffusers 0.40 imports two newer cache helpers.
        All production model paths are local/offline, so a cache-tree miss can
        safely degrade to Diffusers' existing local-entry error path.
        """
        try:
            import huggingface_hub
            from huggingface_hub.utils import LocalEntryNotFoundError
            import huggingface_hub.errors as hub_errors

            if not hasattr(huggingface_hub, "get_cached_repo_tree"):
                def _get_cached_repo_tree(*_args: Any, **_kwargs: Any):
                    raise LocalEntryNotFoundError("cached repository tree is unavailable")

                huggingface_hub.get_cached_repo_tree = _get_cached_repo_tree
            if not hasattr(hub_errors, "CachedRepoTreeNotFoundError"):
                class CachedRepoTreeNotFoundError(Exception):
                    pass

                hub_errors.CachedRepoTreeNotFoundError = CachedRepoTreeNotFoundError
        except Exception as exc:  # pragma: no cover - deployment-only compatibility
            raise BackendUnavailable(f"Hugging Face Hub compatibility setup failed: {exc}") from exc

    def _place_pipeline(self, pipe: Any, device: str, *, sequential: bool = False) -> Any:
        """Place a pipeline directly or install Accelerate CPU-offload hooks."""
        if not self.cfg.enable_model_cpu_offload:
            return pipe.to(device)
        try:
            if sequential and self.cfg.enable_sequential_cpu_offload:
                # Wan's 5B transformer is a single top-level component: plain
                # sequential offload still moves the whole transformer to the
                # GPU for one forward and can exceed a 24 GB RTX 3090. Diffusers
                # 0.40 block-level group offload streams one transformer block
                # at a time and keeps the peak activation+weights bounded.
                if hasattr(pipe, "enable_group_offload"):
                    import torch

                    pipe.enable_group_offload(
                        onload_device=torch.device(device),
                        offload_type="block_level",
                        num_blocks_per_group=1,
                        use_stream=False,
                    )
                else:
                    pipe.enable_sequential_cpu_offload(device=device)
            else:
                pipe.enable_model_cpu_offload(device=device)
        except Exception as exc:
            raise BackendUnavailable(f"model CPU offload setup failed on {device}: {exc}") from exc
        return pipe

    def _reset_peak_vram(self, torch: Any, device: str) -> None:
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            # PyTorch 2.5 rejects the string form after CUDA_VISIBLE_DEVICES
            # masking; reset the current visible card as a compatible fallback.
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def _peak_vram_mb(self, torch: Any, device: str) -> int:
        try:
            return int(torch.cuda.max_memory_allocated(device) / 1024 / 1024)
        except Exception:
            return 0

    def _release_loaded_models(self, keys: tuple[str, ...]) -> list[str]:
        """Release resident pipelines before an isolated multi-GPU worker.

        HunyuanVideo-1.5 intentionally uses physical cards 1 and 2 in two
        child processes.  The normal image/video adapters may have left an
        SDXL/FLUX/Wan pipeline resident on one of those cards, so merely
        setting ``CUDA_VISIBLE_DEVICES`` would still risk an OOM.  Remove
        those optional pipelines from the long-lived gateway first.  The VLM
        stays on card 0 and is not touched.
        """
        released: list[str] = []
        for key in keys:
            entry = self._loaded.pop(key, None)
            if entry is None:
                continue
            pipeline = entry[0] if isinstance(entry, tuple) else entry
            try:
                if hasattr(pipeline, "to"):
                    pipeline.to("cpu")
            except Exception:
                # Deleting the last strong reference is still useful when an
                # offload hook does not support an explicit CPU transfer.
                pass
            del entry, pipeline
            released.append(key)
        if released:
            gc.collect()
            try:
                import torch

                for index in range(torch.cuda.device_count()):
                    with torch.cuda.device(index):
                        torch.cuda.empty_cache()
            except Exception:
                pass
        return released

    def _device_vram_mb(self, device: str) -> int:
        """Read physical-card use for an isolated child runtime, if available."""
        if not device.startswith("cuda:"):
            return 0
        try:
            output = subprocess.check_output(
                ["nvidia-smi", f"--id={device.split(':', 1)[1]}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return int(output.splitlines()[0]) if output else 0
        except Exception:
            return 0

    def _load_dino_identity(self) -> tuple[Any, Any] | None:
        """Load the small local DINOv2 encoder lazily on CPU.

        DINO is used only by the Phase 3 candidate selector.  Keeping it on
        CPU avoids taking capacity from the VLM on GPU 0 or the two Hunyuan
        workers on GPUs 1/2, and the 142M-parameter snapshot is fast enough for
        three sampled frames per candidate.
        """
        if "dino_identity" in self._loaded:
            return self._loaded["dino_identity"]
        if not self.cfg.dino_path:
            self._loaded["dino_identity"] = None
            return None
        path = Path(self.cfg.dino_path)
        if not path.is_dir():
            self._loaded["dino_identity"] = None
            return None
        try:
            from transformers import AutoImageProcessor, AutoModel

            processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
            model = AutoModel.from_pretrained(path, local_files_only=True).eval().to("cpu")
            self._loaded["dino_identity"] = (model, processor)
            return self._loaded["dino_identity"]
        except Exception:
            # Selector falls back to temporal-only ranking when an optional
            # metric cannot be loaded; generation itself must remain usable.
            self._loaded["dino_identity"] = None
            return None

    @staticmethod
    def _mask_product(image: Any) -> Any:
        """Suppress the studio background while retaining a centered product."""
        from PIL import Image, ImageDraw

        image = image.convert("RGB")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            (int(image.width * 0.08), int(image.height * 0.04),
             int(image.width * 0.92), int(image.height * 0.98)), fill=255,
        )
        return Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask)

    def _dino_embedding(self, image: Any) -> Any | None:
        loaded = self._load_dino_identity()
        if loaded is None:
            return None
        model, processor = loaded
        try:
            import torch

            inputs = processor(images=self._mask_product(image), return_tensors="pt")
            with torch.inference_mode():
                output = model(**inputs)
            vector = output.pooler_output[0] if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0][0]
            return torch.nn.functional.normalize(vector.float(), dim=0).cpu()
        except Exception:
            return None

    def _candidate_identity_similarity(self, reference_path: str, video_path: str) -> float | None:
        """Compare reference product to candidate video at three stable anchors."""
        try:
            from PIL import Image
            import av
            import torch

            reference = Image.open(reference_path).convert("RGB")
            anchor = self._dino_embedding(reference)
            if anchor is None:
                return None
            container = av.open(video_path)
            stream = container.streams.video[0]
            target_indices = {0, max(0, int(stream.frames or 0) // 2), max(0, int(stream.frames or 0) - 1)}
            sampled: list[Any] = []
            for index, frame in enumerate(container.decode(video=0)):
                if index in target_indices:
                    sampled.append(frame.to_image().convert("RGB"))
                if index > max(target_indices):
                    break
            container.close()
            scores = []
            for image in sampled:
                vector = self._dino_embedding(image)
                if vector is not None:
                    scores.append(float(torch.dot(anchor, vector).item()))
            if not scores:
                return None
            # Cosine is already close to [0, 1] for DINO image features.  Clamp
            # to keep the combined selector numerically well behaved.
            return max(0.0, min(1.0, sum(scores) / len(scores)))
        except Exception:
            return None

    def _rank_hunyuan_candidates(self, candidates: list[dict[str, Any]], payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        mode = str(payload.get("candidate_ranking") or "temporal")
        if mode != "identity_temporal":
            selected = max(candidates, key=lambda item: float(item.get("temporal_stability", 0.0)))
            for item in candidates:
                item["selection_score"] = float(item.get("temporal_stability", 0.0))
            return selected, "temporal"
        weight = float(payload.get("identity_weight") if payload.get("identity_weight") is not None else self.cfg.hunyuan15_identity_weight)
        for item in candidates:
            identity = self._candidate_identity_similarity(str(payload["input_image"]), str(item["video_path"]))
            item["identity_similarity"] = identity
            temporal = float(item.get("temporal_stability", 0.0))
            if identity is None:
                # A missing optional metric never penalizes a candidate more
                # than the original temporal-only selector.
                item["selection_score"] = temporal
            else:
                item["selection_score"] = weight * identity + (1.0 - weight) * temporal
            item["identity_weight"] = weight
        selected = max(candidates, key=lambda item: float(item.get("selection_score", 0.0)))
        return selected, "identity_temporal"

    def _load_qwen(self):
        if "vlm" in self._loaded:
            return self._loaded["vlm"]
        if not self.cfg.vlm_path:
            raise BackendUnavailable("AIGC_VLM_PATH is unset; stage Qwen3-VL weights on the server first")
        torch = self._torch()
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - deployment-only import
            raise BackendUnavailable("transformers with Qwen3-VL support is not installed") from exc
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg.vlm_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.cfg.device_vlm).eval()
        processor = AutoProcessor.from_pretrained(self.cfg.vlm_path, local_files_only=True)
        self._loaded["vlm"] = (model, processor, torch)
        return self._loaded["vlm"]

    def _open_images(self, values: list[str]):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailable("Pillow is not installed") from exc
        images = []
        for value in values:
            path = Path(value)
            if not path.is_file():
                raise BackendUnavailable(f"Local model worker needs a mounted image path, got: {value}")
            images.append(Image.open(path).convert("RGB"))
        return images

    def _qwen_json(self, prompt: str, image_paths: list[str], schema: Any) -> dict[str, Any]:
        model, processor, torch = self._load_qwen()
        images = self._open_images(image_paths)
        content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processor_kwargs: dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
        if images:
            processor_kwargs["images"] = images
        inputs = processor(**processor_kwargs).to(self.cfg.device_vlm)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=1200, do_sample=False)
        trimmed = generated[:, inputs.input_ids.shape[1] :]
        answer = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        raw = _json_object(answer)
        try:
            return schema.model_validate(raw).model_dump(mode="json", exclude_none=True)
        except Exception as exc:
            raise BackendUnavailable(f"Qwen3-VL output failed {schema.__name__} validation: {exc}") from exc

    def invoke(self, operation: str, payload: dict[str, Any]) -> tuple[dict[str, Any], RunMetadata]:
        started = time.perf_counter()
        if operation == "product.parse":
            result = self._qwen_json(
                """Analyze the product references. Return JSON only with schema_version=product/v1 and fields: product_name, category, visual_description, usage_method, core_selling_points (array), target_audience, usage_scene, pain_point_gain_point, ingredients_material, spec_size. Do not invent brand facts.""",
                list(payload.get("product_image_urls", [])),
                ProductV1Output,
            )
        elif operation == "reference.analyze":
            # The deployment script extracts bounded keyframes from a video before
            # invoking this operation; arbitrary video decoding is intentionally
            # not hidden in the web process.
            frames = list(payload.get("frame_paths", []))
            if not frames:
                raise BackendUnavailable("reference analysis requires frame_paths extracted from the uploaded video")
            result = self._qwen_json(
                """Analyze these sampled frames from a short commercial video. Return JSON only with schema_version=video_analysis/v1, meta_info, narrative_structure.timeline_events, rhythm_and_density, camera_and_composition, on_screen_texts, audio_and_beats, visual_and_color. Use seconds relative to the original clip and omit unverifiable audio details.""",
                frames,
                VideoAnalysisV1Output,
            )
        elif operation == "script.stitch":
            # Text-only storyboard generation is kept deterministic in the
            # existing TypeScript stitcher; a real VLM call is not necessary here.
            raise BackendUnavailable("script.stitch is handled by the TypeScript manifest stitcher")
        elif operation == "product.polish":
            product = dict(payload.get("product") or {})
            dirty_fields = [str(field) for field in payload.get("dirty_fields", [])]
            prompt = (
                "Rewrite only the requested product-description fields for a commercial storyboard. "
                "Return JSON only with schema_version=product/v1 and the same product/v1 fields. "
                "Keep factual visual attributes grounded in the supplied JSON; never invent a brand, "
                "logo, ingredient, specification, or capability. Requested fields: "
                f"{dirty_fields}. Current product JSON: {json.dumps(product, ensure_ascii=False)}"
            )
            polished = self._qwen_json(prompt, [], ProductV1Output)
            result = {**product, **{field: polished.get(field, product.get(field)) for field in dirty_fields}}
            result.update({"schema_version": "product/v1", "polish_source": "qwen3_vl",
                           "dirty_fields": dirty_fields})
        else:
            raise BackendUnavailable(f"unsupported local VLM operation: {operation}")
        return result, self._run(self.cfg.vlm_model, self.cfg.vlm_revision, None, started)

    def _load_sdxl(self):
        if "sdxl" in self._loaded:
            return self._loaded["sdxl"]
        if not self.cfg.sdxl_path:
            raise BackendUnavailable("AIGC_SDXL_PATH is unset; stage SDXL weights on the server first")
        torch = self._torch()
        # Diffusers 0.40 imports newer cache helpers at module import time;
        # install the offline compatibility shim before importing the pipeline.
        self._prepare_diffusers_hub_compat()
        try:
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailable("diffusers is not installed in the image worker environment") from exc
        pipe = AutoPipelineForText2Image.from_pretrained(
            self.cfg.sdxl_path, torch_dtype=torch.float16, local_files_only=True, use_safetensors=True
        ).to(self.cfg.device_image)
        if self.cfg.ip_adapter_repo:
            kwargs: dict[str, Any] = {
                "local_files_only": True,
                "weight_name": self.cfg.ip_adapter_weight_name,
            }
            if self.cfg.ip_adapter_subfolder:
                kwargs["subfolder"] = self.cfg.ip_adapter_subfolder
            pipe.load_ip_adapter(self.cfg.ip_adapter_repo, **kwargs)
            pipe.set_ip_adapter_scale(self.cfg.ip_adapter_scale)
        self._loaded["sdxl"] = (pipe, torch)
        return self._loaded["sdxl"]

    def _load_flux2(self):
        if "flux2" in self._loaded:
            return self._loaded["flux2"]
        if not self.cfg.flux2_path:
            raise BackendUnavailable("AIGC_FLUX2_PATH is unset; stage FLUX.2-klein weights on the server first")
        torch = self._torch()
        self._prepare_diffusers_hub_compat()
        try:
            import diffusers
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailable("diffusers is not installed in the image worker environment") from exc
        cls = getattr(diffusers, "Flux2KleinPipeline", None)
        if cls is None:
            raise BackendUnavailable(
                "installed diffusers does not expose Flux2KleinPipeline; stage a current Diffusers build or use the official FLUX.2 runtime"
            )
        pipe = cls.from_pretrained(self.cfg.flux2_path, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe = self._place_pipeline(pipe, self.cfg.device_image)
        self._loaded["flux2"] = (pipe, torch)
        return self._loaded["flux2"]

    def _save_image(self, image: Any, label: str) -> str:
        filename = f"{label}-{uuid.uuid4().hex[:12]}.png"
        path = self.cfg.output_root / filename
        image.save(path)
        return f"/uploads/local-aigc/{filename}"

    def _generate_image(self, payload: dict[str, Any], run: RunMetadata) -> dict[str, Any]:
        started = time.perf_counter()
        model_name = payload.get("model", "sdxl_ip_adapter")
        if model_name == "sdxl_ip_adapter":
            pipe, torch = self._load_sdxl()
            self._reset_peak_vram(torch, self.cfg.device_image)
            references = list(payload.get("reference_images", []))
            if references and not self.cfg.ip_adapter_repo:
                raise BackendUnavailable("SDXL reference generation requires staged IP-Adapter weights (AIGC_IP_ADAPTER_REPO)")
            kwargs: dict[str, Any] = {
                "prompt": payload["prompt"],
                "negative_prompt": payload.get("negative_prompt") or None,
                "width": payload["width"],
                "height": payload["height"],
                "num_inference_steps": self.cfg.inference_steps_image,
                "generator": torch.Generator(device=self.cfg.device_image).manual_seed(int(payload["seed"])),
            }
            if references:
                kwargs["ip_adapter_image"] = self._open_images(references)[0]
            model_id = self.cfg.image_model
        elif model_name == "flux2_klein":
            pipe, torch = self._load_flux2()
            self._reset_peak_vram(torch, self.cfg.device_image)
            references = list(payload.get("reference_images", []))
            kwargs = {
                "prompt": payload["prompt"],
                "width": payload["width"],
                "height": payload["height"],
                "num_inference_steps": self.cfg.inference_steps_image,
                "generator": torch.Generator(device=self.cfg.device_image).manual_seed(int(payload["seed"])),
            }
            # Flux2Klein supports an optional image-conditioned path.  Passing
            # the same product reference used by SDXL/IP-Adapter makes the
            # model comparison meaningful and materially improves identity
            # preservation for packaging, shape and color.
            if references:
                kwargs["image"] = self._open_images(references)[0]
            model_id = self.cfg.flux2_model
        else:
            raise BackendUnavailable(f"unsupported image model: {model_name}")
        paths = []
        for index in range(int(payload.get("num_candidates", 1))):
            kwargs["generator"] = torch.Generator(device=self.cfg.device_image).manual_seed(int(payload["seed"]) + index)
            image = pipe(**kwargs).images[0]
            paths.append(self._save_image(image, f"{model_name}-{index}"))
        latency = int((time.perf_counter() - started) * 1000)
        return {
            "generated_images": paths,
            "selected_image": paths[0],
            "candidate_scores": [{"index": index, "score": 0.0, "status": "unranked"} for index in range(len(paths))],
            "model_id": model_id,
            "model_revision": run.model_revision,
            "seed": payload["seed"],
            "latency_ms": latency,
            "peak_vram_mb": self._peak_vram_mb(torch, self.cfg.device_image),
        }

    def _video_pipeline(self, key: str, path: str | None, classes: tuple[str, ...], device: str):
        if key in self._loaded:
            return self._loaded[key]
        if not path:
            raise BackendUnavailable(f"AIGC_{key.upper()}_PATH is unset; stage the video model first")
        torch = self._torch()
        self._prepare_diffusers_hub_compat()
        try:
            import diffusers
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailable("diffusers is not installed in the video worker environment") from exc
        if key == "wan22":
            # Diffusers treats ftfy as optional, but the installed Wan pipeline
            # calls its symbol unconditionally during prompt cleanup. Keep the
            # text unchanged when that optional package is absent instead of
            # turning a valid local model into a NameError.
            try:
                from diffusers.pipelines.wan import pipeline_wan_i2v

                if not hasattr(pipeline_wan_i2v, "ftfy"):
                    class _IdentityFtfy:
                        @staticmethod
                        def fix_text(text: str) -> str:
                            return text

                    pipeline_wan_i2v.ftfy = _IdentityFtfy
            except Exception as exc:  # pragma: no cover - optional pipeline layout
                raise BackendUnavailable(f"Wan prompt-cleanup compatibility setup failed: {exc}") from exc
        cls = next((getattr(diffusers, name, None) for name in classes if getattr(diffusers, name, None)), None)
        if cls is None:
            raise BackendUnavailable(f"installed diffusers lacks one of: {', '.join(classes)}")
        pipe = cls.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe = self._place_pipeline(pipe, device, sequential=key == "wan22")
        self._loaded[key] = (pipe, torch)
        return self._loaded[key]

    def _save_video(self, frames: Any, label: str, fps: int) -> str:
        try:
            from diffusers.utils import export_to_video
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailable("diffusers video export support is unavailable") from exc
        filename = f"{label}-{uuid.uuid4().hex[:12]}.mp4"
        path = self.cfg.output_root / filename
        export_to_video(frames, str(path), fps=fps)
        return f"/uploads/local-aigc/{filename}"

    def _run_ltx_official(self, payload: dict[str, Any], run: RunMetadata) -> dict[str, Any]:
        """Run the exact LTXV-2B distilled checkpoint in its isolated runtime.

        LTX's official 0.9.8 runtime pins an older Transformers range than
        Qwen3-VL. Running it via a separately provisioned Python executable is
        intentional: it lets both required model families coexist on the same
        host without dependency pin roulette.
        """
        if not self.cfg.ltx_python or not self.cfg.ltx_inference_script or not self.cfg.ltx_pipeline_config:
            raise BackendUnavailable(
                "official LTXV-2B distilled requires AIGC_LTX_PYTHON, AIGC_LTX_INFERENCE_SCRIPT, and AIGC_LTX_PIPELINE_CONFIG"
            )
        input_image = self._open_images([payload["input_image"]])[0]
        # Validate the path using the same asset policy, but hand the isolated
        # worker the actual file path rather than a PIL image.
        del input_image
        output_dir = self.cfg.output_root / f"ltxv-{run.run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        args = [
            self.cfg.ltx_python,
            self.cfg.ltx_inference_script,
            "--prompt", str(payload["prompt"]),
            "--conditioning_media_paths", str(payload["input_image"]),
            "--conditioning_start_frames", "0",
            "--height", str(payload["height"]),
            "--width", str(payload["width"]),
            "--num_frames", str(payload["num_frames"]),
            "--frame_rate", str(payload["fps"]),
            "--seed", str(payload["seed"]),
            "--pipeline_config", self.cfg.ltx_pipeline_config,
            "--output_path", str(output_dir),
        ]
        if payload.get("negative_prompt"):
            args.extend(["--negative_prompt", str(payload["negative_prompt"])])
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        # One LTX child owns its selected GPU. The parent keeps the physical
        # mapping explicit; the child observes it as cuda:0 after masking.
        if self.cfg.device_video.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = self.cfg.device_video.split(":", 1)[1]
        started = time.perf_counter()
        process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        peak_vram_mb = 0
        deadline = time.monotonic() + self.cfg.ltx_timeout_seconds
        while process.poll() is None:
            peak_vram_mb = max(peak_vram_mb, self._device_vram_mb(self.cfg.device_video))
            if time.monotonic() >= deadline:
                process.kill()
                process.communicate()
                raise BackendUnavailable(f"official LTX timed out after {self.cfg.ltx_timeout_seconds}s")
            time.sleep(0.5)
        stdout, stderr = process.communicate()
        if process.returncode:
            detail = (stderr or stdout).strip()[-4000:]
            raise BackendUnavailable(f"official LTX failed (exit={process.returncode}): {detail}")
        candidates = sorted(output_dir.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            raise BackendUnavailable("official LTX exited successfully but did not produce an MP4")
        output = candidates[0]
        relative = output.relative_to(self.cfg.output_root).as_posix()
        return {
            "video_url": f"/uploads/local-aigc/{relative}",
            "model_id": "Lightricks/LTX-Video:ltxv-2b-0.9.8-distilled",
            "model_revision": "official-0.9.8-distilled",
            "seed": payload["seed"],
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "peak_vram_mb": peak_vram_mb,
            "runtime": "official-isolated",
        }

    def _run_hunyuan15(self, payload: dict[str, Any], run: RunMetadata) -> dict[str, Any]:
        """Run two HunyuanVideo-1.5 candidates concurrently on two 3090s.

        The worker is a subprocess on purpose: it gets a single visible CUDA
        device, loads the 8.3B pipeline with CPU offload, and cannot poison the
        long-lived VLM/LTX Python process.  The request still remains queued
        behind the service's generation lock, while both physical cards are
        busy during this one generation.
        """
        if not self.cfg.hunyuan15_path:
            raise BackendUnavailable("AIGC_HY15_PATH is unset; stage HunyuanVideo-1.5 weights first")
        project_root = Path(__file__).resolve().parents[4]
        worker = Path(self.cfg.hunyuan15_worker or project_root / "scripts" / "run-hunyuan15-worker.py")
        python = self.cfg.hunyuan15_python or sys.executable
        if not worker.is_file():
            raise BackendUnavailable(f"Hunyuan worker script not found: {worker}")
        if len(self.cfg.hunyuan15_gpus) < 2:
            raise BackendUnavailable("HunyuanVideo-1.5 requires two GPU indices")

        released_models = self._release_loaded_models(("sdxl", "flux2", "ltxv", "wan22"))

        output_dir = self.cfg.output_root / f"hunyuan15-{run.run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        children: list[tuple[int, subprocess.Popen[str], Path]] = []
        started = time.perf_counter()
        for rank, gpu in enumerate(self.cfg.hunyuan15_gpus[:2]):
            result_path = output_dir / f"candidate-{rank}.json"
            args = [
                python, str(worker),
                "--model-path", str(self.cfg.hunyuan15_path),
                "--input-image", str(payload["input_image"]),
                "--prompt", str(payload["prompt"]),
                "--output", str(output_dir / f"candidate-{rank}.mp4"),
                "--result-json", str(result_path),
                "--width", str(payload["width"]), "--height", str(payload["height"]),
                "--num-frames", str(payload["num_frames"]), "--fps", str(payload["fps"]),
                "--steps", str(self.cfg.hunyuan15_steps),
                "--seed", str(int(payload["seed"]) + rank),
            ]
            if payload.get("negative_prompt"):
                args.extend(["--negative-prompt", str(payload["negative_prompt"])])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            children.append((rank, process, result_path))

        deadline = time.monotonic() + self.cfg.hunyuan15_timeout_seconds
        while any(process.poll() is None for _, process, _ in children):
            if time.monotonic() >= deadline:
                for _, process, _ in children:
                    if process.poll() is None:
                        process.kill()
                raise BackendUnavailable(f"HunyuanVideo-1.5 dual-GPU run timed out after {self.cfg.hunyuan15_timeout_seconds}s")
            time.sleep(0.5)

        candidates: list[dict[str, Any]] = []
        failures: list[str] = []
        peak_vram: dict[str, int] = {}
        for rank, process, result_path in children:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                failures.append(f"candidate {rank} exit={process.returncode}: {(stderr or stdout).strip()[-1000:]}")
                continue
            try:
                record = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                failures.append(f"candidate {rank} returned invalid metadata: {exc}")
                continue
            if not Path(str(record.get("video_path", ""))).is_file():
                failures.append(f"candidate {rank} did not produce an MP4")
                continue
            record["rank"] = rank
            candidates.append(record)
            peak_vram[str(self.cfg.hunyuan15_gpus[rank])] = int(record.get("peak_vram_mb", 0))
        if not candidates:
            raise BackendUnavailable("HunyuanVideo-1.5 dual-GPU candidates failed: " + " | ".join(failures))
        selected, ranking_mode = self._rank_hunyuan_candidates(candidates, payload)
        selected_path = Path(str(selected["video_path"]))
        relative = selected_path.relative_to(self.cfg.output_root).as_posix()
        return {
            "video_url": f"/uploads/local-aigc/{relative}",
            "model_id": self.cfg.hunyuan15_model,
            "model_revision": "480p-i2v-step-distilled",
            "seed": payload["seed"],
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "peak_vram_mb": max(peak_vram.values(), default=0),
            "peak_vram_by_gpu": peak_vram,
            "runtime": "dual-gpu-candidate-parallel",
            "gpu_indices": list(self.cfg.hunyuan15_gpus[:2]),
            "candidate_count": len(candidates),
            "selected_candidate": int(selected.get("rank", 0)),
            "temporal_stability": float(selected.get("temporal_stability", 0.0)),
            "identity_similarity": selected.get("identity_similarity"),
            "selection_score": float(selected.get("selection_score", selected.get("temporal_stability", 0.0))),
            "candidate_ranking": ranking_mode,
            "identity_weight": selected.get("identity_weight"),
            "released_models_before_launch": released_models,
            "multi_gpu_strategy": "two independent step-distilled candidates in parallel",
            "candidates": candidates,
            "failed_candidates": failures,
        }

    def _generate_video(self, payload: dict[str, Any], run: RunMetadata) -> dict[str, Any]:
        started = time.perf_counter()
        model_name = payload.get("model", "ltxv_2b")
        if model_name == "ltxv_2b":
            if self.cfg.ltx_runtime == "official":
                return self._run_ltx_official(payload, run)
            pipe, torch = self._video_pipeline("ltxv", self.cfg.ltxv_path, ("LTXImageToVideoPipeline", "LTXPipeline"), self.cfg.device_video)
            model_id = self.cfg.video_model
        elif model_name == "wan22_ti2v_5b":
            pipe, torch = self._video_pipeline("wan22", self.cfg.wan22_path, ("WanImageToVideoPipeline", "WanPipeline"), self.cfg.device_video)
            model_id = self.cfg.wan22_model
        elif model_name == "hunyuanvideo_15_i2v":
            return self._run_hunyuan15(payload, run)
        else:
            raise BackendUnavailable(f"unsupported video model: {model_name}")
        self._reset_peak_vram(torch, self.cfg.device_video)
        image = self._open_images([payload["input_image"]])[0]
        kwargs = {
            "prompt": payload["prompt"],
            "image": image,
            "width": payload["width"],
            "height": payload["height"],
            "num_frames": payload["num_frames"],
            "num_inference_steps": self.cfg.inference_steps_video,
            "generator": torch.Generator(device=self.cfg.device_video).manual_seed(int(payload["seed"])),
        }
        if payload.get("negative_prompt"):
            kwargs["negative_prompt"] = payload["negative_prompt"]
        frames = pipe(**kwargs).frames[0]
        url = self._save_video(frames, model_name, int(payload["fps"]))
        return {
            "video_url": url,
            "model_id": model_id,
            "model_revision": run.model_revision,
            "seed": payload["seed"],
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "peak_vram_mb": self._peak_vram_mb(torch, self.cfg.device_video),
        }

    def generate(self, kind: str, payload: dict[str, Any], run: RunMetadata) -> dict[str, Any]:
        try:
            if kind == "image_generation":
                try:
                    return self._generate_image(payload, run)
                except BackendUnavailable as error:
                    if not self.cfg.allow_explicit_fallback or payload.get("model") != "flux2_klein":
                        raise
                    fallback_run = self._run(self.cfg.image_model, "local-pinned-at-deploy", payload.get("seed"), time.perf_counter())
                    fallback = self._generate_image({**payload, "model": "sdxl_ip_adapter"}, fallback_run)
                    fallback["fallback_reason"] = f"FLUX.2-klein unavailable: {error}"
                    fallback["requested_model"] = "flux2_klein"
                    return fallback
            if kind == "video_generation":
                try:
                    return self._generate_video(payload, run)
                except BackendUnavailable as error:
                    if not self.cfg.allow_explicit_fallback or payload.get("model") != "wan22_ti2v_5b":
                        raise
                    fallback_run = self._run(self.cfg.video_model, "local-pinned-at-deploy", payload.get("seed"), time.perf_counter())
                    fallback = self._generate_video({**payload, "model": "ltxv_2b"}, fallback_run)
                    fallback["fallback_reason"] = f"Wan2.2 TI2V unavailable: {error}"
                    fallback["requested_model"] = "wan22_ti2v_5b"
                    return fallback
            raise BackendUnavailable(f"unsupported generation kind: {kind}")
        finally:
            # Keep loaded models resident, but free per-request intermediates.
            gc.collect()
