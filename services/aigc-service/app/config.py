from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str = os.getenv("AIGC_SERVICE_NAME", "metacut-aigc-service")
    backend: str = os.getenv("AIGC_BACKEND", "fake").strip().lower()
    api_token: str | None = os.getenv("AIGC_API_TOKEN") or None
    asset_roots: tuple[Path, ...] = tuple(
        Path(item).expanduser().resolve(strict=False)
        for item in _csv("AIGC_ASSET_ROOTS", "/data/assets,/data/uploads")
    )
    url_allowlist: tuple[str, ...] = _csv(
        "AIGC_URL_ALLOWLIST", "localhost,127.0.0.1"
    )
    allow_data_urls: bool = _bool("AIGC_ALLOW_DATA_URLS", True)
    max_data_url_bytes: int = int(os.getenv("AIGC_MAX_DATA_URL_BYTES", str(10 * 1024 * 1024)))
    fake_job_start_delay_ms: int = int(os.getenv("AIGC_FAKE_JOB_START_DELAY_MS", "20"))
    fake_job_run_delay_ms: int = int(os.getenv("AIGC_FAKE_JOB_RUN_DELAY_MS", "40"))
    output_root: Path = Path(os.getenv("AIGC_OUTPUT_ROOT", "/data/outputs")).expanduser().resolve(strict=False)
    device_vlm: str = os.getenv("AIGC_DEVICE_VLM", "cuda:0")
    device_image: str = os.getenv("AIGC_DEVICE_IMAGE", "cuda:1")
    device_video: str = os.getenv("AIGC_DEVICE_VIDEO", "cuda:2")
    vlm_model: str = os.getenv("AIGC_VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
    vlm_revision: str = os.getenv("AIGC_VLM_REVISION", "unpinned-dev")
    vlm_path: str | None = os.getenv("AIGC_VLM_PATH") or None
    internvl_model: str = os.getenv("AIGC_INTERNVL_MODEL", "OpenGVLab/InternVL3_5-8B-HF")
    internvl_path: str | None = os.getenv("AIGC_INTERNVL_PATH") or None
    image_model: str = os.getenv("AIGC_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
    image_revision: str = os.getenv("AIGC_IMAGE_REVISION", "unpinned-dev")
    sdxl_path: str | None = os.getenv("AIGC_SDXL_PATH") or None
    ip_adapter_repo: str | None = os.getenv("AIGC_IP_ADAPTER_REPO") or None
    ip_adapter_subfolder: str | None = os.getenv("AIGC_IP_ADAPTER_SUBFOLDER") or None
    ip_adapter_weight_name: str = os.getenv("AIGC_IP_ADAPTER_WEIGHT_NAME", "ip-adapter_sdxl.safetensors")
    ip_adapter_scale: float = float(os.getenv("AIGC_IP_ADAPTER_SCALE", "0.65"))
    flux2_path: str | None = os.getenv("AIGC_FLUX2_PATH") or None
    flux2_model: str = os.getenv("AIGC_FLUX2_MODEL", "black-forest-labs/FLUX.2-klein-4B")
    video_model: str = os.getenv("AIGC_VIDEO_MODEL", "Lightricks/LTX-Video")
    video_revision: str = os.getenv("AIGC_VIDEO_REVISION", "unpinned-dev")
    ltxv_path: str | None = os.getenv("AIGC_LTXV_PATH") or None
    # The official LTX 0.9.8 2B-distilled runtime has an older Transformers
    # constraint than Qwen3-VL. Keep it in an optional isolated venv and call
    # it as a subprocess instead of corrupting the Qwen worker environment.
    ltx_runtime: str = os.getenv("AIGC_LTX_RUNTIME", "diffusers").strip().lower()
    ltx_python: str | None = os.getenv("AIGC_LTX_PYTHON") or None
    ltx_inference_script: str | None = os.getenv("AIGC_LTX_INFERENCE_SCRIPT") or None
    ltx_pipeline_config: str | None = os.getenv("AIGC_LTX_PIPELINE_CONFIG") or None
    ltx_timeout_seconds: int = int(os.getenv("AIGC_LTX_TIMEOUT_SECONDS", "900"))
    wan22_model: str = os.getenv("AIGC_WAN22_MODEL", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    wan22_path: str | None = os.getenv("AIGC_WAN22_PATH") or None
    # HunyuanVideo-1.5 480p I2V step-distilled is intentionally isolated from
    # the LTX/Wan pipeline.  A request can launch two independent candidates
    # on two physical cards and select the more temporally stable result.
    hunyuan15_model: str = os.getenv(
        "AIGC_HY15_MODEL",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
    )
    hunyuan15_path: str | None = os.getenv("AIGC_HY15_PATH") or None
    hunyuan15_gpus: tuple[str, ...] = _csv("AIGC_HY15_GPUS", "1,2")
    hunyuan15_steps: int = int(os.getenv("AIGC_HY15_STEPS", "12"))
    hunyuan15_timeout_seconds: int = int(os.getenv("AIGC_HY15_TIMEOUT_SECONDS", "2400"))
    hunyuan15_python: str | None = os.getenv("AIGC_HY15_PYTHON") or None
    hunyuan15_worker: str | None = os.getenv("AIGC_HY15_WORKER") or None
    # Phase 3 can rank the two Hunyuan candidates by product identity as well
    # as temporal smoothness.  The default remains temporal-only so Phase 2.5
    # results stay exactly reproducible.
    dino_path: str | None = os.getenv("AIGC_DINO_PATH") or None
    hunyuan15_identity_weight: float = float(os.getenv("AIGC_HY15_IDENTITY_WEIGHT", "0.55"))
    # Large diffusion/video pipelines exceed a single 24 GB card when fully
    # resident. Accelerate's model CPU offload keeps the public API unchanged
    # while allowing Wan2.2/FLUX.2-klein to run on RTX 3090 hardware.
    enable_model_cpu_offload: bool = _bool("AIGC_ENABLE_MODEL_CPU_OFFLOAD", False)
    enable_sequential_cpu_offload: bool = _bool("AIGC_ENABLE_SEQUENTIAL_CPU_OFFLOAD", False)
    inference_steps_image: int = int(os.getenv("AIGC_IMAGE_STEPS", "20"))
    inference_steps_video: int = int(os.getenv("AIGC_VIDEO_STEPS", "8"))

    def __post_init__(self) -> None:
        if self.backend not in {"fake", "local"}:
            raise ValueError(
                f"Unsupported AIGC_BACKEND={self.backend!r}; use fake or local"
            )
        if self.max_data_url_bytes < 1:
            raise ValueError("AIGC_MAX_DATA_URL_BYTES must be positive")
        if self.ltx_runtime not in {"diffusers", "official"}:
            raise ValueError("AIGC_LTX_RUNTIME must be diffusers or official")
        if self.ltx_timeout_seconds < 1:
            raise ValueError("AIGC_LTX_TIMEOUT_SECONDS must be positive")
        if len(self.hunyuan15_gpus) < 2:
            raise ValueError("AIGC_HY15_GPUS must contain at least two physical GPU indices")
        if self.hunyuan15_steps not in {4, 8, 12, 50}:
            raise ValueError("AIGC_HY15_STEPS must be one of 4, 8, 12, or 50")
        if self.hunyuan15_timeout_seconds < 1:
            raise ValueError("AIGC_HY15_TIMEOUT_SECONDS must be positive")
        if not 0.0 <= self.hunyuan15_identity_weight <= 1.0:
            raise ValueError("AIGC_HY15_IDENTITY_WEIGHT must be between 0 and 1")


settings = Settings()
