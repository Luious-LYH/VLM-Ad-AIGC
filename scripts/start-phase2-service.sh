#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${METACUT_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
MODEL_ROOT="${AIGC_MODEL_ROOT:-${PROJECT_ROOT}/models}"
if [[ -d "/public/lyh/projects/CV_projects/models" ]]; then
  MODEL_ROOT="${AIGC_MODEL_ROOT:-/public/lyh/projects/CV_projects/models}"
fi
export AIGC_SERVICE_PYTHON="${AIGC_SERVICE_PYTHON:-/public/lyh/.conda/envs/vlm/bin/python}"
export AIGC_PYTHONPATH="/public/lyh/.venvs/diffusers-040:/public/lyh/.venvs/vlm-extra"
# The shared runtimes are unpacked site-package trees rather than standalone
# virtualenvs, so make them visible to the selected Python interpreter.
export PYTHONPATH="${AIGC_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
export AIGC_ENABLE_MODEL_CPU_OFFLOAD=true
export AIGC_ENABLE_SEQUENTIAL_CPU_OFFLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AIGC_BACKEND=local
export AIGC_SDXL_PATH="$MODEL_ROOT/sdxl-base-1.0"
export AIGC_IP_ADAPTER_REPO="$MODEL_ROOT/IP-Adapter"
export AIGC_IP_ADAPTER_SUBFOLDER=sdxl_models
export AIGC_IP_ADAPTER_WEIGHT_NAME=ip-adapter_sdxl.safetensors
export AIGC_VLM_PATH="$MODEL_ROOT/Qwen3-VL-4B-Instruct-v3"
export AIGC_LTXV_PATH="$MODEL_ROOT/LTX-Video-2B-0.9.8"
export AIGC_FLUX2_PATH="$MODEL_ROOT/FLUX.2-klein-4B"
export AIGC_WAN22_PATH="$MODEL_ROOT/Wan2.2-TI2V-5B-Diffusers"
export AIGC_HY15_PATH="${AIGC_HY15_PATH:-$MODEL_ROOT/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled}"
export AIGC_HY15_MODEL="${AIGC_HY15_MODEL:-hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled}"
export AIGC_HY15_GPUS="${AIGC_HY15_GPUS:-1,2}"
export AIGC_HY15_STEPS="${AIGC_HY15_STEPS:-12}"
export AIGC_HY15_PYTHON="${AIGC_HY15_PYTHON:-$AIGC_SERVICE_PYTHON}"
export AIGC_HY15_WORKER="${AIGC_HY15_WORKER:-$PROJECT_ROOT/scripts/run-hunyuan15-worker.py}"
export AIGC_DINO_PATH="${AIGC_DINO_PATH:-$MODEL_ROOT/dinov2-small}"
export AIGC_HY15_IDENTITY_WEIGHT="${AIGC_HY15_IDENTITY_WEIGHT:-0.55}"
export AIGC_DEVICE_VLM="${AIGC_DEVICE_VLM:-cuda:0}"
export AIGC_DEVICE_IMAGE="${AIGC_DEVICE_IMAGE:-cuda:1}"
export AIGC_DEVICE_VIDEO="${AIGC_DEVICE_VIDEO:-cuda:2}"
export AIGC_ASSET_ROOTS="$PROJECT_ROOT/public/uploads"
export AIGC_OUTPUT_ROOT="$PROJECT_ROOT/public/uploads/local-aigc"
export AIGC_IMAGE_STEPS=28
export AIGC_VIDEO_STEPS=12
export AIGC_LTX_RUNTIME=official
export AIGC_LTX_PYTHON="$PROJECT_ROOT/scripts/run-ltx-runtime.sh"
export AIGC_LTX_INFERENCE_SCRIPT="$PROJECT_ROOT/third_party/LTX-Video/inference.py"
export AIGC_LTX_PIPELINE_CONFIG="$PROJECT_ROOT/configs/ltxv-2b-0.9.8-distilled.server.yaml"

cd "$PROJECT_ROOT/services/aigc-service"
exec "$AIGC_SERVICE_PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port "${AIGC_PORT:-8100}" \
  2>&1 | tee -a "$PROJECT_ROOT/runs/aigc-service/local-api.log"
