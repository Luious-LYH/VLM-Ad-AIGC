#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
MODEL_ROOT="${AIGC_MODEL_ROOT:-${PROJECT_ROOT}/models}"
if [[ -d "/public/lyh/projects/CV_projects/models" ]]; then
  MODEL_ROOT="${AIGC_MODEL_ROOT:-/public/lyh/projects/CV_projects/models}"
fi
PYTHON="${AIGC_HY15_PYTHON:-/public/lyh/.conda/envs/vlm/bin/python}"
GPU="${1:-1}"
TAG="${2:-single}"
FRAMES="${HY15_SMOKE_FRAMES:-49}"
STEPS="${HY15_SMOKE_STEPS:-8}"

cd "$PROJECT_ROOT"
mkdir -p "runs/phase25/smoke"
export PYTHONPATH="/public/lyh/.venvs/diffusers-040:/public/lyh/.venvs/vlm-extra${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU"

"$PYTHON" scripts/run-hunyuan15-worker.py \
  --model-path "$MODEL_ROOT/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled" \
  --input-image "$PROJECT_ROOT/samples/example/keyframes/flux2_klein.png" \
  --prompt 'Premium beauty product commercial, slow elegant 3/4 turn on a reflective studio pedestal, soft warm key light, controlled specular highlights, shallow depth of field, preserve bottle shape, cap, label layout and brand colors, no text changes, cinematic but physically stable motion' \
  --negative-prompt 'deformed packaging, duplicate product, melted label, unreadable text, watermark, flicker, jitter, camera shake, harsh exposure' \
  --output "$PROJECT_ROOT/runs/phase25/smoke/${TAG}-hy15.mp4" \
  --result-json "$PROJECT_ROOT/runs/phase25/smoke/${TAG}-hy15.json" \
  --width 576 --height 768 --num-frames "$FRAMES" --fps 24 --steps "$STEPS" --seed 20260906 \
  2>&1 | tee "$PROJECT_ROOT/runs/phase25/smoke/${TAG}-hy15.log"
