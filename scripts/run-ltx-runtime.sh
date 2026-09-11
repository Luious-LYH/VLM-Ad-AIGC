#!/usr/bin/env bash
set -euo pipefail

# The shared VLM environment has CUDA Torch and multimedia libraries, while
# this overlay supplies LTX's Transformers pin without changing Qwen3-VL.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${METACUT_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
BASE_PYTHON="${AIGC_LTX_BASE_PYTHON:-/public/lyh/.conda/envs/vlm/bin/python}"
BASE_SITE="${AIGC_LTX_SITE_PACKAGES:-/public/lyh/.conda/envs/vlm/lib/python3.11/site-packages}"
export PYTHONPATH="${PROJECT_ROOT}/ltx-runtime:${BASE_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${BASE_PYTHON}" -S "$@"
