#!/usr/bin/env bash
set -euo pipefail

# Starts the colocated FastAPI gateway on the GPU host.  Keep configuration in
# the process environment so credentials never enter the repository.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${METACUT_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
BACKEND="${1:-${AIGC_BACKEND:-fake}}"
VENV_PYTHON="${AIGC_SERVICE_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

# A GPU host may already provide a CUDA-enabled model environment (for example
# a shared VLM environment).  Keep the gateway's default venv unchanged, but
# allow deployments to select that runtime without copying multi-GB torch
# wheels into the project venv.
if [[ -n "${AIGC_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${AIGC_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi

mkdir -p "${PROJECT_ROOT}/runs/aigc-service" "${PROJECT_ROOT}/public/uploads/local-aigc"
export AIGC_BACKEND="${BACKEND}"
export AIGC_ASSET_ROOTS="${AIGC_ASSET_ROOTS:-${PROJECT_ROOT}/public/uploads}"
export AIGC_OUTPUT_ROOT="${AIGC_OUTPUT_ROOT:-${PROJECT_ROOT}/public/uploads/local-aigc}"
export AIGC_URL_ALLOWLIST="${AIGC_URL_ALLOWLIST:-localhost,127.0.0.1,assets.example.test}"

cd "${PROJECT_ROOT}/services/aigc-service"
exec "${VENV_PYTHON}" -m uvicorn app.main:app --host 127.0.0.1 --port "${AIGC_PORT:-8100}" \
  2>&1 | tee -a "${PROJECT_ROOT}/runs/aigc-service/${BACKEND}-api.log"
