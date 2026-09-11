#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${METACUT_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${AIGC_SERVICE_PYTHON:-/public/lyh/.conda/envs/vlm/bin/python}"
cd "$PROJECT_ROOT"

# Wait for the parallel SDXL+video and InternVL workers.  The polling is
# screen-based so this remains robust if a worker internally spawns children.
while screen -ls 2>/dev/null | grep -qE 'phase2-sdxl-videos|phase2-vlm-internvl'; do
  sleep 30
done

# Reconcile all shared artifacts into the canonical eight combination folders.
# Existing successful records are reused; a missing/failed branch is retried by
# the normal matrix runner after a fresh local API is available.
if ! curl -fsS http://127.0.0.1:8100/health >/dev/null 2>&1; then
  screen -dmS phase2-final-api bash -lc "$PROJECT_ROOT/scripts/start-phase2-service.sh"
  for _ in $(seq 1 60); do
    curl -fsS http://127.0.0.1:8100/health >/dev/null 2>&1 && break
    sleep 2
  done
fi

"$PYTHON" scripts/run-phase2-comparison.py \
  --skip-vlm --skip-existing \
  --config configs/phase2-comparison.json \
  --request-timeout 2400 \
  2>&1 | tee runs/phase2/finalize.log

"$PYTHON" scripts/evaluate-phase2.py \
  --phase2-root public/uploads/phase2 \
  --run-root runs/phase2 \
  --dino-path models/dinov2-small \
  2>&1 | tee runs/phase2/evaluate.log

screen -S phase2-final-api -X quit 2>/dev/null || true
echo "phase2 finalization complete"
