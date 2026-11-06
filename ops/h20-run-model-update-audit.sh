#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec >>"$ROOT/launch-logs/spatial-A-model-update-audit.log" 2>&1
trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" >"$ROOT/control/spatial-A-model-update-audit.exit"; exit "$rc"' EXIT
printf '%s\n' "$$" >"$ROOT/control/spatial-A-model-update-audit.pid"
printf 'source=%s commit=%s\n' "$REPO" "$(cat "$REPO/SOURCE_COMMIT")"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd "$REPO"
"$ROOT/envs/rdm/bin/python" ops/h20-model-update-audit.py
