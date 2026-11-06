#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEP="${1:?checkpoint step}"; GPU="${2:?physical preview GPU}"
[[ "$STEP" == 0 || "$STEP" == 150 || "$STEP" == 200 ]]
[[ "$GPU" == 6 || "$GPU" == 7 ]]
mkdir -p "$ROOT/control" "$ROOT/launch-logs"
exec >>"$ROOT/launch-logs/spatial-A-preview-$STEP.log" 2>&1
trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" >"$ROOT/control/spatial-A-preview-$STEP.exit"; exit "$rc"' EXIT
printf '%s\n' "$$" >"$ROOT/control/spatial-A-preview-$STEP.pid"
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
cd "$REPO"
"$ROOT/envs/rdm/bin/python" ops/h20-spatial-preview.py --step "$STEP" --gpu "$GPU"
