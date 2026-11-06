#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STEP="$1"
GPU="$2"
shift 2
[[ "$GPU" == 6 || "$GPU" == 7 ]]
[[ "$STEP" =~ ^(50|100|150|200|250|300|350|400|450|500)$ ]]
PREFIX="twohead-qib-portraits-$STEP"
mkdir -p "$ROOT/control" "$ROOT/launch-logs"
exec >>"$ROOT/launch-logs/$PREFIX.log" 2>&1
trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" >"$ROOT/control/$PREFIX.exit"; printf "portrait_exit=%s utc=%s\n" "$rc" "$(date -u +%FT%TZ)"; exit "$rc"' EXIT
if [[ -f "$ROOT/control/$PREFIX.pid" ]]; then
  previous="$(cat "$ROOT/control/$PREFIX.pid")"
  if kill -0 "$previous" 2>/dev/null; then echo 'Portrait preview already active'; exit 4; fi
fi
printf '%s\n' "$$" >"$ROOT/control/$PREFIX.pid"
TRAIN_SOURCE="${H20_TRAIN_SOURCE:-/data/rdm-workspace/RDM-twohead-lr10-v1}"
source "$TRAIN_SOURCE/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$TRAIN_SOURCE/z-image:$TRAIN_SOURCE"
cd "$TRAIN_SOURCE"
timeout --signal=TERM --kill-after=30s 1800s "$ROOT/envs/rdm/bin/python" "$REPO/ops/h20-qib-portrait-preview.py" --step "$STEP" --gpu "$GPU" "$@"
