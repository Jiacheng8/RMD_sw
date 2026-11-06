#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GPU="$1"
[[ "$GPU" == 6 || "$GPU" == 7 ]]
PREFIX=twohead-lr10-compare-five-stages200
mkdir -p "$ROOT/control" "$ROOT/launch-logs"
export TMPDIR="$ROOT/.cache/tmp"
exec >>"$ROOT/launch-logs/$PREFIX.log" 2>&1
trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" >"$ROOT/control/$PREFIX.exit"; printf "comparison_exit=%s utc=%s\n" "$rc" "$(date -u +%FT%TZ)"; exit "$rc"' EXIT
if [[ -f "$ROOT/control/$PREFIX.pid" ]]; then
  previous="$(cat "$ROOT/control/$PREFIX.pid")"
  if kill -0 "$previous" 2>/dev/null; then echo 'Comparison already active'; exit 4; fi
fi
printf '%s\n' "$$" >"$ROOT/control/$PREFIX.pid"
source /data/rdm-workspace/RDM-twohead-lr10-v1/ops/h20-data-env.sh
source "$ROOT/env.sh"
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/data/rdm-workspace/RDM-twohead-lr10-v1/z-image:/data/rdm-workspace/RDM-twohead-lr10-v1
cd /data/rdm-workspace/RDM-twohead-lr10-v1
timeout --signal=TERM --kill-after=30s 1800s "$ROOT/envs/rdm/bin/python" "$REPO/ops/h20-preview-five-stages200.py" --gpu "$GPU"
