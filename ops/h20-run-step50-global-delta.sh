#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "$0")/.." && pwd)"
exec >>"$ROOT/launch-logs/twohead-lr10-step50-global-delta.log" 2>&1
trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" >"$ROOT/control/twohead-lr10-step50-global-delta.exit"; exit "$rc"' EXIT
printf '%s\n' "$$" >"$ROOT/control/twohead-lr10-step50-global-delta.pid"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
timeout --signal=TERM --kill-after=15s 600s "$ROOT/envs/rdm/bin/python" "$REPO/ops/h20-step50-global-delta.py"
