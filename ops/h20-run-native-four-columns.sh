#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX=twohead-lr10-native-four-columns
exec >>"$ROOT/launch-logs/$PREFIX.log" 2>&1
trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" >"$ROOT/control/$PREFIX.exit"; exit "$rc"' EXIT
printf '%s\n' "$$" >"$ROOT/control/$PREFIX.pid"
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
timeout --signal=TERM --kill-after=15s 600s "$ROOT/envs/rdm/bin/python" "$REPO/ops/h20-export-native-four-columns.py"
