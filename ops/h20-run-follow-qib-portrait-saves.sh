#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${H20_PORTRAIT_FOLLOW_PREFIX:-twohead-qib-portraits-follow}"
mkdir -p "$ROOT/control" "$ROOT/launch-logs"
exec >>"$ROOT/launch-logs/$PREFIX.log" 2>&1
trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" >"$ROOT/control/$PREFIX.exit"; printf "portrait_follow_exit=%s utc=%s\n" "$rc" "$(date -u +%FT%TZ)"; exit "$rc"' EXIT
if [[ -f "$ROOT/control/$PREFIX.pid" ]]; then
  previous="$(cat "$ROOT/control/$PREFIX.pid")"
  if kill -0 "$previous" 2>/dev/null; then echo 'Portrait follow-up already active'; exit 4; fi
fi
printf '%s\n' "$$" >"$ROOT/control/$PREFIX.pid"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
"$ROOT/envs/rdm/bin/python" "$REPO/ops/h20-follow-qib-portrait-saves.py"
