#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX=qib350-two-gpu
mkdir -p "$ROOT/control" "$ROOT/launch-logs" "$ROOT/qib"
exec >>"$ROOT/launch-logs/$PREFIX.log" 2>&1
CHILD=''
finish() { rc=$?; trap - EXIT; printf '%s\n' "$rc" >"$ROOT/control/$PREFIX.exit"; exit "$rc"; }
stop() { trap '' TERM INT HUP; [[ -z "$CHILD" ]] || kill -TERM "$CHILD" 2>/dev/null || true; set +e; [[ -z "$CHILD" ]] || wait "$CHILD"; exit 143; }
trap finish EXIT
trap stop TERM INT HUP
exec 9>/data/rdm-workspace/rdm-run.lock
flock -n 9 || { echo 'Owned experiment lock is busy'; exit 3; }
[[ ! -f "$ROOT/control/$PREFIX.pid" ]] || { echo 'Prior evaluation exists; inspect before relaunch'; exit 4; }
printf '%s\n' "$$" >"$ROOT/control/$PREFIX.pid"
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONPATH="$REPO/z-image:$REPO"
"$ROOT/envs/rdm/bin/python" "$REPO/ops/h20-qib350-workflow.py" &
CHILD=$!
wait "$CHILD"
