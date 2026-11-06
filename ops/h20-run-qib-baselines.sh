#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX=qib-turbo2-turbo9-gpu67
mkdir -p "$ROOT/control" "$ROOT/launch-logs"
exec >>"$ROOT/launch-logs/$PREFIX.log" 2>&1
CHILD=''
finish() { rc=$?; trap - EXIT; printf '%s\n' "$rc" >"$ROOT/control/$PREFIX.exit"; exit "$rc"; }
stop() { trap '' TERM INT HUP; [[ -z "$CHILD" ]] || kill -TERM "$CHILD" 2>/dev/null || true; set +e; [[ -z "$CHILD" ]] || wait "$CHILD"; exit 143; }
trap finish EXIT
trap stop TERM INT HUP
exec 9>"$ROOT/control/$PREFIX.lock"
flock -n 9 || exit 3
[[ ! -f "$ROOT/control/$PREFIX.pid" ]] || { echo 'Prior baseline attempt exists: inspect before relaunch'; exit 4; }
printf '%s\n' "$$" >"$ROOT/control/$PREFIX.pid"
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO/z-image:$REPO"
"$ROOT/envs/rdm/bin/python" "$REPO/ops/h20-qib-baselines-workflow.py" &
CHILD=$!
wait "$CHILD"
