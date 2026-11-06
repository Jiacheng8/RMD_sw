#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX=dino-qwenvl-mmd2-reference
mkdir -p "$ROOT/control" "$ROOT/launch-logs"
exec >>"$ROOT/launch-logs/$PREFIX.log" 2>&1
OWNS=0
CHILD=''
finish() { rc=$?; trap - EXIT; if [[ "$OWNS" == 1 ]]; then printf '%s\n' "$rc" >"$ROOT/control/$PREFIX.exit"; fi; exit "$rc"; }
stop() { trap '' TERM INT HUP; [[ -z "$CHILD" ]] || kill -TERM "$CHILD" 2>/dev/null || true; set +e; [[ -z "$CHILD" ]] || wait "$CHILD"; exit 143; }
trap finish EXIT
trap stop TERM INT HUP
exec 8>"$ROOT/control/$PREFIX.lock"
flock -n 8 || { echo 'Reference builder already active'; exit 3; }
printf '%s\n' "$$" >"$ROOT/control/$PREFIX.pid"
OWNS=1
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RDM_QWEN3VL_SNAPSHOT="$ROOT/models/qwen3vl-4b-vision/ebb281ec70b05090aa6165b016eac8ec08e71b17"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$ROOT/envs/rdm/bin/python" "$REPO/ops/h20-build-dino-qwenvl-reference.py" &
CHILD=$!
wait "$CHILD"
