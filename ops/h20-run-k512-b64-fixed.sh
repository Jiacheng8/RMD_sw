#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GROUP="${1:-mmd-gan}"
if [[ "$GROUP" == all ]]; then
  # Separate fresh0 runs, serialized on the same two GPUs. Failures stop the pair.
  bash "$0" mmd-only
  bash "$0" mmd-gan
  exit 0
fi
case "$GROUP" in
  mmd-only) PREFIX=dino-qwenvl-k512-b64-mmd-only-from0-wrapper ;;
  mmd-gan) PREFIX=dino-qwenvl-k512-b64-fixed-from0-wrapper ;;
  *) printf 'usage: %s [all|mmd-only|mmd-gan]\n' "$0" >&2; exit 2 ;;
esac
mkdir -p "$ROOT/control" "$ROOT/launch-logs"
exec >>"$ROOT/launch-logs/$PREFIX.log" 2>&1
CHILD=''
finish() { rc=$?; trap - EXIT; printf '%s\n' "$rc" >"$ROOT/control/$PREFIX.exit"; exit "$rc"; }
stop() { trap '' TERM INT HUP; [[ -z "$CHILD" ]] || kill -TERM "$CHILD" 2>/dev/null || true; set +e; [[ -z "$CHILD" ]] || wait "$CHILD"; exit 143; }
trap finish EXIT
trap stop TERM INT HUP
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export PYTHONPATH="$REPO/z-image:$REPO" CUDA_VISIBLE_DEVICES=6,7 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RDM_QWEN3VL_SNAPSHOT="$ROOT/models/qwen3vl-4b-vision/ebb281ec70b05090aa6165b016eac8ec08e71b17"
export RDM_QWEN3VL_SPATIAL_SNAPSHOT="$ROOT/models/qwen3vl-4b-patchgan/ebb281ec70b05090aa6165b016eac8ec08e71b17"
"$ROOT/envs/rdm/bin/python" "$REPO/ops/h20-k512-b64-fixed-workflow.py" --group "$GROUP" &
CHILD=$!
wait "$CHILD"
