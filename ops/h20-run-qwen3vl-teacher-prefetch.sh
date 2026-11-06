#!/usr/bin/env bash
# CPU-only first-shard transfer test; verified tar is retained for feature extraction.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=/data/rdm-workspace/rdm-sets
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=0
exec >>"$ROOT/launch-logs/qwen3vl-teacher-prefetch.log" 2>&1
on_exit() { rc=$?; trap - EXIT; echo "$rc" > "$ROOT/control/qwen3vl-teacher-prefetch.exit"; exit "$rc"; }
trap on_exit EXIT
echo "$$" > "$ROOT/control/qwen3vl-teacher-prefetch.pid"
printf 'source=%s\nmode=CPU-only first teacher shard prefetch\n' "$REPO"
"$ROOT/envs/rdm/bin/python" "$REPO/ops/h20-qwen3vl-reference.py" --root "$ROOT" \
  --dataset-revision 3195cc9e58d83dfe17a6b5d89f9886121872586f --prefetch-only --limit-shards 1
