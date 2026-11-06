#!/usr/bin/env bash
# Asset preparation only. No GPU model, training, inference, or evaluation starts here.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=/data/rdm-workspace/rdm-sets
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=0
exec >>"$ROOT/launch-logs/qwen3vl-prefetch.log" 2>&1
on_exit() { rc=$?; trap - EXIT; echo "$rc" > "$ROOT/control/qwen3vl-prefetch.exit"; exit "$rc"; }
trap on_exit EXIT
echo "$$" > "$ROOT/control/qwen3vl-prefetch.pid"
printf 'source=%s\ncommand=ops/h20-qwen3vl-prefetch.py --root %s\n' "$REPO" "$ROOT"
"$ROOT/envs/rdm/bin/python" "$REPO/ops/h20-qwen3vl-prefetch.py" --root "$ROOT"
