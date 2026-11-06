#!/usr/bin/env bash
# Independent accelerated asset download. Existing HF partials remain untouched.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=/data/rdm-workspace/rdm-sets
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=0
exec >>"$ROOT/launch-logs/qwen3vl-fast-prefetch.log" 2>&1
on_exit() { rc=$?; trap - EXIT; echo "$rc" > "$ROOT/control/qwen3vl-fast-prefetch.exit"; exit "$rc"; }
trap on_exit EXIT
echo "$$" > "$ROOT/control/qwen3vl-fast-prefetch.pid"
printf 'source=%s\nmode=aria2 model prefetch only\n' "$REPO"
"$ROOT/envs/rdm/bin/python" "$REPO/ops/h20-qwen3vl-prefetch.py" --transport aria2 --root "$ROOT"
