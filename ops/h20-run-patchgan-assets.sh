#!/usr/bin/env bash
# CPU-only asset preparation. No formal training, inference, or evaluation.
set -euo pipefail
MODE="${1:?model or images}"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=/data/rdm-workspace/rdm-sets
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=0
case "$MODE" in
  model) ENTRY=ops/h20-patchgan-prefetch.py ;;
  images) ENTRY=ops/h20-patchgan-prepare-images.py ;;
  *) exit 2 ;;
esac
exec >>"$ROOT/launch-logs/patchgan-$MODE.log" 2>&1
on_exit() { rc=$?; trap - EXIT; echo "$rc" > "$ROOT/control/patchgan-$MODE.exit"; exit "$rc"; }
trap on_exit EXIT
echo "$$" > "$ROOT/control/patchgan-$MODE.pid"
printf 'source=%s\ncommand=%s --root %s\n' "$REPO" "$ENTRY" "$ROOT"
"$ROOT/envs/rdm/bin/python" "$REPO/$ENTRY" --root "$ROOT" "$@"
