#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX=mmd2-ten-stage-portraits
mkdir -p "$ROOT/control" "$ROOT/launch-logs"
exec >>"$ROOT/launch-logs/$PREFIX.log" 2>&1
PIDS=()
finish() {
  rc=$?
  trap - EXIT
  if (( rc != 0 )); then
    for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  fi
  printf '%s\n' "$rc" >"$ROOT/control/$PREFIX.exit"
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP
exec 9>"$ROOT/control/$PREFIX.lock"
flock -n 9 || exit 3
[[ ! -f "$ROOT/control/$PREFIX.pid" ]] || { echo 'Prior attempt exists; inspect before retry'; exit 4; }
printf '%s\n' "$$" >"$ROOT/control/$PREFIX.pid"
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH="$REPO/z-image:$REPO"
PYTHON="$ROOT/envs/rdm/bin/python"
for gpu in 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" "$REPO/ops/h20-mmd2-ten-stage-preview.py" --gpu "$gpu" \
    >"$ROOT/launch-logs/$PREFIX-gpu$gpu.log" 2>&1 &
  PIDS+=("$!")
done
fail=0
for i in 0 1; do
  set +e
  wait "${PIDS[$i]}"
  rc=$?
  set -e
  printf '%s\n' "$rc" >"$ROOT/control/$PREFIX-gpu$(( i+6 )).exit"
  (( rc == 0 )) || fail=1
done
(( fail == 0 )) || exit 1
CUDA_VISIBLE_DEVICES='' "$PYTHON" "$REPO/ops/h20-export-mmd2-ten-stages.py"
