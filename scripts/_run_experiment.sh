#!/usr/bin/env bash
# Shared body of the one-command experiment launchers (scripts/run_machine-new*.sh): parse
# --root / --gpus, pick the training GPUs, then hand everything to new_machine.sh with the
# experiment's config and --eval. Not meant to be called directly:
#
#   bash scripts/_run_experiment.sh <launcher path> <config> <B> <title> [launcher args...]
#
# --root DIR   data root (default /root/rdm-sets); needs ~350 GB free.
# --gpus N     training GPUs (default: every visible card, rounded down to 1/2/4/8 so B splits
#              into whole micro-batches of 4). Choose cards with CUDA_VISIBLE_DEVICES.
# Every other option goes to new_machine.sh unchanged (--resume, --gate, --dry-run, ...).
set -euo pipefail
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "bash >= 4.4 required (this is $BASH_VERSION)" >&2; exit 1
fi
[[ $# -ge 4 ]] || { echo "usage: bash scripts/_run_experiment.sh <launcher> <config> <B> <title> [args...]" >&2; exit 2; }
LAUNCHER="$1"; CONFIG="$2"; B="$3"; TITLE="$4"; shift 4
MICRO=4                                     # the configs' memory.micro_batch

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="/root/rdm-sets"
GPUS=""
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)    ROOT="$2"; shift 2 ;;
    --gpus)    GPUS="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$LAUNCHER" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         PASS+=("$1"); shift ;;
  esac
done
[[ -f "$REPO/$CONFIG" ]] || { echo "$CONFIG not found -- is this the geneval-block branch?" >&2; exit 1; }

if [[ -z "$GPUS" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    N=$([[ -z "$CUDA_VISIBLE_DEVICES" ]] && echo 0 || awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
  else
    N=$( (nvidia-smi -L 2>/dev/null || true) | grep -c '^GPU' || true)
  fi
  GPUS=1
  while (( GPUS * 2 <= N && GPUS * 2 <= 8 && B % (GPUS * 2 * MICRO) == 0 )); do GPUS=$(( GPUS * 2 )); done
fi
(( B % (GPUS * MICRO) == 0 )) || {
  echo "--gpus $GPUS does not split B=$B into micro-batches of $MICRO" >&2; exit 2; }

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
printf '\n\033[1m==> %s\033[0m\n' "$TITLE"
printf '    config  %s\n    root    %s\n    gpus    %s for training (grad_accum %s), all visible ones for evaluation\n' \
  "$CONFIG" "$ROOT" "$GPUS" "$(( B / (GPUS * MICRO) ))"
exec bash "$REPO/scripts/new_machine.sh" --root "$ROOT" --config "$CONFIG" --gpus "$GPUS" --eval "${PASS[@]}"
