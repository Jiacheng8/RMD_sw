#!/usr/bin/env bash
# Experiment 1 on a fresh machine, in one command: SW-LMMD + GAN + prompt-grouped windows + the
# COCO & GenEval reference at K=1024 / B=128
# (configs/sw_lmmd_train_h100_gan_grouped_geneval_k1024b128.yaml).
#
# It is scripts/new_machine.sh with this config: conda env, downloads (models, the COCO store AND
# the GenEval block), the Qwen3 context, the GenEval scorer, training on every usable GPU, and when
# training ends GenEval + PickScore + seed diversity for all 10 checkpoints -> <run dir>/eval_summary.md.
#
#   git clone -b geneval-block https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   echo hf_xxx > .hf_token && chmod 600 .hf_token        # jiachengcui888's Read token
#   tmux new -s train
#   bash scripts/run_machine-new1.sh --dry-run            # checks + plan only, changes nothing
#   bash scripts/run_machine-new1.sh                      # the run (root /root/rdm-sets)
#   bash scripts/run_machine-new1.sh --resume             # continue after an interruption
#   bash scripts/run_machine-new1.sh --root /data/me/rdm-sets --gpus 2
#
# --root DIR   data root (default /root/rdm-sets); needs ~350 GB free.
# --gpus N     training GPUs (default: every visible card, rounded down to 1/2/4/8 so B=128 splits
#              into whole micro-batches). Choose cards with CUDA_VISIBLE_DEVICES. 80 GB cards.
# Every other option goes to new_machine.sh unchanged (--gate, --micro-batch 2, --skip-env, ...).
set -euo pipefail
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "bash >= 4.4 required (this is $BASH_VERSION)" >&2; exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/sw_lmmd_train_h100_gan_grouped_geneval_k1024b128.yaml"
B=128
ROOT="/root/rdm-sets"
GPUS=""
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)    ROOT="$2"; shift 2 ;;
    --gpus)    GPUS="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
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
  while (( GPUS * 2 <= N && GPUS * 2 <= 8 )); do GPUS=$(( GPUS * 2 )); done
fi
(( B % (GPUS * 4) == 0 )) || { echo "--gpus $GPUS does not split B=$B into micro-batches of 4" >&2; exit 2; }

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
printf '\n\033[1m==> experiment 1: grouped + GAN + COCO & GenEval reference, K=1024 / B=128\033[0m\n'
printf '    config  %s\n    root    %s\n    gpus    %s for training (grad_accum %s), all visible ones for evaluation\n' \
  "$CONFIG" "$ROOT" "$GPUS" "$(( B / (GPUS * 4) ))"
exec bash "$REPO/scripts/new_machine.sh" --root "$ROOT" --config "$CONFIG" --gpus "$GPUS" --eval "${PASS[@]}"
