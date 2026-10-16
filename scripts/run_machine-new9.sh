#!/usr/bin/env bash
# Experiment 9 on a fresh machine, in one command: SW-LMMD + a BLOCK GAN critic (one logit per
# step's B=32 fresh rows) + prompt-grouped windows (4 seeds per prompt), COCO reference only (no
# GenEval block), K=128 / B=32, 2000 steps (configs/sw_lmmd_train_h100_2gpu_gan_grouped_blockcritic.yaml).
#
# It is scripts/new_machine.sh with this config: conda env, downloads (models and the COCO store;
# no GenEval block), the Qwen3 context, the GenEval scorer, training on every usable GPU, and when
# training ends GenEval + PickScore + seed diversity for all 10 checkpoints -> <run dir>/eval_summary.md.
#
#   git clone -b unbiased-siblings https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   echo hf_xxx > .hf_token && chmod 600 .hf_token        # jiachengcui888's Read token
#   tmux new -s train
#   bash scripts/run_machine-new9.sh --dry-run            # checks + plan only, changes nothing
#   bash scripts/run_machine-new9.sh                      # the run (root /root/rdm-sets)
#   bash scripts/run_machine-new9.sh --resume             # continue after an interruption
#   bash scripts/run_machine-new9.sh --root /data/me/rdm-sets --gpus 2
#
# --root DIR   data root (default /root/rdm-sets); needs ~350 GB free (10 checkpoints).
# --gpus N     training GPUs (default: every visible card, rounded down to 1/2/4/8 so B=32 splits
#              into whole micro-batches). Choose cards with CUDA_VISIBLE_DEVICES. 80 GB cards.
#              (The block critic trains on the fresh 32 rows on every GPU: no extra constraint.)
# --no-eval    train only: skip the evaluation when training ends.
# Every other option goes to new_machine.sh unchanged (--gate, --micro-batch 2, --skip-env, ...).
# (The shared body is scripts/_run_experiment.sh.)
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_run_experiment.sh" "${BASH_SOURCE[0]}" \
  configs/sw_lmmd_train_h100_2gpu_gan_grouped_blockcritic.yaml 32 \
  "experiment 9: grouped + block GAN critic, COCO reference, K=128 / B=32" "$@"
