#!/usr/bin/env bash
# Experiment 3 on a fresh machine, in one command: SW-LMMD + GAN + prompt-grouped windows with TWO
# seeds per prompt and the unbiased sibling repulsion, COCO reference only, K=1024 / B=128
# (configs/sw_lmmd_train_h100_gan_grouped_g2_unbiased_k1024b128.yaml).
#
# It is scripts/new_machine.sh with this config: conda env, downloads (models and the COCO store;
# no GenEval block), the Qwen3 context, the GenEval scorer, training on every usable GPU, and when
# training ends GenEval + PickScore + seed diversity for all 10 checkpoints -> <run dir>/eval_summary.md.
#
#   git clone -b unbiased-siblings https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   echo hf_xxx > .hf_token && chmod 600 .hf_token        # jiachengcui888's Read token
#   tmux new -s train
#   bash scripts/run_machine-new3.sh --dry-run            # checks + plan only, changes nothing
#   bash scripts/run_machine-new3.sh                      # the run (root /root/rdm-sets)
#   bash scripts/run_machine-new3.sh --resume             # continue after an interruption
#   bash scripts/run_machine-new3.sh --root /data/me/rdm-sets --gpus 2
#
# --root DIR   data root (default /root/rdm-sets); needs ~350 GB free.
# --gpus N     training GPUs (default: every visible card, rounded down to 1/2/4/8 so B=128 splits
#              into whole micro-batches). Choose cards with CUDA_VISIBLE_DEVICES. 80 GB cards.
# Every other option goes to new_machine.sh unchanged (--gate, --micro-batch 2, --skip-env, ...).
# (The shared body is scripts/_run_experiment.sh.)
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_run_experiment.sh" "${BASH_SOURCE[0]}" \
  configs/sw_lmmd_train_h100_gan_grouped_g2_unbiased_k1024b128.yaml 128 \
  "experiment 3: grouped, 2 seeds per prompt, unbiased sibling repulsion, COCO reference, K=1024 / B=128" "$@"
