#!/usr/bin/env bash
# Experiment 2 on a fresh machine, in one command: SW-LMMD + GAN + the COCO & GenEval reference with
# grouped and ungrouped steps ALTERNATING (group_pattern GU), K=128 / B=32
# (configs/sw_lmmd_train_h100_2gpu_gan_grouped_geneval_mixGU.yaml).
#
# Everything else equals the earlier K=128 / B=32 runs, so it lands directly between them:
# all-grouped + GenEval (GenEval 0.802, diversity 0.97x the teacher) and ungrouped (0.835, 0.73x).
# It is scripts/new_machine.sh with this config: conda env, downloads (models, the COCO store AND the
# GenEval block), the Qwen3 context, the GenEval scorer, training, and when training ends GenEval +
# PickScore + seed diversity for all 10 checkpoints -> <run dir>/eval_summary.md.
#
#   git clone -b geneval-block https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   echo hf_xxx > .hf_token && chmod 600 .hf_token        # jiachengcui888's Read token
#   tmux new -s train
#   bash scripts/run_machine-new2.sh --dry-run            # checks + plan only, changes nothing
#   bash scripts/run_machine-new2.sh                      # the run (root /root/rdm-sets)
#   bash scripts/run_machine-new2.sh --resume             # continue after an interruption
#
# --root DIR   data root (default /root/rdm-sets); needs ~350 GB free.
# --gpus N     training GPUs (default: every visible card, rounded down to 1/2/4/8 so B=32 splits
#              into whole micro-batches). Choose cards with CUDA_VISIBLE_DEVICES. 80 GB cards.
# Every other option goes to new_machine.sh unchanged (--gate, --micro-batch 2, --skip-env, ...).
# (The shared body is scripts/_run_experiment.sh.)
#
# The training log gains `grouped_step` (1, 0, 1, 0, ...); group_k_* is logged on grouped steps only.
# Experiments 1 and 2 cannot train on the same machine at the same time (new_machine.sh refuses a
# second training): use two machines, or run this one after experiment 1 has finished.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_run_experiment.sh" "${BASH_SOURCE[0]}" \
  configs/sw_lmmd_train_h100_2gpu_gan_grouped_geneval_mixGU.yaml 32 \
  "experiment 2: grouped/ungrouped steps alternating (GU) + GAN + COCO & GenEval reference, K=128 / B=32" "$@"
