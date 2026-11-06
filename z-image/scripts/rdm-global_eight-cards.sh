#!/usr/bin/env bash
# Baseline on a 8-GPU machine, in one command: the original RDM objective (local-to-global matching,
# 10240 fresh rows per step, no cache, no critic) on the same 2-step Z-Image student as ours, for the
# same GPU time as ours, then Qwen-Image-Bench on its last checkpoint.
#
#   bash z-image/scripts/ours_eight-cards.sh --root /root/rdm-sets         # ours first (the budget)
#   bash z-image/scripts/rdm-global_eight-cards.sh --root /root/rdm-sets --dry-run             # checks + plan
#   bash z-image/scripts/rdm-global_eight-cards.sh --root /root/rdm-sets                       # prepare, train, evaluate
#   bash z-image/scripts/rdm-global_eight-cards.sh --root /root/rdm-sets --resume              # after an interruption
#
# Budget: ours' GPU time, 100 GPU-h by default (--match-gpu-hours H, --match-run DIR or --steps N
# to change it). ~6 steps of 10240 rows fit it (one step ~2 h on 8 GPUs, ~4 h on 4, ~8 h on 2 --
# A100 estimates); no need to wait for ours.
# All options: bash z-image/scripts/_rdm_global_run.sh --help
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_rdm_global_run.sh" --gpus 8 "$@"
