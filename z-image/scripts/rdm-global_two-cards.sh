#!/usr/bin/env bash
# Baseline on a 2-GPU machine, in one command: the original RDM objective (local-to-global matching,
# 10240 fresh rows per step, no cache, no critic) on the same 2-step Z-Image student as ours, for the
# same GPU time as ours, then Qwen-Image-Bench on its last checkpoint.
#
#   bash z-image/scripts/ours_two-cards.sh --root /root/rdm-sets         # ours first (the budget)
#   bash z-image/scripts/rdm-global_two-cards.sh --root /root/rdm-sets --dry-run             # checks + plan
#   bash z-image/scripts/rdm-global_two-cards.sh --root /root/rdm-sets                       # prepare, train, evaluate
#   bash z-image/scripts/rdm-global_two-cards.sh --root /root/rdm-sets --resume              # after an interruption
#
# With ours' run under the same --root its GPU time is matched automatically; otherwise pass
# --match-gpu-hours H (or --match-run DIR, --steps N). ~7 steps of 10240 rows fit ours' 2000
# (one step ~1.8 h on 8 GPUs, ~3.5 h on 4, ~7 h on 2 -- A100 estimates).
# All options: bash z-image/scripts/_rdm_global_run.sh --help
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_rdm_global_run.sh" --gpus 2 "$@"
