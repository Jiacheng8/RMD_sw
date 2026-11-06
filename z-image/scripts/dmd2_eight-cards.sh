#!/usr/bin/env bash
# Baseline on a 8-GPU machine (80 GB cards), in one command: DMD2 (official SDXL recipe: batch 128,
# 5 fake-score updates per student update, lr 5e-7, GAN 5e-3 / 1e-2 on the Turbo renders) distilling the
# undistilled Z-Image (CFG 4) into the same 2-step student as ours, for the same GPU time as ours, then
# Qwen-Image-Bench on its last checkpoint.
#
#   bash z-image/scripts/ours_eight-cards.sh --root /root/rdm-sets       # ours first (the budget)
#   bash z-image/scripts/dmd2_eight-cards.sh --root /root/rdm-sets --dry-run           # checks + plan
#   bash z-image/scripts/dmd2_eight-cards.sh --root /root/rdm-sets                     # prepare, train, evaluate
#   bash z-image/scripts/dmd2_eight-cards.sh --root /root/rdm-sets --resume            # after an interruption
#
# Budget: ours' GPU time, 100 GPU-h by default (--match-gpu-hours H, --match-run DIR or
# --iterations N to change it): ~500 iterations, ~100 student updates [estimate]; no need to wait
# for ours. All options: bash z-image/scripts/_dmd2_run.sh --help
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_dmd2_run.sh" --gpus 8 "$@"
