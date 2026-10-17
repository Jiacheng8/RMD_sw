#!/usr/bin/env bash
# Ablation exp 1 on a fresh machine, in one command: the SW-LMMD reference-side ablation, mode RANDOM.
# Each step the attraction is taken against K=128 reference rows drawn uniformly from the whole COCO
# store (each row with its own caption's text block), instead of the window's own 128 rows.
# Everything else is "K128 · GAN · grouped" (configs/sw_lmmd_train_h100_2gpu_gan_grouped.yaml):
# K=128 / B=32, prompt-grouped windows (8 prompts x 4 seeds fresh per step), the per-image GAN critic
# at adaptive weight 1.5, COCO reference only, cosine lr, 2000 steps, seed 0.
#
# It is scripts/new_machine.sh for the conda env, the downloads (models and the COCO store), the Qwen3
# context and the GenEval scorer; then the landmark bundles from the HF dataset, a 10-step check,
# training, and GenEval + PickScore + seed diversity for all 10 checkpoints -> <run dir>/eval_summary.md.
#
#   git clone -b ref-ablation https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   echo hf_xxx > .hf_token && chmod 600 .hf_token        # jiachengcui888's Read token
#   tmux new -s train
#   bash experiments/ref_ablation/run_ablation-exp1.sh --dry-run    # checks + plan only, changes nothing
#   bash experiments/ref_ablation/run_ablation-exp1.sh              # the run (root /root/rdm-sets)
#   bash experiments/ref_ablation/run_ablation-exp1.sh --resume     # continue after an interruption
#   bash experiments/ref_ablation/run_ablation-exp1.sh --root /data/me/rdm-sets --gpus 2
#
# --root DIR       data root (default /root/rdm-sets); needs ~315 GB free (10 checkpoints).
# --gpus N         training GPUs (default: every visible card, rounded down to 1/2/4/8 so B=32 splits
#                  into micro-batches of 4). Choose cards with CUDA_VISIBLE_DEVICES. 80 GB cards.
# --seed N         seed (default 0): the latent noise and the random row sampler.
# --check-steps N  the 10-step check before training (0 skips it).
# --no-eval        train only: skip the evaluation when training ends.
# --resume / --allow-low-disk / --dry-run, and new_machine.sh's --skip-env --skip-download
# --skip-preprocess --skip-geneval: see experiments/ref_ablation/_run_ablation.sh.
#
# Run dir: <root>/sw_lmmd/work_dirs/sw-lmmd-refabl-k128b32-gan-grouped-random-s0. Compare with
# exp 2 (global) and K128 · GAN · grouped (GenEval 0.7915 at step 2000, DreamSim diversity 0.249).
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_run_ablation.sh" "${BASH_SOURCE[0]}" random \
  "ablation exp 1: RANDOM reference rows, K=128 / B=32, grouped + per-image GAN, COCO reference" "$@"
