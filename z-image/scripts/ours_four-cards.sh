#!/usr/bin/env bash
# Ours on a fresh 4-GPU machine, in one command: Z-Image-Turbo (8-step teacher) -> 2-step student,
# SW-LMMD with one MMD + one GAN critic per student step, 1024 px, the public reference pack
# jiachengcui888/zimage-ref-v1, then Qwen-Image-Bench on the last checkpoint and on the teacher.
#
#   git clone -b zimage https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   tmux new -s train
#   bash z-image/scripts/ours_four-cards.sh --root /data/<you>/rdm-sets --dry-run     # checks + plan, changes nothing
#   bash z-image/scripts/ours_four-cards.sh --root /data/<you>/rdm-sets --gate        # set up + 20 steps: memory, s/step
#   bash z-image/scripts/ours_four-cards.sh --root /data/<you>/rdm-sets               # train 2000 steps + evaluate
#   bash z-image/scripts/ours_four-cards.sh --root /data/<you>/rdm-sets --resume      # continue after an interruption
#   bash z-image/scripts/ours_four-cards.sh --root /data/<you>/rdm-sets --eval-only   # Qwen-Image-Bench only
#
# Trains on 4 GPUs (8 of the 32 fresh rows per step each, FSDP); the context rebuild and the evaluation
# use every visible card. Needs 4 x 80 GB cards (H100 / H200 / A100 80GB) and ~330 GB under --root.
# Estimates, not measured -- the --gate run prints the real s/step:
#   H100: ~21 s/step, 2000 steps ~12 h; Qwen-Image-Bench ~20 min per model; contexts ~5 min
#   A100 80GB: about 3x slower
# The trained model does not depend on the GPU count (the same gradient for the same rows).
# Every option of the shared body works here: bash z-image/scripts/_ours_run.sh --help
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ours_run.sh" \
  --gpus 4 --ref-hf jiachengcui888/zimage-ref-v1 "$@"
