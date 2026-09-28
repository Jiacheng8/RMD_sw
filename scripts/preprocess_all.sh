#!/usr/bin/env bash
# Run steps 01 -> 02 -> 03 in order. Each is independently resumable; re-running this script
# after a failure picks up where it stopped rather than redoing finished work.
#
#   bash scripts/preprocess_all.sh                 # full pipeline
#   SEEDS=1 NUM_GPUS=2 bash scripts/preprocess_all.sh    # cheaper variant
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/preprocess_config.sh"

t0=$(date +%s)
for step in preprocess_01_ctx preprocess_02_render preprocess_03_features; do
    say "RUN $step"
    bash "$here/$step.sh"
done
say "pipeline complete in $(( ($(date +%s) - t0) / 60 )) min"
info "reference store: $STORE_DIR"
info "next: point configs/sw_lmmd_flux.yaml reference_root at it"
