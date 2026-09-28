#!/usr/bin/env bash
# Step 01 -- Qwen3 text context for the COCO captions (the generator's conditioning).
#
# Single GPU, ~61 GB out at ctx_len 48. NOT resumable: build_flux2_ctx.py writes one memmap
# in a single pass, so an interrupted run leaves a partial file that this step deletes rather
# than silently reusing (a truncated tail renders blank images -- a known silent failure).
#
# ctx_len MUST match flux_ctx_len at eval time, or the student is evaluated at a sequence
# geometry it never trained on.
source "$(dirname "${BASH_SOURCE[0]}")/preprocess_config.sh"
preflight

N_PROMPTS=$("${PY_RUN[@]}" -c "
from rdm.data.coco import load_coco_pairs; print(len(load_coco_pairs('$COCO_PAIRS')['captions']))" | tail -1)
CAP_ARGS=()
if [[ "${MAX_PROMPTS:-0}" -gt 0 && "${MAX_PROMPTS}" -lt "$N_PROMPTS" ]]; then
    N_PROMPTS="$MAX_PROMPTS"; CAP_ARGS=(--max-images "$MAX_PROMPTS")
    info "MAX_PROMPTS set: capping at $N_PROMPTS prompts"
fi
NEED=$(( N_PROMPTS * CTX_LEN * 7680 * 2 / 1000000000 + 5 ))

say "01  Qwen3 context: $N_PROMPTS prompts, ctx_len=$CTX_LEN -> $CTX_OUT (~${NEED} GB)"
if [[ -f "$CTX_OUT" ]]; then
    ROWS=$("${PY_RUN[@]}" -c "import numpy as np; print(np.load('$CTX_OUT', mmap_mode='r').shape[0])" | tail -1)
    if [[ "$ROWS" == "$N_PROMPTS" ]]; then info "already complete ($ROWS rows), skipping"; exit 0; fi
    info "existing file has $ROWS rows (expected $N_PROMPTS) -- partial, removing"
    rm -f "$CTX_OUT"
fi
require_space "$WORK" "$NEED" "the context pool"

"${PY_RUN[@]}" "$RDM_REPO/scripts/build_flux2_ctx.py" \
    --captions "$COCO_PAIRS" --out "$CTX_OUT" --ctx-len "$CTX_LEN" \
    --model-id Qwen/Qwen3-4B "${CAP_ARGS[@]}" 2>&1 | tee "$LOG_DIR/01_ctx.log"
info "done -> $CTX_OUT"
