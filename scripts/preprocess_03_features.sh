#!/usr/bin/env bash
# Step 03 -- frozen-encoder features over the teacher renders, then assemble the store.
#
# This is what SW-LMMD needs and no one hosts: one feature row per (prompt, seed) reference
# image. The released Nystrom bundles cannot substitute -- they are the reference MEAN
# EMBEDDING, with row identity already integrated out, whereas the window needs y_j for the
# specific rows whose prompts the student just generated from.
#
# Encoders are independent, so they shard across GPUs round-robin. Per-encoder skip-if-exists.
source "$(dirname "${BASH_SOURCE[0]}")/preprocess_config.sh"
preflight
[[ -d "$TEACHER_DIR" ]] || { echo "step 02 output missing: $TEACHER_DIR" >&2; exit 1; }
[[ -f "$TAU_RELEASED" ]] || { echo "released tau table missing: $TAU_RELEASED" >&2; exit 1; }

N_PROMPTS=$("${PY_RUN[@]}" -c "import numpy as np; print(np.load('$CTX_OUT', mmap_mode='r').shape[0])" | tail -1)
IFS=',' read -ra ENCS <<< "$ENCODERS"
say "03  extract ${#ENCS[@]} encoders over $(( N_PROMPTS * KEEP )) rows -> $STORE_DIR"
mkdir -p "$STORE_DIR"

pids=()
for ((g=0; g<NUM_GPUS; g++)); do
    share=""
    for ((i=g; i<${#ENCS[@]}; i+=NUM_GPUS)); do share+="${ENCS[$i]},"; done
    share="${share%,}"
    [[ -z "$share" ]] && continue
    CUDA_VISIBLE_DEVICES=$g "${PY_RUN[@]}" "$RDM_REPO/scripts/_preprocess_extract.py" \
        --teacher-dir "$TEACHER_DIR" --out "$STORE_DIR" --encoders "$share" \
        --num-prompts "$N_PROMPTS" --keep "$KEEP" --img-size "$IMG_SIZE" \
        --batch "$EXTRACT_BATCH" --workers "$NUM_WORKERS" \
        --sigma-scale "$SIGMA_SCALE" --s-txt "$S_TXT" \
        > "$LOG_DIR/03_extract_gpu$g.log" 2>&1 &
    pids+=($!)
    info "GPU $g <- $share"
done
fail=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "GPU $i FAILED -- see $LOG_DIR/03_extract_gpu$i.log" >&2; fail=1; }
done
[[ $fail -eq 0 ]] || exit 1

say "03b assemble + verify the reference store"
"${PY_RUN[@]}" "$RDM_REPO/scripts/_preprocess_finalize_store.py" \
    --store "$STORE_DIR" --tau-released "$TAU_RELEASED" --ctx "$CTX_OUT" \
    --num-prompts "$N_PROMPTS" --keep "$KEEP" --encoders "$ENCODERS" \
    --row-order-seed "$ROW_ORDER_SEED" 2>&1 | tee "$LOG_DIR/03_store.log"
