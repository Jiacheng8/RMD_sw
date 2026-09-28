#!/usr/bin/env bash
# Step 02 -- render the 4-step FLUX.2 klein-4B teacher. THIS is the reference distribution.
#
# The reference is teacher output, not real photos: the student is initialised from this
# teacher, so real photographs are a target neither model can reach and the objective stops
# being distillation. SEEDS renders per prompt keep each prompt's reference a small SAMPLE
# rather than a single point -- with one render per prompt the local MMD degenerates toward
# per-prompt regression and diversity collapses (spec sec. 22.2).
#
# Sharded round-robin across GPUs, skip-if-exists: re-run to resume.
source "$(dirname "${BASH_SOURCE[0]}")/preprocess_config.sh"
preflight
[[ -f "$CTX_OUT" ]] || { echo "step 01 output missing: $CTX_OUT" >&2; exit 1; }

N_PROMPTS=$("${PY_RUN[@]}" -c "import numpy as np; print(np.load('$CTX_OUT', mmap_mode='r').shape[0])" | tail -1)
RENDERED=$(( N_PROMPTS * SEEDS ))
KEPT=$(( N_PROMPTS * KEEP ))
NEED=$(( KEPT * 400 / 1000000 + 10 ))           # only the KEPT images land on disk

say "02  teacher render: $N_PROMPTS prompts x $SEEDS candidates = $RENDERED renders"
info "curation: $CURATE -> keep $KEEP/prompt = $KEPT images on disk (~${NEED} GB)"
info "measured 3.6 img/s per 4090 -> ~$(( RENDERED / 36 / 100 / NUM_GPUS + 1 )) h on $NUM_GPUS GPUs"
mkdir -p "$TEACHER_DIR"
require_space "$TEACHER_DIR" "$NEED" "the teacher renders"

HAVE=$(find "$TEACHER_DIR" -name '*.png' 2>/dev/null | wc -l)
if [[ "$HAVE" -ge "$KEPT" ]]; then info "already complete ($HAVE pngs), skipping"; exit 0; fi
[[ "$HAVE" -gt 0 ]] && info "resuming: $HAVE/$KEPT already kept"

pids=()
for ((g=0; g<NUM_GPUS; g++)); do
    CUDA_VISIBLE_DEVICES=$g "${PY_RUN[@]}" "$RDM_REPO/scripts/_preprocess_render.py" \
        --ctx "$CTX_OUT" --captions "$COCO_PAIRS" --out "$TEACHER_DIR" \
        --rank "$g" --world "$NUM_GPUS" --seeds "$SEEDS" --keep "$KEEP" \
        --curate "$CURATE" --chunk-prompts "$CHUNK_PROMPTS" \
        --steps "$RENDER_STEPS" --batch "$RENDER_BATCH" \
        --img-size "$IMG_SIZE" > "$LOG_DIR/02_render_rank$g.log" 2>&1 &
    pids+=($!)
    info "rank $g -> GPU $g (log: $LOG_DIR/02_render_rank$g.log)"
done
fail=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "rank $i FAILED -- see $LOG_DIR/02_render_rank$i.log" >&2; fail=1; }
done
[[ $fail -eq 0 ]] || exit 1

HAVE=$(find "$TEACHER_DIR" -name '*.png' | wc -l)
info "kept $HAVE/$KEPT images -> $TEACHER_DIR"
[[ "$HAVE" -eq "$KEPT" ]] || { echo "incomplete -- re-run this step to resume" >&2; exit 1; }
