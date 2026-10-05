#!/usr/bin/env bash
# Build the GenEval reference block: the 4-step teacher's GenEval-CORRECT renders on the 553
# GenEval prompts, as an extension of the COCO store (the iRDM "geALLcoco" recipe). The COCO
# store is read, never modified; training picks the block up with `reference_extension:`.
#
#   GPU_IDS=0,4,5 bash scripts/build_geneval_block.sh --geneval-root <scorer root>
#   GPU_IDS=5 LIMIT=4 SEEDS=8 BLOCK_DIR=/tmp/gb RENDER_DIR=/tmp/gr bash scripts/build_geneval_block.sh ...  # smoke test
#
# Stages (each resumable; re-running skips finished work):
#   G1 ctx      Qwen3 context of the 553 prompts (build_flux2_ctx.py)            1 GPU, ~1 min
#   G2 render   SEEDS teacher renders per prompt, one shard per GPU                553*SEEDS images
#   G3 score    the official GenEval scorer on each shard, in parallel
#   G4 block    keep the correct renders (whole groups of 4, <= MAX_PER_PROMPT per prompt),
#               extract the encoder features, write $BLOCK_DIR and read it back with the store
#
# Paths and the conda env come from scripts/preprocess_config.sh (ASSETS, WORK, STORE_DIR, ...).
# The scorer root is the --root of scripts/setup_geneval.sh (it holds geneval_env.sh). Neither
# COCO nor the COCO store is needed: when $STORE_DIR exists the block is also read back as its
# extension, otherwise it is checked on its own (e.g. on a fresh render machine,
# scripts/reference_new_machine.sh).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/preprocess_config.sh"

GENEVAL_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --geneval-root) GENEVAL_ROOT="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[[ -n "$GENEVAL_ROOT" && -f "$GENEVAL_ROOT/geneval_env.sh" ]] || {
  echo "--geneval-root must be the scorer root (with geneval_env.sh); see scripts/setup_geneval.sh" >&2; exit 2; }

GPU_IDS="${GPU_IDS:-0}"
SEEDS="${SEEDS:-128}"                   # candidates per prompt; the teacher is right ~80% of the time
MAX_PER_PROMPT="${MAX_PER_PROMPT:-96}"  # kept per prompt (multiple of 4)
LIMIT="${LIMIT:-0}"                     # >0: only the first N prompts (smoke test)
PROMPTS="${PROMPTS:-$RDM_REPO/assets/geneval_prompts.jsonl}"
GE_CTX="${GE_CTX:-$WORK/qwen3_ctx_geneval553.npy}"
RENDER_DIR="${RENDER_DIR:-$WORK/geneval_renders}"
BLOCK_DIR="${BLOCK_DIR:-$WORK/geneval_block}"
BLOCK_ENCODERS="${BLOCK_ENCODERS:-$ENCODERS}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
WORLD=${#GPUS[@]}
cd "$RDM_REPO"
command -v conda >/dev/null || { echo "conda not on PATH" >&2; exit 1; }
[[ -n "${FLUX2_SRC:-}" && -d "${FLUX2_SRC:-}/flux2" ]] || {
  echo "FLUX2_SRC does not point at an importable flux2 package -- source <root>/env.sh" >&2; exit 1; }
[[ -f "$TAU_RELEASED" ]] || { echo "missing the released tau table: $TAU_RELEASED
  (scripts/fetch_prerequisites.py --group assets)" >&2; exit 1; }
BASE_ARGS=()
[[ -f "$STORE_DIR/metadata.json" ]] && BASE_ARGS=(--base-store "$STORE_DIR")

say "G1  Qwen3 context of the GenEval prompts -> $GE_CTX"
# A failed build_flux2_ctx.py leaves a full-size memmap of zeros, and zero context renders junk
# that the scorer then rejects wholesale. So build into .partial, require every row non-zero,
# and only then move it into place; a stale file that fails the check is rebuilt.
ctx_ok() { "${PY_RUN[@]}" -c "
import numpy as np, sys
a = np.load(sys.argv[1], mmap_mode='r'); n = sum(1 for l in open(sys.argv[2]) if l.strip())
ok = a.shape[0] == n and bool((np.abs(a.reshape(n, -1)).max(1) > 0).all())
print('ok' if ok else f'bad: {a.shape[0]} rows for {n} prompts, or all-zero rows')" "$1" "$PROMPTS" | tail -1; }
if [[ -f "$GE_CTX" && "$(ctx_ok "$GE_CTX")" == ok ]]; then info "present and complete, skipping"; else
  rm -f "$GE_CTX" "$GE_CTX.partial.npy"
  # the same bf16 text model as the COCO context (preprocess_01_ctx.sh): both blocks are
  # encoded identically
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PY_RUN[@]}" scripts/build_flux2_ctx.py \
      --jsonl "$PROMPTS" --out "$GE_CTX.partial.npy" --ctx-len "$CTX_LEN" --model-id Qwen/Qwen3-4B
  check="$(ctx_ok "$GE_CTX.partial.npy")"
  [[ "$check" == ok ]] || { echo "GenEval context check failed: $check" >&2; exit 1; }
  mv "$GE_CTX.partial.npy" "$GE_CTX"
  mv "$GE_CTX.partial_meta.json" "${GE_CTX%.npy}_meta.json"
fi

say "G2  teacher renders: ${SEEDS} per prompt on GPU(s) $GPU_IDS -> $RENDER_DIR"
mkdir -p "$RENDER_DIR" "$LOG_DIR"
pids=()
for r in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$r]}" "${PY_RUN[@]}" scripts/_geneval_render.py \
      --ctx "$GE_CTX" --prompts "$PROMPTS" --out "$RENDER_DIR" --rank "$r" --world "$WORLD" \
      --seeds "$SEEDS" --steps "$RENDER_STEPS" --batch "$RENDER_BATCH" --img-size "$IMG_SIZE" \
      --limit "$LIMIT" > "$LOG_DIR/geneval_render_$r.log" 2>&1 &
  pids+=($!)
  info "rank $r on GPU ${GPUS[$r]} (log: $LOG_DIR/geneval_render_$r.log)"
done
for p in "${pids[@]}"; do wait "$p" || { echo "a render rank failed -- see $LOG_DIR/geneval_render_*.log" >&2; exit 1; }; done

say "G3  GenEval scorer on every shard"
pids=()
for r in "${!GPUS[@]}"; do
  out="$RENDER_DIR/shard_${r}_results.jsonl"
  [[ -d "$RENDER_DIR/shard_$r" ]] || continue              # more GPUs than (limited) prompts
  n_img=$(find "$RENDER_DIR/shard_$r" -path '*/samples/*.png' | wc -l)
  [[ $n_img -gt 0 ]] || continue
  if [[ -f "$out" && $(wc -l < "$out") -eq $n_img ]]; then info "shard $r scored ($n_img images), skipping"; continue; fi
  CUDA_VISIBLE_DEVICES="${GPUS[$r]}" bash scripts/score_geneval.sh "$RENDER_DIR/shard_$r" \
      --root "$GENEVAL_ROOT" --out "$out" > "$LOG_DIR/geneval_score_$r.log" 2>&1 &
  pids+=($!)
  info "shard $r: $n_img images on GPU ${GPUS[$r]} (log: $LOG_DIR/geneval_score_$r.log)"
done
for p in "${pids[@]}"; do wait "$p" || { echo "a scorer failed -- see $LOG_DIR/geneval_score_*.log" >&2; exit 1; }; done

say "G4  select correct renders, extract features -> $BLOCK_DIR"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PY_RUN[@]}" scripts/_geneval_block_build.py \
    --render-dir "$RENDER_DIR" --prompts "$PROMPTS" --ctx "$GE_CTX" \
    --tau-released "$TAU_RELEASED" "${BASE_ARGS[@]}" --out "$BLOCK_DIR" \
    --encoders "$BLOCK_ENCODERS" --max-per-prompt "$MAX_PER_PROMPT" \
    --img-size "$IMG_SIZE" --batch "$EXTRACT_BATCH" --workers "$NUM_WORKERS"
say "done -- train with reference_extension: $(basename "$BLOCK_DIR") (it must sit next to the store)"
