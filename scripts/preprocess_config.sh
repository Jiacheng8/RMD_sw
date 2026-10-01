#!/usr/bin/env bash
# Shared configuration for the SW-LMMD preprocessing pipeline. Edit here, not in the steps.
#
# The three steps are strictly sequential -- 02 conditions on 01's context, 03 encodes 02's
# renders -- and each is independently resumable, so a failed step can be re-run without
# redoing the ones before it.
#
#   01  Qwen3 text context   82,783 prompts            1 GPU    ~61 GB out
#   02  4-step teacher render 82,783 x SEEDS images    N GPUs   ~132 GB out at SEEDS=4
#   03  frozen-encoder features over those renders     N GPUs   ~9 GB out
#
# Measured on this box (1x RTX 4090): render 3.6 img/s, encoders 134-289 img/s.

set -euo pipefail
export PYTHONNOUSERSITE=1   # packages in ~/.local/lib/pythonX.Y must never shadow the conda envs'

# ---- where things live -------------------------------------------------------------
export RDM_REPO="${RDM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # the checkout this file is in
export ASSETS="${ASSETS:-/data/thor/jiacheng/rdm-sets}"
export CONDA_ENV="${CONDA_ENV:-rdm}"

export COCO_PAIRS="${COCO_PAIRS:-$ASSETS/datasets/coco/coco_pairs.npz}"
export WORK="${WORK:-$ASSETS/sw_lmmd}"              # everything this pipeline produces
export CTX_OUT="${CTX_OUT:-$WORK/qwen3_ctx_coco.npy}"
export TEACHER_DIR="${TEACHER_DIR:-$WORK/teacher_renders}"
export STORE_DIR="${STORE_DIR:-$WORK/reference_store}"
# The released SigLIP2 tau(c) table: rows 0..82782 are the COCO captions, row-aligned to
# COCO_PAIRS (verified: matched-row cosine 0.99998, off-diagonal max 0.87). Step 03 slices
# it instead of re-encoding the text tower.
export TAU_RELEASED="${TAU_RELEASED:-$ASSETS/irdm_geall_assets/flux2/siglip2_text_geall_g32.npy}"

# ---- method parameters -------------------------------------------------------------
export CTX_LEN="${CTX_LEN:-48}"                     # MUST match flux_ctx_len at eval time
# The paper's COCO reference block: render 24 candidates per caption, keep the PickScore
# top-3. The curation is the mechanism behind "the 1-step student matches or exceeds its
# 4-step teacher" -- it makes the target the teacher's BEST output, not its average.
export SEEDS="${SEEDS:-24}"                         # candidates rendered per prompt
export KEEP="${KEEP:-4}"                            # kept after scoring -> rows per prompt
                                                    # 4 not the paper's 3: we drop the GenEval block, and
                                                    # keeping a 4th candidate costs NO extra rendering
                                                    # (all 24 are generated either way) -> 331,132 rows
export CURATE="${CURATE:-pickscore}"                # pickscore | none (none => KEEP must == SEEDS)
export CHUNK_PROMPTS="${CHUNK_PROMPTS:-64}"         # streaming granularity; peak RAM ~ CHUNK*SEEDS imgs
export RENDER_STEPS="${RENDER_STEPS:-4}"            # the teacher is the 4-step klein-4B
export IMG_SIZE="${IMG_SIZE:-512}"
export ENCODERS="${ENCODERS:-inception,convnext,mae,clip,dinov3_l,pe_core_l,siglip2,aimv2_huge,webssl_dino_1b,dreamsim}"
export SIGMA_SCALE="${SIGMA_SCALE:-0.25}"           # cold image kernel, as in build_joint_one
export S_TXT="${S_TXT:-1.0}"                        # released bundles have beta == sigma => s_txt = 1
export ROW_ORDER_SEED="${ROW_ORDER_SEED:-3407}"
export MAX_PROMPTS="${MAX_PROMPTS:-0}"   # >0 caps the prompt count (smoke tests / small pilots)

# ---- hardware ----------------------------------------------------------------------
export NUM_GPUS="${NUM_GPUS:-6}"
export RENDER_BATCH="${RENDER_BATCH:-2}"            # measured optimum: 3.60 img/s @ 10.6 GB
export EXTRACT_BATCH="${EXTRACT_BATCH:-64}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

export LOG_DIR="${LOG_DIR:-$WORK/logs}"

# ---- environment -------------------------------------------------------------------
# env.sh sets HF_HOME / HF_HUB_CACHE / FLUX2_SRC. HF_HOME must be the PARENT of the hub
# cache that actually holds the blobs: flux_generator.py::_hub_root() resolves the klein-4B
# and AE weights from HF_HOME alone and never consults HF_HUB_CACHE.
if [[ -f "$ASSETS/env.sh" ]]; then source "$ASSETS/env.sh"; fi

mkdir -p "$WORK" "$LOG_DIR"

# ---- helpers -----------------------------------------------------------------------
PY_RUN=(conda run -n "$CONDA_ENV" --no-capture-output python -u)

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }

preflight() {
    command -v conda >/dev/null || { echo "conda not on PATH" >&2; exit 1; }
    [[ -f "$COCO_PAIRS" ]] || { echo "missing COCO pairing: $COCO_PAIRS
  build it with: python scripts/prepare_datasets.py coco --captions .../captions_train2014.json \\
      --images .../train2014 --out $COCO_PAIRS" >&2; exit 1; }
    [[ -n "${FLUX2_SRC:-}" && -d "${FLUX2_SRC:-}/flux2" ]] || {
        echo "FLUX2_SRC does not point at an importable flux2 package (expected \$FLUX2_SRC/flux2).
  It must be the clone's src/ directory, not the clone root." >&2; exit 1; }
}

free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }

require_space() {   # require_space <path> <needed_gb> <what>
    local avail; avail=$(free_gb "$1")
    if [[ -n "$avail" && "$avail" -lt "$2" ]]; then
        echo "not enough space for $3: need ~${2} GB, $1 has ${avail} GB" >&2; exit 1
    fi
}
