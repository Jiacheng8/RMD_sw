#!/usr/bin/env bash
# Rebuild the reference store's generator context on a NEW machine -- step 01 only.
#
# The reference store travels without qwen_context.npy: it is 61 GB, but it is just the Qwen3
# encoding of the COCO captions and step 01 recomputes it in ~10 min on one GPU. Steps 02/03
# (teacher renders, encoder features) are NOT rerun -- their output is the store you copied.
#
# Needs, on the new machine:
#   $STORE_DIR   the copied reference store, without qwen_context.npy
#   $COCO_PAIRS  the SAME coco_pairs.npz the store was built from (its caption order defines
#                which context row belongs to which prompt -- a different pairing would
#                silently condition every row on another prompt's caption)
#   Qwen3-4B + flux2 source   (fetch_prerequisites.py --group flux --group flux2src)
#
#   ASSETS=/data/me/rdm-sets STORE_DIR=/data/me/reference_store \
#   COCO_PAIRS=/data/me/coco_pairs.npz bash scripts/preprocess_all-new-machine.sh
#
# ASSETS picks the env.sh; every path above defaults as in preprocess_config.sh, except that
# COCO_PAIRS prefers the copy download_all.sh fetches next to the store ($ASSETS/sw_lmmd/).
_assets="${ASSETS:-/data/thor/jiacheng/rdm-sets}"
if [[ -z "${COCO_PAIRS:-}" && -f "$_assets/sw_lmmd/coco_pairs.npz" ]]; then
    export COCO_PAIRS="$_assets/sw_lmmd/coco_pairs.npz"
fi
source "$(dirname "${BASH_SOURCE[0]}")/preprocess_config.sh"
preflight

[[ -f "$STORE_DIR/metadata.json" ]] || { echo "reference store not found: $STORE_DIR
  copy it here first (or set STORE_DIR)" >&2; exit 1; }

# The store must have been built from this caption list: same prompt count.
STORE_PROMPTS=$("${PY_RUN[@]}" -c "import json; print(json.load(open('$STORE_DIR/metadata.json'))['num_prompts'])" | tail -1)
N_CAPTIONS=$("${PY_RUN[@]}" -c "
from rdm.data.coco import load_coco_pairs; print(len(load_coco_pairs('$COCO_PAIRS')['captions']))" | tail -1)
[[ "$STORE_PROMPTS" == "$N_CAPTIONS" ]] || { echo "the store has $STORE_PROMPTS prompts but $COCO_PAIRS has $N_CAPTIONS captions -- not the pairing the store was built from" >&2; exit 1; }
say "store $STORE_DIR: $STORE_PROMPTS prompts, matches $COCO_PAIRS"

# ---- step 01 (skips itself if $CTX_OUT is already complete)
say "RUN preprocess_01_ctx"
bash "$(dirname "${BASH_SOURCE[0]}")/preprocess_01_ctx.sh"

# ---- plug it into the store
ln -sfn "$(realpath "$CTX_OUT")" "$STORE_DIR/qwen_context.npy"
info "linked $STORE_DIR/qwen_context.npy -> $CTX_OUT"

# ---- read it back through the trainer's reader (checks context rows == prompt rows)
"${PY_RUN[@]}" - "$STORE_DIR" <<'PY'
import json, sys
from rdm.sw_lmmd import ReferenceFeatureStore
root = sys.argv[1]
names = sorted(json.load(open(f"{root}/bandwidths.json")))
s = ReferenceFeatureStore(root, names)
rows = s.row_order()[:4]
print(f"[store] OK  {root}\n        {s.num_rows:,} rows, context {tuple(s.context.shape)}, "
      f"sample ctx {tuple(s.generator_context(rows).shape)}")
PY
say "done -- reference store ready: $STORE_DIR (new_machine.sh hands it to training as REFERENCE_ROOT; by hand: --set reference_root=$STORE_DIR)"
