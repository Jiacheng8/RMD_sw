#!/usr/bin/env bash
# Upload reference_v1 as the public training pack jiachengcui888/zimage-ref-v1, once it is built.
#
#   bash z-image/scripts/upload_reference_v1.sh            # asks for a HF Write token (hidden input)
#   GPU_IDS=0,1,2 bash z-image/scripts/upload_reference_v1.sh
#
# 1. If process/launch.py is still running on the root, it prints the progress and stops: run
#    it again when that has finished. (The running `launch.py all` renders, then extracts the
#    features and finalizes the store by itself.)
# 2. If every prompt is rendered but the store is not finalized (launch.py was stopped after the
#    render), it runs `launch.py features` on GPU_IDS (default 0,1,2; ~1-2 h): run this in tmux.
# 3. Packs reference_store/ + the prompt table (~1.6 GB, no PNGs, no contexts) and uploads it,
#    public, to jiachengcui888/zimage-ref-v1 (pack_reference.sh). An interrupted upload resumes
#    when this is run again.
# The training machine then uses: bash z-image/scripts/ours_two-cards.sh   # (ours_four-cards.sh / ours_eight-cards.sh on 4 / 8 GPUs)
set -euo pipefail
export PYTHONNOUSERSITE=1

ROOT="${ROOT:-/data/ironman/jiacheng/RDM/z-image/process/artifacts/reference_v1}"
NAME="${NAME:-zimage-ref-v1}"
REPO_ID="${REPO_ID:-jiachengcui888/zimage-ref-v1}"
GPU_IDS="${GPU_IDS:-0,1,2}"
CONDA_ENV="${CONDA_ENV:-rdm}"
PROCESS="$(cd "$ROOT/../.." && pwd)"                      # .../z-image/process
ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f "$ROOT/prepare.json" ]] || die "$ROOT is not a prepared reference root"
TOTAL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['num_prompts'])" "$ROOT/prepare.json")
DONE=$(find "$ROOT/renders" -name selection.json 2>/dev/null | wc -l)

say "reference_v1: $DONE / $TOTAL prompts rendered"
if pgrep -u "$(id -u)" -f "process/launch.py .*--root $ROOT( |$)" >/dev/null; then
  info "process/launch.py is still running on this root:"
  pgrep -u "$(id -u)" -af "process/launch.py .*--root $ROOT( |$)" | cut -c1-160 | sed 's/^/      /'
  [[ -d "$ROOT/logs" ]] && info "latest: $(ls -t "$ROOT"/logs/*.log | head -1 | xargs tail -n 1)"
  info "run this script again when it has finished (it extracts the features and finalizes by itself)"
  exit 0
fi

if [[ ! -f "$ROOT/reference_store/metadata.json" ]]; then
  (( DONE == TOTAL )) || die "only $DONE / $TOTAL prompts are rendered and nothing is running: resume the
  render first (cd $(dirname "$PROCESS") && GPU_IDS=$GPU_IDS bash process/run_reference.sh)"
  say "features + finalize on GPUs $GPU_IDS (1-2 h)"
  [[ -z "${TMUX:-}${STY:-}" ]] && info "not inside tmux/screen: a dropped SSH session would stop it"
  conda run -n "$CONDA_ENV" --no-capture-output python "$PROCESS/launch.py" features --root "$ROOT" \
    --gpu-ids "$GPU_IDS" || die "feature extraction failed; see $ROOT/logs/features_*.log, finalize_*.log"
  [[ -f "$ROOT/reference_store/metadata.json" ]] || die "finalize did not write metadata.json"
fi
info "store finalized: $ROOT/reference_store"

say "pack + upload -> $REPO_ID (public)"
exec bash "$ZDIR/scripts/pack_reference.sh" --root "$ROOT" --name "$NAME" --upload "$REPO_ID" --public
