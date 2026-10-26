#!/usr/bin/env bash
# Pack a finished Z-Image reference (built by /data/ironman/jiacheng/RDM/z-image/process) for
# training on another machine, and optionally upload it to a Hugging Face dataset.
#
#   bash z-image/scripts/pack_reference.sh --root <render root> --name zimage-ref-v1
#   bash z-image/scripts/pack_reference.sh --root <render root> --name zimage-ref-v1 \
#       --upload <owner>/<dataset>            # asks for a Write token (hidden), private by default
#
# The pack is what training reads, nothing else (~1.6 GB for 125k prompts):
#   <name>/reference_store/   features, tau, bandwidths, row order, prompt ids, z_context.json
#   <name>/prompts.jsonl, prepare.json, render_config.json
# It leaves out the renders: the PNGs (~176 GB, training never reads them) and the per-prompt Qwen3
# contexts (~70 GB), which the new machine rebuilds in minutes (zimage_distill.contexts).
#
# --root DIR    the render root (holds prompts.jsonl and reference_store/metadata.json)
# --name NAME   the pack's directory name (and the folder inside the dataset)
# --out DIR     where the pack is staged (default: <root>/..); files are hard-linked when possible
# --upload REPO upload the staged pack to this HF dataset (created if missing)
# --public      create the dataset public (default private)
# The token is read from $HF_TOKEN or asked for; it never goes on a command line or into a file.
set -euo pipefail
export PYTHONNOUSERSITE=1

ROOT=""; NAME=""; OUT=""; REPO_ID=""; PUBLIC=0
CONDA_ENV="${CONDA_ENV:-rdm}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)   ROOT="$2"; shift 2 ;;
    --name)   NAME="$2"; shift 2 ;;
    --out)    OUT="$2"; shift 2 ;;
    --upload) REPO_ID="$2"; shift 2 ;;
    --public) PUBLIC=1; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[[ -n "$ROOT" && -n "$NAME" ]] || { echo "usage: --root <render root> --name <pack name> (see --help)" >&2; exit 2; }
ROOT="$(realpath "$ROOT")"
OUT="$(realpath -m "${OUT:-$ROOT/..}")"
STORE="$ROOT/reference_store"
PACK="$OUT/$NAME"
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f "$STORE/metadata.json" ]] || die "$STORE/metadata.json missing: the store is not finalized
  (process/launch.py all, or: features.py finalize --root $ROOT)"
for f in prompts.jsonl prepare.json render_config.json; do
  [[ -f "$ROOT/$f" ]] || die "$ROOT/$f missing"
done

say "stage the pack -> $PACK"
link() { ln "$1" "$2" 2>/dev/null || cp -p "$1" "$2"; }      # hard link on the same volume
if [[ -f "$PACK/.staged" ]]; then
  info "already staged (re-using it)"
else
  [[ -e "$PACK" ]] && die "$PACK exists but was not staged completely; remove it and re-run"
  mkdir -p "$PACK/reference_store/encoder_features"
  for f in prompts.jsonl prepare.json render_config.json; do link "$ROOT/$f" "$PACK/$f"; done
  for f in metadata.json bandwidths.json row_order.npy prompt_ids.npy group_ids.npy text_features.npy \
           text_features.done.json z_context.json rows.jsonl; do
    [[ -f "$STORE/$f" ]] && link "$STORE/$f" "$PACK/reference_store/$f"
  done
  for f in "$STORE"/encoder_features/*.npy "$STORE"/encoder_features/*.done.json; do
    [[ "$f" == *.partial.npy ]] && continue
    link "$f" "$PACK/reference_store/encoder_features/$(basename "$f")"
  done
fi
info "$(du -shL "$PACK" | cut -f1) in $(find "$PACK" -type f | wc -l) files"

# read it back exactly as training will (without contexts: those come on the new machine)
ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$ZDIR:$(dirname "$ZDIR"):${PYTHONPATH:-}" conda run -n "$CONDA_ENV" --no-capture-output \
  python - "$PACK" <<'PY' || die "the staged pack does not read back (see above)"
import json, sys
import numpy as np
from zimage_distill.store import ZImageReferenceStore
pack = sys.argv[1]
meta = json.load(open(f"{pack}/reference_store/metadata.json"))
names = sorted(meta["encoder_feature_dims"])
s = ZImageReferenceStore(f"{pack}/reference_store", names)
rows = np.linspace(0, s.num_rows - 1, 64).astype(int)
for n in names:
    assert np.isfinite(s.reference_joint_features(n, rows).numpy()).all(), n
print(f"    ok  {s.num_rows:,} rows / {s.num_prompts:,} prompts, encoders {', '.join(names)}, "
      f"features at {s.feature_image_size()} px; contexts: rebuilt on the training machine")
PY
touch "$PACK/.staged"

if [[ -z "$REPO_ID" ]]; then
  say "done (not uploaded)"
  info "upload later (re-uses the staged pack): bash $0 --root $ROOT --name $NAME --out $OUT --upload <owner>/<dataset>"
  exit 0
fi

say "upload -> HF dataset $REPO_ID ($([[ $PUBLIC -eq 1 ]] && echo public || echo private))"
if [[ -z "${HF_TOKEN:-}" ]]; then
  [[ -t 0 ]] || die "no terminal to ask for the token: export HF_TOKEN=<write token> first"
  read -rs -p "HF write token: " HF_TOKEN; echo; export HF_TOKEN   # lives only in this process
fi
conda run -n "$CONDA_ENV" --no-capture-output python - "$REPO_ID" "$PUBLIC" <<'PY' || die "could not create $REPO_ID"
import os, sys
from huggingface_hub import HfApi
repo, public = sys.argv[1], sys.argv[2] == "1"
api = HfApi(token=os.environ["HF_TOKEN"])
who = api.whoami()
if who.get("auth", {}).get("accessToken", {}).get("role") == "read":
    sys.exit("this is a Read token; uploading needs a Write token")
api.create_repo(repo, repo_type="dataset", private=not public, exist_ok=True)
print(f"    account {who['name']}: dataset {repo} ready")
PY
# upload-large-folder mirrors a folder: hard-link the pack into one, as <NAME>/ (.staged excluded)
STAGE="$OUT/.upload_$NAME"
[[ -d "$STAGE/$NAME" ]] || { mkdir -p "$STAGE" && cp -al "$PACK" "$STAGE/$NAME"; }
conda run -n "$CONDA_ENV" --no-capture-output hf upload-large-folder "$REPO_ID" "$STAGE" \
  --repo-type dataset --exclude "*/.staged" --exclude ".cache/**" || die "upload interrupted; re-run the same command to resume"
say "done: $REPO_ID/$NAME"
info "on the training machine: bash z-image/scripts/run_machine-zimage.sh --ref-hf $REPO_ID --ref-name $NAME"
