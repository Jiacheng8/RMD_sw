#!/usr/bin/env bash
# One entry point for EVERY download this project needs. Run it once, then preprocess.
#
#   bash scripts/download_all.sh --root /data/thor/jiacheng/rdm-sets
#   bash scripts/download_all.sh --root <dir> --dry-run      # plan + footprint, writes nothing
#   bash scripts/download_all.sh --root <dir> --with-eval    # + the 4 held-out eval encoders
#   bash scripts/download_all.sh --root <dir> --with-baseline# + the released 1-step student
#   bash scripts/download_all.sh --root <dir> --no-sw-lmmd-store  # skip our prebuilt reference store
#   bash scripts/download_all.sh --root <dir> --minimal      # only what training from that store needs
#   bash scripts/download_all.sh --root <dir> --sw-lmmd-store-only   # just the store (e.g. a retry)
#
# Also fetches OUR prebuilt SW-LMMD reference store (~8.5 GB) from Google Drive with rclone into
# <root>/sw_lmmd/, so a new machine skips the ~31 h teacher render. It ships without its 61 GB
# Qwen3 context; rebuild that in ~10 min with scripts/preprocess_all-new-machine.sh.
# Needs an rclone remote for that Drive: `rclone config`, a Google Drive remote named "gdrive",
# authorised with the account that owns SW-RDM/ (a folder merely shared with you is not at
# gdrive:SW-RDM). Both files are md5-checked against the published versions.
#   SW_STORE_RCLONE=myremote:SW-RDM bash scripts/download_all.sh ...   # another remote / folder
#   SW_STORE_TAR=/path/reference_store_noctx.tar bash scripts/download_all.sh ...  # archive already here
#
# What it fetches (~50 GB, + 13.5 GB once COCO is extracted):
#
#   encoders  10 frozen training encoders + Inception-FID + DreamSim
#   text      the SigLIP2 SO400M text tower
#   flux      FLUX.2 klein-4B (teacher/student init) + the AE + Qwen3-4B (prompt encoder)
#   pickscore PickScore_v1 + its processor (3.9 GB): eval_checkpoint.sh and the 02 curation
#   flux2src  the native Black Forest Labs flux2 package (git clone -> FLUX2_SRC)
#   coco      COCO train2014 images + captions, extracted and paired
#   assets    the AUTHORS' released reference assets (1.3 GB): 10 joint Nystrom bundles and
#             the SigLIP2 tau(c) table. The bundles drive the iRDM baseline arm; tau's first
#             82,783 rows are the COCO captions, which SW-LMMD reads directly instead of
#             re-encoding the text tower.
#
# What it CANNOT fetch, because nobody hosts it: the per-row reference features. A Nystrom
# bundle is the reference MEAN EMBEDDING with row identity already integrated out, while
# SW-LMMD needs y_j for the specific rows whose prompts the student just generated from.
# Those come from `bash scripts/preprocess_all.sh` -- or from our prebuilt store above.
#
# Idempotent and resumable: HF downloads are cache-backed and the COCO archives resume.
set -euo pipefail
export PYTHONNOUSERSITE=1   # packages in ~/.local/lib/pythonX.Y must never shadow the conda envs'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=""
CONDA_ENV="${CONDA_ENV:-rdm}"
HF_CACHE="${HF_CACHE:-}"
EXTRA=()
PASSTHRU=()
SW_STORE=1
SW_ONLY=0
MINIMAL=0
# The prebuilt SW-LMMD reference store on Google Drive, fetched with rclone.
RCLONE_SRC="${SW_STORE_RCLONE:-gdrive:SW-RDM}"
# md5s of the published files (`rclone md5sum gdrive:SW-RDM`); env-overridable, so a re-uploaded
# store needs no edit here. The only reliable completeness check: GNU `tar -t` exits 0 on an
# archive cut short, mid-member or at a member boundary alike.
SW_COCO_PAIRS_MD5="${SW_COCO_PAIRS_MD5:-8b88b6c5cf7e13183eb027f7e1c5d8a7}"   # coco_pairs.npz -- the caption order the store is built on
SW_STORE_TAR_MD5="${SW_STORE_TAR_MD5:-2165876e5cc25b2694b4f553f7a7a107}"     # reference_store_noctx.tar

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)          ROOT="$2"; shift 2 ;;
    --hf-cache)      HF_CACHE="$2"; shift 2 ;;
    --env)           CONDA_ENV="$2"; shift 2 ;;
    --with-eval)     EXTRA+=(--group evalenc); shift ;;
    --with-baseline) EXTRA+=(--group student); shift ;;
    --with-imagenet) EXTRA+=(--group pmfh); shift ;;
    --dry-run)       PASSTHRU+=(--dry-run); shift ;;
    --no-sw-lmmd-store) SW_STORE=0; shift ;;
    --minimal)       MINIMAL=1; shift ;;
    --sw-lmmd-store-only) SW_ONLY=1; shift ;;
    -h|--help)       sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[[ -n "$ROOT" ]] || { echo "usage: $0 --root <asset-dir> [--dry-run]" >&2; exit 2; }

# Default groups already include encoders/text/flux/pickscore/flux2src/coco/assets; --with-* adds to them.
GROUP_ARGS=()   # NOT "GROUPS": that is a bash builtin array of the caller's gids
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  for g in encoders text flux pickscore flux2src coco assets; do GROUP_ARGS+=(--group "$g"); done
  GROUP_ARGS+=("${EXTRA[@]}")
fi
# --minimal: only what SW-LMMD training from the prebuilt store needs -- the encoders it TRAINS
# with (MINIMAL_ENCODERS; the configs' three), FLUX.2 (klein-4B + AE + Qwen3-4B, which also rebuilds
# the store's context) and the flux2 source -- plus PickScore, so eval_checkpoint.sh runs without
# fetching weights at eval time. No COCO images, text tower, iRDM bundles or the other seven
# encoders (the store already carries their products) -- and no DreamSim / Inception, which come
# from GitHub release assets that rate-limit shared cloud IPs (HTTP 403).
MINIMAL_ENCODERS="${MINIMAL_ENCODERS:-dinov3_l,siglip2,aimv2_huge}"
if [[ $MINIMAL -eq 1 ]]; then
  GROUP_ARGS=(--group encoders --group flux --group pickscore --group flux2src
              --encoders "$MINIMAL_ENCODERS" "${EXTRA[@]}")
fi

CACHE_ARG=()
[[ -n "$HF_CACHE" ]] && CACHE_ARG=(--hf-cache "$HF_CACHE")

RUN=(python)
command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV" \
  && RUN=(conda run -n "$CONDA_ENV" --no-capture-output python -u)

rc=0
if [[ $SW_ONLY -eq 0 ]]; then
  printf '\n\033[1m==> downloading into %s\033[0m\n' "$ROOT"
  "${RUN[@]}" "$REPO_ROOT/scripts/fetch_prerequisites.py" --root "$ROOT" \
      "${CACHE_ARG[@]}" "${GROUP_ARGS[@]}" "${PASSTHRU[@]}" || rc=$?
fi

# ---- our prebuilt SW-LMMD reference store (Google Drive) -------------------------------
SW_DIR="$ROOT/sw_lmmd"
if [[ $SW_STORE -eq 1 ]]; then
  printf '\n\033[1m==> SW-LMMD reference store -> %s\033[0m\n' "$SW_DIR"
  if [[ -f "$SW_DIR/reference_store/metadata.json" ]]; then
    echo "    $SW_DIR/reference_store already present, skipping"
  elif [[ ${#PASSTHRU[@]} -gt 0 ]]; then
    echo "    --dry-run: would rclone coco_pairs.npz (12 MB) and reference_store_noctx.tar (~8.5 GB)"
    echo "    from $RCLONE_SRC, check both md5s and untar into $SW_DIR"
    [[ -n "${SW_STORE_TAR:-}" ]] && echo "    (archive taken from $SW_STORE_TAR instead)"
  else
    mkdir -p "$SW_DIR"
    have_rclone() { command -v rclone >/dev/null 2>&1 \
                      && rclone listremotes 2>/dev/null | grep -qx "${RCLONE_SRC%%:*}:"; }
    md5_is() { [[ -f "$1" && "$(md5sum "$1" | cut -d' ' -f1)" == "$2" ]]; }
    get() {     # get <file name> <dest> <md5>: a VERIFIED copy of $RCLONE_SRC/<file> at dest
      md5_is "$2" "$3" && return 0
      have_rclone || { echo "    no rclone remote '${RCLONE_SRC%%:*}:' -- install rclone and run 'rclone config'" \
                            "(a Google Drive remote named ${RCLONE_SRC%%:*}, authorised with the account" \
                            "that owns ${RCLONE_SRC#*:}/), or set SW_STORE_RCLONE" >&2; return 1; }
      rclone copyto -P "$RCLONE_SRC/$1" "$2" || { rm -f "$2"; return 1; }   # rclone cannot resume it anyway
      md5_is "$2" "$3" && return 0
      echo "    $2: md5 differs from the published file -- deleted" >&2
      rm -f "$2"; return 1
    }
    # Every step is checked explicitly: `set -e` does NOT apply inside a function or subshell
    # whose status is tested with ||, so it would sail past a failed download and report success.
    sw_store() {
      get coco_pairs.npz "$SW_DIR/coco_pairs.npz" "$SW_COCO_PAIRS_MD5" || return 1
      local tar="$SW_DIR/reference_store_noctx.tar"
      if [[ -n "${SW_STORE_TAR:-}" ]]; then           # provided by hand: used as-is, never fetched over
        tar="$SW_STORE_TAR"
        [[ -f "$tar" ]] || { echo "    SW_STORE_TAR=$tar does not exist" >&2; return 1; }
        md5_is "$tar" "$SW_STORE_TAR_MD5" \
          || echo "    note: $tar is not the published archive (md5 differs); using it anyway" >&2
      else
        get reference_store_noctx.tar "$tar" "$SW_STORE_TAR_MD5" || return 1
      fi
      # Extract beside the target, then move it in: an interrupted extraction must not leave a
      # half-written store that the metadata.json check above would later accept as complete.
      local part="$SW_DIR/.reference_store.partial"
      rm -rf "$part" && mkdir -p "$part" && tar -xf "$tar" -C "$part" || return 1
      [[ -f "$part/reference_store/metadata.json" ]] \
        || { echo "    $tar holds no reference_store/metadata.json" >&2; return 1; }
      rm -rf "$SW_DIR/reference_store"                 # leftovers of an older failed attempt
      mv "$part/reference_store" "$SW_DIR/reference_store" && rmdir "$part" || return 1
      [[ -n "${SW_STORE_TAR:-}" ]] || rm -f "$tar"
      echo "    ok  $SW_DIR/reference_store"
    }
    sw_store || { echo "    SW-LMMD store download FAILED -- fix the cause above and re-run" \
                      "(bash scripts/download_all.sh --root $ROOT --sw-lmmd-store-only)" >&2; rc=1; }
  fi
fi

cat <<EOF

==> one item may need you
    black-forest-labs/FLUX.2-dev (ae.safetensors, the native VAE) is a GATED repo. If the run
    above reported it, accept the licence once and re-run this script:
      1) open https://huggingface.co/black-forest-labs/FLUX.2-dev and accept
      2) conda run -n $CONDA_ENV hf auth login --token hf_...   (a "Read" token; or export HF_TOKEN)
    klein-4B itself is NOT gated and needs nothing.

==> next
    source $ROOT/env.sh
    # with the prebuilt store above: rebuild only its 61 GB Qwen3 context (~10 min, 1 GPU)
    ASSETS=$ROOT bash scripts/preprocess_all-new-machine.sh
    # or build everything from scratch (ctx -> renders -> features, ~31 h on 6x4090)
    bash scripts/preprocess_all.sh
EOF
exit $rc
