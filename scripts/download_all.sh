#!/usr/bin/env bash
# One entry point for EVERY download this project needs. Run it once, then preprocess.
#
#   bash scripts/download_all.sh --root /data/thor/jiacheng/rdm-sets
#   bash scripts/download_all.sh --root <dir> --dry-run      # plan + footprint, writes nothing
#   bash scripts/download_all.sh --root <dir> --with-eval    # + the 4 held-out eval encoders
#   bash scripts/download_all.sh --root <dir> --with-baseline# + the released 1-step student
#
# What it fetches (~50 GB, + 13.5 GB once COCO is extracted):
#
#   encoders  10 frozen training encoders + Inception-FID + DreamSim
#   text      the SigLIP2 SO400M text tower
#   flux      FLUX.2 klein-4B (teacher/student init) + the AE + Qwen3-4B (prompt encoder)
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
# Those come from `bash scripts/preprocess_all.sh`, which this script points you at.
#
# Idempotent and resumable: HF downloads are cache-backed and the COCO archives resume.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=""
CONDA_ENV="${CONDA_ENV:-rdm}"
HF_CACHE="${HF_CACHE:-}"
EXTRA=()
PASSTHRU=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)          ROOT="$2"; shift 2 ;;
    --hf-cache)      HF_CACHE="$2"; shift 2 ;;
    --env)           CONDA_ENV="$2"; shift 2 ;;
    --with-eval)     EXTRA+=(--group evalenc); shift ;;
    --with-baseline) EXTRA+=(--group student); shift ;;
    --with-imagenet) EXTRA+=(--group pmfh); shift ;;
    --dry-run)       PASSTHRU+=(--dry-run); shift ;;
    -h|--help)       sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[[ -n "$ROOT" ]] || { echo "usage: $0 --root <asset-dir> [--dry-run]" >&2; exit 2; }

# Default groups already include encoders/text/flux/flux2src/coco/assets; --with-* adds to them.
GROUP_ARGS=()   # NOT "GROUPS": that is a bash builtin array of the caller's gids
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  for g in encoders text flux flux2src coco assets; do GROUP_ARGS+=(--group "$g"); done
  GROUP_ARGS+=("${EXTRA[@]}")
fi

CACHE_ARG=()
[[ -n "$HF_CACHE" ]] && CACHE_ARG=(--hf-cache "$HF_CACHE")

RUN=(python)
command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV" \
  && RUN=(conda run -n "$CONDA_ENV" --no-capture-output python -u)

printf '\n\033[1m==> downloading into %s\033[0m\n' "$ROOT"
rc=0
"${RUN[@]}" "$REPO_ROOT/scripts/fetch_prerequisites.py" --root "$ROOT" \
    "${CACHE_ARG[@]}" "${GROUP_ARGS[@]}" "${PASSTHRU[@]}" || rc=$?

cat <<EOF

==> one item may need you
    black-forest-labs/FLUX.2-dev (ae.safetensors, the native VAE) is a GATED repo. If the run
    above reported it, accept the licence once and re-run this script:
      1) open https://huggingface.co/black-forest-labs/FLUX.2-dev and accept
      2) hf auth login
    klein-4B itself is NOT gated and needs nothing.

==> next
    source $ROOT/env.sh
    bash scripts/preprocess_all.sh        # builds what no one hosts (ctx -> renders -> features)
EOF
exit $rc
