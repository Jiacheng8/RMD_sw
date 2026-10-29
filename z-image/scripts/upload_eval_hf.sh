#!/usr/bin/env bash
# Upload finished evaluations to one public Hugging Face dataset, one folder per evaluation, so they
# outlive the rented machine and the whole group can read them. Run it on the machine that has them.
#
#   bash z-image/scripts/upload_eval_hf.sh --root /root/rdm-sets turbo9 turbo2      # these runs
#   bash z-image/scripts/upload_eval_hf.sh --root /root/rdm-sets --all              # every finished run
#   bash z-image/scripts/upload_eval_hf.sh --root /root/rdm-sets --all --dry-run    # list only
#
# Layout in the dataset (default jiachengcui888/evaluation-result):
#
#   qwen-image-bench/<run>/report.md                 this run's table, next to the paper's leaderboard
#                          input_bench_scores.json   total, 5 pillars, 15 sub-capabilities
#                          input_judged.jsonl        the judge's verdict per image
#                          input.jsonl               ID, prompt, image per row
#                          generation.json           what rendered the images (checkpoint / steps, seed)
#                          images/{ID:04d}.png       the 1000 renders (~1.5 GB; --no-images leaves them out)
#                          logs/                     render + judge logs
#
# <run> is the name it was evaluated under (turbo9, turbo2, zimage-2step-1024_s2000, ...); a run that
# is already in the dataset is updated in place. Only finished runs (input_bench_scores.json) go up;
# --all leaves out smoke-test runs (*_limitN). The upload resumes when run again after an
# interruption. It needs a HF Write token: $HF_TOKEN, else asked for (hidden; never stored).
#
# --root DIR     data root (default /root/rdm-sets); the runs are in <root>/qib/runs
# --repo REPO    the dataset (default jiachengcui888/evaluation-result; created public if missing)
# --all          every finished run under <root>/qib/runs, smoke tests (*_limitN) left out
# --no-images    scores and logs only
# --dry-run      list what would go up, upload nothing
# --env NAME     conda env with the hf CLI (default rdm)
set -euo pipefail
export PYTHONNOUSERSITE=1
ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ZDIR/.." && pwd)"
ROOT="/root/rdm-sets"; REPO_ID="jiachengcui888/evaluation-result"; ALL=0; IMAGES=1; DRY=0; ENV_NAME="rdm"
NAMES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)      ROOT="$2"; shift 2 ;;
    --repo)      REPO_ID="$2"; shift 2 ;;
    --all)       ALL=1; shift ;;
    --no-images) IMAGES=0; shift ;;
    --dry-run)   DRY=1; shift ;;
    --env)       ENV_NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    *)  NAMES+=("$1"); shift ;;
  esac
done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
ROOT="$(realpath -m "$ROOT")"
RUNS="$ROOT/qib/runs"
[[ -d "$RUNS" ]] || die "$RUNS does not exist (wrong --root?)"
[[ "$REPO_ID" == */* ]] || die "--repo <owner>/<dataset>"
if (( ALL )); then
  for d in "$RUNS"/*/; do
    n=$(basename "$d")
    [[ -f "$d/input_bench_scores.json" && "$n" != *_limit* ]] && NAMES+=("$n")
  done
fi
(( ${#NAMES[@]} )) || die "no runs given: name them (e.g. turbo9 turbo2) or pass --all"

say "Qwen-Image-Bench runs -> HF dataset $REPO_ID (public)$( (( IMAGES )) || echo ', without images')"
for n in "${NAMES[@]}"; do
  d="$RUNS/$n"
  [[ -f "$d/input_bench_scores.json" ]] || die "$n: not a finished evaluation ($d/input_bench_scores.json missing)"
  total=$(python3 -c "import json,sys; print(round(json.load(open(sys.argv[1]))['total'], 2))" "$d/input_bench_scores.json")
  imgs=$(find "$d/images" -name '[0-9]*.png' 2>/dev/null | wc -l)
  size=$(du -sh --exclude=shards $( (( IMAGES )) || echo --exclude=images ) "$d" | cut -f1)
  info "$(printf '%-34s total %6s | %4s images | %s' "$n" "$total" "$imgs" "$size")"
done
(( DRY )) && { say "dry run: nothing uploaded"; exit 0; }

if [[ -z "${HF_TOKEN:-}" ]]; then
  [[ -t 0 ]] || die "no terminal to ask for the token: export HF_TOKEN=<write token> first"
  read -rs -p "HF write token (for $REPO_ID): " HF_TOKEN; echo; export HF_TOKEN
fi
crun() { ( [[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"; export PYTHONPATH="$ZDIR:$REPO:${PYTHONPATH:-}"
           cd "$REPO" && conda run -n "$ENV_NAME" --no-capture-output "$@" ); }
crun python - "$REPO_ID" <<'PY' || die "could not create $REPO_ID"
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
who = api.whoami()
if who.get("auth", {}).get("accessToken", {}).get("role") == "read":
    sys.exit("this is a Read token; uploading needs a Write token")
api.create_repo(sys.argv[1], repo_type="dataset", private=False, exist_ok=True)
print(f"    account {who['name']}: dataset {sys.argv[1]} ready")
PY

# upload-large-folder mirrors a folder: hard-link each run into <stage>/qwen-image-bench/<run>/
# (kept between runs of this script: it holds the upload's resume state)
STAGE="$ROOT/qib/.hf_upload"
for n in "${NAMES[@]}"; do
  d="$RUNS/$n"; s="$STAGE/qwen-image-bench/$n"
  rm -rf "$s"; mkdir -p "$s"
  for f in input_bench_scores.json input_judged.jsonl input.jsonl generation.json; do
    [[ -f "$d/$f" ]] && ln "$d/$f" "$s/$f"
  done
  [[ -d "$d/logs" ]] && cp -al "$d/logs" "$s/logs"
  (( IMAGES )) && [[ -d "$d/images" ]] && { mkdir -p "$s/images"; find "$d/images" -maxdepth 1 -name '[0-9]*.png' -exec ln {} "$s/images/" \; ; }
  crun python -m eval_qib.report "$d" --out "$s/report.md" >/dev/null || info "$n: no report.md (scores still go up)"
done
crun hf upload-large-folder "$REPO_ID" "$STAGE" --repo-type dataset --exclude ".cache/**" \
  || die "upload interrupted; run the same command again to resume"
say "done: https://huggingface.co/datasets/$REPO_ID/tree/main/qwen-image-bench"
for n in "${NAMES[@]}"; do info "$n"; done
