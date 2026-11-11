#!/usr/bin/env bash
# Qwen-Image-Bench scores for runs whose images are already in the evaluation dataset, in one command
# on a fresh machine: rdm env, judge env + Q-Judger, the images from Hugging Face, a smoke test, the
# judge, a table next to ours and the teacher, and the scores back to the dataset. Run it inside tmux.
#
#   bash z-image/scripts/judge_hub_runs.sh --root /data/<you>/rdm-sets --dry-run    # checks + plan
#   bash z-image/scripts/judge_hub_runs.sh --root /data/<you>/rdm-sets              # TwinFlow + DIDR
#   bash z-image/scripts/judge_hub_runs.sh --root ... zimage-2step-1024_s500 --suffix _rejudge
#                                                     # the judge's own noise: s500 judged again
#
# RUN ... are folder names under qwen-image-bench/ in --eval-repo (default: the two released
# distillations rendered by qib_render_released.sh, twinflow-zimage-turbo-exp_2nfe_1024 and
# didr-zimage_1nfe_1024). The images are not rendered again: only images/, input.jsonl and
# generation.json are downloaded, judged here with the official judge.py (thinking on, as the
# leaderboard), and the scores uploaded into the same folder (images are not uploaded again).
# A run that already has scores in the dataset is refused: judge it under another name (--suffix).
#
# Run again after an interruption: every finished stage is kept (judge parts too).
#
# Stages:
#   1 env        scripts/setup_env.sh (conda env rdm), unless it exists
#   2 judge      eval_qib/run_eval.sh --setup: judge env + Qwen-Image-Bench + Q-Judger (65 GB) -> <root>/qib
#   3 images     qwen-image-bench/<run>/images from --eval-repo -> <root>/qib/runs/<run><suffix>, ~1.5 GB each
#   4 smoke      the first run's first --smoke images judged, so a broken chain fails in minutes
#   5 judge      each run in turn; each finished run's scores go up right away
#   6 table      <root>/qib/runs/report_hub_runs.md: these runs next to --ref (scores from the dataset)
#
# Time: one judge copy per 80 GB card (or 3 x 24 GB), copies in parallel. Per 1000 images 3.6-6 GPU-h
# measured on A100 / H-class cards (s500, s2000): ~1-1.5 h per run on 4 x 80 GB, twice that on 40 GB cards.
#
# --root DIR       data root (default /root/rdm-sets); a fresh machine needs ~70 GB free
# --eval-repo REPO HF dataset with the images, gets the scores (default jiachengcui888/evaluation-result)
# --suffix S       judge (and upload) as <run>S, e.g. _rejudge; needed for a run already judged there
# --ref LIST       runs in --eval-repo whose scores join the table (default
#                  zimage-2step-1024_s500,turbo9,turbo2; "" for none)
# --smoke N        images of the smoke test (default 20; 0 skips it)
# --no-upload      keep the scores on this machine (no token needed); they die with it
# --gpus IDS       the cards to use (default all visible)
# --env NAME       the rdm conda env (default rdm)
# --dry-run        checks and the plan, change nothing
#
# Needs conda, the NVIDIA driver and, unless --no-upload, a HF Write token: $HF_TOKEN, else asked for
# once at the start (hidden; never stored), so the upload cannot stop the run hours later.
# Log: <root>/qib/judge_hub_runs.log.
set -euo pipefail
export PYTHONNOUSERSITE=1
ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ZDIR/.." && pwd)"
ROOT="/root/rdm-sets"; EVAL_REPO="jiachengcui888/evaluation-result"; SUFFIX=""
REFS="zimage-2step-1024_s500,turbo9,turbo2"; SMOKE=20; UPLOAD=1; GPUS=""; ENV_NAME="rdm"; DRY=0
RUN_LIST=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)      ROOT="$2"; shift 2 ;;
    --eval-repo) EVAL_REPO="$2"; shift 2 ;;
    --suffix)    SUFFIX="$2"; shift 2 ;;
    --ref)       REFS="$2"; shift 2 ;;
    --smoke)     SMOKE="$2"; shift 2 ;;
    --no-upload) UPLOAD=0; shift ;;
    --gpus)      GPUS="$2"; shift 2 ;;
    --env)       ENV_NAME="$2"; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    *)  RUN_LIST+=("$1"); shift ;;
  esac
done
(( ${#RUN_LIST[@]} )) || RUN_LIST=(twinflow-zimage-turbo-exp_2nfe_1024 didr-zimage_1nfe_1024)
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
ROOT="$(realpath -m "$ROOT")"; QIB="$ROOT/qib"; RUNS="$QIB/runs"
IFS=, read -r -a REF_LIST <<< "$REFS"
[[ "$SMOKE" =~ ^[0-9]+$ ]] || die "--smoke takes a number of images"
[[ "$SUFFIX" =~ ^[A-Za-z0-9._-]*$ ]] || die "--suffix: letters, digits, . _ - only"
local_name() { echo "$1$SUFFIX"; }
judged() { [[ -f "$RUNS/$(local_name "$1")/input_bench_scores.json" ]]; }

# --- checks: everything that can be known now fails now, not hours into the run ------------------
command -v conda >/dev/null 2>&1 || die "conda not found on PATH"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: no NVIDIA driver?"
if [[ -n "$GPUS" ]]; then    # nvidia-smi -i skips a card that does not exist without a word
  CARDS=",$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | paste -sd, -),"
  for g in ${GPUS//,/ }; do [[ "$CARDS" == *",$g,"* ]] || die "--gpus: no card $g here (cards ${CARDS:1:-1})"; done
fi
USE="${GPUS:-${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | paste -sd, -)}}"
MIN_MIB=$(nvidia-smi -i "$USE" --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
PER=$(( (66000 + MIN_MIB - 1) / MIN_MIB )); N=$(awk -F, '{print NF}' <<< "$USE")
(( N >= PER )) || die "the judge needs $PER card(s) of $(( MIN_MIB / 1024 )) GB for one copy; only $N ($USE)"
COPIES=$(( N / PER ))
info "cards $USE ($(( MIN_MIB / 1024 )) GB): judge $COPIES cop$( (( COPIES > 1 )) && echo ies || echo y) x $PER card(s)"
info "root $ROOT | runs ${RUN_LIST[*]}$( [[ -n "$SUFFIX" ]] && echo " (as *$SUFFIX)") | smoke $SMOKE images"
if (( UPLOAD )); then info "scores -> dataset $EVAL_REPO"; else info "scores stay on this machine (--no-upload)"; fi
mkdir -p "$ROOT"
NEED=0
[[ -f "$QIB/env/.ready" && -f "$QIB/judge/config.json" ]] || NEED=$(( NEED + 65 ))
for r in "${RUN_LIST[@]}"; do [[ -f "$RUNS/$(local_name "$r")/input.jsonl" ]] || NEED=$(( NEED + 2 )); done
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
info "disk: ~$NEED GB needed, $FREE GB free"
(( FREE >= NEED )) || die "not enough space under $ROOT: choose another --root"
if (( DRY )); then
  say "dry run: nothing changed"
  for r in "${RUN_LIST[@]}"; do
    info "would download $EVAL_REPO qwen-image-bench/$r/images, judge as $(local_name "$r")$( (( UPLOAD )) && echo ", upload its scores")"
  done
  (( ${#REF_LIST[@]} )) && info "table with: ${REF_LIST[*]}"
  exit 0
fi

crun() { ( [[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"; export PYTHONPATH="$ZDIR:$REPO:${PYTHONPATH:-}"
           cd "$REPO" && conda run -n "$ENV_NAME" --no-capture-output "$@" ); }
if (( UPLOAD )) && [[ -z "${HF_TOKEN:-}" ]]; then
  [[ -t 0 ]] || die "no terminal to ask for the token: export HF_TOKEN=<write token> first, or --no-upload"
  read -rs -p "HF write token (for $EVAL_REPO): " HF_TOKEN; echo; export HF_TOKEN   # lives only in this process
fi
mkdir -p "$QIB" "$RUNS"
exec > >(tee -a "$QIB/judge_hub_runs.log") 2>&1
say "start $(date '+%F %T') on $(hostname)"

# 1 ------------------------------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  info "conda env $ENV_NAME present"
else
  say "1/6 conda env $ENV_NAME"
  bash "$REPO/scripts/setup_env.sh" --name "$ENV_NAME"
fi
if (( UPLOAD )); then    # the env is here now: reject a bad token before the hours of work
  crun python - <<'PY' || die "the HF token cannot upload (rejected or Read only); export a Write token, or --no-upload"
import os, sys
from huggingface_hub import whoami
w = whoami(token=os.environ["HF_TOKEN"])
role = w.get("auth", {}).get("accessToken", {}).get("role", "?")
print(f"    token account: {w['name']} (role: {role})")
sys.exit(1 if role == "read" else 0)
PY
fi
# a run judged before in the dataset is not judged again under the same name (its scores would be
# overwritten by a second, different draw of the judge)
crun python - "$EVAL_REPO" "$SUFFIX" "${RUN_LIST[@]}" <<'PY' || die "see above"
import sys
from huggingface_hub import HfApi
repo, suffix, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
api, bad = HfApi(), []
for r in runs:
    try:
        have = {e.path.rsplit("/", 1)[-1] for e in api.list_repo_tree(repo, repo_type="dataset", path_in_repo=f"qwen-image-bench/{r}")}
    except Exception as e:
        if "404" not in str(e) and "not found" not in str(e).lower():
            raise
        have = set()
    if "images" not in have or "input.jsonl" not in have:
        bad.append(f"{r}: no images/input.jsonl in {repo}")
    if suffix:
        try:
            if any(True for _ in api.list_repo_tree(repo, repo_type="dataset", path_in_repo=f"qwen-image-bench/{r}{suffix}")):
                bad.append(f"{r}{suffix}: already in {repo}; choose another --suffix")
        except Exception as e:
            if "404" not in str(e) and "not found" not in str(e).lower():
                raise
    elif "input_bench_scores.json" in have:
        bad.append(f"{r}: already judged in {repo}; judge it again under another name with --suffix")
for b in bad:
    print("    " + b)
sys.exit(1 if bad else 0)
PY

# 2 ------------------------------------------------------------------------------------------------
EVAL=(env ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" bash "$ZDIR/eval_qib/run_eval.sh" --root "$QIB")
EV=("${EVAL[@]}" --skip-generate)
[[ -n "$GPUS" ]] && EV+=(--judge-gpus "$GPUS")
say "2/6 judge env + Qwen-Image-Bench + Q-Judger -> $QIB"
"${EVAL[@]}" --setup || die "the judge setup failed (see above); re-run to retry"

# 3 ------------------------------------------------------------------------------------------------
say "3/6 images from $EVAL_REPO -> $RUNS"
DL="$QIB/.hf_download"
for r in "${RUN_LIST[@]}"; do
  n="$(local_name "$r")"; d="$RUNS/$n"
  [[ -f "$d/input.jsonl" ]] && { info "$n: images present"; continue; }
  # one --include per pattern: file names given next to --include make hf ignore the --include
  crun hf download "$EVAL_REPO" --repo-type dataset --include "qwen-image-bench/$r/images/*" \
    --include "qwen-image-bench/$r/input.jsonl" --include "qwen-image-bench/$r/generation.json" --local-dir "$DL" >/dev/null \
    || die "image download incomplete; re-run to resume"
  src="$DL/qwen-image-bench/$r"
  rows=$(grep -c . "$src/input.jsonl"); got=$(find "$src/images" -name '[0-9]*.png' | wc -l)
  (( got == rows )) || die "$r: $got images for $rows rows of input.jsonl in $EVAL_REPO"
  rm -rf "${d:?}.tmp"; mkdir -p "$d.tmp"
  cp -al "$src/images" "$d.tmp/images"
  cp "$src/generation.json" "$d.tmp/" 2>/dev/null || true
  cp "$src/input.jsonl" "$d.tmp/input.jsonl"     # run_eval.sh re-points image_path at $d/images
  mv "$d.tmp" "$d"
  info "$n: $got images"
done

# 4 ------------------------------------------------------------------------------------------------
if (( SMOKE )); then
  r="${RUN_LIST[0]}"; from="$RUNS/$(local_name "$r")"; n="$(local_name "$r")_limit$SMOKE"; d="$RUNS/$n"
  say "4/6 smoke test: $n"
  if [[ ! -f "$d/input.jsonl" ]]; then
    rm -rf "${d:?}"; mkdir -p "$d/images"
    head -n "$SMOKE" "$from/input.jsonl" > "$d/input.tmp"
    python3 - "$d/input.tmp" "$from/images" "$d/images" <<'PY'
import json, os, sys
rows, src, dst = sys.argv[1:]
for line in open(rows, encoding="utf-8"):
    if line.strip():
        name = os.path.basename(json.loads(line)["image_path"])
        os.link(os.path.join(src, name), os.path.join(dst, name))
PY
    [[ -f "$from/generation.json" ]] && cp "$from/generation.json" "$d/"
    mv "$d/input.tmp" "$d/input.jsonl"
  fi
  "${EV[@]}" --name "$n" || die "the smoke test failed; see $d/logs"
fi

# 5 ------------------------------------------------------------------------------------------------
upload() { bash "$ZDIR/scripts/upload_eval_hf.sh" --root "$ROOT" --repo "$EVAL_REPO" --env "$ENV_NAME" --no-images "$@"; }
NAMES=(); FAILED=()
for r in "${RUN_LIST[@]}"; do
  n="$(local_name "$r")"; NAMES+=("$n")
  say "5/6 judge $n"
  "${EV[@]}" --name "$n" || die "$n failed; re-run the same command (finished parts are kept)"
  (( UPLOAD )) || continue
  upload "$n" || { info "upload of $n failed; tried again at the end"; FAILED+=("$n"); }
done

# 6 ------------------------------------------------------------------------------------------------
say "6/6 table"
DIRS=()
for n in "${NAMES[@]}"; do DIRS+=("$RUNS/$n"); done
if (( ${#REF_LIST[@]} )); then
  REFD="$QIB/hub_scores"
  for ref in "${REF_LIST[@]}"; do
    crun hf download "$EVAL_REPO" "qwen-image-bench/$ref/input_bench_scores.json" \
      "qwen-image-bench/$ref/generation.json" "qwen-image-bench/$ref/input_judged.jsonl" --repo-type dataset --local-dir "$REFD" >/dev/null \
      && [[ -f "$REFD/qwen-image-bench/$ref/input_bench_scores.json" ]] \
      && DIRS+=("$REFD/qwen-image-bench/$ref") || info "$ref: no scores in $EVAL_REPO, left out of the table"
  done
fi
crun python -m eval_qib.report "${DIRS[@]}" --out "$RUNS/report_hub_runs.md" | sed -n '/^| model/,/^$/p'
if (( UPLOAD && ${#FAILED[@]} )); then
  upload "${FAILED[@]}" || die "the upload failed; re-run the same command before giving the machine back"
fi
if (( UPLOAD )); then
  say "done: scores in https://huggingface.co/datasets/$EVAL_REPO/tree/main/qwen-image-bench (safe to give the machine back)"
else
  say "done: scores only in $RUNS -- copy them before giving the machine back"
fi
info "table: $RUNS/report_hub_runs.md"
