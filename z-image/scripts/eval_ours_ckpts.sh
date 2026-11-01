#!/usr/bin/env bash
# Qwen-Image-Bench for ours' intermediate checkpoints on a fresh rented machine, in one command: env,
# Z-Image-Turbo, judge, the checkpoints from Hugging Face, a smoke test, then render + judge + upload
# per checkpoint, and the table. Run it inside tmux.
#
#   bash z-image/scripts/eval_ours_ckpts.sh --root /data/<you>/rdm-sets --dry-run   # checks + plan
#   bash z-image/scripts/eval_ours_ckpts.sh --root /data/<you>/rdm-sets             # s500, s1000, s1500
#   bash z-image/scripts/eval_ours_ckpts.sh --root ... --steps 1000                 # this one only
#
# Rendering and judging can run on two machines, the images travelling through --eval-repo:
#
#   bash z-image/scripts/eval_ours_ckpts.sh --root ... --render-only   # checkpoints, render, upload images
#   bash z-image/scripts/eval_ours_ckpts.sh --root ... --judge-only    # download images, judge, upload scores
#
# Run again after an interruption: every finished stage is skipped, downloads and images resume.
#
# Stages:
#   1 env        scripts/setup_env.sh (conda env rdm), unless it exists
#   2 model      Z-Image-Turbo (31 GB) -> <root>/hf, unless <root>/env.sh already finds it
#   3 judge      eval_qib/run_eval.sh --setup: judge env + Qwen-Image-Bench + Q-Judger (65 GB) -> <root>/qib
#   4 ckpts      ours/step_*.pth (25 GB each) from --repo -> <root>/ckpts (--judge-only: the images instead)
#   5 smoke      the first checkpoint on the first --smoke prompts, so a broken chain fails in minutes
#   6 evaluate   per step: 1000 prompts, run <root>/qib/runs/zimage-2step-1024_s<step>, ~1.5 GB; each
#                finished run goes up to --eval-repo right away, so it outlives the machine
#   7 table      <root>/qib/runs/report.md, next to the paper's leaderboard (s2000 is not on this
#                machine: its 47.45 is in the dataset as zimage-2step-1024_s2000)
#
# --root DIR       data root (default /root/rdm-sets); a fresh machine needs ~177 GB free, ~112 GB with
#                  --render-only, ~71 GB with --judge-only
# --steps LIST     checkpoints to judge (default 500,1000,1500)
# --repo REPO      HF model repo with the checkpoints (default jiachengcui888/z-image)
# --eval-repo REPO HF dataset for the results (default jiachengcui888/evaluation-result)
# --smoke N        prompts of the smoke test (default 20; 0 skips it)
# --render-only    no judge and no judge weights, any card that renders is enough; the images go to
#                  --eval-repo (a run that already has scores there is not touched)
# --judge-only     the images from --eval-repo, then the judge (no Turbo, no checkpoints)
# --no-upload      keep the results on this machine (no token needed); they die with it
# --gpus IDS       the cards to use (default all visible)
# --env NAME       the rdm conda env (default rdm)
# --dry-run        checks and the plan, change nothing
#
# Needs conda, the NVIDIA driver and, unless --no-upload, a HF Write token: $HF_TOKEN, else asked for
# once at the start (hidden; never stored), so the upload cannot stop the run hours later.
# Log: <root>/qib/eval_ours_ckpts.log.
set -euo pipefail
export PYTHONNOUSERSITE=1
ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ZDIR/.." && pwd)"
ROOT="/root/rdm-sets"; STEPS="500,1000,1500"; CKPT_REPO="jiachengcui888/z-image"
EVAL_REPO="jiachengcui888/evaluation-result"; SMOKE=20; UPLOAD=1; GPUS=""; ENV_NAME="rdm"; DRY=0
DO_RENDER=1; DO_JUDGE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)      ROOT="$2"; shift 2 ;;
    --steps)     STEPS="$2"; shift 2 ;;
    --repo)      CKPT_REPO="$2"; shift 2 ;;
    --eval-repo) EVAL_REPO="$2"; shift 2 ;;
    --smoke)     SMOKE="$2"; shift 2 ;;
    --render-only) DO_JUDGE=0; shift ;;
    --judge-only)  DO_RENDER=0; shift ;;
    --no-upload) UPLOAD=0; shift ;;
    --gpus)      GPUS="$2"; shift 2 ;;
    --env)       ENV_NAME="$2"; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
ROOT="$(realpath -m "$ROOT")"; QIB="$ROOT/qib"; RUNS="$QIB/runs"; CKPTS="$ROOT/ckpts"
IFS=, read -r -a STEP_LIST <<< "$STEPS"
for s in "${STEP_LIST[@]}"; do [[ "$s" =~ ^[1-9][0-9]*$ ]] || die "--steps takes step numbers, e.g. 500,1000"; done
[[ "$SMOKE" =~ ^[0-9]+$ ]] || die "--smoke takes a number of prompts"
(( DO_RENDER || DO_JUDGE )) || die "--render-only and --judge-only exclude each other"
(( DO_JUDGE || UPLOAD )) || die "--render-only --no-upload: the judge machine could not get the images"
ckpt_rel() { printf 'ours/step_%07d.pth' "$1"; }
run_name() { printf 'zimage-2step-1024_s%s' "$1"; }
rendered() { [[ -f "$RUNS/$(run_name "$1")/input.jsonl" ]]; }

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
if (( DO_JUDGE )); then
  (( N >= PER )) || die "the judge needs $PER card(s) of $(( MIN_MIB / 1024 )) GB for one copy; only $N ($USE)"
  info "cards $USE ($(( MIN_MIB / 1024 )) GB): $( (( DO_RENDER )) && echo "render on all $N, ")judge $(( N / PER )) cop$( (( N / PER > 1 )) && echo ies || echo y) x $PER card(s)"
else
  info "cards $USE ($(( MIN_MIB / 1024 )) GB): render on all $N"
fi
info "root $ROOT | checkpoints ${STEP_LIST[*]} | smoke $SMOKE prompts"
if (( UPLOAD )); then info "results -> dataset $EVAL_REPO (public)"; else info "results stay on this machine (--no-upload)"; fi

mkdir -p "$ROOT"
NEED=0; MISSING=()
if (( DO_RENDER )) && [[ ! -d "$ROOT/hf" ]]; then NEED=$(( NEED + 31 )); fi
if (( DO_JUDGE )) && [[ ! -f "$QIB/env/.ready" || ! -f "$QIB/judge/config.json" ]]; then NEED=$(( NEED + 65 )); fi
for s in "${STEP_LIST[@]}"; do
  rendered "$s" && continue
  NEED=$(( NEED + 2 ))
  if (( DO_RENDER )) && [[ ! -f "$CKPTS/$(ckpt_rel "$s")" ]]; then
    NEED=$(( NEED + 25 )); MISSING+=("$(ckpt_rel "$s")")
  fi
done
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
info "disk: ~$NEED GB needed, $FREE GB free"
(( FREE >= NEED )) || die "not enough space under $ROOT: choose another --root"
if (( DRY )); then
  say "dry run: nothing changed"
  (( DO_RENDER )) && info "would download: ${MISSING[*]:-(no checkpoint needed)}"
  for s in "${STEP_LIST[@]}"; do
    if (( DO_RENDER )); then info "would render: $(run_name "$s") <- $CKPTS/$(ckpt_rel "$s")"
    else info "would download images: $EVAL_REPO qwen-image-bench/$(run_name "$s")"; fi
    (( DO_JUDGE )) && info "would judge: $(run_name "$s")"
  done
  exit 0
fi

crun() { ( [[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"; export PYTHONPATH="$ZDIR:$REPO:${PYTHONPATH:-}"
           cd "$REPO" && conda run -n "$ENV_NAME" --no-capture-output "$@" ); }
if (( UPLOAD )) && [[ -z "${HF_TOKEN:-}" ]]; then
  [[ -t 0 ]] || die "no terminal to ask for the token: export HF_TOKEN=<write token> first, or --no-upload"
  read -rs -p "HF write token (for $EVAL_REPO): " HF_TOKEN; echo; export HF_TOKEN   # lives only in this process
fi
mkdir -p "$QIB"
exec > >(tee -a "$QIB/eval_ours_ckpts.log") 2>&1

# 1 ------------------------------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  info "conda env $ENV_NAME present"
else
  say "1/7 conda env $ENV_NAME"
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

# 2 ------------------------------------------------------------------------------------------------
if (( DO_RENDER )); then
  have_turbo() {
    crun python -c "from zimage_distill.model import resolve_snapshot; print(resolve_snapshot())" 2>/dev/null | tail -1
  }
  if TURBO=$(have_turbo) && [[ -d "$TURBO" ]]; then
    info "Z-Image-Turbo present: $TURBO"
  else
    say "2/7 Z-Image-Turbo -> $ROOT/hf"
    crun python "$REPO/scripts/fetch_prerequisites.py" --root "$ROOT" --group zimage \
      || die "download incomplete; re-run to resume"
    TURBO=$(have_turbo) && [[ -d "$TURBO" ]] || die "Z-Image-Turbo still not found after the download"
  fi
fi

# 3 ------------------------------------------------------------------------------------------------
EVAL=(env ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" bash "$ZDIR/eval_qib/run_eval.sh" --root "$QIB")
EV=("${EVAL[@]}")
[[ -n "$GPUS" ]] && EV+=(--gen-gpus "$GPUS" --judge-gpus "$GPUS")
(( DO_JUDGE )) || EV+=(--skip-judge)
(( DO_RENDER )) || EV+=(--skip-generate)
if (( DO_JUDGE )); then
  say "3/7 judge env + Qwen-Image-Bench + Q-Judger -> $QIB"
  "${EVAL[@]}" --setup || die "the judge setup failed (see above); re-run to retry"
else
  say "3/7 Qwen-Image-Bench prompts -> $QIB/bench (--render-only: no judge)"
  crun python -c "import sys; from eval_qib.bench import fetch_bench; print('    bench ->', fetch_bench(sys.argv[1]))" "$QIB/bench" \
    || die "could not fetch the benchmark's prompts"
fi

# 4 ------------------------------------------------------------------------------------------------
if (( DO_RENDER )); then
  if (( ${#MISSING[@]} )); then
    say "4/7 checkpoints -> $CKPTS: ${MISSING[*]}"
    crun hf download "$CKPT_REPO" "${MISSING[@]}" --local-dir "$CKPTS" \
      || die "checkpoint download incomplete; re-run to resume"
    for f in "${MISSING[@]}"; do [[ -f "$CKPTS/$f" ]] || die "$f not downloaded (is it in $CKPT_REPO?)"; done
  fi
else
  say "4/7 images from $EVAL_REPO -> $RUNS"
  mkdir -p "$RUNS"
  DL="$QIB/.hf_download"
  for s in "${STEP_LIST[@]}"; do
    n="$(run_name "$s")"
    rendered "$s" && { info "$n: images present"; continue; }
    crun hf download "$EVAL_REPO" --repo-type dataset --include "qwen-image-bench/$n/*" --local-dir "$DL" >/dev/null \
      || die "image download incomplete; re-run to resume"
    got=$(find "$DL/qwen-image-bench/$n/images" -name '[0-9]*.png' 2>/dev/null | wc -l)
    (( got == 1000 )) && [[ -f "$DL/qwen-image-bench/$n/input.jsonl" ]] \
      || die "$n: $got of 1000 images in $EVAL_REPO (run --render-only first)"
    rm -rf "${RUNS:?}/${n:?}.tmp"; cp -al "$DL/qwen-image-bench/$n" "$RUNS/$n.tmp"; mv "$RUNS/$n.tmp" "$RUNS/$n"
    info "$n: 1000 images"
  done
fi

# 5 ------------------------------------------------------------------------------------------------
if (( SMOKE )); then
  s="${STEP_LIST[0]}"; n="$(run_name "$s")_limit$SMOKE"; d="$RUNS/$n"
  say "5/7 smoke test: $n"
  if (( DO_RENDER )); then
    if [[ -f "$d/input.jsonl" ]]; then
      "${EV[@]}" --name "$n" || die "the smoke test failed; see $d/logs"
    else
      "${EV[@]}" --name "$n" --limit "$SMOKE" --ckpt "$CKPTS/$(ckpt_rel "$s")" || die "the smoke test failed; see $d/logs"
    fi
  else
    if [[ ! -f "$d/input.jsonl" ]]; then      # the first prompts of the run downloaded above
      from="$RUNS/$(run_name "$s")"
      mkdir -p "$d/images"
      mapfile -t IMGS < <(find "$from/images" -maxdepth 1 -name '[0-9]*.png' | sort)
      cp -l "${IMGS[@]:0:$SMOKE}" "$d/images/"
      head -n "$SMOKE" "$from/input.jsonl" > "$d/input.tmp"; mv "$d/input.tmp" "$d/input.jsonl"
    fi
    "${EV[@]}" --name "$n" || die "the smoke test failed; see $d/logs"
  fi
fi

# 6 ------------------------------------------------------------------------------------------------
upload() { bash "$ZDIR/scripts/upload_eval_hf.sh" --root "$ROOT" --repo "$EVAL_REPO" --env "$ENV_NAME" "$@"; }
# --images-only deletes the scores a run has in the dataset unless they are on this machine: a
# repeated --render-only here must not erase what the judge machine uploaded
has_scores() {
  crun python - "$EVAL_REPO" "$1" <<'PY'
import sys
from huggingface_hub import HfApi
repo, run = sys.argv[1:]
try:
    files = {f.path for f in HfApi().list_repo_tree(repo, repo_type="dataset", path_in_repo=f"qwen-image-bench/{run}")}
except Exception as e:
    sys.exit(1 if "404" in str(e) else 2)
sys.exit(0 if f"qwen-image-bench/{run}/input_bench_scores.json" in files else 1)
PY
}
NAMES=(); TO_UPLOAD=()
for s in "${STEP_LIST[@]}"; do
  n="$(run_name "$s")"; NAMES+=("$n")
  say "6/7 $n"
  CK=(); if (( DO_RENDER )) && ! rendered "$s"; then CK=(--ckpt "$CKPTS/$(ckpt_rel "$s")"); fi
  "${EV[@]}" --name "$n" "${CK[@]}" || die "$n failed; re-run the same command (finished parts are kept)"
  (( UPLOAD )) || continue
  if (( DO_JUDGE )); then
    upload "$n" || info "upload of $n failed; tried again at the end"
    TO_UPLOAD+=("$n")
  else
    rc=0; has_scores "$n" || rc=$?
    if (( rc == 0 )); then
      info "$n: already judged in $EVAL_REPO: left as it is"
    elif (( rc == 1 )); then
      upload --images-only "$n" || info "upload of $n failed; tried again at the end"
      TO_UPLOAD+=("$n")
    else
      die "could not read $EVAL_REPO; re-run the same command"
    fi
  fi
done

# 7 ------------------------------------------------------------------------------------------------
if (( ! DO_JUDGE )); then
  if (( ${#TO_UPLOAD[@]} )); then
    upload --images-only "${TO_UPLOAD[@]}" || die "the upload failed; re-run the same command"
  fi
  say "rendered and uploaded: https://huggingface.co/datasets/$EVAL_REPO"
  info "next, on the judge machine: bash z-image/scripts/eval_ours_ckpts.sh --root <dir> --judge-only --steps $STEPS"
  info "the checkpoints are no longer needed here"
  exit 0
fi
say "7/7 table"
"${EVAL[@]}" --report "${NAMES[@]}" | sed -n '/^| model/,/^$/p'
info "full table: $RUNS/report.md"
if (( UPLOAD )); then
  upload "${TO_UPLOAD[@]}" || die "the upload failed; re-run the same command before giving the machine back"
  say "done: https://huggingface.co/datasets/$EVAL_REPO (safe to give the machine back)"
else
  say "done: results only in $RUNS -- copy them before giving the machine back"
fi
