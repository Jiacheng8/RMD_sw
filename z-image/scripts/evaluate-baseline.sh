#!/usr/bin/env bash
# Qwen-Image-Bench for Z-Image-Turbo itself (the baseline the 2-step student is compared with), in one
# command: the rdm env, Z-Image-Turbo, the judge env + Q-Judger, then render + judge + table.
#
#   bash z-image/scripts/evaluate-baseline.sh --root /data/<you>/rdm-sets              # teacher, 9 steps
#   bash z-image/scripts/evaluate-baseline.sh --root /data/<you>/rdm-sets --steps 9,2  # + the teacher at 2 steps
#   bash z-image/scripts/evaluate-baseline.sh --root /data/<you>/rdm-sets --limit 20   # smoke test, 20 prompts
#   CUDA_VISIBLE_DEVICES=0,2,3 bash z-image/scripts/evaluate-baseline.sh --root ...   # which cards
#
# Stages (each skipped when already done):
#   1 env       scripts/setup_env.sh (conda env rdm), unless it exists
#   2 model     Z-Image-Turbo (31 GB) -> <root>/hf, unless <root>/env.sh already finds it (then that
#               env.sh is left untouched)
#   3 judge     eval_qib/run_eval.sh --setup: the judge env + Qwen-Image-Bench + Q-Judger (54.7 GB)
#               -> <root>/qib
#   4 evaluate  per --steps value N: render the 1000 prompts with Turbo on its N-step schedule (Chinese
#               prompts, as the leaderboard), judge them -> <root>/qib/runs/turboN, ~1.5 GB
#   5 table     <root>/qib/runs/report.md, next to the paper's leaderboard
#
# --root DIR     data root (default /root/rdm-sets); ~100 GB: Z-Image 31 + judge env and weights 65
# --steps LIST   Turbo's step counts to evaluate (default 9 = the official setting; 2 = the student's
#                budget without distillation)
# --limit N      the first N prompts only (a smoke test; scores not comparable with the leaderboard,
#                kept apart as runs/turboN_limitN)
# --env NAME     the rdm conda env (default rdm)
# The rendering and the judge use every visible card: one judge copy per 80 GB card (or per 3 x 24 GB),
# copies in parallel. Estimates per model: ~10-15 min on 8 x H100, ~40 min on 2 x H100, ~5 h on
# 4 x 4090 (one copy) -- not measured yet.
set -euo pipefail
export PYTHONNOUSERSITE=1
ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ZDIR/.." && pwd)"
ROOT="/root/rdm-sets"; STEPS="9"; LIMIT=0; ENV_NAME="rdm"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)  ROOT="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --env)   ENV_NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
ROOT="$(realpath -m "$ROOT")"
mkdir -p "$ROOT"
command -v conda >/dev/null 2>&1 || die "conda not found on PATH"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.cache/pip}" CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$ROOT/.cache/conda-pkgs}"
IFS=, read -r -a STEP_LIST <<< "$STEPS"
for n in "${STEP_LIST[@]}"; do [[ "$n" =~ ^[1-9][0-9]*$ ]] || die "--steps takes step counts, e.g. 9,2"; done
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then GPUS="$CUDA_VISIBLE_DEVICES"
else GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | paste -sd, -); fi
[[ -n "$GPUS" ]] || die "no GPU visible"
info "root $ROOT | GPUs $GPUS | Turbo steps ${STEP_LIST[*]}${LIMIT:+$( (( LIMIT )) && echo " | first $LIMIT prompts")}"

# 1 ------------------------------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  info "conda env $ENV_NAME present"
else
  say "1/5 conda env $ENV_NAME"
  bash "$REPO/scripts/setup_env.sh" --name "$ENV_NAME"
fi

# 2 ------------------------------------------------------------------------------------------------
have_turbo() {
  ( [[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
    PYTHONPATH="$ZDIR" conda run -n "$ENV_NAME" --no-capture-output python -c \
      "from zimage_distill.model import resolve_snapshot; print(resolve_snapshot())" ) 2>/dev/null | tail -1
}
if TURBO=$(have_turbo) && [[ -d "$TURBO" ]]; then
  info "Z-Image-Turbo present: $TURBO"
else
  say "2/5 Z-Image-Turbo -> $ROOT/hf"
  conda run -n "$ENV_NAME" --no-capture-output python "$REPO/scripts/fetch_prerequisites.py" \
    --root "$ROOT" --group zimage || die "download incomplete; re-run to resume"
  TURBO=$(have_turbo) && [[ -d "$TURBO" ]] || die "Z-Image-Turbo still not found after the download"
fi

# 3-5 ----------------------------------------------------------------------------------------------
EVAL=(env ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" bash "$ZDIR/eval_qib/run_eval.sh" --root "$ROOT/qib")
say "3/5 judge env + Qwen-Image-Bench + Q-Judger -> $ROOT/qib"
"${EVAL[@]}" --setup || die "the judge setup failed (see above); re-run to retry"

NAMES=()
for n in "${STEP_LIST[@]}"; do
  name="turbo$n"; (( LIMIT )) && name="${name}_limit$LIMIT"
  say "4/5 Z-Image-Turbo, $n steps -> $ROOT/qib/runs/$name"
  "${EVAL[@]}" --name "$name" --teacher-steps "$n" --limit "$LIMIT" --gen-gpus "$GPUS" --judge-gpus "$GPUS" \
    || die "the evaluation of $name failed; re-run the same command (finished parts are kept)"
  NAMES+=("$name")
done

say "5/5 table"
"${EVAL[@]}" --report "${NAMES[@]}" | sed -n '/^| model/,/^$/p'
info "full table: $ROOT/qib/runs/report.md"
