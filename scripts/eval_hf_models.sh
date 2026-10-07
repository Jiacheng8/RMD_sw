#!/usr/bin/env bash
# Evaluate every checkpoint of the SW-LMMD runs on the Hub (jiachengcui888/*), then the 4-step
# klein-4B teacher and the released iRDM s180, and write one report.
#
#   tmux new -s eval
#   bash scripts/eval_hf_models.sh --dry-run             # the plan: runs, checkpoints, GPUs; nothing runs
#   bash scripts/eval_hf_models.sh                       # everything (resumable: re-run after any stop)
#   bash scripts/eval_hf_models.sh --gpus 3,4            # only these GPUs
#   bash scripts/eval_hf_models.sh --report-only         # rebuild the report from what is finished
#
# For every checkpoint: download it (one ahead of the GPUs), evaluate it with
# scripts/eval_checkpoint.sh -- GenEval (553 prompts x 4 seeds), PickScore (Pick-a-Pic 499), seed
# diversity (how different the 4 images of one prompt are: pixel, DreamSim, DINOv3) -- and delete it
# as soon as its evaluation is complete. A failed evaluation is retried once at the end of the queue,
# and keeps its checkpoint if it fails again; a re-run picks up exactly where things stopped.
# A GPU is used only while it has >= 22 GB free (other jobs on it are waited for, not crashed into).
#
# Output, under --out (default /data/ironman/jiacheng/rdm-eval):
#   report.md       table 1: each run's best checkpoint (by GenEval) vs the teacher and s180; the best
#                   checkpoint with diversity >= 0.9x the teacher; GenEval per task; figures; all steps
#   curves.html     interactive: x = step, y = GenEval / PickScore / seed diversity; tick any runs
#   fig_*.png       the three curves for report.md;   results.csv: every number
#   evals/          per checkpoint: summary.json, GenEval renders + results, PickScore, diversity
#   logs/           the driver log and one log per evaluation
#
# Time: ~25 min per checkpoint on an idle RTX 4090 (the teacher ~35), slower on a shared card;
# 75 checkpoints + 2 baselines on 4 GPUs ~ 8-12 h. Disk: ~1 GB of renders per evaluation (77 GB,
# --drop-images to delete them once scored) + at most (GPUs + 1) x 15.5 GB of checkpoints in flight.
#
# Options: --out DIR  --root DIR (data root with env.sh; default /data/thor/jiacheng/rdm-sets)
#          --geneval-root DIR (default <root>/geneval, else <root>/geneval_test)  --env NAME (rdm)
#          --gpus LIST  --runs REPO,REPO  --only-steps 200,2000  --no-baselines  --drop-images
#          --prefetch N (1)  --report-only  --dry-run
set -euo pipefail
export PYTHONNOUSERSITE=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="/data/ironman/jiacheng/rdm-eval"
ROOT="/data/thor/jiacheng/rdm-sets"
GROOT=""
ENV_NAME="rdm"
REPORT_ONLY=0
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)           OUT="$2"; shift 2 ;;
    --root)          ROOT="$2"; shift 2 ;;
    --geneval-root)  GROOT="$2"; shift 2 ;;
    --env)           ENV_NAME="$2"; shift 2 ;;
    --gpus|--runs|--only-steps|--prefetch|--attempts)
                     PASS+=("$1" "$2"); shift 2 ;;
    --no-baselines|--drop-images|--dry-run|--no-geneval)
                     PASS+=("$1"); shift ;;
    --report-only)   REPORT_ONLY=1; shift ;;
    -h|--help)       sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
OUT="$(realpath -m "$OUT")"
ROOT="$(realpath -m "$ROOT")"
die() { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
command -v conda >/dev/null 2>&1 || die "conda not found on PATH"
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || die "conda env '$ENV_NAME' not found"
PY=(conda run -n "$ENV_NAME" --no-capture-output python -u)
mkdir -p "$OUT/logs"
# one driver per --out: two would download and evaluate the same checkpoints into the same dirs
exec 9>"$OUT/.eval_hf_models.lock"
flock -n 9 || die "another eval_hf_models.sh is already running on $OUT (it holds $OUT/.eval_hf_models.lock)"

if [[ $REPORT_ONLY -eq 0 ]]; then
  [[ -f "$ROOT/env.sh" ]] || die "$ROOT/env.sh not found -- pass --root <the data root with the models>"
  if [[ -z "$GROOT" ]]; then
    for g in "$ROOT/geneval" "$ROOT/geneval_test"; do [[ -f "$g/geneval_env.sh" ]] && { GROOT="$g"; break; }; done
  fi
  [[ -n "$GROOT" && -f "$GROOT/geneval_env.sh" ]] || die "no GenEval scorer env found: pass --geneval-root"
  [[ -z "${TMUX:-}${STY:-}" ]] && printf '\033[33mWARNING:\033[0m not inside tmux/screen -- this runs for hours\n'
  LOG="$OUT/logs/eval_hf_models_$(date +%Y%m%d_%H%M%S).log"
  echo "==> evaluating into $OUT (log: $LOG)"
  echo "    data root $ROOT, GenEval scorer $GROOT, conda env $ENV_NAME"
  set +e
  ( source "$ROOT/env.sh" && cd "$REPO" && "${PY[@]}" scripts/_eval_hf_models.py --out "$OUT" \
      --root "$ROOT" --geneval-root "$GROOT" --env "$ENV_NAME" "${PASS[@]}" ) 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}
  set -e
  [[ $RC -eq 130 ]] && exit 130
  for a in "${PASS[@]}"; do [[ "$a" == --dry-run ]] && exit "$RC"; done
else
  RC=0
fi

echo "==> report"
( cd "$REPO" && "${PY[@]}" scripts/eval_hf_report.py --out "$OUT" ) || die "report generation failed"
echo "    $OUT/report.md   (open $OUT/curves.html in a browser for the interactive curves)"
exit "$RC"
