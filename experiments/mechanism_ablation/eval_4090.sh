#!/usr/bin/env bash
# Score the mechanism-ablation arms on the 6 x 4090 paper evaluation box, after the H100 machines uploaded them.
#
#   tmux new -s mecheval
#   bash experiments/mechanism_ablation/eval_4090.sh --dry-run                # the plan
#   bash experiments/mechanism_ablation/eval_4090.sh                          # every arm that is on the Hub
#   bash experiments/mechanism_ablation/eval_4090.sh --runs window-nodiag     # one arm
#   bash experiments/mechanism_ablation/eval_4090.sh --anchor                 # + re-score main run step 3400
#
# For each run (window-nodiag, global-plus-self-b1): download step_0003400.pth from the model repo
# jiachengcui888/<run> and the training-side files from datasets/jiachengcui888/evaluation-result/
# mechanism-ablation/<run>/train_side/, score the checkpoint with the paper's pipeline on this box
# (<eval repo>/scripts/eval_checkpoint.sh --steps 1 --no-diversity, eval root /data/thor/jiacheng/rdm-sets-eval:
# GenEval 553 x 4, shared noise, official detector; PickScore 499 Pick-a-Pic, 1 image each -- the setup that
# scored Table 2's step-3400 numbers), write report.md / scores.json / run_info.json next to Table 2
# (Local 0.854 / 21.90, Global 0.746 / 21.81), and upload them to
# datasets/jiachengcui888/evaluation-result/mechanism-ablation/<run>/ (--no-upload keeps them local).
# One job per idle GPU (~22 min each); the downloaded checkpoint is deleted after its job. Re-running skips
# finished evaluations; a run not on the Hub yet is skipped with a note.
#
# --runs LIST     default "window-nodiag global-plus-self-b1"
# --anchor        also re-score the main run's step 3400 here (a check of this box's pipeline vs the paper)
# --gpus LIST     default: every GPU with < 1 GB in use
# --root DIR      eval root (default /data/thor/jiacheng/rdm-sets-eval)
# --eval-repo DIR the checkout whose eval scripts score (default /home/jiacheng/RDM, the paper's)
# --no-upload / --dry-run
set -euo pipefail
export PYTHONNOUSERSITE=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OWNER=jiachengcui888
RESULT_DATASET="$OWNER/evaluation-result"
RESULT_PATH="mechanism-ablation"
MAIN_REPO="$OWNER/w128_s32_2000steps_w-gan"
CKPT="step_0003400.pth"

RUNS="window-nodiag global-plus-self-b1"; ANCHOR=0; GPUS=""; ROOT=/data/thor/jiacheng/rdm-sets-eval
EVAL_REPO=/home/jiacheng/RDM; UPLOAD=1; DRY=0; ENV_NAME="${CONDA_ENV:-rdm}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)      RUNS="$2"; shift 2 ;;
    --anchor)    ANCHOR=1; shift ;;
    --gpus)      GPUS="$2"; shift 2 ;;
    --root)      ROOT="$2"; shift 2 ;;
    --eval-repo) EVAL_REPO="$2"; shift 2 ;;
    --no-upload) UPLOAD=0; shift ;;
    --dry-run)   DRY=1; shift ;;
    -h|--help)   sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           echo "unknown option '$1' (see --help)" >&2; exit 2 ;;
  esac
done
WORK="$ROOT/sw_lmmd/work_dirs/mechanism_ablation"
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m    WARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m    ERROR: %s\033[0m\n' "$*" >&2; exit 1; }
[[ -f "$ROOT/env.sh" ]] || die "no $ROOT/env.sh"
[[ -f "$ROOT/geneval/geneval_env.sh" ]] || die "no GenEval scorer env at $ROOT/geneval"
grep -q -- "--no-diversity" "$EVAL_REPO/scripts/eval_checkpoint.sh" 2>/dev/null || die "$EVAL_REPO/scripts/eval_checkpoint.sh missing or too old"
source "$ROOT/env.sh"
pyrun() { conda run -n "$ENV_NAME" --no-capture-output python "$@"; }
mkdir -p "$WORK/logs"
[[ -n "$GPUS" ]] || GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2 < 1000 {print $1}' | paste -sd, -)
[[ -n "$GPUS" ]] || die "no idle GPU; pass --gpus"
IFS=, read -r -a GPU_ARR <<< "$GPUS"

say "mechanism ablation: scoring on $(hostname)"
info "eval root  $ROOT | eval repo $EVAL_REPO @ $(git -C "$EVAL_REPO" rev-parse --short HEAD) | GPUs $GPUS"
info "runs       $RUNS$( (( ANCHOR )) && echo ' + anchor (main run step 3400)')"

# ---- fetch: training-side files and the step-3400 checkpoint of each run -------------------------------
fetch() {   # <run> -> prints the checkpoint path, or MISSING
  pyrun - "$OWNER/$1" "$CKPT" "$WORK/$1" "$RESULT_DATASET" "$RESULT_PATH/$1/train_side" <<'PY' | tail -1
import os, shutil, sys
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
repo, ckpt, out, ds, side = sys.argv[1:6]
try:
    d = snapshot_download(ds, repo_type="dataset", allow_patterns=[side + "/*"])
    if os.path.isdir(os.path.join(d, side)):
        shutil.copytree(os.path.join(d, side), os.path.join(out, "train_side"), dirs_exist_ok=True)
    print(hf_hub_download(repo, ckpt, local_dir=out))
except (EntryNotFoundError, RepositoryNotFoundError):
    print("MISSING")
PY
}

JOBS=()   # "<name> <hub repo> <out dir>"
for r in $RUNS; do
  [[ -s "$WORK/$r/eval_step_0003400/summary.json" ]] && { info "$r: scored before"; continue; }
  JOBS+=("$r $OWNER/$r $WORK/$r/eval_step_0003400")
done
if (( ANCHOR )) && [[ ! -s "$WORK/anchor_main_step3400/summary.json" ]]; then
  JOBS+=("anchor $MAIN_REPO $WORK/anchor_main_step3400")
fi
say "scoring ${#JOBS[@]} checkpoint(s), ${#GPU_ARR[@]} at a time"
for j in "${JOBS[@]}"; do info "$j"; done
if (( DRY )); then say "dry run: nothing scored or uploaded"; exit 0; fi

run_job() {   # <gpu> <name> <hub repo> <out>
  local gpu="$1" name="$2" repo="$3" out="$4" ckpt
  if [[ "$name" == anchor ]]; then
    ckpt="$(pyrun -c "from huggingface_hub import hf_hub_download as d; print(d('$repo', '$CKPT', local_dir='$WORK/anchor'))" | tail -1)"
  else
    ckpt="$(fetch "$name")"
  fi
  if [[ "$ckpt" == MISSING || ! -s "$ckpt" ]]; then echo "[$(date +%T)] $name: $CKPT not on $repo yet, skipped"; return 0; fi
  echo "[$(date +%T)] $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" bash "$EVAL_REPO/scripts/eval_checkpoint.sh" "$ckpt" --steps 1 --root "$ROOT" \
    --geneval-root "$ROOT/geneval" --out "$out" --no-diversity --env "$ENV_NAME" || echo "[$(date +%T)] $name FAILED"
  rm -f "$ckpt"                                  # this script's own download
  echo "[$(date +%T)] $name done"
}
pids=()
for gi in "${!GPU_ARR[@]}"; do
  (
    for ji in "${!JOBS[@]}"; do
      (( ji % ${#GPU_ARR[@]} == gi )) || continue
      read -r n rp o <<< "${JOBS[$ji]}"
      run_job "${GPU_ARR[$gi]}" "$n" "$rp" "$o" >> "$WORK/logs/eval_$n.log" 2>&1
      tail -1 "$WORK/logs/eval_$n.log"
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done

# ---- report + upload, per run ----------------------------------------------------------------------------
for r in $RUNS; do
  ev="$WORK/$r/eval_step_0003400"
  if [[ ! -s "$ev/summary.json" ]]; then warn "$r: not scored (see $WORK/logs/eval_$r.log)"; continue; fi
  [[ -d "$WORK/$r/train_side" ]] || fetch "$r" >/dev/null || true
  say "report: $r"; mkdir -p "$WORK/$r/report"
  ANC=(); [[ -s "$WORK/anchor_main_step3400/summary.json" ]] && ANC=(--anchor-eval "$WORK/anchor_main_step3400")
  (cd "$REPO" && pyrun -m experiments.mechanism_ablation.report scored --run-name "$r" --train-side "$WORK/$r/train_side" \
     --run-eval "$ev" ${ANC[@]+"${ANC[@]}"} --out "$WORK/$r/report" --eval-root "$ROOT" --eval-repo "$EVAL_REPO") \
    | tee "$WORK/$r/report/stdout.txt"
  if (( UPLOAD )); then
    pyrun - "$RESULT_DATASET" "$RESULT_PATH/$r" "$WORK/$r/report" <<'PY'
import sys
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id=sys.argv[1], repo_type="dataset", path_in_repo=sys.argv[2], folder_path=sys.argv[3],
                      allow_patterns=["report.md", "scores.json", "run_info.json"],
                      commit_message=f"mechanism ablation: {sys.argv[2]} scored on the paper evaluation box")
print(f"    -> https://huggingface.co/datasets/{sys.argv[1]}/tree/main/{sys.argv[2]}")
PY
  fi
done
say "summary"
grep -h '^==> SUMMARY' "$WORK"/*/report/stdout.txt 2>/dev/null || info "nothing scored yet"
