#!/usr/bin/env bash
# No-cache control on a fresh machine, in one command: K128 · GAN with K = B = 128 -- every step
# generates all 128 window rows fresh and back-propagates them against those rows' references
# (experiments/no_cache/configs/nocache_k128_b128.yaml; trainer experiments/no_cache/train.py).
#
#   git clone -b no-cache https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   echo hf_xxx > .hf_token && chmod 600 .hf_token        # jiachengcui888's Read token
#   tmux new -s nocache
#   bash experiments/no_cache/run_nocache.sh --dry-run    # plan only, changes nothing
#   bash experiments/no_cache/run_nocache.sh              # same STEPS as K128 · GAN (2000)
#   bash experiments/no_cache/run_nocache.sh --match-hours 2   # same GPU TIME as K128 · GAN
#   bash experiments/no_cache/run_nocache.sh --resume     # continue after an interruption
#
# Stages, all re-runnable:
#   1 setup   scripts/new_machine.sh --no-train: conda env, downloads (models + COCO store),
#             the Qwen3 context, and (with --eval) the GenEval scorer
#   2 check   10 real-model steps in a scratch dir: memory and seconds per step
#   3 train   torchrun -m experiments.no_cache.train, `steps` steps (cosine lr over all of them),
#             10 checkpoints (save_freq = steps / 10)
#   4 eval    only with --eval: scripts/eval_run.sh (GenEval + PickScore + seed diversity)
#
# --root DIR     data root (default /root/rdm-sets); ~315 GB free
# --gpus N       training GPUs (default: every visible card, rounded down to 1/2/4/8); 80 GB cards
# --steps N      training steps (default 2000). The lr schedule and the 10 checkpoints follow N.
#                For the same-GPU-time comparison pick N from stage 2's seconds per step:
#                N = 2 h / (s per step); K128 · GAN does 2000 steps in ~2 h (3.6 s/step).
# --match-hours H  pick --steps automatically: stage 2 measures s/step and sets
#                N = H x 3600 / (s per step), rounded down to a multiple of 10. The choice is saved
#                in <root>/sw_lmmd/no_cache_steps.txt so --resume finds the same run.
# --eval         also evaluate every checkpoint when training ends (default: train only --
#                evaluate later on the 4090s; --no-eval is accepted and is the default)
# --resume       continue the stopped run from its resume.pth (skips stages 1-2)
# --dry-run      print the plan, change nothing
set -euo pipefail
export PYTHONNOUSERSITE=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CONFIG_REL="experiments/no_cache/configs/nocache_k128_b128.yaml"
B=128; MICRO=4
ROOT="/root/rdm-sets"; GPUS=""; STEPS=2000; EVAL=0; RESUME=0; DRY=0; MATCH_HOURS=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)    ROOT="$2"; shift 2 ;;
    --gpus)    GPUS="$2"; shift 2 ;;
    --steps)   STEPS="$2"; shift 2 ;;
    --match-hours) MATCH_HOURS="$2"; shift 2 ;;
    --no-eval) EVAL=0; shift ;;
    --eval)    EVAL=1; shift ;;
    --resume)  RESUME=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         echo "unknown option '$1' (see --help)" >&2; exit 2 ;;
  esac
done
[[ "$STEPS" =~ ^[0-9]+$ && "$STEPS" -ge 10 ]] || { echo "--steps must be an integer >= 10" >&2; exit 2; }
ROOT="$(realpath -m "$ROOT")"
SW="$ROOT/sw_lmmd"; STORE="$SW/reference_store"; OUT="$SW/work_dirs"; CHECK_DIR="$SW/no_cache_check"
STEPS_FILE="$SW/no_cache_steps.txt"
if [[ -n "$MATCH_HOURS" ]] && (( RESUME )); then
  [[ -s "$STEPS_FILE" ]] || { echo "--resume --match-hours: $STEPS_FILE missing; pass --steps N" >&2; exit 2; }
  STEPS="$(cat "$STEPS_FILE")"
fi
set_run() {     # derived from STEPS
  SAVE_FREQ=$(( STEPS / 10 ))
  EXP="nocache-k128b128"; [[ "$STEPS" != 2000 ]] && EXP="$EXP-s$STEPS"
  RUN_DIR="$OUT/$EXP"
}
set_run
ENV_NAME="${CONDA_ENV:-rdm}"
if [[ -z "${HF_TOKEN:-}" && -s "$REPO/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$REPO/.hf_token")"; export HF_TOKEN
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\033[31m    ERROR: %s\033[0m\n' "$*" >&2; exit 1; }
cd "$REPO"
if [[ -z "$GPUS" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    N=$([[ -z "$CUDA_VISIBLE_DEVICES" ]] && echo 0 || awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
  else
    N=$( (nvidia-smi -L 2>/dev/null || true) | grep -c '^GPU' || true)
  fi
  GPUS=1
  while (( GPUS * 2 <= N && GPUS * 2 <= 8 && B % (GPUS * 2 * MICRO) == 0 )); do GPUS=$(( GPUS * 2 )); done
fi
(( B % (GPUS * MICRO) == 0 )) || die "--gpus $GPUS does not split B=$B into micro-batches of $MICRO"
say "no-cache control: K = B = 128, $STEPS steps"
info "config    $CONFIG_REL"
info "root      $ROOT"
info "gpus      $GPUS (grad_accum $(( B / (GPUS * MICRO) )))$( (( EVAL )) && echo ', eval on all visible GPUs' || echo ', no evaluation')"
if [[ -n "$MATCH_HOURS" ]] && ! (( RESUME )); then
  info "steps     chosen after the 10-step check: ${MATCH_HOURS} h of training at the measured s/step"
else
  info "run dir   $RUN_DIR (checkpoints every $SAVE_FREQ steps)"
fi

train_cmd() {   # <output dir> <steps> <save_freq> [--set ...]
  local out="$1" steps="$2" freq="$3"; shift 3
  env ASSETS="$ROOT" conda run -n "$ENV_NAME" --no-capture-output torchrun --nproc_per_node="$GPUS" \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29500}" -m experiments.no_cache.train \
    "$CONFIG_REL" --set "reference_root=$STORE" --set "output_dir=$out" --set "steps=$steps" \
    --set "save_freq=$freq" --set "exp_name=$EXP" "$@"
}

# ---- 1 setup -----------------------------------------------------------------------------------
if (( RESUME )); then
  say "1 setup: skipped (--resume)"
else
  say "1 setup: scripts/new_machine.sh --no-train"
  NM=(bash "$REPO/scripts/new_machine.sh" --root "$ROOT" --config "$CONFIG_REL" --gpus "$GPUS"
      --env "$ENV_NAME" --no-train)
  (( EVAL )) || NM+=(--skip-geneval)
  (( DRY )) && NM+=(--dry-run)
  "${NM[@]}"
fi
[[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
if (( DRY )); then
  say "dry run: would run a 10-step check, then $STEPS steps -> $RUN_DIR$( (( EVAL )) && echo ', then eval' )"
  exit 0
fi

# ---- 2 check -----------------------------------------------------------------------------------
if (( RESUME )) || { [[ -z "$MATCH_HOURS" ]] && [[ -s "$RUN_DIR/train_log.jsonl" ]]; }; then
  say "2 check: skipped"
else
  say "2 check: 10 real-model steps -> $CHECK_DIR (deleted afterwards)"
  rm -rf "$CHECK_DIR"
  train_cmd "$CHECK_DIR" 10 1000 --set "save_resume=false" || die "the 10-step check failed (log above)"
  SPS=$(conda run -n "$ENV_NAME" --no-capture-output python - "$CHECK_DIR/$EXP/train_log.jsonl" <<'PY' | tail -1
import json, sys, statistics
rows = [json.loads(l) for l in open(sys.argv[1])]
print(statistics.median(r["seconds"] for r in rows[2:]) if len(rows) > 3 else rows[-1]["seconds"])
PY
)
  info "$(printf '%.1f' "$SPS") s per step (K128 · GAN: 3.6 s) -> 2000 steps ~ $(python3 -c "print(round(2000*$SPS/3600,1))") h"
  rm -rf "$CHECK_DIR"
  if [[ -n "$MATCH_HOURS" ]]; then
    STEPS=$(python3 -c "print(int($MATCH_HOURS*3600/$SPS)//10*10)")
    (( STEPS >= 10 )) || die "only $STEPS steps fit in $MATCH_HOURS h"
    echo "$STEPS" > "$STEPS_FILE"
    set_run
    info "--match-hours $MATCH_HOURS -> $STEPS steps, checkpoints every $SAVE_FREQ, run dir $RUN_DIR"
  fi
fi

# ---- 3 train -----------------------------------------------------------------------------------
say "3 train: $STEPS steps -> $RUN_DIR"
RES=()
if (( RESUME )); then
  [[ -f "$RUN_DIR/resume.pth" ]] || die "--resume: $RUN_DIR/resume.pth does not exist"
  RES=(--set "resume_from=$RUN_DIR/resume.pth")
elif [[ -s "$RUN_DIR/train_log.jsonl" ]]; then
  die "$RUN_DIR already holds a run: --resume to continue it, or move it away"
fi
train_cmd "$OUT" "$STEPS" "$SAVE_FREQ" ${RES[@]+"${RES[@]}"}

# ---- 4 eval ------------------------------------------------------------------------------------
if (( EVAL )); then
  say "4 eval: scripts/eval_run.sh $RUN_DIR"
  env ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" bash "$REPO/scripts/eval_run.sh" "$RUN_DIR" --root "$ROOT" --env "$ENV_NAME"
else
  say "done (no eval): checkpoints in $RUN_DIR"
fi
