#!/usr/bin/env bash
# Continue the K128 · GAN run (jiachengcui888/w128_s32_2000steps_w-gan, trained to step 2000 at commit
# 71e96e9) for 2000 more steps, from an empty machine, in one command. 2 x H100 / A100 80 GB.
#
#   git clone -b main https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   echo hf_xxx > .hf_token && chmod 600 .hf_token     # gated FLUX.2 VAE + the private reference store
#   tmux new -s k128
#   bash scripts/continue_k128_gan.sh --root /root/rdm-sets --dry-run     # plan only
#   bash scripts/continue_k128_gan.sh --root /root/rdm-sets
#
# Stages (a re-run continues: finished stages are skipped, a stopped run goes on from its resume.pth):
#   1 setup    scripts/new_machine.sh --no-train --skip-geneval: conda env rdm, encoders + FLUX.2 klein-4B,
#              the COCO reference store, its Qwen3 context (~105 GB)
#   2 resume   resume.pth (weights, AdamW8bit moments, cache, critic at step 2000; 23.5 GB) and train_log.jsonl
#              from the public HF repo -> <root>/sw_lmmd/work_dirs/sw-lmmd-flux-h100-2gpu-gan/
#   3 train    scripts/new_machine.sh --resume --steps <2000 + --more-steps> --micro-batch 4
#              (checkpoints step_0002200 .. every 200 steps, resume.pth every 400 steps and at the end)
#
# Learning rate. The original cosine decay ended at step 2000 at its floor, 2.83e-7.
#   --lr restart  (default) the decay is stretched to the new last step, as any run of that length:
#                 the lr jumps at step 2000 (1.61e-6 for 2000 more steps) and decays to 2.83e-7 at the end
#   --lr floor    the original schedule stays: constant 2.83e-7 for all the new steps
#
# --root DIR        data root (default /root/rdm-sets); ~320 GB free
# --more-steps N    steps to add (default 2000; a multiple of 200, the checkpoint interval)
# --gpus N          training GPUs (default 2)
# --micro-batch N   rows per forward per GPU (default 4: what the run used; 8 runs out of memory on 80 GB)
# --lr MODE         restart | floor (see above)
# --env NAME        conda env (default rdm)
# --skip-setup      skip stage 1 (the machine is already set up under --root)
# --dry-run         print the plan, change nothing
set -euo pipefail
export PYTHONNOUSERSITE=1
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "bash >= 4.4 required (this is $BASH_VERSION)" >&2; exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="/root/rdm-sets"; MORE=2000; GPUS=2; MICRO=4; LR_MODE="restart"; ENV_NAME="${CONDA_ENV:-rdm}"; SETUP=1; DRY=0
HF_REPO="jiachengcui888/w128_s32_2000steps_w-gan"
CONFIG="configs/sw_lmmd_train_h100_2gpu_gan.yaml"
EXP="sw-lmmd-flux-h100-2gpu-gan"       # the config's exp_name: the run directory
START=2000; RESUME_BYTES=23504415310

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)        ROOT="$2"; shift 2 ;;
    --more-steps)  MORE="$2"; shift 2 ;;
    --gpus)        GPUS="$2"; shift 2 ;;
    --micro-batch) MICRO="$2"; shift 2 ;;
    --lr)          LR_MODE="$2"; shift 2 ;;
    --env)         ENV_NAME="$2"; shift 2 ;;
    --skip-setup)  SETUP=0; shift ;;
    --dry-run)     DRY=1; shift ;;
    -h|--help)     sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)             echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$MORE" =~ ^[0-9]+$ ]] && (( MORE >= 200 && MORE % 200 == 0 )) || die "--more-steps must be a positive multiple of 200"
[[ "$LR_MODE" == restart || "$LR_MODE" == floor ]] || die "--lr must be restart or floor"
(( GPUS >= 1 && 32 % (GPUS * MICRO) == 0 )) || die "--gpus $GPUS x --micro-batch $MICRO must divide B = 32"
TARGET=$(( START + MORE ))
ROOT="$(realpath -m "$ROOT")"
RUN_DIR="$ROOT/sw_lmmd/work_dirs/$EXP"
cd "$REPO"

if ! command -v conda >/dev/null 2>&1; then       # a non-interactive shell (an agent's, tmux -c)
  for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /root/miniconda3; do
    [[ -x "$c/bin/conda" ]] && { export PATH="$c/bin:$PATH"; break; }
  done
fi
command -v conda >/dev/null 2>&1 || die "conda not found: install Miniconda3 first"
crun() { conda run -n "$ENV_NAME" --no-capture-output "$@"; }

# the lr of this plan, from the same formula as rdm/sw_lmmd/config.py LRSchedule (base 2.83e-6, warm-up 100, floor 0.1)
LR_PLAN=$(python3 - "$START" "$TARGET" "$LR_MODE" <<'PY'
import math, sys
start, target, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
horizon = target if mode == "restart" else start
def lr(s):
    p = min(1.0, (s - 100) / max(1, horizon - 100))
    return 2.83e-6 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
print(f"{lr(start):.3g} at step {start} -> {lr((start + target) // 2):.3g} at step {(start + target) // 2} -> {lr(target - 1):.3g} at step {target}")
PY
)

say "continue K128 · GAN: step $START -> $TARGET | repo $(git rev-parse --short HEAD 2>/dev/null || echo '?') ($(git branch --show-current 2>/dev/null || echo '?'))"
info "root        $ROOT"
info "run dir     $RUN_DIR"
info "batching    $GPUS GPUs x micro $MICRO x accum $(( 32 / (GPUS * MICRO) )) = 32 rows per update"
info "lr ($LR_MODE)  $LR_PLAN"
info "checkpoints $(( MORE / 200 )) x 15.5 GB (step_$(printf %07d $(( START + 200 ))) ... step_$(printf %07d $TARGET))"
[[ $DRY -eq 1 || -n "${TMUX:-}${STY:-}" ]] || info "not inside tmux/screen: a dropped SSH session would stop this"

# The model and critic of this run came from commit 71e96e9; main only adds options that are off by default.
if [[ $DRY -eq 0 ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: no GPUs"
  mkdir -p "$ROOT"
  mapfile -t ROWS < <(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits)
  (( ${#ROWS[@]} >= GPUS )) || die "need $GPUS GPUs, found ${#ROWS[@]}"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=, read -r -a SEL <<< "$CUDA_VISIBLE_DEVICES"; else mapfile -t SEL < <(seq 0 $(( GPUS - 1 ))); fi
  (( ${#SEL[@]} >= GPUS )) || die "CUDA_VISIBLE_DEVICES has fewer than $GPUS GPUs"
  export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${SEL[*]:0:$GPUS}")"
  for g in "${SEL[@]:0:$GPUS}"; do
    IFS=', ' read -r _ total free <<< "$(printf '%s\n' "${ROWS[@]}" | awk -F', ' -v g="$g" '$1 == g')"
    info "GPU $g       $free / $total MiB free"
    (( total >= 79000 )) || die "GPU $g has $total MiB: this run needs 80 GB cards"
    (( free >= 70000 )) || die "GPU $g has only $free MiB free: stop what is running there, or choose idle cards with CUDA_VISIBLE_DEVICES"
  done
  FREE_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  have_kb=$(du -sk "$ROOT" 2>/dev/null | cut -f1 || echo 0)
  NEED_GB=$(( 26 + 17 + 61 + 8 + 47 + MORE / 200 * 16 + 5 - have_kb / 1048576 ))
  (( NEED_GB < 60 )) && NEED_GB=60
  info "disk        $FREE_GB GB free, ~$NEED_GB GB still needed"
  (( FREE_GB >= NEED_GB )) || die "under ~$NEED_GB GB free on $ROOT: choose a bigger --root"
fi

# ---------------------------------------------------------------------------- 1 setup
say "1/3 setup (environment, models, reference store, Qwen3 context)"
NM=(bash scripts/new_machine.sh --root "$ROOT" --gpus "$GPUS" --config "$CONFIG" --env "$ENV_NAME" --allow-low-disk)
if (( SETUP )); then
  S=("${NM[@]}" --no-train --skip-geneval); (( DRY )) && S+=(--dry-run)
  "${S[@]}"
else
  info "skipped (--skip-setup)"
fi
if (( ! DRY )); then
  [[ -f "$ROOT/env.sh" ]] || die "$ROOT/env.sh missing: stage 1 did not finish"
  [[ -f "$ROOT/sw_lmmd/reference_store/qwen_context.npy" ]] || [[ -e "$ROOT/sw_lmmd/reference_store/qwen_context" ]] \
    || die "the reference store has no Qwen3 context: stage 1 did not finish"
  # shellcheck disable=SC1091
  source "$ROOT/env.sh"
fi

# ---------------------------------------------------------------------------- 2 resume state
say "2/3 resume state from $HF_REPO -> $RUN_DIR"
if [[ -f "$RUN_DIR/resume.pth" ]]; then
  info "resume.pth present: keeping it (never overwritten, so a stopped continuation goes on from where it stopped)"
elif (( DRY )); then
  info "would download resume.pth (23.5 GB) and train_log.jsonl"
else
  mkdir -p "$RUN_DIR"
  crun python - "$HF_REPO" "$RUN_DIR" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
repo, out = sys.argv[1:]
print("   ", hf_hub_download(repo, "resume.pth", local_dir=out), flush=True)
if not os.path.exists(os.path.join(out, "train_log.jsonl")):   # the first 2000 steps, so the log stays whole
    print("   ", hf_hub_download(repo, "train_log.jsonl", local_dir=out), flush=True)
PY
  rm -rf "$RUN_DIR/.cache"
  got=$(stat -c %s "$RUN_DIR/resume.pth")
  [[ "$got" == "$RESUME_BYTES" ]] || die "$RUN_DIR/resume.pth is $got bytes, expected $RESUME_BYTES"
fi
AT=$START
if (( ! DRY )); then
  AT=$(crun python - "$RUN_DIR/resume.pth" <<'PY' | tail -1
import sys, torch
s = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
missing = {"model", "step", "schedule", "optimizer", "cache", "critic"} - set(s)
if missing:
    raise SystemExit(f"resume.pth has no {sorted(missing)}: it cannot continue the run exactly")
print(int(s["step"]))
PY
  ) || die "could not read $RUN_DIR/resume.pth"
  [[ "$AT" =~ ^[0-9]+$ ]] || die "could not read the step of $RUN_DIR/resume.pth (got '$AT')"
  info "resume.pth is at step $AT"
  (( AT >= START )) || die "$RUN_DIR/resume.pth is at step $AT, before $START: not this run's state"
  if (( AT >= TARGET )); then say "already at step $AT (target $TARGET): nothing to train"; exit 0; fi
fi

# ---------------------------------------------------------------------------- 3 train
say "3/3 training: step $AT -> $TARGET"
TRAIN_CONFIG="$CONFIG"
if [[ "$LR_MODE" == floor ]]; then
  # `extends` merges top-level keys only, so the whole block is restated; horizon 2000 = the original decay
  TRAIN_CONFIG="$ROOT/sw_lmmd/continue_k128_gan_floor_lr.yaml"
  if (( DRY )); then info "would write $TRAIN_CONFIG (lr_schedule.total_steps: $START)"; TRAIN_CONFIG="$CONFIG"
  else cat > "$TRAIN_CONFIG" <<EOF
# Written by scripts/continue_k128_gan.sh --lr floor: the original lr schedule (it ends at step $START)
extends: $REPO/$CONFIG
lr_schedule:
  name: cosine
  warmup_steps: 100
  min_lr_ratio: 0.1
  total_steps: $START
EOF
  fi
fi
T=(bash scripts/new_machine.sh --root "$ROOT" --gpus "$GPUS" --config "$TRAIN_CONFIG" --env "$ENV_NAME" --allow-low-disk
   --micro-batch "$MICRO" --steps "$TARGET" --resume)
(( DRY )) && T+=(--dry-run)
"${T[@]}"

(( DRY )) && { say "dry run: nothing was changed"; exit 0; }
[[ -f "$RUN_DIR/step_$(printf %07d "$TARGET").pth" ]] || die "training ended without step_$(printf %07d "$TARGET").pth; see above"
say "done: step $TARGET -> $RUN_DIR"
ls -l "$RUN_DIR" | grep -E 'step_|resume' | tail -12
