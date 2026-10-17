#!/usr/bin/env bash
# Shared body of the reference-ablation launchers (experiments/ref_ablation/run_ablation-exp*.sh), as
# scripts/_run_experiment.sh is for scripts/run_machine-new*.sh. Not meant to be called directly:
#
#   bash experiments/ref_ablation/_run_ablation.sh <launcher path> <ref_mode> <title> [launcher args...]
#
# Stages, all re-runnable:
#   1 setup      scripts/new_machine.sh --no-train: conda env, downloads (models + the COCO store),
#                the 61 GB Qwen3 context, the GenEval scorer env
#   2 landmarks  the global mode's Nystrom bundles from the private HF dataset
#                jiachengcui888/sw-rdm-reference-store (ref_ablation/nystrom_m4096, 0.5 GB, checksummed);
#                rebuilt from the store on GPU 0 (~1 min) only if that download fails
#   3 check      a 10-step real-model run in its own directory (memory, s/step, the diagnostics)
#   4 train      experiments/ref_ablation/run.sh <mode> <seed>: 2000 steps, ~2.4 h on 2 x H100
#   5 eval       scripts/eval_run.sh: GenEval + PickScore + seed diversity on all 10 checkpoints
#
# --root DIR        data root (default /root/rdm-sets); ~315 GB free, as new_machine.sh
# --gpus N          training GPUs (default: every visible card, rounded down to 1/2/4/8 so B=32
#                   splits into micro-batches of 4). Choose cards with CUDA_VISIBLE_DEVICES.
# --seed N          the run's seed (default 0)
# --check-steps N   length of the stage-3 check (default 10; 0 skips it)
# --resume          continue the stopped run from its resume.pth (skips stages 1-3)
# --no-eval         train only
# --allow-low-disk  train although the checkpoints will not all fit (you prune them)
# --dry-run         print the plan, change nothing
# Passed on to new_machine.sh (stage 1): --skip-env --skip-download --skip-preprocess --skip-geneval
#                   --full-download --allow-hf-drift --hf-cache DIR
set -euo pipefail
export PYTHONNOUSERSITE=1
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "bash >= 4.4 required (this is $BASH_VERSION)" >&2; exit 1
fi
[[ $# -ge 3 ]] || { echo "usage: bash experiments/ref_ablation/_run_ablation.sh <launcher> <mode> <title> [args...]" >&2; exit 2; }
LAUNCHER="$1"; MODE="$2"; TITLE="$3"; shift 3
case "$MODE" in local|random|global) ;; *) echo "unknown ref_mode '$MODE'" >&2; exit 2 ;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
B=32; MICRO=4; STEPS=2000; SAVE_FREQ=200
CONFIG_REL="experiments/ref_ablation/configs/ref_${MODE}.yaml"
RUN_PREFIX="sw-lmmd-refabl-k128b32-gan-grouped"        # exp_name in configs/ref_ablation_base.yaml
HF_DATASET="jiachengcui888/sw-rdm-reference-store"
ENV_NAME="${CONDA_ENV:-rdm}"

ROOT="/root/rdm-sets"; GPUS=""; SEED=0; CHECK_STEPS=10
RESUME=0; EVAL=1; LOW_DISK=0; DRY=0; SETUP=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)           ROOT="$2"; shift 2 ;;
    --gpus)           GPUS="$2"; shift 2 ;;
    --seed)           SEED="$2"; shift 2 ;;
    --check-steps)    CHECK_STEPS="$2"; shift 2 ;;
    --resume)         RESUME=1; shift ;;
    --no-eval)        EVAL=0; shift ;;
    --allow-low-disk) LOW_DISK=1; shift ;;
    --dry-run)        DRY=1; shift ;;
    --skip-env|--skip-download|--skip-preprocess|--skip-geneval|--full-download|--allow-hf-drift)
                      SETUP+=("$1"); shift ;;
    --hf-cache)       SETUP+=("$1" "$2"); shift 2 ;;
    -h|--help)        sed -n '2,/^set -euo pipefail/p' "$LAUNCHER" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                echo "unknown option '$1' (see --help)" >&2; exit 2 ;;
  esac
done
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "--seed must be a non-negative integer" >&2; exit 2; }
ROOT="$(realpath -m "$ROOT")"
SW="$ROOT/sw_lmmd"
STORE="$SW/reference_store"
NYS="$SW/ref_ablation/nystrom_m4096"
OUT="$SW/work_dirs"
RUN_DIR="$OUT/$RUN_PREFIX-$MODE-s$SEED"
CHECK_DIR="$SW/ref_ablation/check_${MODE}_s$SEED"
# This machine's HF token (needed for the private dataset), as new_machine.sh reads it.
if [[ -z "${HF_TOKEN:-}" && -s "$REPO/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$REPO/.hf_token")"; export HF_TOKEN
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m    WARNING: %s\033[0m\n' "$*" >&2; }
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
RUN_ENV=(ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" REFERENCE_ROOT="$STORE" NYSTROM_DIR="$NYS")

say "$TITLE"
info "mode      $MODE (configs: $CONFIG_REL), seed $SEED"
info "root      $ROOT"
info "gpus      $GPUS for training (grad_accum $(( B / (GPUS * MICRO) )))$( (( EVAL )) && echo ', all visible ones for evaluation' || echo ', no evaluation (--no-eval)')"
info "run dir   $RUN_DIR"
MIN_MIB=$( (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true) | sort -n | head -1)
if [[ -n "$MIN_MIB" && "$MIN_MIB" -lt 79000 ]]; then
  warn "the smallest GPU has ${MIN_MIB} MiB: these configs are sized for 80 GB cards (shard none, fp32 battery)"
fi

# ---- 1 setup -----------------------------------------------------------------------------------
if (( RESUME )); then
  say "1 setup: skipped (--resume)"
else
  say "1 setup: scripts/new_machine.sh --no-train"
  NM=(bash "$REPO/scripts/new_machine.sh" --root "$ROOT" --config "$CONFIG_REL" --gpus "$GPUS"
      --env "$ENV_NAME" --no-train ${SETUP[@]+"${SETUP[@]}"})
  (( DRY )) && NM+=(--dry-run)
  "${NM[@]}"
fi
[[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"

# ---- 2 landmarks -------------------------------------------------------------------------------
say "2 landmark bundles -> $NYS"
landmarks_ok() { [[ -f "$NYS/SHA256SUMS" ]] && (cd "$NYS" && sha256sum --quiet -c SHA256SUMS) >/dev/null 2>&1; }
if (( DRY )); then
  info "would download $HF_DATASET:ref_ablation/nystrom_m4096 (3 x ~170 MB) and check SHA256SUMS"
elif landmarks_ok; then
  info "present, checksums OK"
else
  mkdir -p "$NYS"
  if conda run -n "$ENV_NAME" --no-capture-output python - "$HF_DATASET" "$SW" <<'PY' && landmarks_ok; then
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type="dataset", local_dir=sys.argv[2],
                  allow_patterns=["ref_ablation/nystrom_m4096/*"])
PY
    info "downloaded from $HF_DATASET, checksums OK"
  else
    warn "could not fetch the bundles from $HF_DATASET (token without access?): rebuilding them from the"
    warn "store on GPU 0 (~1 min). k-means on a GPU is not bit-reproducible, so they differ slightly"
    warn "from the published ones."
    rm -f "$NYS/SHA256SUMS"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" conda run -n "$ENV_NAME" --no-capture-output \
      python -m experiments.ref_ablation.build_landmarks "$CONFIG_REL" \
      --set "reference_root=$STORE" --out "$NYS" || die "landmark build failed"
  fi
fi

# ---- 3 check -----------------------------------------------------------------------------------
if (( RESUME )) || (( CHECK_STEPS == 0 )) || [[ -f "$RUN_DIR/step_$(printf '%07d' $STEPS).pth" ]]; then
  say "3 check: skipped"
else
  say "3 check: $CHECK_STEPS real-model steps -> $CHECK_DIR (no checkpoints kept)"
  CHECK=(bash "$HERE/run.sh" "$MODE" "$SEED" --gpus "$GPUS" --steps "$CHECK_STEPS"
         --set "save_final=false" --set "save_resume=false")
  if (( DRY )); then
    env "${RUN_ENV[@]}" OUTPUT_DIR="$CHECK_DIR" "${CHECK[@]}" --dry-run
  else
    rm -rf "$CHECK_DIR"
    env "${RUN_ENV[@]}" OUTPUT_DIR="$CHECK_DIR" "${CHECK[@]}" || die "the $CHECK_STEPS-step check failed (log above)"
    conda run -n "$ENV_NAME" --no-capture-output python - "$CHECK_DIR" <<'PY'
import glob, json, sys
log = glob.glob(f"{sys.argv[1]}/*/train_log.jsonl")[0]
steps = [r for r in map(json.loads, open(log)) if "seconds" in r]
s = sorted(r["seconds"] for r in steps[1:]) or [steps[0]["seconds"]]
print(f"    check OK: {len(steps)} steps, {s[len(s) // 2]:.1f} s/step (median after the first), "
      f"peak {max(r.get('mem_peak_gb', 0) for r in steps):.1f} GB allocated on rank 0")
PY
  fi
fi

# ---- 4 train -----------------------------------------------------------------------------------
say "4 train: $MODE, seed $SEED, $STEPS steps"
FINAL="$RUN_DIR/step_$(printf '%07d' $STEPS).pth"
TRAIN=(bash "$HERE/run.sh" "$MODE" "$SEED" --gpus "$GPUS")
if [[ -f "$FINAL" ]]; then
  info "$FINAL exists: training already finished"
elif (( DRY )); then
  env "${RUN_ENV[@]}" OUTPUT_DIR="$OUT" "${TRAIN[@]}" --dry-run
else
  TRAIN_PAT='^[^ ]*python[0-9.]* .*-m (experiments[.]ref_ablation[.]train|rdm[.]sw_lmmd[.]launch)'
  if pgrep -u "$(id -u)" -f "$TRAIN_PAT" >/dev/null 2>&1; then
    die "a training process is already running:
$(pgrep -u "$(id -u)" -af "$TRAIN_PAT" | cut -c1-160)"
  fi
  RESUME_FROM=""
  if (( RESUME )); then
    [[ -f "$RUN_DIR/resume.pth" ]] || die "--resume: $RUN_DIR/resume.pth does not exist"
    RESUME_FROM="$RUN_DIR/resume.pth"
  elif [[ -f "$RUN_DIR/resume.pth" ]]; then
    die "$RUN_DIR holds a stopped run: re-run with --resume to continue it, or move it away"
  elif [[ -s "$RUN_DIR/train_log.jsonl" ]]; then
    die "$RUN_DIR already holds a run (without resume.pth): move it away (mv $RUN_DIR{,_old})"
  fi
  N_HAVE=0
  [[ -d "$RUN_DIR" ]] && N_HAVE=$(find "$RUN_DIR" -maxdepth 1 -name 'step_*.pth' | wc -l)
  N_LEFT=$(( STEPS / SAVE_FREQ - N_HAVE )); (( N_LEFT < 0 )) && N_LEFT=0
  NEED_GB=$(( N_LEFT * 31 / 2 + 47 + 10 ))                 # 15.5 GB each; resume.pth x2 while replaced
  mkdir -p "$OUT"
  FREE_GB=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')
  info "disk      $FREE_GB GB free for $N_LEFT more checkpoints + resume.pth -> need ~$NEED_GB GB"
  if (( FREE_GB < NEED_GB )); then
    (( LOW_DISK )) || die "not enough disk under $OUT; free space or pass --allow-low-disk (then prune step_*.pth)"
    warn "not enough disk for every checkpoint (--allow-low-disk): delete old step_*.pth as it trains"
  fi
  env "${RUN_ENV[@]}" OUTPUT_DIR="$OUT" RESUME_FROM="$RESUME_FROM" "${TRAIN[@]}"
  [[ -f "$FINAL" ]] || die "training ended without $FINAL (log above)"
  info "logs + checkpoints: $RUN_DIR"
fi

# ---- 5 eval ------------------------------------------------------------------------------------
if (( EVAL )); then
  say "5 eval: GenEval + PickScore + seed diversity on every checkpoint"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then EVAL_GPUS="$CUDA_VISIBLE_DEVICES"
  else EVAL_GPUS=$( (nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true) | tr -d ' ' | paste -sd, -); fi
  PER_GPU=1; [[ "${MIN_MIB:-0}" -ge 48000 ]] && PER_GPU=2     # one evaluation peaks at ~21 GB
  EV=(bash "$REPO/scripts/eval_run.sh" "$RUN_DIR" --root "$ROOT" --env "$ENV_NAME"
      --gpus "$EVAL_GPUS" --per-gpu "$PER_GPU")
  if [[ -f "$ROOT/geneval/geneval_env.sh" ]]; then EV+=(--geneval-root "$ROOT/geneval")
  else warn "no GenEval scorer env at $ROOT/geneval (stage 1 skipped or failed): PickScore only"; EV+=(--no-geneval); fi
  if (( DRY )); then
    info "would run: env ASSETS=$ROOT CONDA_ENV=$ENV_NAME ${EV[*]}"
  elif env ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" "${EV[@]}"; then
    info "table: $RUN_DIR/eval_summary.md"
  else
    warn "some evaluations failed (see above); re-run the same command to retry only those"
  fi
fi
(( DRY )) && say "dry run: nothing was changed" || say "done: $RUN_DIR"
