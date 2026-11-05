#!/usr/bin/env bash
# DMD2 baseline on the one-step FLUX.2 klein-4B student (rdm/baselines/dmd2_flux.py), the iRDM
# paper's recipe (appendix E.2): batch 128 at 512 px, fake score every iteration / student every
# 5th, lr 5e-7 with warm-up, t = klein's SNR shift of U[0.02, 0.98], no GAN, no EMA.
#
#   bash scripts/train_dmd2_flux.sh --root /root/rdm-sets                  # every visible GPU
#   bash scripts/train_dmd2_flux.sh --root /root/rdm-sets --eval           # + GenEval / PickScore of each checkpoint
#   bash scripts/train_dmd2_flux.sh --root /root/rdm-sets --dry-run        # print the plan only
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_dmd2_flux.sh --root /root/rdm-sets
#
# Needs what SW-LMMD training needs: <root>/env.sh, klein-4B, and the reference store (its COCO
# prompts' Qwen3 contexts are the prompts here) -- new_machine.sh / download_all.sh set them up.
# -> <root>/sw_lmmd/work_dirs/<exp>/  step_0000250.pth ... step_0001000.pth (student, fp32, 15.5 GB
#    each), resume.pth (+ fake score, 31 GB), train_log.jsonl, train.log, args.json
# Run it again after an interruption: it resumes from resume.pth (the last checkpoint; the
# optimizers restart). --fresh deletes the run and starts over.
#
# Memory: student + fake score trained (fp32 masters) + the frozen teacher, FSDP over all ranks.
# Per card ~ (2 x state + 7.8 GB) / GPUs + activations; state = 62 GB with AdamW, 38.8 GB with
# 8-bit AdamW (picked automatically when AdamW does not fit). 4 x 80 GB: AdamW, ~40 GB per card.
# The paper's run: 4 x H200, best at 500 iterations, ~10 H200 GPU-hours.
#
# --root DIR         data root (default $ASSETS or /data/thor/jiacheng/rdm-sets)
# --iterations N     default 1000 (the paper's Table 8 range; its peak was 500)
# --save-every N     default 250 -> checkpoints 250 / 500 / 750 / 1000
# --batch N          global batch, default 128;  --micro-batch N  rows per forward per GPU, default 4
# --lr X             student and fake score, default 5e-7
# --optimizer K      adamw | adamw8bit (default: adamw if it fits, else adamw8bit)
# --exp-name NAME    default dmd2-flux-klein
# --budget-gpu-hours H  stop after H GPU-hours instead (the iterations are then an upper bound; the first 20
#                    are timed, and 4 checkpoints are spread over the rest)
# --eval             after training, evaluate every checkpoint (scripts/eval_checkpoint.sh: GenEval
#                    + PickScore + seed diversity), one GPU each, in parallel
# --geneval-root D   for --eval (default $GENEVAL_ROOT, else <root>/geneval)
# --fresh            delete an existing run of this name first
# --env NAME         conda env (default $CONDA_ENV or rdm)
# --dry-run          print the plan, run nothing
set -euo pipefail
export PYTHONNOUSERSITE=1   # packages in ~/.local/lib/pythonX.Y must never shadow the conda envs'

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ASSETS:-/data/thor/jiacheng/rdm-sets}"
CONDA_ENV="${CONDA_ENV:-rdm}"
ITERS=1000; SAVE_EVERY=250; BATCH=128; MICRO=4; LR=5e-7; OPT=""; EXP="dmd2-flux-klein"
EVAL=0; GROOT="${GENEVAL_ROOT:-}"; FRESH=0; DRY=0; BUDGET=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)         ROOT="$2"; shift 2 ;;
    --iterations)   ITERS="$2"; shift 2 ;;
    --save-every)   SAVE_EVERY="$2"; shift 2 ;;
    --batch)        BATCH="$2"; shift 2 ;;
    --micro-batch)  MICRO="$2"; shift 2 ;;
    --lr)           LR="$2"; shift 2 ;;
    --optimizer)    OPT="$2"; shift 2 ;;
    --exp-name)     EXP="$2"; shift 2 ;;
    --budget-gpu-hours) BUDGET=(--budget-gpu-hours "$2"); shift 2 ;;
    --eval)         EVAL=1; shift ;;
    --geneval-root) GROOT="$2"; shift 2 ;;
    --fresh)        FRESH=1; shift ;;
    --env)          CONDA_ENV="$2"; shift 2 ;;
    --dry-run)      DRY=1; shift ;;
    -h|--help)      sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

ROOT="$(realpath -m "$ROOT")"
[[ -f "$ROOT/env.sh" ]] || die "$ROOT/env.sh missing (wrong --root? new_machine.sh writes it)"
source "$ROOT/env.sh"
STORE="$ROOT/sw_lmmd/reference_store"
OUT="$ROOT/sw_lmmd/work_dirs"
RUN="$OUT/$EXP"
GROOT="${GROOT:-$ROOT/geneval}"
[[ -f "$STORE/metadata.json" ]] || die "reference store missing: $STORE (bash scripts/new_machine.sh --root $ROOT)"

# ---------------------------------------------------------------- GPUs, memory, batching
say "plan"
mapfile -t MEM < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=, read -ra VIS <<< "$CUDA_VISIBLE_DEVICES"; GPUS=${#VIS[@]}
else GPUS=${#MEM[@]}; fi
(( GPUS > 0 )) || die "no GPU visible"
CARD_GB=$(( ${MEM[0]:-0} / 1024 ))
(( BATCH % (GPUS * MICRO) == 0 )) || die "--batch $BATCH must split into $GPUS GPUs x --micro-batch $MICRO (x accumulation)"
est() {  # est <optimizer>: GB per card
  python3 -c "import sys; s = 62.0 if sys.argv[1] == 'adamw' else 38.8; \
print(round((2 * s + 7.8) / int(sys.argv[2]) + 4 + 1.5 * int(sys.argv[3])))" "$1" "$GPUS" "$MICRO"
}
if [[ -z "$OPT" ]]; then
  OPT=adamw; (( $(est adamw) > CARD_GB * 9 / 10 )) && OPT=adamw8bit
fi
[[ "$OPT" == adamw || "$OPT" == adamw8bit ]] || die "--optimizer adamw | adamw8bit"
EST=$(est "$OPT")
info "GPUs          $GPUS x ${CARD_GB} GB, FSDP | optimizer $OPT | est. ~${EST} GB per card [ESTIMATE]"
(( EST > CARD_GB )) && die "does not fit: ~${EST} GB per card on ${CARD_GB} GB cards -- more GPUs, or a smaller --micro-batch"
info "batch         $BATCH = $GPUS GPUs x $MICRO micro x $(( BATCH / GPUS / MICRO )) accumulation"
info "recipe        lr $LR (student and fake), fake every iteration / student every 5th, warm-up 20, clip 10"
info "iterations    $ITERS, checkpoints every $SAVE_EVERY -> $RUN"
info "prompts       COCO captions of $STORE"

RESUME=()
if [[ -d "$RUN" ]] && (( FRESH )); then
  info "--fresh       deleting $RUN"; (( DRY )) || rm -rf "$RUN"
elif [[ -f "$RUN/resume.pth" ]]; then
  RESUME=(--resume "$RUN/resume.pth"); info "RESUMING      $RUN/resume.pth (optimizers restart)"
elif compgen -G "$RUN/step_*.pth" >/dev/null; then
  die "$RUN has checkpoints but no resume.pth: --fresh to start over, or --exp-name another"
fi
NEED=$(( (ITERS / SAVE_EVERY + 1) * 16 + 31 ))
(( ${#BUDGET[@]} )) && NEED=$(( 5 * 16 + 31 ))
FREE=$(( $(df -Pk "$OUT" 2>/dev/null | awk 'NR==2 {print $4}' || df -Pk "$ROOT" | awk 'NR==2 {print $4}') / 1048576 ))
info "disk          ~${NEED} GB of checkpoints, ${FREE} GB free under $OUT"
(( FREE < NEED )) && info "WARNING       not enough disk for every checkpoint"
if (( EVAL )); then
  [[ -d "$GROOT" ]] || die "--eval: GenEval not set up at $GROOT (bash scripts/setup_geneval.sh --root $GROOT, or --geneval-root)"
  info "eval          every checkpoint after training (GenEval at $GROOT)"
fi
(( DRY )) && { say "dry run: nothing started"; exit 0; }

# ---------------------------------------------------------------- train
say "train"
mkdir -p "$RUN"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
conda run -n "$CONDA_ENV" --no-capture-output torchrun --nproc_per_node="$GPUS" \
  --master_port="${MASTER_PORT:-29517}" -m rdm.baselines.dmd2_flux \
  --reference-root "$STORE" --output-dir "$OUT" --exp-name "$EXP" \
  --iterations "$ITERS" --save-every "$SAVE_EVERY" --batch "$BATCH" --micro-batch "$MICRO" \
  --lr "$LR" --fake-lr "$LR" --optimizer "$OPT" "${RESUME[@]}" "${BUDGET[@]}" 2>&1 | tee -a "$RUN/train.log"
(( PIPESTATUS[0] == 0 )) || die "training failed (log $RUN/train.log); run the same command again to resume"

# ---------------------------------------------------------------- evaluate
CKPTS=("$RUN"/step_*.pth)
if (( ! EVAL )); then
  say "done: $RUN"
  info "evaluate a checkpoint (one 24 GB+ GPU each):"
  for c in "${CKPTS[@]}"; do info "  CUDA_VISIBLE_DEVICES=0 bash scripts/eval_checkpoint.sh $c --root $ROOT --geneval-root $GROOT"; done
  exit 0
fi
say "evaluate ${#CKPTS[@]} checkpoints on $GPUS GPUs"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IDS=("${VIS[@]}"); else IDS=($(seq 0 $(( GPUS - 1 )))); fi
FAIL=0; PIDS=()
for i in "${!CKPTS[@]}"; do
  c="${CKPTS[$i]}"; g="${IDS[$(( i % GPUS ))]}"
  log="$RUN/eval_$(basename "$c" .pth).log"
  info "$(basename "$c") on GPU $g -> $log"
  CUDA_VISIBLE_DEVICES="$g" bash "$REPO/scripts/eval_checkpoint.sh" "$c" --root "$ROOT" \
    --geneval-root "$GROOT" --env "$CONDA_ENV" > "$log" 2>&1 &
  PIDS+=($!)
  if (( ${#PIDS[@]} == GPUS )); then for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done; PIDS=(); fi
done
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
say "results"
for c in "${CKPTS[@]}"; do
  s="$(dirname "$c")/eval_$(basename "$c" .pth)/summary.json"
  if [[ -f "$s" ]]; then
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); g=d.get('geneval') or {}; \
print(f\"    {sys.argv[2]:22s} GenEval {g.get('overall', float('nan')):.4f}  PickScore {d.get('pickscore', float('nan')):.3f}\")" \
      "$s" "$(basename "$c" .pth)" || info "$(basename "$c"): $s unreadable"
  else
    info "$(basename "$c"): no summary (see $RUN/eval_$(basename "$c" .pth).log)"
  fi
done
(( FAIL == 0 )) || die "some evaluations failed (logs in $RUN/eval_*.log)"
