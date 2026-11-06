#!/usr/bin/env bash
# Shared body of rdm-global_two-cards.sh / _four-cards.sh / _eight-cards.sh: the original RDM
# (local-to-global) baseline, z-image/zimage_distill/rdm_global.py, in one command, at the same GPU
# time as ours (ours_*-cards.sh). Run one of the launchers:
#
#   bash z-image/scripts/rdm-global_eight-cards.sh --dry-run          # checks + plan, changes nothing
#   bash z-image/scripts/rdm-global_eight-cards.sh                    # prepare, train, evaluate
#   bash z-image/scripts/rdm-global_eight-cards.sh --resume           # continue after an interruption
#   bash z-image/scripts/rdm-global_eight-cards.sh --eval-only        # Qwen-Image-Bench only
#
# Stages (each skipped when already done):
#   1 prepare    _ours_run.sh --no-train --no-eval: env, Z-Image + encoders, the reference pack, the
#                Qwen3 contexts -- nothing to do on a machine that already ran ours with this --root
#   2 judge env  eval_qib/run_eval.sh --setup (skipped with --no-eval)
#   3 train      torchrun -m zimage_distill.rdm_global: N fresh rows per step (default 10240, iRDM's
#                rollout), each matched against the whole reference set (Nystrom, built once from the
#                store); the step count spends ours' GPU time: 100 GPU-h by default (ours' 2000
#                steps, training only), or --match-gpu-hours H, or --match-run <ours' run dir>
#   4 eval       Qwen-Image-Bench on the last checkpoint; the table also lists whatever ours / teacher
#                runs <root>/qib/runs holds -> <run dir>/qwen_image_bench.md
#
# Budget arithmetic: ours' 2000 steps are ~100 GPU-h of training; one 10240-row step costs ~16 GPU-h
# (every row is rolled out and back-propagated through both student steps; the 4 x A100 smoke test
# ran 128 rows in ~180 s), so the baseline gets ~6 steps (~2 h each on 8 GPUs). That IS the comparison: same compute, iRDM's
# batch. --rows N changes it (N must split over the GPUs).
#
# --root DIR          data root (default /root/rdm-sets; the same as ours to reuse everything)
# --gpus N            training GPUs (set by the launcher)
# --rows N            fresh rows per step (default 10240)
# --match-gpu-hours H the budget in GPU-hours (default 100 = ours' 2000 training steps)
# --match-run DIR     read the budget from ours' run dir instead (its train_log.jsonl)
# --steps N           a fixed step count instead of a budget
# --output-dir DIR    where the run dir goes (default <root>/zimage/work_dirs)
# --checkpoints N     evenly spaced step checkpoints (default 4)
# --eval-limit N      judge the first N prompts only (a smoke test; named *_limitN)
# --oom-probe         only test whether a --rows step fits in memory (minutes): every buffer of the
#                     step at full size, 2 micro-batches really rolled out, one optimizer step;
#                     prints 'oom-probe: PASS|FAIL' + peak memory per GPU, saves nothing, no eval
# --resume --eval-only --no-eval --dry-run --micro-batch N --env NAME (rdm)
# W&B is required: no wandb login (or a refused key / not in sw-lmmd) stops the launcher before
# anything runs; --no-wandb runs without it (z-image/scripts/_wandb.sh)
set -euo pipefail
export PYTHONNOUSERSITE=1
ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ZDIR/.." && pwd)"
ROOT="/root/rdm-sets"; GPUS=""; ROWS=10240; MATCH_RUN=""; MATCH_H="100"; STEPS=""; RESUME=0
EVAL_ONLY=0; DO_EVAL=1; DRY=0; MICRO=""; ENV_NAME="rdm"; REF_HF=""; PASS=(); OUT=""; CKPTS=4; EVAL_LIMIT=0; OOM_PROBE=0
CONFIG="$ZDIR/configs/zimage_2step_1024_h100.yaml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)            ROOT="$2"; shift 2 ;;
    --gpus)            GPUS="$2"; shift 2 ;;
    --rows)            ROWS="$2"; shift 2 ;;
    --match-run)       MATCH_RUN="$2"; shift 2 ;;
    --match-gpu-hours) MATCH_H="$2"; shift 2 ;;
    --steps)           STEPS="$2"; shift 2 ;;
    --resume)          RESUME=1; shift ;;
    --eval-only)       EVAL_ONLY=1; shift ;;
    --no-eval)         DO_EVAL=0; shift ;;
    --dry-run)         DRY=1; shift ;;
    --no-wandb)        export WANDB_MODE=disabled; shift ;;
    --micro-batch)     MICRO="$2"; shift 2 ;;
    --env)             ENV_NAME="$2"; shift 2 ;;
    --ref-hf)          REF_HF="$2"; shift 2 ;;
    --output-dir)      OUT="$2"; shift 2 ;;
    --checkpoints)     CKPTS="$2"; shift 2 ;;
    --eval-limit)      EVAL_LIMIT="$2"; shift 2 ;;
    --oom-probe)       OOM_PROBE=1; DO_EVAL=0; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
source "$ZDIR/scripts/_wandb.sh"
[[ -n "$GPUS" ]] || die "--gpus N (use the rdm-global_*-cards.sh launchers)"
(( ROWS % GPUS == 0 )) || die "--rows $ROWS does not split over $GPUS GPUs"
ROOT="$(realpath -m "$ROOT")"
ZROOT="$ROOT/zimage"; PACK="$ZROOT/zimage-ref-v1"; STORE="$PACK/reference_store"
OURS_OUT="$ZROOT/work_dirs"; OUT="$(realpath -m "${OUT:-$OURS_OUT}")"; QIB="$ROOT/qib"
EXP=$(python3 -c "import re,sys; print(re.search(r'^exp_name:\s*(\S+)', open(sys.argv[1]).read(), re.M).group(1))" "$CONFIG")
RUN="$OUT/$EXP-rdm-global-N$ROWS"
OURS_RUN="$OURS_OUT/$EXP"

# the budget: --steps, else ours' run when given, else --match-gpu-hours (default 100)
MATCH=()
if (( OOM_PROBE )); then MATCH=(--oom-probe)
elif [[ -n "$STEPS" ]]; then MATCH=(--steps "$STEPS")
elif [[ -n "$MATCH_RUN" ]]; then MATCH=(--match-run "$MATCH_RUN")
else MATCH=(--match-gpu-hours "$MATCH_H")
fi
(( OOM_PROBE == 0 )) && wandb_require "$DRY"
say "RDM (local-to-global) baseline: $ROWS fresh rows/step on $GPUS GPU(s) -> $RUN"
info "budget      ${MATCH[*]}"
if [[ $RESUME -eq 1 && -f "$RUN/plan.json" ]]; then
  info "plan        $(tr -d '\n ' < "$RUN/plan.json" | cut -c1-160)"
elif [[ $EVAL_ONLY -eq 0 && ${MATCH[0]} == --match-run && $DRY -eq 0 ]]; then
  [[ -s "$MATCH_RUN/train_log.jsonl" ]] || die "no ours run at $MATCH_RUN to match the GPU time of:
  run ours first (ours_*-cards.sh, same --root), or pass --match-run DIR / --match-gpu-hours H / --steps N"
  BUDGET=$(PYTHONPATH="$ZDIR:$REPO" conda run -n "$ENV_NAME" --no-capture-output python -c \
    "from zimage_distill.rdm_global import gpu_seconds_of_run as g; s, w, n = g('$MATCH_RUN'); print(f'{s/3600:.1f} GPU-h ({n} steps on {w} GPUs)')" | tail -1)
  info "ours        $BUDGET"
fi

# 1 ------------------------------------------------------------------------------------------------
if [[ $EVAL_ONLY -eq 0 && $RESUME -eq 0 ]]; then
  say "1/4 prepare (ours' stages: env, downloads, reference pack, contexts)"
  PREP=(bash "$ZDIR/scripts/_ours_run.sh" --root "$ROOT" --gpus "$GPUS" --env "$ENV_NAME" --no-train --no-eval
        --ref-hf "${REF_HF:-jiachengcui888/zimage-ref-v1}")
  [[ $DRY -eq 1 ]] && PREP+=(--dry-run)
  "${PREP[@]}" | grep -v "^\s*$" | sed '/dry run: nothing was changed/d' || die "preparation failed (see above)"
fi

# 2 ------------------------------------------------------------------------------------------------
EVAL=(env ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" bash "$ZDIR/eval_qib/run_eval.sh" --root "$QIB")
if [[ $DO_EVAL -eq 1 ]]; then
  say "2/4 judge env + Q-Judger -> $QIB"
  if [[ $DRY -eq 1 ]]; then info "would run: ${EVAL[*]} --setup"
  else "${EVAL[@]}" --setup || die "the judge setup failed (see above); --no-eval trains without it"; fi
fi

# 3 ------------------------------------------------------------------------------------------------
(( DRY == 0 && OOM_PROBE == 0 )) && wandb_start "$RUN" rdm-global "$GPUS" "$STEPS" "$CONFIG"
if [[ $EVAL_ONLY -eq 0 ]]; then
  say "3/4 training"
  TRAIN=(torchrun --nproc_per_node="$GPUS" --master_port="${MASTER_PORT:-29502}" -m zimage_distill.rdm_global
         "$CONFIG" --rows "$ROWS" "${MATCH[@]}" --checkpoints "$CKPTS"
         --set "reference_root=$STORE" --set "output_dir=$OUT")
  [[ -n "$MICRO" ]] && TRAIN+=(--micro-batch "$MICRO")
  if [[ $RESUME -eq 1 ]]; then
    [[ -f "$RUN/resume.pth" ]] || die "--resume: $RUN/resume.pth does not exist"
    TRAIN+=(--resume "$RUN/resume.pth")
  elif [[ -s "$RUN/train_log.jsonl" && $DRY -eq 0 && $OOM_PROBE -eq 0 ]]; then
    die "$RUN already holds a run. Continue it with --resume, or move it away (mv $RUN{,_old})"
  fi
  if [[ $DRY -eq 1 ]]; then
    info "would run: ${TRAIN[*]}"
  else
    [[ -f "$PACK/contexts/manifest.json" ]] || die "$PACK/contexts missing: stage 1 has not completed"
    ( source "$ROOT/env.sh"; export PYTHONPATH="$ZDIR:$REPO:${PYTHONPATH:-}"
      cd "$REPO" && conda run -n "$ENV_NAME" --no-capture-output "${TRAIN[@]}" ) \
      || die "training stopped; continue with --resume"
  fi
fi

# 4 ------------------------------------------------------------------------------------------------
if [[ $DO_EVAL -eq 1 ]]; then
  say "4/4 Qwen-Image-Bench"
  if [[ $DRY -eq 1 ]]; then
    info "would judge the last step_*.pth of $RUN, table -> $RUN/qwen_image_bench.md"
  else
    LAST=$(find "$RUN" -maxdepth 1 -name 'step_*.pth' | sort | tail -1)
    [[ -n "$LAST" ]] || die "no step_*.pth in $RUN to evaluate"
    step=$(basename "$LAST" .pth); step=$((10#${step#step_}))
    NAME="rdm-global-N${ROWS}_s$step"; LIM=()
    (( EVAL_LIMIT )) && { NAME+="_limit$EVAL_LIMIT"; LIM=(--limit "$EVAL_LIMIT"); }
    wandb_eval "$NAME" "$EVAL_LIMIT"
    "${EVAL[@]}" --name "$NAME" --ckpt "$LAST" "${LIM[@]}" || die "Qwen-Image-Bench failed on $LAST (re-run with --eval-only)"
    NAMES=("$NAME")
    for d in "$QIB"/runs/*/; do                    # ours and the teacher, when already judged
      n=$(basename "$d")
      [[ "$n" != "$NAME" && -f "$d/input_bench_scores.json" && "$n" != *_limit* ]] && NAMES+=("$n")
    done
    "${EVAL[@]}" --report "${NAMES[@]}" >/dev/null
    cp "$QIB/runs/report.md" "$RUN/qwen_image_bench.md"
    sed -n '/^| model/,/^$/p' "$RUN/qwen_image_bench.md" | sed 's/^/    /'
    info "table: $RUN/qwen_image_bench.md"
  fi
fi
[[ $DRY -eq 1 ]] && say "dry run: nothing was changed"
exit 0
