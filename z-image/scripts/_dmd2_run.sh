#!/usr/bin/env bash
# Shared body of dmd2_four-cards.sh / dmd2_eight-cards.sh: the DMD2 baseline (z-image/zimage_distill/
# dmd2.py) in one command, at the same GPU time as ours (ours_*-cards.sh). Run one of the launchers:
#
#   bash z-image/scripts/dmd2_eight-cards.sh --dry-run          # checks + plan, changes nothing
#   bash z-image/scripts/dmd2_eight-cards.sh                    # prepare, train, evaluate
#   bash z-image/scripts/dmd2_eight-cards.sh --resume           # continue after an interruption
#   bash z-image/scripts/dmd2_eight-cards.sh --eval-only        # Qwen-Image-Bench only
#
# Stages (each skipped when already done):
#   1 prepare     _ours_run.sh --no-train --no-eval: env, Z-Image-Turbo + encoders, the reference
#                 pack, the Qwen3 contexts -- nothing to redo on the --root ours ran on
#   2 base model  Tongyi-MAI/Z-Image (the undistilled base, its transformer: ~25 GB): the real score
#                 and the fake score's initialization
#   3 real data   zimage_distill.dmd2_latents on every GPU: --real-shards of the 87 tar shards of
#                 jiachengcui888/zimage-turbo-reference (the Turbo renders ours trains with), VAE-encoded
#                 to latents (24 -> ~34k images, ~18 GB; each tar deleted once encoded) + the empty-
#                 prompt context for the CFG negative; with --latents-hf REPO, downloaded instead (~17 GB,
#                 made once by prepare_dmd2_latents.sh)
#   4 judge env   eval_qib/run_eval.sh --setup (skipped with --no-eval)
#   5 train       torchrun -m zimage_distill.dmd2: the official SDXL recipe (batch 128, ratio 5,
#                 lr 5e-7, GAN 5e-3 / 1e-2, clip 10), real-score guidance 4; the iteration count spends
#                 ours' GPU time (--match-run, default ours' run under --root) or --match-gpu-hours
#   6 eval        Qwen-Image-Bench on the last checkpoint, tabulated with whatever ours / teacher /
#                 RDM runs are already judged -> <run dir>/qwen_image_bench.md
#
# --root DIR          data root (default /root/rdm-sets; the same as ours to reuse everything)
# --gpus N            training GPUs (set by the launcher: 4 or 8 x 80 GB; three 6 B models)
# --match-run DIR     ours' run dir (default: the one under --root)
# --match-gpu-hours H the budget in GPU-hours instead;  --iterations N  a fixed count instead
# --real-shards K     tar shards of real renders to encode (default 24)
# --guidance G        real-score CFG scale (default 4)
# --batch N           prompts per iteration (default 128, DMD2's)
# --latents-dir DIR   where the real latents go (default <root>/zimage/dmd2_latents)
# --latents-hf REPO   download them from this HF dataset (prepare_dmd2_latents.sh) instead of encoding
# --output-dir DIR    where the run dir goes (default <root>/zimage/work_dirs)
# --checkpoints N     evenly spaced step checkpoints (default 4)
# --eval-limit N      judge the first N prompts only (a smoke test; named *_limitN)
# --resume --eval-only --no-eval --dry-run --micro-batch N --env NAME (rdm)
set -euo pipefail
export PYTHONNOUSERSITE=1
ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ZDIR/.." && pwd)"
ROOT="/root/rdm-sets"; GPUS=""; MATCH_RUN=""; MATCH_H=""; ITERS=""; SHARDS=24; GUIDANCE=4
RESUME=0; EVAL_ONLY=0; DO_EVAL=1; DRY=0; MICRO=""; ENV_NAME="rdm"
BATCH=128; LAT=""; LAT_HF=""; OUT=""; CKPTS=4; EVAL_LIMIT=0
CONFIG="$ZDIR/configs/zimage_2step_1024_h100.yaml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)            ROOT="$2"; shift 2 ;;
    --gpus)            GPUS="$2"; shift 2 ;;
    --match-run)       MATCH_RUN="$2"; shift 2 ;;
    --match-gpu-hours) MATCH_H="$2"; shift 2 ;;
    --iterations)      ITERS="$2"; shift 2 ;;
    --real-shards)     SHARDS="$2"; shift 2 ;;
    --guidance)        GUIDANCE="$2"; shift 2 ;;
    --resume)          RESUME=1; shift ;;
    --eval-only)       EVAL_ONLY=1; shift ;;
    --no-eval)         DO_EVAL=0; shift ;;
    --dry-run)         DRY=1; shift ;;
    --micro-batch)     MICRO="$2"; shift 2 ;;
    --env)             ENV_NAME="$2"; shift 2 ;;
    --batch)           BATCH="$2"; shift 2 ;;
    --latents-dir)     LAT="$2"; shift 2 ;;
    --latents-hf)      LAT_HF="$2"; shift 2 ;;
    --output-dir)      OUT="$2"; shift 2 ;;
    --checkpoints)     CKPTS="$2"; shift 2 ;;
    --eval-limit)      EVAL_LIMIT="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
[[ -n "$GPUS" ]] || die "--gpus N (use the dmd2_*-cards.sh launchers)"
(( GPUS >= 4 )) || die "--gpus $GPUS: DMD2 needs 4 or 8 GPUs (three 6 B models)"
(( BATCH % GPUS == 0 )) || die "--batch $BATCH does not split over $GPUS GPUs"
ROOT="$(realpath -m "$ROOT")"
ZROOT="$ROOT/zimage"; PACK="$ZROOT/zimage-ref-v1"; STORE="$PACK/reference_store"
OURS_OUT="$ZROOT/work_dirs"; OUT="$(realpath -m "${OUT:-$OURS_OUT}")"; QIB="$ROOT/qib"
LAT="$(realpath -m "${LAT:-$ZROOT/dmd2_latents}")"
EXP=$(python3 -c "import re,sys; print(re.search(r'^exp_name:\s*(\S+)', open(sys.argv[1]).read(), re.M).group(1))" "$CONFIG")
RUN="$OUT/$EXP-dmd2"
crun() { ( [[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"; export PYTHONPATH="$ZDIR:$REPO:${PYTHONPATH:-}"
           cd "$REPO" && conda run -n "$ENV_NAME" --no-capture-output "$@" ); }
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' '); fi

MATCH=()
if [[ -n "$ITERS" ]]; then MATCH=(--iterations "$ITERS")
elif [[ -n "$MATCH_H" ]]; then MATCH=(--match-gpu-hours "$MATCH_H")
else MATCH_RUN="${MATCH_RUN:-$OURS_OUT/$EXP}"; MATCH=(--match-run "$MATCH_RUN"); fi
say "DMD2 baseline on $GPUS GPU(s) -> $RUN"
info "budget      ${MATCH[*]}"
if [[ $EVAL_ONLY -eq 0 && $RESUME -eq 0 && ${MATCH[0]} == --match-run && $DRY -eq 0 ]]; then
  [[ -s "$MATCH_RUN/train_log.jsonl" ]] || die "no ours run at $MATCH_RUN to match the GPU time of:
  run ours first (ours_*-cards.sh, same --root), or pass --match-gpu-hours H / --iterations N"
fi

# 1 ------------------------------------------------------------------------------------------------
if [[ $EVAL_ONLY -eq 0 && $RESUME -eq 0 ]]; then
  say "1/6 prepare (ours' stages: env, downloads, reference pack, contexts)"
  PREP=(bash "$ZDIR/scripts/_ours_run.sh" --root "$ROOT" --gpus "$GPUS" --env "$ENV_NAME" --no-train --no-eval
        --ref-hf jiachengcui888/zimage-ref-v1)
  [[ $DRY -eq 1 ]] && PREP+=(--dry-run)
  "${PREP[@]}" | grep -v "^\s*$" | sed '/dry run: nothing was changed/d' || die "preparation failed (see above)"

# 2 ------------------------------------------------------------------------------------------------
  say "2/6 the undistilled base Tongyi-MAI/Z-Image (real + fake score)"
  if [[ $DRY -eq 1 ]]; then info "would download its transformer into the HF cache of $ROOT/env.sh"
  else
    crun python -c "
from huggingface_hub import snapshot_download
print('    ok ', snapshot_download('Tongyi-MAI/Z-Image', allow_patterns=['model_index.json', 'transformer/*']))" \
      || die "could not download Tongyi-MAI/Z-Image"
  fi

# 3 ------------------------------------------------------------------------------------------------
  say "3/6 real data: ${LAT_HF:-$SHARDS tar shards of Turbo renders} -> latents in $LAT"
  if [[ -f "$LAT/latents/manifest.json" ]]; then info "present"
  elif [[ $DRY -eq 1 ]]; then
    [[ -n "$LAT_HF" ]] && info "would download $LAT_HF" || info "would encode on GPUs ${GPU_IDS[*]}, then finalize"
  elif [[ -n "$LAT_HF" ]]; then
    crun hf download "$LAT_HF" --repo-type dataset --local-dir "$LAT" >/dev/null \
      || die "could not download $LAT_HF (re-run to resume)"
    crun python -c "
import sys
from zimage_distill.dmd2_latents import Latents
from zimage_distill.store import read_prompts
l, n = Latents('$LAT/latents'), len(read_prompts('$PACK/prompts.jsonl'))
if l.index.shape[0] != n: sys.exit(f'$LAT_HF has latents for {l.index.shape[0]} prompts, the pack {n}')
x = l.get(l.available[:2]).float()
assert x.isfinite().all() and x.std() > 0.1, 'latents look wrong'
print(f'    {l.available.size} latents from $LAT_HF')" || die "the latents of $LAT_HF do not fit this pack"
    [[ -f "$LAT/negative_context.pt" ]] || die "$LAT_HF has no negative_context.pt"
  else
    mkdir -p "$LAT/logs"; pids=()
    for r in "${!GPU_IDS[@]}"; do
      CUDA_VISIBLE_DEVICES="${GPU_IDS[$r]}" crun python -m zimage_distill.dmd2_latents encode --out "$LAT" \
        --shards "$SHARDS" --rank "$r" --world "${#GPU_IDS[@]}" --pack "$PACK" > "$LAT/logs/encode_$r.log" 2>&1 &
      pids+=($!)
    done
    fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || { info "rank $i failed: $LAT/logs/encode_$i.log"; fail=1; }; done
    [[ $fail -eq 0 ]] || die "encoding incomplete; re-run (finished ranks are kept)"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" crun python -m zimage_distill.dmd2_latents finalize --out "$LAT" \
      --world "${#GPU_IDS[@]}" || die "could not finalize the latents"
  fi
fi

# 4 ------------------------------------------------------------------------------------------------
EVAL=(env ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" bash "$ZDIR/eval_qib/run_eval.sh" --root "$QIB")
if [[ $DO_EVAL -eq 1 ]]; then
  say "4/6 judge env + Q-Judger -> $QIB"
  if [[ $DRY -eq 1 ]]; then info "would run: ${EVAL[*]} --setup"
  else "${EVAL[@]}" --setup || die "the judge setup failed (see above); --no-eval trains without it"; fi
fi

# 5 ------------------------------------------------------------------------------------------------
if [[ $EVAL_ONLY -eq 0 ]]; then
  say "5/6 training"
  TRAIN=(torchrun --nproc_per_node="$GPUS" --master_port="${MASTER_PORT:-29503}" -m zimage_distill.dmd2 "$CONFIG"
         --latents "$LAT" --guidance "$GUIDANCE" --batch "$BATCH" --checkpoints "$CKPTS" "${MATCH[@]}"
         --set "reference_root=$STORE" --set "output_dir=$OUT")
  [[ -n "$MICRO" ]] && TRAIN+=(--micro-batch "$MICRO")
  if [[ $RESUME -eq 1 ]]; then
    [[ -f "$RUN/resume.pth" ]] || die "--resume: $RUN/resume.pth does not exist"
    TRAIN+=(--resume "$RUN/resume.pth")
  elif [[ -s "$RUN/train_log.jsonl" && $DRY -eq 0 ]]; then
    die "$RUN already holds a run. Continue it with --resume, or move it away (mv $RUN{,_old})"
  fi
  if [[ $DRY -eq 1 ]]; then info "would run: ${TRAIN[*]}"
  else
    [[ -f "$LAT/latents/manifest.json" ]] || die "$LAT is missing: stage 3 has not completed"
    crun "${TRAIN[@]}" || die "training stopped; continue with --resume"
  fi
fi

# 6 ------------------------------------------------------------------------------------------------
if [[ $DO_EVAL -eq 1 ]]; then
  say "6/6 Qwen-Image-Bench"
  if [[ $DRY -eq 1 ]]; then info "would judge the last step_*.pth of $RUN, table -> $RUN/qwen_image_bench.md"
  else
    LAST=$(find "$RUN" -maxdepth 1 -name 'step_*.pth' | sort | tail -1)
    [[ -n "$LAST" ]] || die "no step_*.pth in $RUN to evaluate"
    it=$(basename "$LAST" .pth); it=$((10#${it#step_}))
    NAME="dmd2_it$it"; LIM=()
    (( EVAL_LIMIT )) && { NAME+="_limit$EVAL_LIMIT"; LIM=(--limit "$EVAL_LIMIT"); }
    "${EVAL[@]}" --name "$NAME" --ckpt "$LAST" "${LIM[@]}" || die "Qwen-Image-Bench failed on $LAST (re-run with --eval-only)"
    NAMES=("$NAME")
    for d in "$QIB"/runs/*/; do
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
