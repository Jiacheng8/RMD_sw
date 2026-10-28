#!/usr/bin/env bash
# Qwen-Image-Bench for a Z-Image model, end to end on this machine: render the 1000 prompts, judge
# them with the official judge.py + Q-Judger (Qwen3.6-27B), tabulate next to the paper's leaderboard.
#
#   bash z-image/eval_qib/run_eval.sh --setup                                   # once: env + 60 GB
#   bash z-image/eval_qib/run_eval.sh --name s2000 --ckpt <run>/step_0002000.pth
#   bash z-image/eval_qib/run_eval.sh --name turbo9 --teacher-steps 9           # the 8-step teacher
#   bash z-image/eval_qib/run_eval.sh --name turbo2 --teacher-steps 2           # teacher at 2 steps
#   bash z-image/eval_qib/run_eval.sh --report s2000 turbo9 turbo2              # table only
#
# --root DIR        everything goes under it (default /data/thor/jiacheng/rdm-sets/qib):
#                   env/ (judge env), bench/ (release code + prompts), judge/ (54.7 GB weights),
#                   runs/<name>/ (images, input.jsonl, judge outputs, scores)
# --gen-gpus IDS    GPUs for rendering, one process each (default: all visible)
# --judge-gpus IDS  GPUs for the judge (default: all visible). One copy of it (54.7 GB bf16) takes
#                   1 x 80 GB card or 3 x 24 GB; the cards hold as many copies as fit, each judges
#                   its share of the images at the same time (eval_qib/judge_parallel.py), so
#                   2 x H100 = 2 copies, twice as fast as one. A copy over several cards runs them
#                   one after another (layer split), no faster than one card.
# --gpus-per-judge N  cards per copy (default: from the card memory)
# --lang zh|en      prompt language (default zh = the leaderboard's; see eval_qib/bench.py)
# --batch N         judge batch (ms-swift max_batch_size; default 24 as the paper, lower if it OOMs)
# --limit N         the first N prompts only: a smoke test of the whole chain (use its own --name;
#                   its scores cover N prompts and are not comparable with the leaderboard)
# --skip-generate / --skip-judge   re-use a stage's output
# A stage whose output exists is skipped: images are resumed per file, judge parts per part (a part
# interrupted mid-way starts over: judge.py writes its results only at the end).
set -euo pipefail
export PYTHONNOUSERSITE=1
ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ZDIR/.." && pwd)"
ROOT="/data/thor/jiacheng/rdm-sets/qib"
NAME=""; CKPT=""; TEACHER=0; LANG_="zh"; BATCH=24; SETUP=0; REPORT=0; LIMIT=0; PER_JUDGE=""
GEN_GPUS=""; JUDGE_GPUS=""; DO_GEN=1; DO_JUDGE=1
CONDA_ENV="${CONDA_ENV:-rdm}"
ASSETS="${ASSETS:-/data/thor/jiacheng/rdm-sets}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)          ROOT="$2"; shift 2 ;;
    --name)          NAME="$2"; shift 2 ;;
    --ckpt)          CKPT="$(realpath "$2")"; shift 2 ;;
    --teacher-steps) TEACHER="$2"; shift 2 ;;
    --lang)          LANG_="$2"; shift 2 ;;
    --limit)         LIMIT="$2"; shift 2 ;;
    --gpus-per-judge) PER_JUDGE="$2"; shift 2 ;;
    --batch)         BATCH="$2"; shift 2 ;;
    --gen-gpus)      GEN_GPUS="$2"; shift 2 ;;
    --judge-gpus)    JUDGE_GPUS="$2"; shift 2 ;;
    --skip-generate) DO_GEN=0; shift ;;
    --skip-judge)    DO_JUDGE=0; shift ;;
    --setup)         SETUP=1; shift ;;
    --report)        REPORT=1; shift; break ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
ENV="$ROOT/env"; BENCH="$ROOT/bench"; JUDGE="$ROOT/judge"; RUNS="$ROOT/runs"
[[ -f "$ASSETS/env.sh" ]] && source "$ASSETS/env.sh"
export PYTHONPATH="$ZDIR:$REPO:${PYTHONPATH:-}"
rdm() { conda run -n "$CONDA_ENV" --no-capture-output "$@"; }
visible() {
  if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then echo "$CUDA_VISIBLE_DEVICES"
  else nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | paste -sd, -; fi
}

# ---------------------------------------------------------------------------
if [[ $REPORT -eq 1 ]]; then
  [[ $# -gt 0 ]] || die "--report <run name> [<run name> ...]"
  dirs=(); for n in "$@"; do dirs+=("$RUNS/$n"); done
  rdm python -m eval_qib.report "${dirs[@]}" --out "$RUNS/report.md"
  exit 0
fi

# ---------------------------------------------------------------------------
if [[ $SETUP -eq 1 ]]; then
  say "setup -> $ROOT"
  mkdir -p "$ROOT"
  # download caches under --root too, never ~/.cache on the (full) system disk
  export PIP_CACHE_DIR="$ROOT/.cache/pip" CONDA_PKGS_DIRS="$ROOT/.cache/conda-pkgs"
  if [[ ! -f "$ENV/.ready" ]]; then              # absent after an interrupted install: redo it
    info "judge env $ENV (python 3.11, torch 2.8.0+cu126, ms-swift 4.6.0, transformers 5.17.0)"
    # the env comes from the defaults channels, whose Terms of Service must be accepted first
    # (older conda has no `tos` subcommand and needs none)
    for ch in https://repo.anaconda.com/pkgs/main https://repo.anaconda.com/pkgs/r; do
      conda tos accept --override-channels --channel "$ch" >/dev/null 2>&1 || true
    done
    [[ -x "$ENV/bin/python" ]] || conda create -y -p "$ENV" python=3.11 >/dev/null
    PIP=(env -u PYTHONPATH "$ENV/bin/pip" install -q)       # this repo is not the judge env's
    "${PIP[@]}" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
    # transformers pinned to the rdm env's (Qwen3.5 architecture needs >= 5.2; ms-swift 4.6 < 5.20)
    "${PIP[@]}" ms-swift==4.6.0 transformers==5.17.0 "qwen-vl-utils>=0.0.14" \
      "pandas>=2.0" "openpyxl>=3.1" "Pillow>=10" tqdm "huggingface_hub>=0.23" accelerate
    # Qwen3.5's linear-attention layers: without these Triton kernels transformers falls back to
    # a slow PyTorch path (same results, much slower prefill, and OOMs at batch 24 on 80 GB);
    # fla 0.6 needs a newer triton than torch 2.8 ships (3.4), so it fails to import -> pin 0.5.2
    "${PIP[@]}" "flash-linear-attention==0.5.2" "fla-core==0.5.2" || info "flash-linear-attention did not install: the judge runs on the slow path"
    touch "$ENV/.ready"
  fi
  # envs built before the pin above may still carry fla 0.6, which fails to import: repair them
  if ! env -u PYTHONPATH "$ENV/bin/python" -c "import fla.ops.gated_delta_rule" >/dev/null 2>&1; then
    info "flash-linear-attention missing or broken in $ENV: installing 0.5.2"
    env -u PYTHONPATH "$ENV/bin/pip" install -q "flash-linear-attention==0.5.2" "fla-core==0.5.2" \
      || info "flash-linear-attention did not install: the judge runs on the slow path"
  fi
  env -u PYTHONPATH "$ENV/bin/python" -c "import swift, transformers, torch; print('    ok  ms-swift', swift.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__)"
  rdm python - "$BENCH" "$JUDGE" <<'PY'
import sys
from huggingface_hub import snapshot_download
from eval_qib.bench import BENCH_REPO, JUDGE_REVISION, fetch_bench
bench, judge = sys.argv[1:]
print("    bench ->", fetch_bench(bench))
print("    judge ->", snapshot_download(BENCH_REPO, revision=JUDGE_REVISION, local_dir=judge))
PY
  # ms-swift 4.6 finds several model and template types for Qwen3_5ForConditionalGeneration and refuses to
  # guess; pin both in the bench backend (re-applied because the fetch above restores the file)
  sed -i 's/TransformersEngine(model_path, max_batch_size=max_batch_size)$/TransformersEngine(model_path, max_batch_size=max_batch_size, model_type="qwen3_5", template_type="qwen3_5")/' \
    "$BENCH/backends/ms_swift_backend.py"
  grep -q 'template_type="qwen3_5"' "$BENCH/backends/ms_swift_backend.py" \
    || die "could not pin model_type/template_type in $BENCH/backends/ms_swift_backend.py (upstream changed it?)"
  info "done"
  [[ -n "$NAME" ]] || exit 0
fi

# ---------------------------------------------------------------------------
[[ -n "$NAME" ]] || die "--name <run name> (and --ckpt or --teacher-steps); see --help"
[[ -f "$BENCH/judge.py" && -f "$JUDGE/config.json" ]] || die "run --setup first"
RUN="$RUNS/$NAME"
mkdir -p "$RUN/logs"
say "Qwen-Image-Bench: $NAME -> $RUN"

if [[ $DO_GEN -eq 1 && ! -f "$RUN/input.jsonl" ]]; then
  [[ -n "$CKPT" || "$TEACHER" -gt 0 ]] || die "--ckpt <step_*.pth> or --teacher-steps N"
  GEN=(--bench "$BENCH" --out "$RUN" --lang "$LANG_" --limit "$LIMIT")
  if [[ -n "$CKPT" ]]; then GEN+=(--ckpt "$CKPT"); else GEN+=(--teacher-steps "$TEACHER"); fi
  IFS=, read -r -a GG <<< "${GEN_GPUS:-$(visible)}"
  say "render 1000 prompts ($LANG_) on GPUs ${GG[*]}"
  pids=()
  for r in "${!GG[@]}"; do
    CUDA_VISIBLE_DEVICES="${GG[$r]}" rdm python -m eval_qib.generate "${GEN[@]}" \
      --rank "$r" --world "${#GG[@]}" > "$RUN/logs/generate_$r.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { info "rank $i failed -- $RUN/logs/generate_$i.log"; fail=1; }
  done
  [[ $fail -eq 0 ]] || die "rendering incomplete; re-run the same command to resume"
  tail -n 1 "$RUN"/logs/generate_*.log | grep "s/image" | sed 's/^/    /' || true
  rdm python -m eval_qib.generate --bench "$BENCH" --out "$RUN" --lang "$LANG_" --limit "$LIMIT" --finalize
fi

if [[ $DO_JUDGE -eq 1 && ! -f "$RUN/input_bench_scores.json" ]]; then
  [[ -f "$RUN/input.jsonl" ]] || die "$RUN/input.jsonl missing: render first"
  IFS=, read -r -a JG <<< "${JUDGE_GPUS:-$(visible)}"
  # one model copy needs ~55 GB of weights + ~10 GB to work in: 1 x 80 GB card, 3 x 24 GB
  if [[ -z "$PER_JUDGE" ]]; then
    MIN_MIB=$(nvidia-smi -i "$(IFS=,; echo "${JG[*]}")" --query-gpu=memory.total \
              --format=csv,noheader,nounits | sort -n | head -1)
    PER_JUDGE=$(( (66000 + MIN_MIB - 1) / MIN_MIB ))
  fi
  COPIES=$(( ${#JG[@]} / PER_JUDGE ))
  (( COPIES >= 1 )) || die "${#JG[@]} GPU(s) cannot hold the judge ($PER_JUDGE needed per copy)"
  CALLS=$(python3 -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip()))" "$RUN/input.jsonl")
  say "judge $CALLS images with Q-Judger: $COPIES cop$( (( COPIES > 1 )) && echo ies || echo y) x $PER_JUDGE GPU(s), batch $BATCH"
  # the official judge.py, unchanged, once per copy on its share of the images
  judge_part() {   # judge_part <input jsonl> <gpu list> <log>
    ( cd "$BENCH" && CUDA_VISIBLE_DEVICES="$2" env -u PYTHONPATH "$ENV/bin/python" judge.py \
        --input "$1" --model "$JUDGE" --local-metadata "$BENCH/metadata/bench_metadata.json" \
        --max-batch-size "$BATCH" ) > "$3" 2>&1
  }
  if (( COPIES == 1 )); then
    judge_part "$RUN/input.jsonl" "$(IFS=,; echo "${JG[*]:0:$PER_JUDGE}")" "$RUN/logs/judge.log" || true
  else
    env -u PYTHONPATH "$ENV/bin/python" "$ZDIR/eval_qib/judge_parallel.py" split --input "$RUN/input.jsonl" \
      --parts "$COPIES" --metadata "$BENCH/metadata/bench_metadata.json" | sed 's/^/    /'
    pids=()
    for ((k=0; k<COPIES; k++)); do
      [[ -f "$RUN/shards/part_${k}_judged.jsonl" ]] && { info "part $k: judged already"; continue; }
      GPUS_K="$(IFS=,; echo "${JG[*]:$(( k * PER_JUDGE )):$PER_JUDGE}")"
      info "part $k -> GPUs $GPUS_K (log: $RUN/logs/judge_$k.log)"
      judge_part "$RUN/shards/part_$k.jsonl" "$GPUS_K" "$RUN/logs/judge_$k.log" &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    for ((k=0; k<COPIES; k++)); do
      [[ -f "$RUN/shards/part_${k}_judged.jsonl" ]] || die "part $k did not finish; see $RUN/logs/judge_$k.log
  (re-run the same command: finished parts are kept)"
    done
    ( cd "$BENCH" && env -u PYTHONPATH "$ENV/bin/python" "$ZDIR/eval_qib/judge_parallel.py" merge \
        --run "$RUN" --parts "$COPIES" --bench "$BENCH" ) > "$RUN/logs/judge_merge.log" 2>&1 \
      || die "merge failed; see $RUN/logs/judge_merge.log"
  fi
  grep -h "Parse failures\|Skipped\|\[merge\]" "$RUN"/logs/judge*.log 2>/dev/null | sed 's/^/    /' || true
  [[ -f "$RUN/input_bench_scores.json" ]] || die "the judge did not finish; see $RUN/logs/judge*.log"
fi

say "scores"
rdm python -m eval_qib.report "$RUN" --out "$RUN/report.md" | sed -n '1,20p'
