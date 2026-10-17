#!/usr/bin/env bash
# Smoke test: the unmodified pipeline and the three modes, a few steps each, same machine settings.
#
#   bash experiments/ref_ablation/smoke.sh [--gpus N] [--steps N] [--runs "baseline local random global"]
#                                          [--out DIR] [--set KEY=VALUE ...]
#
#   # 2 x H100, the configs as they are:
#   bash experiments/ref_ablation/smoke.sh --gpus 2
#   # 4 x RTX 4090 (24 GB): the training state must be sharded
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/ref_ablation/smoke.sh --gpus 4 \
#       --set memory.shard=fsdp --set memory.grad_reduce=none --set memory.compute_dtype=bf16 \
#       --set memory.micro_batch=1 --set memory.battery_bf16=true
#
# baseline = python -m rdm.sw_lmmd.launch on ref_local.yaml: the existing trainer, same config.
# local / random / global = python -m experiments.ref_ablation.train. Every run gets the same
# --set overrides, STEPS steps (default 5), no checkpoint kept (the baseline's final one is
# deleted), and a 0.5 s nvidia-smi memory trace of its GPUs. The summary gives s/step (steps 2..N,
# step 1 warms up) and the peak memory per GPU above what was in use before the run; the three
# modes also log torch's own peak (mem_peak_gb) and print their per-encoder diagnostics.
set -euo pipefail
export PYTHONNOUSERSITE=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
GPUS="${GPUS:-2}"
STEPS=5
RUNS="baseline local random global"
OUT="/data/thor/jiacheng/rdm-sets/sw_lmmd/ref_ablation/smoke_$(date +%Y%m%d_%H%M%S)"
SETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)  GPUS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --runs)  RUNS="$2"; shift 2 ;;
    --out)   OUT="$2"; shift 2 ;;
    --set)   SETS+=("--set" "$2"); shift 2 ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option '$1'" >&2; exit 2 ;;
  esac
done
CONDA_ENV="${CONDA_ENV:-rdm}"
ASSETS="${ASSETS:-/data/thor/jiacheng/rdm-sets}"
[[ -f "$ASSETS/env.sh" ]] && source "$ASSETS/env.sh"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then GPU_LIST="$CUDA_VISIBLE_DEVICES"; else GPU_LIST="$(seq -s, 0 $((GPUS - 1)))"; fi
mkdir -p "$OUT"
COMMON=(--set "steps=$STEPS" --set "save_freq=1000000" --set "save_resume=false"
        --set "output_dir=$OUT" ${SETS[@]+"${SETS[@]}"})
printf '\n\033[1m==> smoke: %s | %s steps | GPUs %s | out %s\033[0m\n' "$RUNS" "$STEPS" "$GPU_LIST" "$OUT"
cd "$REPO"

for run in $RUNS; do
  port=$(( 29600 + RANDOM % 1000 ))
  case "$run" in
    baseline) cmd=(-m rdm.sw_lmmd.launch "$HERE/configs/ref_local.yaml" "${COMMON[@]}"
                   --set "exp_name=smoke-baseline") ;;
    local|random|global) cmd=(-m experiments.ref_ablation.train "$HERE/configs/ref_${run}.yaml"
                              "${COMMON[@]}" --set "exp_name=smoke" --set "save_final=false") ;;
    *) echo "unknown run '$run'" >&2; exit 2 ;;
  esac
  printf '\n\033[1m--- %s ---\033[0m\n' "$run"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPU_LIST" \
    > "$OUT/mem_${run}.csv"                                   # the pre-run usage, one sample
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$GPU_LIST" -lms 500 \
    >> "$OUT/mem_${run}.csv" &
  sampler=$!
  set +e
  conda run -n "$CONDA_ENV" --no-capture-output torchrun --nproc_per_node="$GPUS" \
    --master_addr=127.0.0.1 --master_port="$port" "${cmd[@]}" 2>&1 | tee "$OUT/${run}.log"
  rc=${PIPESTATUS[0]}
  set -e
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  find "$OUT" -name 'step_*.pth' -delete                     # smoke checkpoints are never kept
  echo "$rc" > "$OUT/${run}.rc"
  (( rc == 0 )) || echo "!!! $run exited with $rc (log: $OUT/${run}.log)"
done

conda run -n "$CONDA_ENV" --no-capture-output python - "$OUT" $RUNS <<'PY'
import csv, glob, json, os, sys
out, runs = sys.argv[1], sys.argv[2:]
print(f"\n{'run':<10}{'rc':>4}{'steps':>7}{'s/step (2..N)':>15}{'torch peak GB':>15}   nvidia-smi peak above pre-run, MiB per GPU")
for run in runs:
    rc = open(os.path.join(out, f"{run}.rc")).read().strip() if os.path.exists(os.path.join(out, f"{run}.rc")) else "?"
    logs = sorted(glob.glob(os.path.join(out, "smoke-baseline" if run == "baseline" else f"smoke-{run}-*", "train_log.jsonl")))
    steps = []
    if logs:
        steps = [r for r in map(json.loads, open(logs[-1])) if "step" in r and "seconds" in r]
    secs = [r["seconds"] for r in steps[1:]]
    peak = max((r.get("mem_peak_gb", 0.0) for r in steps), default=0.0)
    used = {}
    with open(os.path.join(out, f"mem_{run}.csv")) as f:
        rows = [(int(a), int(b)) for a, b in csv.reader(f, skipinitialspace=True)]
    gpus = sorted({g for g, _ in rows})
    base = {g: next(v for gg, v in rows if gg == g) for g in gpus}
    for g, v in rows:
        used[g] = max(used.get(g, 0), v - base[g])
    print(f"{run:<10}{rc:>4}{len(steps):>7}{(sum(secs) / len(secs)) if secs else float('nan'):>15.1f}"
          f"{peak if peak else float('nan'):>15.2f}   " + "  ".join(f"{g}:{used[g]}" for g in gpus))
PY
