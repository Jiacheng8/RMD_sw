#!/usr/bin/env bash
# E1 cache drift, one command: environment -> downloads -> smoke -> 4 measurements -> report.
#
#   bash tools/run_cache_drift.sh --root /root/rdm-sets              # 2 x H100 80 GB
#   bash tools/run_cache_drift.sh --root /root/rdm-sets --dry-run    # print the plan, change nothing
#
# Stages (each one is skipped when its output already exists, so a re-run continues):
#   1 setup     scripts/new_machine.sh --no-train --skip-geneval: conda env rdm, encoders + FLUX.2,
#               the reference store, its Qwen3 context
#   2 ckpt      step_0000200.pth, step_0001000.pth, resume.pth, train_log.jsonl from the public HF
#               repo jiachengcui888/w128_s32_2000steps_w-gan -> <root>/e1/ckpt (55 GB)
#   3 gpu       2 visible GPUs, each with >= 75 GB free
#   4 smoke     1 measured window on step 200 -> <root>/e1/smoke (not in the report)
#   5 measure   step_0200, step_1000, step_2000, age_curve_step2000 -> outputs/cache_drift/*.json
#   6 report    summary.md / summary.csv / age_curve_step2000.png / notes.md
#
# --root DIR        data root (env.sh, models, store, checkpoints); ~180 GB free
# --out DIR         report directory (default <repo>/outputs/cache_drift)
# --env NAME        conda env (default rdm)
# --port N          torchrun master port (default 29500)
# --skip-setup      skip stage 1 (the machine is already set up under --root)
# --no-smoke        skip stage 4
# --force           redo stages 4-6 even if their outputs exist
# --dry-run         print the plan
#
# HF token (stage 1 needs it for the gated FLUX.2-dev VAE and the private reference store): put it
# in <repo>/.hf_token (git-ignored) or export HF_TOKEN before running. This script never prints it.
set -euo pipefail
export PYTHONNOUSERSITE=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=""; OUT=""; ENV_NAME="${CONDA_ENV:-rdm}"; PORT="${MASTER_PORT:-29500}"
SETUP=1; SMOKE=1; FORCE=0; DRY=0
HF_REPO="jiachengcui888/w128_s32_2000steps_w-gan"
CFG="configs/sw_lmmd_train_h100_2gpu_gan.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)       ROOT="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    --env)        ENV_NAME="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --skip-setup) SETUP=0; shift ;;
    --no-smoke)   SMOKE=0; shift ;;
    --force)      FORCE=1; shift ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
run()  { info "\$ $*"; (( DRY )) || "$@"; }

[[ -n "$ROOT" ]] || die "usage: bash tools/run_cache_drift.sh --root <data root> (see --help)"
ROOT="$(realpath -m "$ROOT")"
OUT="$(realpath -m "${OUT:-$REPO/outputs/cache_drift}")"
CK="$ROOT/e1/ckpt"
STORE="$ROOT/sw_lmmd/reference_store"
cd "$REPO"

# conda in a non-interactive shell (an agent's): put it on PATH if `conda init` did not reach us
if ! command -v conda >/dev/null 2>&1; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /root/miniconda3; do
    [[ -x "$c/bin/conda" ]] && { export PATH="$c/bin:$PATH"; break; }
  done
fi
command -v conda >/dev/null 2>&1 || die "conda not found: install Miniconda3 (README step 2)"

say "E1 cache drift | repo $REPO ($(git branch --show-current 2>/dev/null || echo '?') @ $(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
info "data root $ROOT | report $OUT | env $ENV_NAME"
[[ -f tools/measure_cache_drift.py ]] || die "tools/measure_cache_drift.py missing: check out the cache-drift branch"

# ---------------------------------------------------------------------------- 1 setup
say "1/6 setup (environment, models, reference store, Qwen3 context)"
if (( SETUP )); then
  run bash scripts/new_machine.sh --root "$ROOT" --gpus 2 --config "$CFG" \
      --no-train --skip-geneval --allow-low-disk --env "$ENV_NAME"
else
  info "skipped (--skip-setup)"
fi
if (( ! DRY )); then
  [[ -f "$ROOT/env.sh" ]] || die "$ROOT/env.sh missing: stage 1 did not finish"
  [[ -f "$STORE/bandwidths.json" ]] || die "$STORE/bandwidths.json missing: stage 1 did not finish"
  [[ -e "$STORE/qwen_context.npy" || -e "$STORE/qwen_context/manifest.json" ]] \
    || die "the store's Qwen3 context is missing ($STORE/qwen_context*): stage 1 did not finish"
  # shellcheck disable=SC1091
  source "$ROOT/env.sh"
fi
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
crun() { conda run -n "$ENV_NAME" --no-capture-output "$@"; }

# ---------------------------------------------------------------------------- 2 checkpoints
say "2/6 checkpoints from $HF_REPO -> $CK"
declare -A SIZE=([step_0000200.pth]=15502249848 [step_0001000.pth]=15502249848
                 [resume.pth]=23504415310 [train_log.jsonl]=3325990)
need=()
for f in step_0000200.pth step_0001000.pth resume.pth train_log.jsonl; do
  have=$(stat -c %s "$CK/$f" 2>/dev/null || echo 0)
  if [[ "$have" == "${SIZE[$f]}" ]]; then info "$f present"; else need+=("$f"); fi
done
if (( ${#need[@]} )); then
  info "downloading ${need[*]} (public repo; the three checkpoints are 55 GB together)"
  if (( ! DRY )); then
    mkdir -p "$CK"
    crun python - "$HF_REPO" "$CK" "${need[@]}" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
for f in files:
    print("   ", hf_hub_download(repo, f, local_dir=out), flush=True)
PY
    for f in "${need[@]}"; do
      have=$(stat -c %s "$CK/$f" 2>/dev/null || echo 0)
      [[ "$have" == "${SIZE[$f]}" ]] || die "$CK/$f is $have bytes, expected ${SIZE[$f]}"
    done
  fi
fi

# ---------------------------------------------------------------------------- 3 GPUs
say "3/6 GPUs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
if (( ! DRY )); then
  IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
  (( ${#DEVS[@]} >= 2 )) || die "need 2 GPUs, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  for d in "${DEVS[@]:0:2}"; do
    name=$(nvidia-smi -i "$d" --query-gpu=name --format=csv,noheader) || die "nvidia-smi cannot see GPU $d"
    free=$(nvidia-smi -i "$d" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')
    total=$(nvidia-smi -i "$d" --query-gpu=memory.total --format=csv,noheader,nounits | tr -dc '0-9')
    info "GPU $d: $name, $free / $total MiB free"
    (( free >= 75000 )) || die "GPU $d has only $free MiB free: this needs an idle 80 GB card (H100)"
  done
fi
TR=(crun torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port="$PORT"
    tools/measure_cache_drift.py measure "$CFG"
    --set "reference_root=$STORE" --set "output_dir=$ROOT/sw_lmmd/work_dirs")
(( DRY )) || mkdir -p "$OUT/logs" "$ROOT/e1/smoke"

measure() {   # measure <name> <ckpt> <out json> <log> [extra args]
  local name="$1" ckpt="$2" json="$3" log="$4"; shift 4
  if [[ -s "$json" && $FORCE -eq 0 ]]; then info "$name: done before ($json)"; return 0; fi
  info "$name: $(basename "$ckpt") -> $json"
  local t0=$SECONDS
  if (( DRY )); then info "\$ ${TR[*]} --ckpt $ckpt --out $json $*"; return 0; fi
  if ! "${TR[@]}" --ckpt "$ckpt" --out "$json" "$@" 2>&1 | tee "$log"; then
    die "$name failed -- see $log"
  fi
  [[ -s "$json" ]] || die "$name wrote no $json -- see $log"
  info "$name: $(( (SECONDS - t0) / 60 )) min"
}

# ---------------------------------------------------------------------------- 4 smoke
say "4/6 smoke test (1 window, step 200)"
if (( SMOKE )); then
  measure smoke "$CK/step_0000200.pth" "$ROOT/e1/smoke/smoke.json" "$ROOT/e1/smoke/smoke.log" --windows 1
  if (( ! DRY )); then
    crun python - "$ROOT/e1/smoke/smoke.json" <<'PY' || die "smoke test failed its checks"
import json, sys
j = json.load(open(sys.argv[1]))
s = j["sanity"]["after_warmup"]
ok = s["passed"] and len(j["per_window"]) == 1 and j["meta"]["step"] == 200
print(f"    smoke: step {j['meta']['step']}, ages {s['age_counts']}, floor (a) identical "
      f"{j['sanity']['floor_a_bitwise_identical']}, {j['seconds'] / 60:.1f} min -> {'OK' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
PY
  fi
else
  info "skipped (--no-smoke)"
fi

# ---------------------------------------------------------------------------- 5 measurements
say "5/6 measurements (one at a time)"
measure step_0200 "$CK/step_0000200.pth" "$OUT/step_0200.json" "$OUT/logs/step_0200.log"
measure step_1000 "$CK/step_0001000.pth" "$OUT/step_1000.json" "$OUT/logs/step_1000.log"
measure step_2000 "$CK/resume.pth" "$OUT/step_2000.json" "$OUT/logs/step_2000.log"
measure age_curve "$CK/resume.pth" "$OUT/age_curve_step2000.json" "$OUT/logs/age_curve_step2000.log" \
        --long-ages 31

# ---------------------------------------------------------------------------- 6 report
say "6/6 report"
run crun python tools/measure_cache_drift.py summarize --dir "$OUT" --train-log "$CK/train_log.jsonl"
(( DRY )) || ls -l "$OUT"
say "done: $OUT/summary.md, notes.md (fill in 'Agent remarks'), age_curve_step2000.png"
