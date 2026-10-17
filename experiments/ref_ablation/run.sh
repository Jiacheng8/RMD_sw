#!/usr/bin/env bash
# One run of the reference-side ablation: one ref_mode, one seed.
#
#   bash experiments/ref_ablation/run.sh <local|random|global> <seed> [--gpus N] [--steps N]
#                                        [--dry-run] [--diagnose-only] [--set KEY=VALUE ...]
#
#   bash experiments/ref_ablation/run.sh random 1                  # 2 GPUs, 2000 steps
#   for s in 0 1 2; do for m in local random global; do            # the whole ablation, in turn
#     bash experiments/ref_ablation/run.sh $m $s; done; done
#
# The seed sets `seed` (each row's latent noise, torch's global seed) and `ref_ablation.ref_seed`
# (the random mode's row sampler). The window order stays `window.order_seed` = 3407 for every
# mode and seed unless ORDER_SEED is set, so the three modes of one seed see the same windows and
# the same noise and differ only in the attraction's reference side. Run directory:
# <output_dir>/sw-lmmd-refabl-k128b32-gan-<mode>-s<seed>.
#
# Environment, as scripts/train_sw_lmmd.sh: GPUS (default 2; B=32 must split into micro-batches
# of memory.micro_batch=4 per rank), REFERENCE_ROOT, OUTPUT_DIR, NYSTROM_DIR (the landmark
# bundles, needed by global), ORDER_SEED, FSDP=1, MICRO_BATCH, RESUME_FROM, MASTER_PORT,
# CONDA_ENV (rdm), ASSETS (sourced env.sh).
set -euo pipefail
export PYTHONNOUSERSITE=1   # packages in ~/.local/lib/pythonX.Y must never shadow the conda envs'

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
usage() { sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
[[ $# -ge 2 ]] || { usage >&2; exit 2; }
MODE="$1"; SEED="$2"; shift 2
case "$MODE" in local|random|global) ;; *) echo "mode must be local, random or global, got '$MODE'" >&2; exit 2 ;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer, got '$SEED'" >&2; exit 2; }

GPUS="${GPUS:-2}"
DRY=0
SETS=()      # --set KEY=VALUE pairs, passed on after the ones built from the environment
FLAGS=()     # flags of the training entry point itself
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)          GPUS="$2"; shift 2 ;;
    --steps)         STEPS="$2"; shift 2 ;;
    --dry-run)       DRY=1; shift ;;
    --diagnose-only) FLAGS+=("--diagnose-only"); shift ;;
    --set)           SETS+=("--set" "$2"); shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *)               echo "unknown option '$1'" >&2; exit 2 ;;
  esac
done
MASTER_PORT="${MASTER_PORT:-29500}"
CONDA_ENV="${CONDA_ENV:-rdm}"
ASSETS="${ASSETS:-/data/thor/jiacheng/rdm-sets}"
CONFIG="$HERE/configs/ref_${MODE}.yaml"
[[ -f "$ASSETS/env.sh" ]] && source "$ASSETS/env.sh"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OVERRIDES=(--set "seed=$SEED" --set "ref_ablation.ref_seed=$SEED")
[[ -n "${ORDER_SEED:-}" ]] && OVERRIDES+=(--set "window.order_seed=$ORDER_SEED")
[[ -n "${FSDP:-}" && "${FSDP}" != "0" ]] && OVERRIDES+=(--set "memory.shard=fsdp" --set "memory.grad_reduce=none")
[[ -n "${STEPS:-}" ]] && OVERRIDES+=(--set "steps=$STEPS")
[[ -n "${MICRO_BATCH:-}" ]] && OVERRIDES+=(--set "memory.micro_batch=$MICRO_BATCH")
[[ -n "${REFERENCE_ROOT:-}" ]] && OVERRIDES+=(--set "reference_root=$REFERENCE_ROOT")
[[ -n "${OUTPUT_DIR:-}" ]] && OVERRIDES+=(--set "output_dir=$OUTPUT_DIR")
[[ -n "${NYSTROM_DIR:-}" ]] && OVERRIDES+=(--set "ref_ablation.nystrom_dir=$NYSTROM_DIR")
if [[ -n "${RESUME_FROM:-}" ]]; then
  [[ -f "$RESUME_FROM" ]] || { echo "RESUME_FROM=$RESUME_FROM does not exist" >&2; exit 1; }
  OVERRIDES+=(--set "resume_from=$RESUME_FROM" --set "load_from=$RESUME_FROM" --set "strict_load=true")
fi
OVERRIDES+=(${SETS[@]+"${SETS[@]}"})

printf '\n\033[1m==> reference ablation: %s, seed %s, %s GPU(s)\033[0m\n' "$MODE" "$SEED" "$GPUS"
cd "$REPO"
conda run -n "$CONDA_ENV" --no-capture-output python - "$CONFIG" "$GPUS" "${OVERRIDES[@]}" <<'PY'
import os, sys
from rdm.train.launch import load_config
from rdm.sw_lmmd.launch import (apply_override, gan_from_config, lr_schedule_from_config,
                                memory_from_config, resolve_batching, window_from_config)
from experiments.ref_ablation.config import ref_ablation_from_config
from experiments.ref_ablation.train import check_nystrom, run_name

cfg_path, gpus = sys.argv[1], int(sys.argv[2])
cfg = load_config(cfg_path)
for tok in sys.argv[3:]:
    if tok != "--set":
        apply_override(cfg, tok)
w, m, ref = window_from_config(cfg), memory_from_config(cfg), ref_ablation_from_config(cfg)
b = resolve_batching(cfg, m, w, world_size=gpus)          # raises if B does not split
root = getattr(cfg, "reference_root", None)
if not root or not os.path.exists(os.path.join(root, "metadata.json")):
    raise SystemExit(f"    reference store missing: {root}")
have = check_nystrom(cfg, ref)                            # global: bundles exist and match the store
g, s = gan_from_config(cfg), lr_schedule_from_config(cfg)
print(f"    ref_mode      {ref.ref_mode} (ref_seed {ref.ref_seed}) | Nystrom bundles "
      f"{'OK, ' + ref.nystrom_dir if have else 'none (' + str(ref.nystrom_dir) + ')'}")
print(f"    window        K={w.size} B={w.stride} grouped={w.group_by_prompt} order_seed={w.order_seed}")
print(f"    batching      {gpus} ranks x {b['micro_batch']} micro x {b['grad_accum']} accum = {w.stride}")
print(f"    memory        shard={m.shard} opt={m.optimizer} battery_bf16={m.battery_bf16}")
print(f"    gan           {'on, ' + g.critic + ' critic, weight ' + str(g.weight) if g.enabled else 'off'}")
print(f"    lr            {cfg.lr:g} {s.name} (warm-up {s.warmup_steps}, floor {s.min_lr_ratio:g}x)")
print(f"    seed          {cfg.seed} | steps {cfg.steps} | encoders {','.join(cfg.encoders)}")
print(f"    run dir       {os.path.join(getattr(cfg, 'output_dir', './work_dirs'), run_name(cfg, ref))}")
PY
(( DRY )) && { echo "    --dry-run: not launching"; exit 0; }
exec conda run -n "$CONDA_ENV" --no-capture-output torchrun --nproc_per_node="$GPUS" \
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
  -m experiments.ref_ablation.train "$CONFIG" "${OVERRIDES[@]}" ${FLAGS[@]+"${FLAGS[@]}"}
