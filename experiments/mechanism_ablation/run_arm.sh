#!/usr/bin/env bash
# One training run of the own-reference (mechanism) ablation, as experiments/ref_ablation/run.sh.
#
#   bash experiments/mechanism_ablation/run_arm.sh <config> [--gpus N] [--dry-run] [--set KEY=VALUE ...]
#
#   <config>: window_nodiag | global_plus_self | main_probe   (experiments/mechanism_ablation/configs/<config>.yaml)
#   bash experiments/mechanism_ablation/run_arm.sh window_nodiag
#   bash experiments/mechanism_ablation/run_arm.sh global_plus_self --set own_term.beta=0.0079
#
# Environment: GPUS (default 2), REFERENCE_ROOT, OUTPUT_DIR, NYSTROM_DIR, RESUME_FROM, MASTER_PORT,
# CONDA_ENV (rdm), ASSETS (its env.sh is sourced).
set -euo pipefail
export PYTHONNOUSERSITE=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
usage() { sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
CONFIG="$HERE/configs/$1.yaml"; shift
[[ -f "$CONFIG" ]] || { echo "no config $CONFIG" >&2; exit 2; }

GPUS="${GPUS:-2}"; DRY=0; SETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)    GPUS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --set)     SETS+=("--set" "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)         echo "unknown option '$1'" >&2; exit 2 ;;
  esac
done
MASTER_PORT="${MASTER_PORT:-29500}"
CONDA_ENV="${CONDA_ENV:-rdm}"
[[ -n "${ASSETS:-}" && -f "$ASSETS/env.sh" ]] && source "$ASSETS/env.sh"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OVERRIDES=()
[[ -n "${REFERENCE_ROOT:-}" ]] && OVERRIDES+=(--set "reference_root=$REFERENCE_ROOT")
[[ -n "${OUTPUT_DIR:-}" ]] && OVERRIDES+=(--set "output_dir=$OUTPUT_DIR")
[[ -n "${NYSTROM_DIR:-}" && "$(basename "$CONFIG")" != main_probe.yaml ]] && \
  OVERRIDES+=(--set "ref_ablation.nystrom_dir=$NYSTROM_DIR")
if [[ -n "${RESUME_FROM:-}" ]]; then
  [[ -f "$RESUME_FROM" ]] || { echo "RESUME_FROM=$RESUME_FROM does not exist" >&2; exit 1; }
  OVERRIDES+=(--set "resume_from=$RESUME_FROM" --set "load_from=$RESUME_FROM" --set "strict_load=true")
fi
OVERRIDES+=(${SETS[@]+"${SETS[@]}"})

cd "$REPO"
conda run -n "$CONDA_ENV" --no-capture-output python - "$CONFIG" "$GPUS" "${OVERRIDES[@]}" <<'PY'
import os, sys
from rdm.train.launch import load_config
from rdm.sw_lmmd.launch import (apply_override, gan_from_config, lr_schedule_from_config,
                                memory_from_config, resolve_batching, window_from_config)
from experiments.ref_ablation.config import ref_ablation_from_config
from experiments.ref_ablation.train import check_nystrom
from experiments.mechanism_ablation.config import mechanism_from_config
from experiments.mechanism_ablation.train import run_name

cfg_path, gpus = sys.argv[1], int(sys.argv[2])
cfg = load_config(cfg_path)
for tok in sys.argv[3:]:
    if tok != "--set":
        apply_override(cfg, tok)
w, m, ref, mech = window_from_config(cfg), memory_from_config(cfg), ref_ablation_from_config(cfg), \
    mechanism_from_config(cfg)
b = resolve_batching(cfg, m, w, world_size=gpus)
root = getattr(cfg, "reference_root", None)
if not root or not os.path.exists(os.path.join(root, "metadata.json")):
    raise SystemExit(f"    reference store missing: {root}")
have = check_nystrom(cfg, ref)
g, s = gan_from_config(cfg), lr_schedule_from_config(cfg)
print(f"    variant       {mech.variant} | attraction.skip_diagonal {mech.skip_diagonal} | own_term.beta "
      f"{mech.self_beta:g} | ref_mode {ref.ref_mode} | Nystrom {'OK' if have else 'none'}")
print(f"    window        K={w.size} B={w.stride} grouped={w.group_by_prompt} order_seed={w.order_seed}")
print(f"    batching      {gpus} ranks x {b['micro_batch']} micro x {b['grad_accum']} accum = {w.stride}")
print(f"    gan           {'on, ' + g.critic + ' critic, weight ' + str(g.weight) if g.enabled else 'off'}"
      f" (warm-up {g.g_start_step}, min_norm_ratio {g.min_norm_ratio:g})")
print(f"    lr            {cfg.lr:g} {s.name} (warm-up {s.warmup_steps}, floor {s.min_lr_ratio:g}x, "
      f"decay horizon {s.total_steps})")
print(f"    steps         to {cfg.steps}, save_freq {cfg.save_freq}, resume "
      f"{getattr(cfg, 'resume_from', None) or 'no'} | seed {cfg.seed}")
print(f"    run dir       {os.path.join(getattr(cfg, 'output_dir', './work_dirs'), run_name(cfg))}")
PY
(( DRY )) && { echo "    --dry-run: not launching"; exit 0; }
exec conda run -n "$CONDA_ENV" --no-capture-output torchrun --nproc_per_node="$GPUS" \
  --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
  -m experiments.mechanism_ablation.train "$CONFIG" "${OVERRIDES[@]}"
