#!/usr/bin/env bash
# torchrun launcher for the Z-Image few-step student.
#
#   GPUS=4 bash z-image/scripts/train.sh                                   # default config
#   GPUS=4 bash z-image/scripts/train.sh <config.yaml> --set steps=5       # smoke run
#   RESUME_FROM=<run dir>/resume.pth GPUS=4 bash z-image/scripts/train.sh  # continue a run
#   REFERENCE_ROOT=/mnt/store OUTPUT_DIR=/mnt/runs GPUS=4 bash z-image/scripts/train.sh
#
# Extra arguments after the config are passed to the trainer (--set KEY=VALUE ...). GPUS must
# divide B = 32. Under FSDP the optimizer moments are sharded and not saved, so a resume keeps
# the weights, the window position, both caches and both critics, and restarts Adam's moments.
set -euo pipefail
export PYTHONNOUSERSITE=1

ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ZDIR/.." && pwd)"
CONFIG="${1:-$ZDIR/configs/zimage_2step_1024_h100.yaml}"
[[ $# -gt 0 ]] && shift
GPUS="${GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"
CONDA_ENV="${CONDA_ENV:-rdm}"
ASSETS="${ASSETS:-/data/thor/jiacheng/rdm-sets}"
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 1; }
[[ -f "$ASSETS/env.sh" ]] && source "$ASSETS/env.sh"
export PYTHONPATH="$ZDIR:$REPO_ROOT:${PYTHONPATH:-}"

OVERRIDES=("$@")
[[ -n "${REFERENCE_ROOT:-}" ]] && OVERRIDES+=("--set" "reference_root=$REFERENCE_ROOT")
[[ -n "${OUTPUT_DIR:-}" ]] && OVERRIDES+=("--set" "output_dir=$OUTPUT_DIR")
if [[ -n "${RESUME_FROM:-}" ]]; then
  [[ -f "$RESUME_FROM" ]] || { echo "RESUME_FROM=$RESUME_FROM does not exist" >&2; exit 1; }
  # under FSDP the weights come in through load_from (before sharding), the rest through resume_from
  OVERRIDES+=("--set" "resume_from=$RESUME_FROM" "--set" "load_from=$RESUME_FROM")
fi

# preflight: the batching and the store, before 4 x 25 GB of weights load
conda run -n "$CONDA_ENV" --no-capture-output python - "$CONFIG" "$GPUS" "${OVERRIDES[@]}" <<'PY'
import logging, os, sys
logging.basicConfig(level=logging.INFO, format="    %(message)s")
from rdm.train.launch import load_config
from rdm.sw_lmmd.launch import apply_override, memory_from_config, resolve_batching, window_from_config
from zimage_distill.launch import student_from_config

cfg_path, gpus = sys.argv[1], int(sys.argv[2])
cfg = load_config(cfg_path)
for tok in sys.argv[3:]:
    if tok != "--set":
        apply_override(cfg, tok)
w, m = window_from_config(cfg), memory_from_config(cfg)
b = resolve_batching(cfg, m, w, world_size=gpus)
s = student_from_config(cfg)
root = cfg.reference_root
if not os.path.exists(os.path.join(root, "metadata.json")):
    raise SystemExit(f"    Z-Image reference store missing or unfinished: {root}\n"
                     f"    build it with /data/ironman/jiacheng/RDM/z-image/process (launch.py all)")
from zimage_distill.launch import check_render_config, resolve_feature_pool
from zimage_distill.store import ZImageReferenceStore
store = ZImageReferenceStore(root, list(cfg.encoders))
pool = resolve_feature_pool(store, int(cfg.img_size), dict(cfg.zimage).get("feature_pool", "auto"))
check_render_config(store, int(cfg.img_size))
print(f"    store       {store.num_rows} rows / {store.num_prompts} prompts | contexts "
      f"{'precomputed' if store.has_context else 'online'} | feature pool {pool}x")
print(f"    student     {s['num_steps']} steps, sigmas {[round(x, 4) for x in s['sampling_args']['sigmas']]}, "
      f"renoise {s['sampling_args']['renoise']}, step weights {s['step_weights']}")
print(f"    window      K={w.size} B={w.stride} | {gpus} ranks x {b['micro_batch']} micro x {b['grad_accum']} accum")
print(f"    memory      shard={m.shard} opt={m.optimizer} battery_bf16={m.battery_bf16}")
print(f"    reference   {root}")
print(f"    steps       {cfg.steps} -> {cfg.output_dir}/{cfg.exp_name}")
PY

cd "$REPO_ROOT"
exec conda run -n "$CONDA_ENV" --no-capture-output torchrun --nproc_per_node="$GPUS" \
  --master_port="$MASTER_PORT" -m zimage_distill.launch "$CONFIG" "${OVERRIDES[@]}"
