#!/usr/bin/env bash
# torchrun launcher for SW-LMMD training.
#
#   GPUS=2 bash scripts/train_sw_lmmd.sh configs/sw_lmmd_h100_2gpu.yaml
#   GPUS=4 FSDP=1 bash scripts/train_sw_lmmd.sh configs/sw_lmmd_debug_4x4090.yaml
#   GPUS=2 STEPS=5 bash scripts/train_sw_lmmd.sh <config>        # short smoke run
#   REFERENCE_ROOT=/mnt/store bash scripts/train_sw_lmmd.sh <config>   # store moved/copied
#
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT, as in scripts/train.sh.
#
# The window (K=1024, B=128) is fixed by the config and is the same on every machine;
# tests/test_sw_lmmd_parity.py shows the parameter gradient is identical at world 1/2/4, so
# GPUS only changes throughput, never the objective. It must divide B.
#
# FSDP=1 shards params/grads/optimizer state across ranks (ZeRO-3). Only needed when the
# 46.5 GB training state does not fit one card -- an 80 GB card is FASTER without it, since
# every MM-DiT block costs an extra all-gather. It also forces grad_reduce=none, because
# FSDP's reduce-scatter already averages the gradient.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: GPUS=<n> bash scripts/train_sw_lmmd.sh <config.yaml>}"
GPUS="${GPUS:-2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONDA_ENV="${CONDA_ENV:-rdm}"
ASSETS="${ASSETS:-/data/thor/jiacheng/rdm-sets}"

[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 1; }
[[ -f "$ASSETS/env.sh" ]] && source "$ASSETS/env.sh"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Preflight. Each of these fails silently or very late if left unchecked: a missing store
# surfaces only after both models are on the GPU, and an indivisible B surfaces only at the
# first all-gather.
# ---------------------------------------------------------------------------
say "preflight"
OVERRIDES=()
[[ -n "${FSDP:-}" && "${FSDP}" != "0" ]] && OVERRIDES+=("--set" "memory.shard=fsdp" "--set" "memory.grad_reduce=none")
[[ -n "${STEPS:-}" ]] && OVERRIDES+=("--set" "steps=$STEPS")
[[ -n "${MICRO_BATCH:-}" ]] && OVERRIDES+=("--set" "memory.micro_batch=$MICRO_BATCH")
# The configs carry an absolute reference_root, so a new machine would otherwise need the YAML
# edited before it could run at all.
[[ -n "${REFERENCE_ROOT:-}" ]] && OVERRIDES+=("--set" "reference_root=$REFERENCE_ROOT")

# The overrides are passed here too, so the numbers printed are the numbers the run uses.
conda run -n "$CONDA_ENV" --no-capture-output python - "$CONFIG" "$GPUS" "${OVERRIDES[@]}" <<'PY'
import sys, os
from rdm.train.launch import load_config
from rdm.sw_lmmd.launch import apply_override, memory_from_config, resolve_batching, window_from_config

cfg_path, gpus = sys.argv[1], int(sys.argv[2])
cfg = load_config(cfg_path)
for tok in sys.argv[3:]:
    if tok != "--set":
        apply_override(cfg, tok)
if getattr(cfg, "method", None) != "sw_lmmd":
    raise SystemExit(f"    {cfg_path} is not a SW-LMMD config (method={getattr(cfg,'method',None)!r})")

w, m = window_from_config(cfg), memory_from_config(cfg)
b = resolve_batching(cfg, m, w, world_size=gpus)      # raises if B is not divisible by GPUS
root = getattr(cfg, "reference_root", None)
if not root or not os.path.exists(os.path.join(root, "metadata.json")):
    raise SystemExit(f"    reference store missing: {root}\n"
                     f"    build it first: bash scripts/preprocess_all.sh")

P = 3.875
state = P * (2 if m.param_dtype.itemsize == 2 else 4) * 2 + (P * 2 if m.optimizer == "adamw8bit" else P * 8)
if m.shard == "fsdp":
    state /= gpus
enc = 0.95 * len(cfg.encoders) if not m.battery_bf16 else 0.78 * len(cfg.encoders)
est = state + enc + 0.2 + 2.0 * m.micro_batch
print(f"    window        K={w.size} B={w.stride} overlap={w.overlap} max_cache_age={w.max_age_steps}")
print(f"    batching      {gpus} ranks x {b['micro_batch']} micro x {b['grad_accum']} accum = {w.stride}")
print(f"    memory        shard={m.shard} reduce={m.grad_reduce} opt={m.optimizer} "
      f"param={str(m.param_dtype).replace('torch.','')} battery_bf16={m.battery_bf16}")
print(f"    est. per-card ~{est:.0f} GB  (activation term is an estimate; halve micro_batch if it OOMs)")
print(f"    encoders      {len(cfg.encoders)}: {','.join(cfg.encoders)}")
print(f"    reference     {root}")
print(f"    steps         {cfg.steps}  ->  {getattr(cfg,'output_dir','./work_dirs')}/{cfg.exp_name}")
PY

say "launching $GPUS rank(s)"
[[ ${#OVERRIDES[@]} -gt 0 ]] && info "overrides: ${OVERRIDES[*]}"
exec conda run -n "$CONDA_ENV" --no-capture-output torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --nproc_per_node="$GPUS" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m rdm.sw_lmmd.launch "$CONFIG" "${OVERRIDES[@]}"
