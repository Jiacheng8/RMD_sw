#!/usr/bin/env bash
# torchrun launcher for SW-LMMD training.
#
#   GPUS=2 bash scripts/train_sw_lmmd.sh configs/sw_lmmd_h100_2gpu.yaml
#   GPUS=4 FSDP=1 bash scripts/train_sw_lmmd.sh configs/sw_lmmd_debug_4x4090.yaml
#   GPUS=2 STEPS=5 bash scripts/train_sw_lmmd.sh <config>        # short smoke run
#   REFERENCE_ROOT=/mnt/store bash scripts/train_sw_lmmd.sh <config>   # store moved/copied
#   OUTPUT_DIR=/mnt/work_dirs bash scripts/train_sw_lmmd.sh <config>   # checkpoints/logs elsewhere
#   RESUME_FROM=<run dir>/resume.pth bash scripts/train_sw_lmmd.sh <config>   # continue a stopped run
#
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT, as in scripts/train.sh.
#
# The window is fixed by the config (K=1024, B=128 by default; the H100 configs use K=128, B=32);
# tests/test_sw_lmmd_parity.py shows the parameter gradient is identical at world 1/2/4, so
# GPUS only changes throughput, never the objective. It must divide B.
#
# FSDP=1 shards params/grads/optimizer state across ranks (ZeRO-3). Only needed when the
# 46.5 GB training state does not fit one card -- an 80 GB card is FASTER without it, since
# every MM-DiT block costs an extra all-gather. It also forces grad_reduce=none, because
# FSDP's reduce-scatter already averages the gradient.
set -euo pipefail
export PYTHONNOUSERSITE=1   # packages in ~/.local/lib/pythonX.Y must never shadow the conda envs'

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
[[ -n "${OUTPUT_DIR:-}" ]] && OVERRIDES+=("--set" "output_dir=$OUTPUT_DIR")
# Resume (configs with save_resume: true keep <run dir>/resume.pth). The weights come in through
# load_from, strictly: a wrong path must stop the run, not silently restart from the base model.
if [[ -n "${RESUME_FROM:-}" ]]; then
  [[ -f "$RESUME_FROM" ]] || { echo "RESUME_FROM=$RESUME_FROM does not exist" >&2; exit 1; }
  OVERRIDES+=("--set" "resume_from=$RESUME_FROM" "--set" "load_from=$RESUME_FROM" "--set" "strict_load=true")
fi

# The overrides are passed here too, so the numbers printed are the numbers the run uses.
conda run -n "$CONDA_ENV" --no-capture-output python - "$CONFIG" "$GPUS" "${OVERRIDES[@]}" <<'PY'
import sys, os
from rdm.train.launch import load_config
from rdm.sw_lmmd.launch import (apply_override, gan_from_config, lr_schedule_from_config,
                                memory_from_config, resolve_batching, window_from_config)

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
sch = lr_schedule_from_config(cfg)
print(f"    lr            {cfg.lr:g}, {sch.name}" + (f", warm-up {sch.warmup_steps}" if sch.warmup_steps else "")
      + (f", decays to {sch.min_lr_ratio:g}x by step {sch.total_steps}" if sch.name != "constant" else ""))
g = gan_from_config(cfg)
if g.enabled:
    print(f"    gan           critic on {','.join(g.encoders or cfg.encoders)} | weight {g.weight} "
          f"({'adaptive' if g.adaptive else 'fixed'}) | {g.loss} | lr {g.lr} | from step {g.g_start_step}")
    if m.battery_bf16:
        print("    WARNING       gan with battery_bf16=true: the critic can learn the bf16-vs-fp32 "
              "encoder pipeline gap (see rdm/sw_lmmd/adversarial.py); prefer battery_bf16: false")
print(f"    reference     {root}")
print(f"    steps         {cfg.steps}  ->  {getattr(cfg,'output_dir','./work_dirs')}/{cfg.exp_name}")
if getattr(cfg, "save_resume", False):
    print(f"    resume state  every {getattr(cfg, 'resume_every', 0) or cfg.save_freq} steps -> <run dir>/resume.pth")
if getattr(cfg, "resume_from", None):
    import torch
    st = torch.load(cfg.resume_from, map_location="cpu", weights_only=False, mmap=True)
    if int(st["step"]) >= int(cfg.steps):
        raise SystemExit(f"    {cfg.resume_from} is at step {st['step']}: the {cfg.steps}-step run is already complete")
    print(f"    RESUMING      {cfg.resume_from} at step {st['step']} (optimizer "
          f"{'yes' if st.get('optimizer') is not None else 'NO'}, cache {'yes' if st.get('cache') else 'NO'}), "
          f"{int(cfg.steps) - int(st['step'])} steps to go")
PY

say "launching $GPUS rank(s)"
[[ ${#OVERRIDES[@]} -gt 0 ]] && info "overrides: ${OVERRIDES[*]}"
exec conda run -n "$CONDA_ENV" --no-capture-output torchrun \
  --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --nproc_per_node="$GPUS" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m rdm.sw_lmmd.launch "$CONFIG" "${OVERRIDES[@]}"
