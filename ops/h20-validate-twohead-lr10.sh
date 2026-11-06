#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec >>"$ROOT/launch-logs/twohead-lr10-validation.log" 2>&1
trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" >"$ROOT/control/twohead-lr10-validation.exit"; exit "$rc"' EXIT
printf '%s\n' "$$" >"$ROOT/control/twohead-lr10-validation.pid"
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RDM_QWEN3VL_SPATIAL_SNAPSHOT="$ROOT/models/qwen3vl-4b-patchgan/ebb281ec70b05090aa6165b016eac8ec08e71b17"
export CUDA_VISIBLE_DEVICES=""
cd "$REPO"
set +e
"$ROOT/envs/rdm/bin/python" -m pytest -q \
  tests/test_qwen3_vl_spatial.py tests/test_reference_images.py tests/test_pixel_grad_caching.py \
  tests/test_spatial_cache.py tests/test_spatial_configs.py tests/test_benchmark_acceptance.py \
  z-image/tests/test_dinov3_spatial.py z-image/tests/test_spatial_adversarial.py \
  z-image/tests/test_spatial_critic_runtime.py z-image/tests/test_spatial_trainer.py \
  z-image/tests/test_zimage_multistep.py z-image/tests/test_checkpoint_groups.py \
  z-image/tests/test_qwen_text_store.py z-image/tests/test_adaptive_d_lr.py \
  z-image/tests/test_adaptive_d_lr_distributed.py z-image/tests/test_owned_relay_recovery.py \
  z-image/tests/test_adaptive_resume.py
rc=$?
printf '%s\n' "$rc" >"$ROOT/control/twohead-lr10-cpu.exit"
set -e
test "$rc" = 0
exec 9>/data/rdm-workspace/rdm-run.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=4,5 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
"$ROOT/envs/rdm/bin/python" ops/h20-select-lr10-benchmark.py
