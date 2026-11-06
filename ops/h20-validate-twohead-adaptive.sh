#!/usr/bin/env bash
# CPU/Gloo integration only. Does not take the training lock or allocate GPUs.
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/launch-logs" "$ROOT/control"
exec >>"$ROOT/launch-logs/twohead-adaptive-validation.log" 2>&1
trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" >"$ROOT/control/twohead-adaptive-validation.exit"; exit "$rc"' EXIT
printf '%s\n' "$$" >"$ROOT/control/twohead-adaptive-validation.pid"
printf 'source=%s commit=%s utc=%s\n' "$REPO" "$(cat "$REPO/SOURCE_COMMIT")" "$(date -u +%FT%TZ)"
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO"
"$ROOT/envs/rdm/bin/python" -m pytest -q \
  tests/test_qwen3_vl_spatial.py tests/test_reference_images.py \
  tests/test_pixel_grad_caching.py tests/test_spatial_cache.py tests/test_spatial_configs.py \
  tests/test_benchmark_acceptance.py z-image/tests/test_dinov3_spatial.py \
  z-image/tests/test_spatial_adversarial.py z-image/tests/test_spatial_critic_runtime.py \
  z-image/tests/test_spatial_trainer.py z-image/tests/test_zimage_multistep.py \
  z-image/tests/test_checkpoint_groups.py z-image/tests/test_qwen_text_store.py \
  z-image/tests/test_adaptive_d_lr.py z-image/tests/test_adaptive_d_lr_distributed.py \
  z-image/tests/test_owned_relay_recovery.py z-image/tests/test_adaptive_resume.py
