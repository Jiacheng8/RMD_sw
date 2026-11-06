#!/usr/bin/env bash
# Latest softplus/D1e-5 checks, followed immediately by formal A500; no B/eval.
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
PYTHON="$ROOT/envs/rdm/bin/python"
exec >>"$ROOT/launch-logs/spatial-A-preflight.log" 2>&1
trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" >"$ROOT/control/spatial-A-preflight.exit"; exit "$rc"' EXIT
echo "$$" >"$ROOT/control/spatial-A-preflight.pid"
printf 'source=%s\nmode=latest softplus D1e-5 G2.83e-6, then formal500\n' "$REPO"
cd "$REPO"
exec 9>/data/rdm-workspace/rdm-run.lock
flock -n 9 || exit 3
CUDA_VISIBLE_DEVICES="" "$PYTHON" -m pytest -q tests/test_qwen3_vl_spatial.py tests/test_reference_images.py \
  tests/test_pixel_grad_caching.py tests/test_spatial_cache.py tests/test_spatial_configs.py \
  tests/test_benchmark_acceptance.py z-image/tests/test_dinov3_spatial.py \
  z-image/tests/test_spatial_adversarial.py z-image/tests/test_spatial_critic_runtime.py \
  z-image/tests/test_spatial_trainer.py z-image/tests/test_zimage_multistep.py \
  z-image/tests/test_checkpoint_groups.py z-image/tests/test_qwen_text_store.py
check_idle() {
  "$PYTHON" - <<'PY'
import subprocess,sys,time
sys.path.insert(0,'ops')
from h20_benchmark_acceptance import wait_for_idle_gpus
def read_state():
    text=subprocess.check_output(['nvidia-smi','-i','4,5',
      '--query-gpu=index,memory.used,memory.total,utilization.gpu',
      '--format=csv,noheader,nounits'],text=True,timeout=10)
    return [list(map(int,row.split(','))) for row in text.splitlines()]
state=wait_for_idle_gpus(read_state,time.sleep,time.monotonic)
assert all(memory<=32 and util==0 for _,memory,total,util in state),state
PY
}
check_idle
CUDA_VISIBLE_DEVICES=4 "$PYTHON" ops/h20-patchgan-gpu-check.py --root "$ROOT" --loss ns \
  --image-manifest "$ROOT/zimage/zimage-ref-v1/reference_images_patchgan/manifest.json" \
  --image-row 0 --text-row 0 --output "$ROOT/control/spatial-softplus-real-png-check.json"
check_idle
"$ROOT/envs/rdm/bin/torchrun" --standalone --nproc-per-node=2 \
  ops/h20-spatial-critic-diagnose.py --root "$ROOT" \
  --out "$ROOT/control/spatial-critic-loss-lr-diagnose.json"
check_idle
printf '0\n' >"$ROOT/control/spatial-A-preflight.exit"
printf 'checks_complete; starting formal A500 with unchanged configured warmup\n'
trap - EXIT
flock -u 9
exec 9>&-
exec bash "$REPO/ops/h20-run-spatial-a.sh"
