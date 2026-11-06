#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=/data/rdm-workspace/rdm-sets
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec >>"$ROOT/launch-logs/checkpoint-sweep.log" 2>&1
on_exit() { rc=$?; trap - EXIT; echo "$rc" > "$ROOT/control/checkpoint-sweep.exit"; exit "$rc"; }
trap on_exit EXIT
echo "$$" > "$ROOT/control/checkpoint-sweep.pid"
while [[ ! -f "$ROOT/control/validate-optimization.exit" ]]; do sleep 10; done
test "$(cat "$ROOT/control/validate-optimization.exit")" = 0
exec 9>/data/rdm-workspace/rdm-run.lock
flock -n 9 || exit 3
python3 - <<'PY'
import subprocess
text = subprocess.check_output(['nvidia-smi','-i','4,5','--query-gpu=memory.used,utilization.gpu',
                               '--format=csv,noheader,nounits'], text=True)
assert all(int(memory) <= 32 and int(util) == 0 for memory, util in
           (row.split(',') for row in text.splitlines())), text
PY
cd "$REPO"
"$ROOT/envs/rdm/bin/python" -m pytest z-image/tests/test_checkpoint_groups.py \
  z-image/tests/test_zimage_multistep.py z-image/tests/test_qwen_text_store.py \
  z-image/tests/test_vae_upsample.py -q
CUDA_VISIBLE_DEVICES=4 "$ROOT/envs/rdm/bin/python" ops/h20-vae-upsampling-check.py \
  --out "$ROOT/control/vae-upsampling-cuda-check.json"
"$ROOT/envs/rdm/bin/torchrun" --nproc_per_node=2 --master_port=29541 \
  ops/h20-checkpoint-parity.py --out "$ROOT/control/checkpoint-fsdp-parity.json"
"$ROOT/envs/rdm/bin/python" ops/h20-checkpoint-sweep.py "$@"
