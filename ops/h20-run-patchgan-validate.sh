#!/usr/bin/env bash
# Page-spec acceptance only. No formal500-step A/B, checkpoints, W&B or evaluation.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=/data/rdm-workspace/rdm-sets
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec >>"$ROOT/launch-logs/patchgan-validate.log" 2>&1
on_exit() { rc=$?; trap - EXIT; echo "$rc" > "$ROOT/control/patchgan-validate.exit"; exit "$rc"; }
trap on_exit EXIT
echo "$$" > "$ROOT/control/patchgan-validate.pid"
printf 'source=%s\nmode=Page spatial PatchGAN bounded acceptance\n' "$REPO"
PYTHON="$ROOT/envs/rdm/bin/python"
MODEL_RECEIPT="$ROOT/control/patchgan-qwen3vl-prefetch-verified.json"
while [[ ! -s "$MODEL_RECEIPT" ]]; do
  if [[ -f "$ROOT/control/patchgan-model.exit" ]] && \
     [[ "$(cat "$ROOT/control/patchgan-model.exit")" != 0 ]]; then exit 20; fi
  sleep 10
done
export RDM_QWEN3VL_SPATIAL_SNAPSHOT
RDM_QWEN3VL_SPATIAL_SNAPSHOT="$($PYTHON -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["all_official_hashes_verified"]; print(r["snapshot"])' "$MODEL_RECEIPT")"
export HF_HUB_OFFLINE=1
cd "$REPO"
CUDA_VISIBLE_DEVICES="" "$PYTHON" -m pytest -q tests/test_qwen3_vl_spatial.py \
  tests/test_reference_images.py tests/test_pixel_grad_caching.py tests/test_spatial_cache.py \
  tests/test_spatial_configs.py tests/test_benchmark_acceptance.py z-image/tests/test_dinov3_spatial.py \
  z-image/tests/test_spatial_adversarial.py z-image/tests/test_spatial_critic_runtime.py \
  z-image/tests/test_spatial_trainer.py z-image/tests/test_zimage_multistep.py \
  z-image/tests/test_checkpoint_groups.py z-image/tests/test_qwen_text_store.py
CUDA_VISIBLE_DEVICES="" "$ROOT/envs/rdm/bin/torchrun" --standalone --nproc-per-node=2 \
  ops/h20-spatial-distributed-parity.py --json "$ROOT/control/patchgan-distributed-parity.json"
exec 9>/data/rdm-workspace/rdm-run.lock
flock -n 9 || exit 3
check_idle() {
  "$PYTHON" - <<'PY'
import subprocess, sys, time
sys.path.insert(0, 'ops')
from h20_benchmark_acceptance import wait_for_idle_gpus
def read_state():
    rows=subprocess.check_output(['nvidia-smi','-i','4,5',
        '--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10)
    return [list(map(int,row.split(','))) for row in rows.splitlines()]
rows=wait_for_idle_gpus(read_state,time.sleep,time.monotonic)
assert all(m<=32 and u==0 for _,m,total,u in rows),rows
PY
}
check_idle
CUDA_VISIBLE_DEVICES=4 "$PYTHON" ops/h20-patchgan-gpu-check.py --root "$ROOT" \
  --loss ns \
  --output "$ROOT/control/patchgan-gpu-check.json"
flock -u 9

# Waiting for CPU download/indexing never keeps a GPU context or the run lock.
IMAGES="$ROOT/zimage/zimage-ref-v1/reference_images_patchgan/manifest.json"
while [[ ! -s "$IMAGES" ]]; do
  if [[ -f "$ROOT/control/patchgan-images.exit" ]] && \
     [[ "$(cat "$ROOT/control/patchgan-images.exit")" != 0 ]]; then exit 21; fi
  sleep 10
done
"$PYTHON" - "$IMAGES" <<'PY'
import sys
from rdm.sw_lmmd.reference_images import ReferenceImageStore
store=ReferenceImageStore(sys.argv[1])
assert store.is_complete and store.available_rows==125808
store.close()
PY
flock -n 9 || exit 3
check_idle
CUDA_VISIBLE_DEVICES=4 "$PYTHON" ops/h20-patchgan-gpu-check.py --root "$ROOT" \
  --loss ns \
  --image-manifest "$IMAGES" --image-row 0 --text-row 0 \
  --output "$ROOT/control/patchgan-real-png-gpu-check.json"
export CUDA_VISIBLE_DEVICES=4,5
CONFIG=z-image/configs/zimage_2step_1024_dlr1e-4_spatial_patchgan_h20_fa3.yaml
STORE="$ROOT/zimage/zimage-ref-v1/reference_store_qwen"
for MICRO in 2 1; do
  "$PYTHON" ops/h20-checkpoint-sweep.py --only-case 2 "$MICRO" --config "$CONFIG" \
    --reference-root "$STORE" --state-prefix spatial-patchgan-sweep
  if "$PYTHON" - "$ROOT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);s=json.loads((root/'control/spatial-patchgan-sweep-latest.json').read_text())
c=s['cases'][-1]
if c['status']=='oom': raise SystemExit(1)
assert c['status']=='passed' and c['exit']==0
result={'passed':True,'formal_training_started':False,'gate':c,
        'image_manifest':str(root/'zimage/zimage-ref-v1/reference_images_patchgan/manifest.json'),
        'config':s['config'],'source_commit':s['source_commit']}
tmp=root/'control/patchgan-acceptance.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n')
tmp.replace(root/'control/patchgan-acceptance.json')
print(json.dumps(result))
PY
  then exit 0; fi
done
echo 'No candidate completed full updates and probe/refresh' >&2
exit 22
