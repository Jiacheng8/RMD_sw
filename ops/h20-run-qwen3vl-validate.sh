#!/usr/bin/env bash
# Complete preparation and bounded acceptance only. Never start formal A/B or eval.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=/data/rdm-workspace/rdm-sets
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
exec >>"$ROOT/launch-logs/qwen3vl-validate.log" 2>&1
on_exit() { rc=$?; trap - EXIT; echo "$rc" > "$ROOT/control/qwen3vl-validate.exit"; exit "$rc"; }
trap on_exit EXIT
echo "$$" > "$ROOT/control/qwen3vl-validate.pid"
printf 'source=%s\nmode=Qwen3VL preparation and bounded validation only\n' "$REPO"
PYTHON="$ROOT/envs/rdm/bin/python"
RECEIPT="$ROOT/control/qwen3vl-4b-vision-prefetch-verified.json"
while [[ ! -s "$RECEIPT" ]]; do
  # A failed download is actionable; retain its real exit and logs.
  if [[ -f "$ROOT/control/qwen3vl-fast-prefetch.exit" ]] && \
     [[ "$(cat "$ROOT/control/qwen3vl-fast-prefetch.exit")" != 0 ]] && \
     [[ -f "$ROOT/control/qwen3vl-prefetch.exit" ]] && \
     [[ "$(cat "$ROOT/control/qwen3vl-prefetch.exit")" != 0 ]]; then exit 20; fi
  sleep 10
done
export RDM_QWEN3VL_SNAPSHOT
RDM_QWEN3VL_SNAPSHOT="$($PYTHON -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["all_official_hashes_verified"]; print(x["snapshot"])' "$RECEIPT")"
export RDM_QWEN3VL_ATTN_IMPLEMENTATION=sdpa
cd "$REPO"
CUDA_VISIBLE_DEVICES="" "$PYTHON" -m pytest -q tests/test_qwen3_vl_backbone.py \
  tests/test_panel_split.py tests/test_representation.py z-image/tests/test_qwen3vl_reference_builder.py
exec 9>/data/rdm-workspace/rdm-run.lock
flock -n 9 || exit 3
"$PYTHON" - <<'PY'
import subprocess
rows = subprocess.check_output(['nvidia-smi','-i','4,5','--query-gpu=memory.used,utilization.gpu',
                                '--format=csv,noheader,nounits'], text=True)
assert all(int(m) <= 32 and int(u) == 0 for m,u in (r.split(',') for r in rows.splitlines())), rows
PY
CUDA_VISIBLE_DEVICES=4 "$PYTHON" ops/h20-qwen3vl-check.py \
  --snapshot "$RDM_QWEN3VL_SNAPSHOT" --out "$ROOT/control/qwen3vl-image-gradient-check.json"
export HF_HUB_OFFLINE=0
CUDA_VISIBLE_DEVICES=4 "$PYTHON" ops/h20-qwen3vl-reference.py --root "$ROOT" \
  --dataset-revision 3195cc9e58d83dfe17a6b5d89f9886121872586f --limit-shards 1
# Same immutable contract and committed per-shard rows resume the pilot directly.
CUDA_VISIBLE_DEVICES=4 "$PYTHON" ops/h20-qwen3vl-reference.py --root "$ROOT" \
  --dataset-revision 3195cc9e58d83dfe17a6b5d89f9886121872586f
export CUDA_VISIBLE_DEVICES=4,5
export HF_HUB_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
CONFIG=z-image/configs/zimage_2step_1024_dlr1e-4_qwen3vl_h20_fa3.yaml
STORE="$ROOT/zimage/zimage-ref-v1/reference_store_qwen3vl"
for MICRO in 16 8 4; do
  "$PYTHON" ops/h20-checkpoint-sweep.py --only-case 2 "$MICRO" \
    --config "$CONFIG" --reference-root "$STORE" --state-prefix qwen3vl-sweep
  # Preserve each gate; controller exit0 also covers an expected OOM.
  "$PYTHON" - "$ROOT" "$MICRO" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);m=int(sys.argv[2])
s=json.loads((r/'control/qwen3vl-sweep-latest.json').read_text())
c=s['cases'][-1]
assert c['status'] in ('passed','oom'), c
(r/f'control/qwen3vl-micro{m}-gate.json').write_text(json.dumps(s,indent=2)+'\n')
PY
done
"$PYTHON" - "$ROOT" <<'PY'
import datetime,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);cases=[]
for micro in (16,8,4):
    s=json.loads((r/f'control/qwen3vl-micro{micro}-gate.json').read_text())
    cases.append(s['cases'][-1])
passed=[x for x in cases if x['status']=='passed']
assert passed, 'No full iteration/probe/refresh gate passed'
stable=[x for x in passed if max(x['nvml_peak_mib'].values()) <= 143771*.95]
assert stable, 'No measured point leaves 5% memory margin'
winner=min(stable,key=lambda x:x['mean_seconds'])
result={'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'formal_training_started':False,'cases':cases,'stable_candidate':winner,
        'max_passed_micro':max(x['micro_batch'] for x in passed)}
tmp=r/'control/qwen3vl-acceptance.tmp'
tmp.write_text(json.dumps(result,indent=2)+'\n')
tmp.replace(r/'control/qwen3vl-acceptance.json')
print(json.dumps(result))
PY
