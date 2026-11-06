#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG=z-image/configs/zimage_2step_1024_twohead_glr10_h20_fa3.yaml
EXP="$ROOT/zimage/work_dirs/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A"
ARCHIVE=/mnt/airs-business/shaoshitong/rdm-sets/checkpoints/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A
ARCHIVE_CONTROL="$ROOT/control/twohead-retain50-archive"
PREFIX=twohead-lr10-resume200-500-v3
mkdir -p "$ROOT/control" "$ROOT/launch-logs" "$ARCHIVE_CONTROL"
exec >>"$ROOT/launch-logs/$PREFIX.log" 2>&1
RUN_DIR="$ROOT/control/$PREFIX-runs/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$RUN_DIR"
OWNS_LATEST=0
CHILD_PID=''
on_exit() {
  rc=$?; trap - EXIT
  printf '%s\n' "$rc" >"$RUN_DIR/exit"
  if [[ "$OWNS_LATEST" == 1 ]]; then printf '%s\n' "$rc" >"$ROOT/control/$PREFIX.exit"; fi
  printf 'resume200_exit=%s utc=%s\n' "$rc" "$(date -u +%FT%TZ)"
  exit "$rc"
}
on_signal() {
  trap '' TERM INT HUP
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    set +e; wait "$CHILD_PID"; exit "$?"
  fi
  exit 143
}
trap on_exit EXIT
trap on_signal TERM INT HUP
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export CUDA_VISIBLE_DEVICES=4,5 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="$REPO/z-image:$REPO"
PYTHON="$ROOT/envs/rdm/bin/python"
cd "$REPO"
exec 9>/data/rdm-workspace/rdm-run.lock
flock -n 9 || { echo 'Training lock is held; no duplicate resume'; exit 3; }
if [[ -f "$ROOT/control/$PREFIX.pid" ]]; then
  prior="$(cat "$ROOT/control/$PREFIX.pid")"
  if kill -0 "$prior" 2>/dev/null; then echo 'Resume already active'; exit 4; fi
fi
for file in pid exit command.json preflight.json; do
  if [[ -f "$ROOT/control/$PREFIX.$file" ]]; then mv "$ROOT/control/$PREFIX.$file" "$RUN_DIR/previous-$file"; fi
done
printf '%s\n' "$$" >"$ROOT/control/$PREFIX.pid"
OWNS_LATEST=1
"$PYTHON" - "$REPO" "$EXP" "$ARCHIVE" "$ARCHIVE_CONTROL" "$PREFIX" <<'PY'
import pathlib,json,os,sys,time,hashlib,yaml,subprocess
repo,exp,archive,archive_control=map(pathlib.Path,sys.argv[1:5]);prefix=sys.argv[5]
root=pathlib.Path('/data/rdm-workspace/rdm-sets');control=root/'control'
stop=json.loads((control/'twohead-resume200-stop-verified.json').read_text())
assert stop['passed'] and stop['classification']=='intentional-user-authorized-handoff-not-training-failure'
old_process=pathlib.Path('/proc/1677394/cmdline')
assert not old_process.exists() or not old_process.read_bytes(), 'Old launcher still live'
audit=json.loads((control/'twohead-resume200-checkpoint-audit.json').read_text());assert audit['passed'] and audit['step']==200
assert (exp/'resume.pth').stat().st_size==audit['bytes']
assert (exp/'resume.pth').stat().st_mtime_ns==audit['mtime_ns']
cfg=yaml.safe_load((repo/'z-image/configs/zimage_2step_1024_twohead_glr10_h20_fa3.yaml').read_text())
assert cfg['steps']==500 and cfg['resume_every']==50 and cfg['save_resume']
assert cfg['lr']==2.83e-5 and cfg['gan']['lr']==1e-5 and cfg['gan']['loss']=='ns'
assert cfg['memory']['micro_batch']==2 and cfg['memory']['optimizer']=='adamw8bit'
assert cfg['encoders']==['dinov3_l','siglip2','aimv2_huge'] and cfg['spatial_gan']['global_gan_encoders']==[]
old=pathlib.Path('/data/rdm-workspace/RDM-twohead-lr10-v1')
for subtree in ['rdm','z-image/zimage_distill']:
 for p in (old/subtree).rglob('*.py'):
  relative=p.relative_to(old)
  if str(relative)=='rdm/sw_lmmd/launch.py':continue
  assert p.read_bytes()==(repo/relative).read_bytes(), 'Algorithm change outside saving:'+str(relative)
assert str(exp.resolve()).startswith(str(root)+'/' ) and str(archive).startswith('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints/')
archive.mkdir(parents=True,exist_ok=True)
archive_proof=archive_control/'checkpoint-archive-200.json'
if archive_proof.exists():
 saved=json.loads(archive_proof.read_text())
 assert saved['passed'] and pathlib.Path(saved['destination']).stat().st_size==saved['bytes']
else:
 probe=archive/'resume200-write-test.json';probe.write_text('{"passed":true}\n');assert json.loads(probe.read_text())['passed']
sys.path.insert(0,str(repo/'ops'))
from h20_benchmark_acceptance import wait_for_idle_gpus
def read_gpus():
 return [list(map(int,x.split(','))) for x in subprocess.check_output(['nvidia-smi','-i','4,5','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
idle=wait_for_idle_gpus(read_gpus,time.sleep,time.monotonic)
assert all(memory<=32 and util==0 for _,memory,total,util in idle)
command=[str(root/'envs/rdm/bin/torchrun'),'--standalone','--nproc-per-node=2','-m','zimage_distill.launch','z-image/configs/zimage_2step_1024_twohead_glr10_h20_fa3.yaml','--set','memory.micro_batch=2','--set','save_freq=50','--set','resume_from='+str(exp/'resume.pth'),'--set','load_from='+str(exp/'resume.pth'),'--set','checkpoint_archive_dir='+str(archive),'--set','checkpoint_archive_control='+str(archive_control)]
receipt=dict(source=str(repo),source_commit=(repo/'SOURCE_COMMIT').read_text().strip(),command=command,
 created_epoch=time.time(),formal_training=True,physical_gpus=[4,5],steps=500,resume_from_step=200,
 prior_source=str(old),prior_source_commit='a25e4c42794b13d5c881d3f1994944929b019190',
 log=str(root/'launch-logs'/f'{prefix}.log'),archive_dir=str(archive),weights_every=50,resume_every=50,
 G_Adam_reset_disclosed=True,critics_controller_cache_schedule_preserved=True)
(control/f'{prefix}.command.json').write_text(json.dumps(receipt,indent=2)+'\n')
(control/f'{prefix}.preflight.json').write_text(json.dumps(dict(passed=True,idle_gpus=idle,checkpoint_step=200,archive_dir=str(archive),all_caches_remote=True),indent=2)+'\n')
print(json.dumps(receipt),flush=True)
PY
if [[ ! -f "$EXP/step_0000200.pth" && ! -f "$ARCHIVE_CONTROL/checkpoint-archive-200.json" ]]; then
  CUDA_VISIBLE_DEVICES='' "$PYTHON" "$REPO/ops/h20-export-student200.py"
fi
ARCHIVE_ACTIVE=0
if [[ -f "$ARCHIVE_CONTROL/checkpoint-archive-worker.pid" ]]; then
  prior="$(cat "$ARCHIVE_CONTROL/checkpoint-archive-worker.pid")"
  if kill -0 "$prior" 2>/dev/null; then
    "$PYTHON" - "$prior" "$EXP" "$ARCHIVE" <<'PY'
import pathlib,sys
cmd=pathlib.Path('/proc/'+sys.argv[1]+'/cmdline').read_bytes().split(b'\0')
assert b'rdm.sw_lmmd.checkpoint_archive' in cmd and sys.argv[2].encode() in cmd and sys.argv[3].encode() in cmd
PY
    ARCHIVE_ACTIVE=1
  fi
fi
if [[ "$ARCHIVE_ACTIVE" == 0 ]]; then
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 setsid "$PYTHON" -m rdm.sw_lmmd.checkpoint_archive \
  --source-dir "$EXP" --archive-dir "$ARCHIVE" --control-dir "$ARCHIVE_CONTROL" \
  >>"$ROOT/launch-logs/twohead-retain50-archive.log" 2>&1 < /dev/null 9>&- &
printf '%s\n' "$!" >"$ARCHIVE_CONTROL/checkpoint-archive-worker.pid"
fi
"$ROOT/envs/rdm/bin/torchrun" --standalone --nproc-per-node=2 \
  -m zimage_distill.launch "$CONFIG" --set memory.micro_batch=2 --set save_freq=50 \
  --set "resume_from=$EXP/resume.pth" --set "load_from=$EXP/resume.pth" --set "checkpoint_archive_dir=$ARCHIVE" \
  --set "checkpoint_archive_control=$ARCHIVE_CONTROL" &
CHILD_PID=$!
printf '%s\n' "$CHILD_PID" >"$RUN_DIR/torchrun.pid"
wait "$CHILD_PID"
