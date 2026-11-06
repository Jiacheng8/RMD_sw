#!/usr/bin/env bash
# Resume the approved frozen A snapshot with adaptive D LR; full target remains500.
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=z-image/configs/zimage_2step_1024_spatial_patchgan_adaptive_h20_fa3.yaml
RESUME_FROM="${1:?Pass the approved frozen checkpoint path}"
RESUME_STEP="${2:?Pass its actual saved step}"
export RESUME_FROM RESUME_STEP
CONTROL="$ROOT/control"
mkdir -p "$CONTROL" "$ROOT/launch-logs"
exec >>"$ROOT/launch-logs/spatial-A-500-adaptive.log" 2>&1
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="$CONTROL/spatial-A-500-adaptive-runs/$RUN_ID"
mkdir -p "$RUN_DIR"
OWNS_LATEST=0
CHILD_PID=""
on_exit() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" >"$RUN_DIR/exit"
  if [[ "$OWNS_LATEST" == 1 ]]; then
    printf '%s\n' "$rc" >"$CONTROL/spatial-A-500-adaptive.exit.tmp-$$"
    mv "$CONTROL/spatial-A-500-adaptive.exit.tmp-$$" "$CONTROL/spatial-A-500-adaptive.exit"
  fi
  printf 'formal_A_exit=%s run=%s utc=%s\n' "$rc" "$RUN_ID" "$(date -u +%FT%TZ)"
  exit "$rc"
}
on_signal() {
  sig="$1"; fallback="$2"
  trap '' TERM INT HUP
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -s "$sig" "$CHILD_PID" 2>/dev/null || true
    set +e
    wait "$CHILD_PID"
    rc=$?
    exit "$rc"
  fi
  exit "$fallback"
}
trap on_exit EXIT
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT
trap 'on_signal HUP 129' HUP
printf 'formal_A_launch run=%s source=%s config=%s utc=%s\n' "$RUN_ID" "$REPO" "$CONFIG" "$(date -u +%FT%TZ)"
source "$REPO/ops/h20-data-env.sh"
source "$ROOT/env.sh"
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
PYTHON="$ROOT/envs/rdm/bin/python"
cd "$REPO"
exec 9>/data/rdm-workspace/rdm-run.lock
flock -n 9 || { echo 'Another task owns the H20 run lock; no formal launch'; exit 3; }
if [[ -f "$CONTROL/spatial-A-500-adaptive.pid" ]]; then
  previous="$(cat "$CONTROL/spatial-A-500-adaptive.pid")"
  if [[ "$previous" =~ ^[0-9]+$ ]] && kill -0 "$previous" 2>/dev/null; then
    echo 'Existing spatial-A-500-adaptive launcher PID is active; no duplicate launch'
    exit 4
  fi
fi
for previous in pid exit command.json preflight.json; do
  if [[ -f "$CONTROL/spatial-A-500-adaptive.$previous" ]]; then
    mv "$CONTROL/spatial-A-500-adaptive.$previous" "$RUN_DIR/previous-$previous"
  fi
done
printf '%s\n' "$$" >"$CONTROL/spatial-A-500-adaptive.pid"
printf '%s\n' "$$" >"$RUN_DIR/launcher.pid"
OWNS_LATEST=1
"$PYTHON" - "$ROOT" "$CONFIG" "$RUN_DIR" "$CONTROL" "$RESUME_FROM" "$RESUME_STEP" <<'PY'
import json, os, pathlib, sys, time
import yaml
root, config, run, control = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
frozen, expected_step = pathlib.Path(sys.argv[5]).resolve(strict=True), int(sys.argv[6])
freeze_receipt=json.loads((frozen.parent/'frozen.json').read_text())
assert freeze_receipt['checkpoint']==str(frozen) and freeze_receipt['step']==expected_step
assert freeze_receipt['accept_generator_optimizer_reset'] is True
assert 0 < expected_step < 500 and expected_step % 50 == 0
assert frozen.name != 'resume.pth', 'Both readers must use the same frozen snapshot'
assert frozen.is_relative_to(root.resolve(strict=True)/'control/spatial-adaptive-migrations')
assert sys.platform.startswith('linux'), 'Formal H20 launch is Linux-only'
root = root.resolve(strict=True)
cfg = yaml.safe_load(config.read_text())
assert not cfg.get('extends'), 'Preflight expects the published standalone A config'
assert cfg['steps'] == 500 and cfg['spatial_gan']['enabled'] is True
assert cfg['student']['num_steps'] == 2 and cfg['student']['step_weights'] == [1.0, 1.0]
assert cfg['gan']['enabled'] is True and cfg['gan']['g_start_step'] == 50
assert cfg['gan']['loss'] == 'ns' and float(cfg['gan']['lr']) == 1e-5
assert float(cfg['lr']) == 2.83e-6
assert cfg['resume_every'] == 50 and cfg['save_freq'] == 500 and cfg['save_resume'] is True
assert not cfg.get('resume_from') and not cfg.get('load_from')
assert cfg['adaptive_d_lr']['enabled'] is True
import torch
state=torch.load(frozen,map_location='cpu',weights_only=False,mmap=True)
assert state['step']==expected_step and state['num_steps']==2
assert all(k in state for k in ('model','schedule','cache','spatial_cache','critics'))
assert frozen.stat().st_size==freeze_receipt['checkpoint_bytes']
del state
def remote(path):
    value = pathlib.Path(path)
    assert value.is_absolute(), 'All data/output/cache paths must be absolute'
    resolved = value.resolve()
    assert os.path.commonpath([str(root), str(resolved)]) == str(root), 'Path escapes remote RDM_ROOT'
    return str(resolved)
paths = {name: remote(cfg[name]) for name in ('output_dir', 'reference_root')}
paths.update({name: remote(cfg['spatial_gan'][name]) for name in ('reference_images', 'qwen_snapshot')})
for name in ('HF_HOME', 'TORCH_HOME', 'XDG_CACHE_HOME', 'TMPDIR', 'TRITON_CACHE_DIR',
             'CUDA_CACHE_PATH', 'TORCHINDUCTOR_CACHE_DIR', 'PIP_CACHE_DIR'):
    assert os.environ.get(name), 'Missing explicit remote cache path:' + name
    paths[name] = remote(os.environ[name])
assert os.environ['CUDA_VISIBLE_DEVICES'] == '4,5'
experiment = pathlib.Path(paths['output_dir']) / cfg['exp_name']
paths['experiment'] = remote(experiment)
listing = sorted(p.name for p in experiment.iterdir()) if experiment.exists() else []
print(json.dumps({'event':'formal_A_output_preflight', 'experiment':str(experiment), 'existing_entries':listing}), flush=True)
rows=[json.loads(s) for s in (experiment/'train_log.jsonl').read_text().splitlines() if s.strip()]
assert [r['step'] for r in rows]==list(range(1,expected_step+1)), 'Must preserve exactly the frozen prefix before append'
paths['frozen_checkpoint']=str(frozen)
receipt = {'checked_epoch':time.time(), 'config':str(config.resolve()), 'steps':500,
           'gan_g_start_step':cfg['gan']['g_start_step'], 'resume_every':50, 'weights_every':500,
           'physical_gpus':[4,5], 'paths':paths, 'existing_entries':listing,
           'resume_step':expected_step,'accept_generator_optimizer_reset':True,
           'adaptive_controller_bootstrap':'explicit legacy migration or restored state'}
for destination in (run/'preflight.json', control/'spatial-A-500-adaptive.preflight.json'):
    destination.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
PY
"$PYTHON" - <<'PY'
import subprocess, sys, time
sys.path.insert(0, 'ops')
from h20_benchmark_acceptance import wait_for_idle_gpus
def read_state():
    rows = subprocess.check_output(['nvidia-smi','-i','4,5',
        '--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    return [list(map(int,row.split(','))) for row in rows.splitlines()]
state=wait_for_idle_gpus(read_state,time.sleep,time.monotonic)
assert all(memory<=32 and utilization==0 for _,memory,total,utilization in state), state
print({'formal_A_idle_gpus':state}, flush=True)
PY
"$PYTHON" - "$REPO" "$CONFIG" "$RUN_DIR" "$CONTROL" <<'PY'
import json,pathlib,sys,time,os
repo,config,run,control = map(pathlib.Path,sys.argv[1:])
root = pathlib.Path('/data/rdm-workspace/rdm-sets')
command=[str(root/'envs/rdm/bin/torchrun'),'--standalone','--nproc-per-node=2',
         '-m','zimage_distill.launch',str(config),'--set','load_from='+os.environ['RESUME_FROM'],
         '--set','resume_from='+os.environ['RESUME_FROM'],'--set','adaptive_d_lr_allow_legacy_resume=true']
receipt={'created_epoch':time.time(),'source':str(repo), 'source_commit':(repo/'SOURCE_COMMIT').read_text().strip(),
         'command':command,'log':str(root/'launch-logs/spatial-A-500-adaptive.log'),'formal_training':True,
         'setting':'A','steps':500,'physical_gpus':[4,5]}
for path in (run/'command.json',control/'spatial-A-500-adaptive.command.json'):
    path.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(json.dumps(receipt,sort_keys=True),flush=True)
PY
"$ROOT/envs/rdm/bin/torchrun" --standalone --nproc-per-node=2 \
  -m zimage_distill.launch "$CONFIG" \
  --set "load_from=$RESUME_FROM" --set "resume_from=$RESUME_FROM" \
  --set adaptive_d_lr_allow_legacy_resume=true &
CHILD_PID=$!
printf '%s\n' "$CHILD_PID" >"$RUN_DIR/torchrun.pid"
wait "$CHILD_PID"
