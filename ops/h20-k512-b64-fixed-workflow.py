#!/usr/bin/env python3
"""Source-bound successor: CPU checks, reference/storage/idle GPU gates, real smoke, fresh0 train500.

This never restarts the old A, loads its weights, changes an active runtime,
retries failed training, or touches GPUs other than6/7. All artifacts stay remote.
"""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--group', choices=['mmd-only','mmd-gan'], default='mmd-gan')
ARGS=parser.parse_args()
WITH_GAN=ARGS.group=='mmd-gan'
ROOT=Path('/data/rdm-workspace/rdm-sets')
CODE=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'control'
PREFIX='dino-qwenvl-k512-b64-'+('fixed' if WITH_GAN else 'mmd-only')+'-from0'
STATE=CONTROL/(PREFIX+'-status.json')
CONFIG=CODE/'z-image/configs'/('zimage_2step_1024_dino_qwenvl_k512_b64_'+('fixed' if WITH_GAN else 'mmd_only')+'_from0.yaml')
REFERENCE=ROOT/'zimage/zimage-ref-v1/reference_store_dino_qwenvl'
EXP=ROOT/'zimage/work_dirs'/('zimage-2step-1024-'+PREFIX)
ARCHIVE=Path('/mnt/airs-business/rdm-experiments/checkpoints')/EXP.name
ARCHIVE_CONTROL=CONTROL/(PREFIX+'-archive')
PYTHON=ROOT/'envs/rdm/bin/python'
TORCHRUN=ROOT/'envs/rdm/bin/torchrun'
CHILDREN={}
SOURCE=None  # Bound to the immutable deployment at main(), before any launch.
OWNS=False
NAMES=['dinov3_l','qwen3vl_4b']


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.partial-{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def status(phase,**kwargs):
    value=dict(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(CODE),source_commit=SOURCE,physical_GPUs=[6,7],experiment_group=ARGS.group,**kwargs)
    write(STATE,value);print(json.dumps(value),flush=True)


def gpu():
    return [list(map(int,x.split(','))) for x in subprocess.check_output(['nvidia-smi','-i','6,7',
        '--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],
        text=True,timeout=15).splitlines()]


def spawn(name,command,gpus=''):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpus
    log=ROOT/'launch-logs'/(PREFIX+'-'+name+'.log')
    with log.open('ab') as stream:
        p=subprocess.Popen([str(x) for x in command],cwd=CODE,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    p.name,p.log=name,log;CHILDREN[p.pid]=p
    write(CONTROL/(PREFIX+'-'+name+'.command.json'),dict(command=[str(x) for x in command],pid=p.pid,
        source=str(CODE),source_commit=SOURCE,CUDA_VISIBLE_DEVICES=gpus,log=str(log),started_epoch=time.time()))
    return p


def record_exit(p):
    code=p.wait();write(CONTROL/(PREFIX+'-'+p.name+'.exit.json'),dict(actual_exit=code,completed_epoch=time.time()))
    CHILDREN.pop(p.pid,None);return code


def require_success(p):
    code=record_exit(p)
    if code:
        text=p.log.read_text(errors='replace')
        write(CONTROL/(PREFIX+'-failure.json'),dict(stage=p.name,actual_exit=code,last30=text.splitlines()[-30:],
            command=json.loads((CONTROL/(PREFIX+'-'+p.name+'.command.json')).read_text()),
            GPU6_7=gpu(),disk_free_bytes=shutil.disk_usage(ROOT).free))
        raise RuntimeError(f'{p.name} failed with actual exit{code}; command/last30/GPU/disk preserved')


def check_failed_dependency(prefix):
    p=CONTROL/(prefix+'.exit')
    if p.exists() and p.read_text().strip()!='0':
        raise RuntimeError(f'{prefix} has real exit{p.read_text().strip()}; preserve outputs, inspect original logs')


def ready():
    ref=json.loads((CONTROL/'dino-qwenvl-mmd2-reference-verified.json').read_text())
    assert ref['passed'] and ref['rows']==125808 and ref['MMD_encoders']==NAMES
    assert ref['raw_image_features_finite'] and ref['text_norm_verified']
    free=shutil.disk_usage(ROOT).free
    if free<100_000_000_000:return False,dict(waiting='100GB checkpoint staging gate',disk_free_bytes=free)
    return True,dict(disk_free_bytes=free,reference_rows=ref['rows'])


def preflight():
    import yaml
    cfg=yaml.safe_load(CONFIG.read_text())
    assert cfg['gan']['enabled'] is WITH_GAN and cfg['spatial_gan']['enabled'] is WITH_GAN
    assert cfg['exp_name']==EXP.name
    assert cfg['encoders']==NAMES and cfg['spatial_gan']['global_gan_encoders']==[]
    assert cfg['steps']==500 and cfg['save_freq']==cfg['resume_every']==50 and cfg['save_resume']
    assert cfg['lr']==2.83e-5 and cfg['gan']['lr']==1e-5 and cfg['gan']['loss']=='ns'
    assert cfg['gan']['g_start_step']==50 and not cfg['adaptive_d_lr']['enabled']
    assert not cfg['gan']['adaptive'] and cfg['gan']['weight']==5e-5
    assert cfg['window']['size']==512 and cfg['window']['stride']==64
    assert cfg['lr_schedule']['min_lr_ratio']==cfg['discriminator_lr_schedule']['min_lr_ratio']==.5
    assert cfg['gan']['grad_clip']==4
    assert cfg['memory']['micro_batch']==2 and cfg['active_global_batch']==64
    assert cfg['zimage']['checkpoint_group_size']==2 and cfg['zimage']['attention_backend']=='_flash_3_varlen_hub'
    assert cfg['grad_clip']==4. and 'gradient_telemetry' not in cfg
    assert cfg['reference_root']==str(REFERENCE) and cfg['output_dir']==str(ROOT/'zimage/work_dirs')
    assert not cfg.get('load_from') and not cfg.get('resume_from')
    assert not EXP.exists() or not list(EXP.iterdir()),'Preserve existing successor output; never silently restart'
    for name in ['HF_HOME','TORCH_HOME','XDG_CACHE_HOME','TMPDIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH',
                 'TORCHINDUCTOR_CACHE_DIR','PIP_CACHE_DIR','RDM_QWEN3VL_SNAPSHOT','RDM_QWEN3VL_SPATIAL_SNAPSHOT']:
        path=Path(os.environ[name]).resolve();assert path.is_relative_to(ROOT.resolve()),name
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6,7'
    ARCHIVE.mkdir(parents=True,exist_ok=True);ARCHIVE_CONTROL.mkdir(exist_ok=True)
    assert not list(ARCHIVE.glob('step_*.pth')),'Existing successor archive requires deliberate recovery'
    probe=ARCHIVE/f'.write-test-{os.getpid()}';probe.write_text(SOURCE);assert probe.read_text()==SOURCE;probe.unlink()
    write(CONTROL/(PREFIX+'-preflight.json'),dict(passed=True,source_commit=SOURCE,init='official pretrained0',
        MMD_encoders=NAMES,GAN_encoders=NAMES if WITH_GAN else [],steps=500,weights_every=50,expected_milestones=list(range(50,501,50)),
        all_data_cache_outputs_remote=True,global_batch=64,micro_batch=2,grad_accum=16,
        G_base_lr=2.83e-5,D_base_lr=1e-5,original_grad_norm_only=True))


def finite(value):
    if isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,list):
        for v in value:finite(v)
    elif isinstance(value,(int,float)):assert math.isfinite(value),'Nonfinite formal metric'


def stop(signum,frame):
    for p in list(CHILDREN.values()):
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    raise SystemExit(128+signum)


def main():
    global SOURCE,OWNS
    SOURCE=(CODE/'SOURCE_COMMIT').read_text().strip()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    lock=(CONTROL/(PREFIX+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists(),'Existing controller status must be inspected before starting another attempt'
    OWNS=True
    (CONTROL/(PREFIX+'.pid')).write_text(str(os.getpid())+'\n')
    status('CPU_checks_before_handoff')
    tests=['tests/test_pixel_grad_caching.py','tests/test_fixed_k512_config.py',
        'z-image/tests/test_spatial_trainer.py','z-image/tests/test_adaptive_d_lr_distributed.py',
        'tests/test_benchmark_acceptance.py','tests/test_qwen3_vl_backbone.py','tests/test_checkpoint_archive.py','tests/test_mmd2_handoff.py',
        'z-image/tests/test_spatial_critic_runtime.py',
        'z-image/tests/test_qwen3vl_reference_builder.py','z-image/tests/test_qwen_text_store.py']
    cpu=spawn('cpu',[PYTHON,'-m','pytest','-q',*tests]);require_success(cpu)
    last=None
    while True:
        passed,evidence=ready()
        if passed:break
        serialized=json.dumps(evidence,sort_keys=True)
        if serialized!=last:status('waiting_evaluation_reference_storage',**evidence);last=serialized
        time.sleep(5)
    # Serialize this task's workflows; foreign GPU processes are never modified.
    runlock=Path('/data/rdm-workspace/rdm-run.lock').open('a')
    status('waiting_owned_run_lock',**evidence)
    fcntl.flock(runlock,fcntl.LOCK_EX)
    preflight()
    from h20_benchmark_acceptance import wait_for_idle_gpus,validate_benchmark_record
    while True:
        idle=wait_for_idle_gpus(gpu,time.sleep,time.monotonic)
        if all(mem<=32 and util==0 for _,mem,total,util in idle):break
        status('waiting_GPU6_7',GPU6_7=idle)
        time.sleep(30)
    status('GPU6_7_idle_verified',GPU6_7=idle)
    status('real_twoGPU_DG_smoke' if WITH_GAN else 'real_twoGPU_MMD_smoke')
    smoke=spawn('smoke',[PYTHON,CODE/'ops/h20-checkpoint-sweep.py','--only-case','2','2','--gpus','6,7','--case-timeout-seconds','3600',
        '--config',CONFIG,'--reference-root',REFERENCE,'--state-prefix',PREFIX+'-smoke'],'6,7')
    require_success(smoke)
    summary=json.loads((CONTROL/(PREFIX+'-smoke-latest.json')).read_text());case=summary['cases'][-1]
    assert summary['source_commit']==SOURCE and case['status']=='passed'
    record=json.loads(Path(case['result']).read_text())
    validate_benchmark_record(record,expected_spatial=WITH_GAN,expected_iterations=3,expected_heads=NAMES if WITH_GAN else None,
        expected_probe_request=64,expected_retained_rows=448)
    assert record['MMD_encoders']==NAMES and record['source_commit']==SOURCE
    assert record['G_base_lr']==2.83e-5 and record['D_base_lr']==1e-5 and not record['adaptive_D_lr']
    assert record['global_batch']==64 and record['generator_gan_active'] is WITH_GAN
    for ranks in record['steps']:
        for row in ranks:
            if WITH_GAN:assert row['logs']['pixel_gc/balance_mode']=='fixed'
            else:assert not any('/gan/' in key for key in row['logs'])
            assert row['logs']['active_rows']==64
            assert row['logs']['active_rows_per_rank']==32
    assert max(case['nvml_peak_mib'].values())<min(r[2] for r in gpu())*.95
    for ranks in record['steps']:
        for row in ranks:
            logs=row['logs']
            assert not any('siglip' in k or 'aimv2' in k for k in logs)
            assert all(logs[f's{k}/qwen3vl_4b/mmd2']>=0 for k in [1,2])
    if not WITH_GAN:
        assert len(record['probe_refresh'])==2
        for rank in record['probe_refresh']:
            assert rank['stats']['drift_probe_rows']==64
            assert rank['stats']['refreshed_rows']==448
    write(CONTROL/(PREFIX+'-smoke-verified.json'),dict(passed=True,source_commit=SOURCE,
        benchmark=str(case['result']),MMD_encoders=NAMES,three_complete_updates=True,GAN_active=WITH_GAN,
        steady_seconds_excluding_first=case['mean_seconds'],GPU_peak_MiB=case['nvml_peak_mib']))
    # Smoke lives in a separate process. Formal launch reloads official pretrained0;
    # no state, optimizer, cache, RNG cursor or warmup updates are carried over.
    preflight()
    archive=spawn('archive',[PYTHON,'-m','rdm.sw_lmmd.checkpoint_archive','--source-dir',EXP,
        '--archive-dir',ARCHIVE,'--control-dir',ARCHIVE_CONTROL,'--first-step','50','--last-step','500',
        '--interval','50','--verify-final-named'])
    (ARCHIVE_CONTROL/'checkpoint-archive-worker.pid').write_text(str(archive.pid)+'\n')
    status('formal_training_fresh0',steps=500,expected_milestones=list(range(50,501,50)))
    formal=spawn('formal',[TORCHRUN,'--standalone','--nproc-per-node=2','-m','zimage_distill.launch',CONFIG,
        '--set','checkpoint_archive_dir='+str(ARCHIVE),'--set','checkpoint_archive_control='+str(ARCHIVE_CONTROL)],'6,7')
    while formal.poll() is None:
        if archive.poll() is not None:
            if archive.pid in CHILDREN:require_success(archive)
            assert (ARCHIVE_CONTROL/'checkpoint-archive-complete.json').exists(),'Archive exited before retained weights complete'
        p=EXP/'train_log.jsonl';lastrow=None
        if p.exists():
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-131072));lines=f.read().splitlines()
            if lines:
                try:lastrow=json.loads(lines[-1])
                except json.JSONDecodeError:pass
        if lastrow and (not STATE.exists() or json.loads(STATE.read_text()).get('completed_iteration')!=lastrow['step']):
            finite(lastrow);assert lastrow.get('skipped') is False
            status('formal_training_progress',completed_iteration=lastrow['step'],grad_norm=lastrow.get('grad_norm'),
                disk_free_bytes=shutil.disk_usage(ROOT).free)
        time.sleep(5)
    require_success(formal)
    status('training_exit0_waiting_final_archive')
    if archive.pid in CHILDREN:require_success(archive)
    complete=json.loads((ARCHIVE_CONTROL/'checkpoint-archive-complete.json').read_text())
    assert complete['passed'] and [r['step'] for r in complete['weights']]==list(range(50,501,50))
    assert all(r['final_named_sha_readback'] for r in complete['weights'])
    lines=[json.loads(x) for x in (EXP/'train_log.jsonl').read_text().splitlines() if x.strip()]
    assert [r['step'] for r in lines]==list(range(1,501))
    for row in lines:finite(row);assert row['skipped'] is False
    write(CONTROL/(PREFIX+'-training-complete.json'),dict(passed=True,steps=500,actual_training_exit=0,
        all10_weights_final_named_SHA_verified=True,archive_dir=str(ARCHIVE),source_commit=SOURCE,
        reference_root=str(REFERENCE),MMD_encoders=NAMES,init='official pretrained0',completed_epoch=time.time(),
        remaining_final_acceptance='Inspect latest full resume500 model/controller/cache identity; final curves and user report'))
    status('training500_and_weights_complete_pending_final_resume_audit',completed_iteration=500)


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        traceback.print_exc();code=1
        try:
            if OWNS:status('failed_preserve_outputs',GPU6_7=gpu(),disk_free_bytes=shutil.disk_usage(ROOT).free)
        except Exception:pass
    finally:
        for p in list(CHILDREN.values()):
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            try:p.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
            record_exit(p)
        if OWNS:(CONTROL/(PREFIX+'.exit')).write_text(str(code)+'\n')
    raise SystemExit(code)
