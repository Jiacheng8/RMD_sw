#!/usr/bin/env python3
"""Read source-bound MMD2 process, completed updates, saves/exits; no model/tensor loads.

Execute via the existing h20-recovery-ssh.py on stdin. No launch/restart/mutation.
"""
import datetime
import json
import math
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/data/rdm-workspace/rdm-sets')
C=ROOT/'control'
PREFIX='dino-qwenvl-mmd2-from0'
EXP=ROOT/'zimage/work_dirs/zimage-2step-1024-dino-qwenvl-mmd2-from0'


def read(path):
    return json.loads(path.read_text().replace('/data/shaoshitong/', '/data/rdm-workspace/')) if path.exists() else None


def live(pid):
    p=Path(f'/proc/{pid}/cmdline')
    try:return p.read_bytes().replace(b'\0',b' ').decode() or None
    except FileNotFoundError:return None


def complete_rows(path):
    if not path.exists():return []
    raw=path.read_bytes();raw=raw[:raw.rfind(b'\n')+1]
    return [json.loads(s) for s in raw.splitlines() if s.strip()]


def safe(value):
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,list):return [safe(v) for v in value]
    return value


command=read(C/'dino-qwenvl-mmd2-handoff-command.json');source=Path(command['source'])
assert (source/'SOURCE_COMMIT').read_text().strip()==command['source_commit']
status=read(C/(PREFIX+'-status.json'))
assert status is None or status['source_commit']==command['source_commit']
rows=complete_rows(EXP/'train_log.jsonl');numeric_errors=[]
def numeric(value,path='record'):
    if isinstance(value,float) and not math.isfinite(value):numeric_errors.append(path)
    elif isinstance(value,dict):
        for k,v in value.items():numeric(v,path+'/'+k)
    elif isinstance(value,list):
        for i,v in enumerate(value):numeric(v,path+'/'+str(i))
for row in rows:numeric(row,'step'+str(row['step']))
summary=read(C/(PREFIX+'-smoke-latest.json'));smoke=[];case=None
if summary:
    assert summary['source_commit']==command['source_commit']
    case=summary['cases'][-1] if summary['cases'] else None
    if case:
        for line in Path(case['log']).read_text(errors='replace').splitlines():
            try:v=json.loads(line)
            except json.JSONDecodeError:continue
            if 'ranks' in v:
                smoke.append(dict(iteration=v['ranks'][0]['iteration'],ranks=[dict(rank=x['rank'],seconds=x['seconds'],
                    grad_norm=x['logs'].get('grad_norm'),skipped=x['logs'].get('skipped'),D_updates=x.get('critic_updates_after')) for x in v['ranks']]))
stages={}
for name in ['cpu','smoke','formal','archive']:
    cmd=read(C/(PREFIX+'-'+name+'.command.json'))
    if cmd:assert cmd['source_commit']==command['source_commit']
    stages[name]=dict(command=cmd,live_command=live(cmd['pid']) if cmd else None,
        actual_exit=read(C/(PREFIX+'-'+name+'.exit.json')))
archive=read(C/(PREFIX+'-archive/checkpoint-archive-progress.json'))
resume=EXP/'resume.pth'
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    pipeline=dict(wrapper_pid=command['pid'],live_command=live(command['pid']),source=str(source),source_commit=command['source_commit'],actual_exit=read(C/(PREFIX+'.exit'))),
    status=status,stages=stages,completed_formal_iterations=len(rows),
    formal_step_ids_contiguous=[r['step'] for r in rows]==list(range(1,len(rows)+1)),
    G_skips=[r['step'] for r in rows if r.get('skipped') is not False],
    D_skips=[r['step'] for r in rows if any(r.get(f's{k}/gan/d_skipped') is not False for k in [1,2])],
    nonfinite_numeric_paths=numeric_errors,last_formal=rows[-1] if rows else None,
    formal_last10=[{k:r[k] for k in ['step','seconds','grad_norm','lr','loss_total','force','gan/d_lr_used','gan/d_lr_next'] if k in r} for r in rows[-10:]],
    smoke_iterations=smoke,smoke_status=summary['status'] if summary else None,
    smoke_verified=read(C/(PREFIX+'-smoke-verified.json')),
    historically_verified_weights=archive['verified_steps'] if archive else [],
    retained_weights=[s for s in archive['verified_steps'] if Path(read(C/(PREFIX+f'-archive/checkpoint-archive-{s}.json'))['destination']).is_file()] if archive else [],
    rolling_resume=dict(bytes=resume.stat().st_size,mtime_ns=resume.stat().st_mtime_ns) if resume.exists() else None,
    training_complete=read(C/(PREFIX+'-training-complete.json')),
    final_integrity=read(C/(PREFIX+'-final-integrity.json')),
    finalizer_status=read(C/'qib350-mmd2-finalization-status.json'),finalizer_exit=read(C/'qib350-mmd2-finalization.exit'),
    GPU4_5=subprocess.check_output(['nvidia-smi','-i','4,5','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15),
    disk_free_bytes=shutil.disk_usage(ROOT).free)
if out['pipeline']['actual_exit'] not in (None,0):
    failure=read(C/(PREFIX+'-failure.json'));out['failure']=failure
print(json.dumps(safe(out),allow_nan=False))
