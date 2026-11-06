#!/usr/bin/env python3
"""Actual1000 ID image coverage, completed official judge batches and real exits."""
import datetime
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

ROOT=Path('/data/rdm-workspace/rdm-sets')
C=ROOT/'control';PREFIX='qib-mmd2-s500-gpu4567-v1'
RUN=ROOT/'qib/runs/mmd2-from0-s500-zh-1024-seed0-v1'

def read(p):return json.loads(p.read_text()) if p.exists() else None

def live(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return None

metadata=read(ROOT/'qib/bench/metadata/bench_metadata.json')
expected={int(r['ID']) for r in metadata};assert len(expected)==1000
images=[];bad=[]
for ident in expected:
    p=RUN/'images'/f'{ident:04d}.png'
    if p.exists():
        with p.open('rb') as f:b=f.read(24)
        if len(b)==24 and b[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',b[16:24])==(1024,1024):images.append(ident)
        else:bad.append(ident)
logs=[]
for p in sorted((RUN/'logs').glob('*.log')):
    lines=p.read_text(errors='replace').splitlines()
    pairs=[tuple(map(int,m.groups())) for s in lines if 'Batch inference' in s for m in [re.search(r'\b(\d+)/(\d+)\b',s)] if m]
    command=read(p.with_suffix('.command.json'))
    logs.append(dict(name=p.name,actual_exit=read(p.with_suffix('.exit.json')),
        completed_batches=max([a for a,b in pairs],default=0),total_batches=pairs[-1][1] if pairs else None,
        pid=command['pid'] if command else None,live_command=live(command['pid']) if command else None,
        errors=[s for s in lines if any(e in s for e in ['Traceback (most recent last)','Traceback (most recent call last)','OutOfMemoryError','CUDA out of memory','ChildFailedError','Bus error'])][-4:],tail=lines[-2:]))
command=read(C/(PREFIX+'-command.json'))
wrapper=read(C/(PREFIX+'.pid'))
value=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,wrapper_pid=wrapper,
    wrapper_live=live(wrapper) if wrapper else None,status=read(C/(PREFIX+'-status.json')),actual_exit=read(C/(PREFIX+'.exit')),
    failure=read(C/(PREFIX+'-failure.json')),expected_images=1000,actual_valid_expected_images=len(images),bad_image_headers=bad,
    generators=[read(RUN/f'generate-rank{k}-progress.json') for k in range(4)],
    student_RAM_copy=read(C/'qib-mmd2-s500-student-ram-progress.json'),student_RAM_verified=read(C/'qib-mmd2-s500-student-ram-verified.json'),
    completed_judge_parts=[k for k in range(8) if (RUN/'shards'/f'part_{k}_judged.jsonl').exists()],logs=logs,
    scores_complete=read(C/(PREFIX+'-scores-complete.json')),completion=read(C/(PREFIX+'-complete.json')),
    GPU4_7=subprocess.check_output(['nvidia-smi','-i','4,5,6,7','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15),
    disk_free_bytes=shutil.disk_usage(ROOT).free)
print(json.dumps(value,ensure_ascii=False))
