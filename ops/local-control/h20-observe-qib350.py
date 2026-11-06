#!/usr/bin/env python3
"""Read-only remote QIB350 evidence. Formal ID PNGs and completed judge batches only."""
import datetime
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

ROOT = Path('/data/rdm-workspace/rdm-sets')
RUN = ROOT/'qib/runs/twohead-lr10-A-s350-zh-v1'
CONTROL = ROOT/'control'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def tail(path, limit=200000):
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-limit))
        return stream.read().decode(errors='replace')


def live(pid):
    path = Path(f'/proc/{pid}/cmdline')
    return path.read_bytes().replace(b'\0',b' ').decode()[:500] if path.exists() else None


def main():
    command = read(CONTROL/'qib350-two-gpu-command.json')
    pid_path = CONTROL/'qib350-two-gpu.pid'
    pid = int(pid_path.read_text()) if pid_path.exists() else None
    exit_path = CONTROL/'qib350-two-gpu.exit'
    actual_exit = int(exit_path.read_text()) if exit_path.exists() else None
    meta = ROOT/'qib/bench/metadata/bench_metadata.json'
    expected = {int(r['ID']) for r in json.loads(meta.read_text())} if meta.exists() else set()
    actual, invalid = set(), []
    for path in (RUN/'images').glob('*.png'):
        match = re.fullmatch(r'(\d{4})\.png',path.name)
        if not match or int(match[1]) not in expected:
            continue
        with path.open('rb') as image:
            header = image.read(24)
        if header[:8] == b'\x89PNG\r\n\x1a\n' and len(header)==24 and struct.unpack('>II',header[16:24])==(1024,1024):
            actual.add(int(match[1]))
        else:
            invalid.append(path.name)
    stages = {}
    for name in ('assets','environment','generate_0','generate_1','merge','check-thinking','report'):
        c = read(RUN/'logs'/f'{name}.command.json')
        log = RUN/'logs'/f'{name}.log'
        stages[name] = dict(pid=c['pid'] if c else None,
            live_command=live(c['pid']) if c else None, actual_exit=read(RUN/'logs'/f'{name}.exit.json'),
            log_bytes=log.stat().st_size if log.exists() else 0)
        if name.startswith('generate'):
            events=[]
            for line in tail(log,50000).splitlines():
                try:
                    value=json.loads(line)
                except ValueError:
                    continue
                if isinstance(value,dict) and 'event' in value:
                    events.append(value)
            stages[name]['latest_actual_event']=events[-1] if events else None
    assets_text = tail(RUN/'logs/assets.log')
    aria = re.findall(r'\[#[^\r\n]*?\d+(?:\.\d+)?[KMGT]?i?B/[^\r\n]*?\]',assets_text)
    weights = [dict(name=p.name,logical_bytes=p.stat().st_size,allocated_bytes=p.stat().st_blocks*512)
               for p in (ROOT/'qib/judge').glob('*.download')]
    generation = [read(RUN/f'generate-rank{k}-progress.json') for k in (0,1)]
    judge_attempts = []
    completed = []
    for p in sorted((RUN/'logs').glob('judge_part*_attempt*.log')):
        match=re.fullmatch(r'judge_part(\d+)_gpu([45])_b(\d+)_attempt(\d+)\.log',p.name)
        if not match:
            continue
        text=tail(p)
        batches=re.findall(r'Batch inference:[^\r\n]*?(\d+)/(\d+)',text)
        name=p.stem
        e=read(p.with_suffix('.exit.json'))
        c=read(p.with_suffix('.command.json'))
        judge_attempts.append(dict(part=int(match[1]),gpu=int(match[2]),batch=int(match[3]),
            actual_exit=e, live_command=live(c['pid']) if c else None,
            completed_batches=int(batches[-1][0]) if batches else 0,
            total_batches=int(batches[-1][1]) if batches else None,
            memory=read(p.with_suffix('.memory.json')),log=str(p)))
        result=RUN/'shards'/f'part_{int(match[1])}_judged.jsonl'
        thinking=read(RUN/'logs'/f'{name}-thinking.exit.json')
        if e and e['actual_exit']==0 and thinking and thinking['actual_exit']==0 and result.exists():
            completed.append(int(match[1]))
    complete = read(CONTROL/'qib350-two-gpu-complete.json')
    full = bool(actual_exit==0 and complete and complete['passed'] and complete['checkpoint_step']==350
        and complete['images']==1000 and complete['actual_raw_dimension_answers']==4105
        and len(actual)==len(expected)==1000 and len(set(completed))==8 and not invalid)
    print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pipeline=dict(wrapper_pid=pid, live_command=live(pid) if pid else None,
            actual_exit=actual_exit, source_commit=command.get('code_commit') if command else None),
        scope='latest manual stop418 then fullQIB350; never restart old500 training',
        expected_images=len(expected),actual_valid_ID_PNGs=len(actual),invalid_PNGs=invalid,
        generation=generation,stages=stages,judge_attempts=judge_attempts,
        completed_judge_parts=sorted(set(completed)),assets_actual_aria2_progress=aria[-1:] if aria else [],
        assets_sparse_files=weights,assets_verified=read(CONTROL/'qib350-assets-complete.json'),
        judge_environment_verified=read(ROOT/'qib/env/.ready'),
        GPU4_5=subprocess.check_output(['nvidia-smi','-i','4,5',
            '--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip(),
        disk_free_bytes=shutil.disk_usage(ROOT).free,completion=complete,full_benchmark_completion_proven=full),ensure_ascii=False))


if __name__ == '__main__':
    main()
