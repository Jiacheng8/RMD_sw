#!/usr/bin/env python3
"""Read-only actual baseline image IDs, completed judge batches, and real exits."""
import datetime
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

ROOT = Path('/data/rdm-workspace/rdm-sets')
CONTROL = ROOT/'control'
PREFIX = 'qib-turbo2-turbo9-gpu67'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def alive(pid):
    try: return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError: return None


deployment = read(CONTROL/(PREFIX+'-deployment.json'))
expected = {int(r['ID']) for r in read(ROOT/'qib/bench/metadata/bench_metadata.json')}
runs = []
for steps in [2,9]:
    run = ROOT/'qib/runs'/f'turbo{steps}-zh-1024-seed0-v1'
    images = []
    for ident in expected:
        path = run/'images'/f'{ident:04d}.png'
        if path.exists():
            with path.open('rb') as stream: header = stream.read(24)
            assert header[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', header[16:24]) == (1024,1024)
            images.append(ident)
    logs = []
    for path in sorted((run/'logs').glob('*.log')):
        lines = path.read_text(errors='replace').splitlines()
        batches = [line for line in lines if 'Batch inference' in line and re.search(r'\b\d+/\d+\b', line)]
        counts = [tuple(map(int, re.search(r'\b(\d+)/(\d+)\b', line).groups())) for line in batches]
        completed = max((done for done, total in counts), default=0)
        errors = [line for line in lines if any(x in line for x in
            ['Traceback (most recent call last)', 'OutOfMemoryError', 'CUDA out of memory', 'ChildFailedError'])]
        command = read(path.with_suffix('.command.json'))
        logs.append(dict(name=path.name, actual_exit=read(path.with_suffix('.exit.json')),
            pid=command['pid'] if command else None,
            live_command=alive(command['pid']) if command else None,
            completed_batch_lines=batches[-3:], completed_batch_events=completed,
            total_batches=counts[-1][1] if counts else None,
            errors=errors[-3:], tail=lines[-3:]))
    runs.append(dict(teacher_steps=steps, actual_expected_1024_images=len(images),
        generator_progress=[read(run/f'generate-rank{k}-progress.json') for k in [0,1]],
        judged_parts=[k for k in range(8) if (run/'shards'/f'part_{k}_judged.jsonl').exists()],
        logs=logs, completion=read(CONTROL/(run.name+'-complete.json')),
        preservation=read(CONTROL/(run.name+'-preserved.json'))))
result = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), deployment=deployment,
    wrapper_live=alive(deployment['wrapper_pid']) if deployment else None,
    status=read(CONTROL/(PREFIX+'-status.json')), actual_exit=read(CONTROL/(PREFIX+'.exit')),
    failure=read(CONTROL/(PREFIX+'-failure.json')), completion=read(CONTROL/(PREFIX+'-complete.json')),
    runs=runs, GPU6_7=subprocess.check_output(['nvidia-smi','-i','6,7',
        '--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True, timeout=15),
    disk_free_bytes=shutil.disk_usage(ROOT).free)
print(json.dumps(result, ensure_ascii=False))
