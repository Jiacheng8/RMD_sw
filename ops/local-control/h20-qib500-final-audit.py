#!/usr/bin/env python3
"""Read-only final audit through actual judge exits, official scores and denominators.

Run using ROOT/qib/env/bin/python, CUDA_VISIBLE_DEVICES='', TMPDIR on/data.
Raw judge outputs and images never leave the remote host; print only small audit.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import time

ROOT=Path('/data/rdm-workspace/rdm-sets');C=ROOT/'control';PREFIX='qib-mmd2-s500-gpu4567-v1'
RUN=ROOT/'qib/runs/mmd2-from0-s500-zh-1024-seed0-v1'

def read(p):return json.loads(p.read_text())

def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

assert read(C/(PREFIX+'.exit'))==0
completion=read(C/(PREFIX+'-complete.json'));assert completion['passed'] and completion['checkpoint_step']==500
command=read(C/(PREFIX+'-command.json'));source=Path(command['source'])
assert (source/'SOURCE_COMMIT').read_text().strip()==command['source_commit']
assert completion['checkpoint_sha256']=='7a189cf104b9c4b7f09ef7e27b090e1824ca520a35c9626ff734bba2d47bae32'
for rank in range(4):assert read(RUN/'logs'/f'generate_{rank}.exit.json')['actual_exit']==0
for name in ['split','merge','check-thinking','report','comparison-report']:
    assert read(RUN/'logs'/f'{name}.exit.json')['actual_exit']==0
for part in range(8):
    exits=list((RUN/'logs').glob(f'judge_part{part}_*.exit.json'))
    assert any(read(p)['actual_exit']==0 for p in exits)
    assert read(RUN/'logs'/f'part{part}-thinking.exit.json')['actual_exit']==0
metadata=read(ROOT/'qib/bench/metadata/bench_metadata.json');wanted={int(r['ID']) for r in metadata}
proof=read(RUN/'generation-verified.json');assert len(proof['images'])==1000
for row in proof['images']:
    p=RUN/'images'/f"{row['ID']:04d}.png";b=p.read_bytes()
    assert hashlib.sha256(b).hexdigest()==row['sha256'] and len(b)==row['bytes']
    assert b[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',b[16:24])==(1024,1024)
assert {r['ID'] for r in proof['images']}==wanted
sys.path[:0]=[str(ROOT/'qib/bench'),str(source/'z-image')]
import judge
from score_utils import compute_dimension_score,aggregate_total_score
utility=module('qib500_final_parallel',source/'z-image/eval_qib/judge_parallel.py')

def audit(run):
    rows=[json.loads(s) for s in (run/'input_judged.jsonl').read_text().splitlines() if s.strip()]
    assert len(rows)==1000 and len({int(r['ID']) for r in rows})==1000 and {int(r['ID']) for r in rows}==wanted
    actual=utility.bench_scores(rows,judge);stored=read(run/'input_bench_scores.json');assert actual==stored
    raw,without=utility.thinking_stats(rows)
    parsed=0;total_valid=0;l1=Counter();l2={};empty=0
    for row in rows:
        value=row.get('judge_model_output');value=json.loads(value) if isinstance(value,str) and value else value or {}
        empty+=not bool(value);parsed+=len(value)
        dim={k:compute_dimension_score(v) for k,v in value.items() if v is not None}
        total_valid+=aggregate_total_score(dim) is not None
        for k,v in dim.items():
            l1[k]+=v['level1_score'] is not None
            for sub,score in v['level2_scores'].items():l2.setdefault(k,Counter())[sub]+=score is not None
    return dict(run=run.name,rows=1000,raw_answers=raw,parsed_L1=parsed,raw_without_thinking=without,
        empty_parsed_images=empty,total_effective_denominator=total_valid,L1_effective_denominators=dict(l1),
        L2_effective_denominators={k:dict(v) for k,v in l2.items()},scores=stored,official_aggregation_recomputed=True)

current=audit(RUN);assert current['raw_answers']==4105
others=[audit(ROOT/'qib/runs'/name) for name in ['turbo2-zh-1024-seed0-v1','turbo9-zh-1024-seed0-v1','twohead-lr10-A-s350-zh-v1']]
archive=completion['preservation'];p=Path(archive['archive']);assert p.stat().st_size==archive['bytes']
h=hashlib.sha256()
with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
assert h.hexdigest()==archive['sha256']
cleanup=read(C/'user-mmd2-delete-50-100-150-200-20261106.json');assert cleanup['passed']
assert all(not Path(v['path']).exists() for v in cleanup['files'])
for path,identity in cleanup['preserved'].items():assert Path(path).stat().st_size==identity[0]
print(json.dumps(dict(passed=True,epoch=time.time(),actual_wrapper_exit=0,checkpoint_step=500,
    checkpoint_sha256=completion['checkpoint_sha256'],source_commit=command['source_commit'],
    current=current,comparisons=others,preservation=archive,deleted_steps=cleanup['deleted'],retained_steps=cleanup['retained_steps'],
    deleted_object_bytes=cleanup['deleted_object_bytes'],official_protocol='Chinese1000;1024;seed0;thinkingTrue;unchanged official aggregation')))
