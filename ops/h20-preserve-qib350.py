#!/usr/bin/env python3
"""After fullQIB350 exit0, preserve the complete benchmark run to shared storage.

No source removal, no Mac datasets/images, no model loads. Source files are
immutable after the exact original pipeline exits successfully. Only small
acceptance/scores/progress receipts are read back to the Mac.
"""
import hashlib
import gzip
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tarfile
import time

ROOT=Path('/data/rdm-workspace/rdm-sets')
CONTROL=ROOT/'control'
RUN=ROOT/'qib/runs/twohead-lr10-A-s350-zh-v1'
DEST=Path('/mnt/airs-business/shaoshitong/rdm-sets/evaluations')/(RUN.name+'.tar.gz')
PREFIX='qib350-full-preservation'


def write(path,value):
    temp=path.with_name(path.name+f'.partial-{os.getpid()}')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def finite_scores(value):
    import math
    if isinstance(value,dict):
        for x in value.values():finite_scores(x)
    elif isinstance(value,list):
        for x in value:finite_scores(x)
    elif isinstance(value,(int,float)):assert math.isfinite(value),'Nonfinite official benchmark score'


class HashWriter:
    def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.bytes=0;self.reported=-1
    def write(self,b):
        n=self.f.write(b);assert n==len(b);self.h.update(b);self.bytes+=n
        if self.bytes//1024**3!=self.reported:
            self.reported=self.bytes//1024**3
            write(CONTROL/(PREFIX+'-progress.json'),dict(phase='streaming_compressed_full_run',actual_bytes=self.bytes,epoch=time.time()))
        return n
    def flush(self):self.f.flush()


@contextmanager
def archive_stream(fileobj):
    # Explicit gzip supports both local Python3.9 and remote Python3.12;
    # tarfile's streaming mode does not consistently accept compresslevel.
    with gzip.GzipFile(fileobj=fileobj,mode='wb',compresslevel=1,mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w|') as tar:yield tar


def preserve():
    assert (CONTROL/'qib350-two-gpu.exit').read_text().strip()=='0'
    completion=json.loads((CONTROL/'qib350-two-gpu-complete.json').read_text())
    assert completion['passed'] and completion['checkpoint_step']==350
    assert completion['images']==1000 and completion['judge_parts']==8 and completion['actual_raw_dimension_answers']==4105
    source=json.loads((CONTROL/'qib350-two-gpu-command.json').read_text())
    assert source['code_commit']=='31bbce8253dfee657a6399e5e23d8946e9d65bd1'
    assert source['physical_GPUs']==[4,5]
    assert json.loads((RUN/'generation-verified.json').read_text())['passed']
    for part in range(8):
        commands=list((RUN/'logs').glob(f'judge_part{part}_gpu*_attempt*-thinking.command.json'))
        assert len(commands)==1,'Require exactly one successful thinking check per part'
        command=commands[0];stem=command.name.removesuffix('.command.json')
        assert json.loads(command.with_name(stem+'.exit.json').read_text())['actual_exit']==0
        judged=[json.loads(s) for s in (RUN/'shards'/f'part_{part}_judged.jsonl').read_text().splitlines() if s.strip()]
        assert len(judged)==125
    merged=[json.loads(s) for s in (RUN/'input_judged.jsonl').read_text().splitlines() if s.strip()]
    assert len(merged)==1000 and len({int(x['ID']) for x in merged})==1000
    assert sum(1 for r in merged for k,v in r.items() if k.endswith('_judge_output') and isinstance(v,str) and v)==4105
    score=json.loads((RUN/'input_bench_scores.json').read_text());assert score==completion['scores'];finite_scores(score)
    assert (RUN/'report.md').stat().st_size>0
    for name in ['merge','check-thinking','report']:
        assert json.loads((RUN/'logs'/(name+'.exit.json')).read_text())['actual_exit']==0
    files=[]
    for path in sorted(RUN.rglob('*')):
        if path.is_file():
            assert not path.is_symlink(),'Inspect external benchmark links before preservation'
            st=path.stat();files.append((path,st.st_size,st.st_mtime_ns))
    DEST.parent.mkdir(parents=True,exist_ok=True)
    assert not DEST.exists(),'Existing benchmark archive must be verified, not overwritten'
    temp=DEST.with_name(DEST.name+f'.partial-{os.getpid()}')
    with temp.open('xb') as f:
        hashed=HashWriter(f)
        with archive_stream(hashed) as tar:
            for path,size,mtime in files:
                tar.add(path,arcname=str(Path(RUN.name)/path.relative_to(RUN)),recursive=False)
                st=path.stat();assert (st.st_size,st.st_mtime_ns)==(size,mtime),'Source run changed during archival'
        hashed.flush();os.fsync(f.fileno());expected=hashed.h.hexdigest();size=hashed.bytes
    assert temp.stat().st_size==size;temp.replace(DEST)
    actual=hashlib.sha256();n=0;reported=-1
    with DEST.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            actual.update(block);n+=len(block)
            if n//1024**3!=reported:
                reported=n//1024**3;write(CONTROL/(PREFIX+'-progress.json'),dict(phase='final_named_SHA_readback',actual_bytes=n,total_bytes=size,epoch=time.time()))
    assert n==size and actual.hexdigest()==expected
    receipt=dict(passed=True,checkpoint_step=350,images=1000,judge_parts=8,actual_raw_dimension_answers=4105,
        source_commit=source['code_commit'],source_run=str(RUN),archive=str(DEST),archive_bytes=size,archive_sha256=expected,
        files=len(files),source_uncompressed_bytes=sum(r[1] for r in files),source_retained=True,
        images_without_parsed_judgement=completion['images_without_parsed_judgement'],scores=score,completed_epoch=time.time())
    write(CONTROL/(PREFIX+'-verified.json'),receipt);write(DEST.with_suffix('.verified.json'),receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    preserve()
