#!/usr/bin/env python3
"""Preserve a verified cached judge shard remotely, release its local bytes after evaluation.

The active judge path stays untouched through full benchmark completion. The
49.8GB copy and final-named SHA readback run concurrently on CPU/shared storage.
Only after actual benchmark exit0 and8 successful judge parts is the original
local file replaced with its verified remote symlink. No weights are deleted.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path('/data/rdm-workspace/rdm-sets')
CONTROL=ROOT/'control'
PREFIX='dino-qwenvl-mmd2-judge-cache-offload'
NAME='model-00001-of-00002.safetensors'
SOURCE=ROOT/'qib/judge'/NAME
DEST=Path('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints/qjudger-1b77ff83564ac4e4e8140769eacff7ff73f35f3c')/NAME
SIZE=49_825_162_976
SHA='cd93ba04b9dad1af628ecffdcdce16554621cae6ef6f00334c2baf28316aff3e'


def write(path,value):
    p=path.with_name(path.name+f'.partial-{os.getpid()}')
    p.write_text(json.dumps(value,indent=2)+'\n');p.replace(path)


def progress(phase,n):
    value=dict(phase=phase,actual_bytes=n,total_bytes=SIZE,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(CONTROL/f'{PREFIX}-progress.json',value);print(json.dumps(value),flush=True)


def verify(path):
    h=hashlib.sha256();n=0;reported=-1
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block);n+=len(block)
            if n//1024**3!=reported:reported=n//1024**3;progress('final_named_SHA_readback',n)
    assert n==SIZE and h.hexdigest()==SHA,'Final shared judge shard differs from official source'


def main():
    lock=(CONTROL/f'{PREFIX}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (CONTROL/f'{PREFIX}.pid').write_text(str(os.getpid())+'\n')
    receipt_path=CONTROL/f'{PREFIX}-verified.json'
    if SOURCE.is_symlink():
        receipt=json.loads(receipt_path.read_text());assert receipt['passed'] and SOURCE.resolve()==DEST.resolve()
        assert DEST.stat().st_size==SIZE;return
    official=json.loads((CONTROL/'qib350-judge-verified.json').read_text())
    entry=next(f for f in official['files'] if Path(f['path']).name==NAME)
    assert official['passed'] and entry['bytes']==SIZE and entry['digest']==SHA
    assert json.loads((ROOT/'qib/runs/twohead-lr10-A-s350-zh-v1/generation-verified.json').read_text())['passed']
    before=SOURCE.stat();stamp=(before.st_ino,before.st_size,before.st_mtime_ns)
    assert before.st_size==SIZE and before.st_nlink==1,'Inspect hardlinks before promising local capacity'
    DEST.parent.mkdir(parents=True,exist_ok=True)
    if not DEST.exists():
        temporary=DEST.with_name(DEST.name+f'.partial-{os.getpid()}')
        h=hashlib.sha256();n=0;reported=-1
        with SOURCE.open('rb') as inp,temporary.open('xb') as out:
            for block in iter(lambda:inp.read(8*1024**2),b''):
                out.write(block);h.update(block);n+=len(block)
                if n//1024**3!=reported:reported=n//1024**3;progress('copy_inactive_destination_only_source_untouched',n)
            out.flush();os.fsync(out.fileno())
        assert n==SIZE and h.hexdigest()==SHA and temporary.stat().st_size==SIZE
        assert not DEST.exists();temporary.replace(DEST)
    verify(DEST)
    write(DEST.with_suffix('.verified.json'),dict(passed=True,sha256=SHA,bytes=SIZE,source=str(SOURCE),destination=str(DEST),local_source_untouched=True))
    progress('waiting_full_evaluation_exit0_before_local_link',SIZE)
    while True:
        exit_path=CONTROL/'qib350-two-gpu.exit';complete_path=CONTROL/'qib350-two-gpu-complete.json'
        if exit_path.exists() and exit_path.read_text().strip()!='0':
            raise RuntimeError('Benchmark failed; preserve active judge cache and inspect original exit/log')
        if exit_path.exists() and exit_path.read_text().strip()=='0' and complete_path.exists():
            complete=json.loads(complete_path.read_text())
            if complete['passed'] and complete['images']==1000 and complete['judge_parts']==8 and complete['actual_raw_dimension_answers']==4105:break
        time.sleep(5)
    run=ROOT/'qib/runs/twohead-lr10-A-s350-zh-v1'
    for command in (run/'logs').glob('judge_part*_attempt*.command.json'):
        data=json.loads(command.read_text());pid=data['pid']
        proc=Path(f'/proc/{pid}/cmdline')
        live=proc.read_bytes() if proc.exists() else b''
        assert b'h20-qib350-judge-part.py' not in live,'A judge subprocess is still attached to local weights'
    after=SOURCE.stat();assert stamp==(after.st_ino,after.st_size,after.st_mtime_ns)
    assert DEST.stat().st_size==SIZE
    link=SOURCE.with_name(SOURCE.name+f'.object-link-{os.getpid()}');link.symlink_to(DEST)
    link.replace(SOURCE)
    assert SOURCE.is_symlink() and SOURCE.resolve()==DEST.resolve() and SOURCE.stat().st_size==SIZE
    receipt=dict(passed=True,bytes=SIZE,sha256=SHA,source=str(SOURCE),destination=str(DEST),
        actual_benchmark_exit=0,local_path_is_verified_symlink=True,freed_local_payload_bytes=SIZE,
        disk_free_bytes=shutil.disk_usage(ROOT).free,completed_epoch=time.time())
    write(receipt_path,receipt);write(DEST.with_suffix('.verified.json'),receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:traceback.print_exc();code=1
    finally:(CONTROL/f'{PREFIX}.exit').write_text(str(code)+'\n')
    raise SystemExit(code)
