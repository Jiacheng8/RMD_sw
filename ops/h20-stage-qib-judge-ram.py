#!/usr/bin/env python3
"""Stage the offloaded Q-Judger shard in remote tmpfs before safetensors mmap.

The persistent verified source stays on same-cluster object storage. Never
change ROOT/qib/judge, active training assets, or any Mac cache.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path('/data/rdm-workspace/rdm-sets')
REVISION = '1b77ff83564ac4e4e8140769eacff7ff73f35f3c'
DEST = Path('/dev/shm')/('rdm-qib-judge-'+REVISION)
NAME = 'model-00001-of-00002.safetensors'
SIZE = 49825162976
SHA = 'cd93ba04b9dad1af628ecffdcdce16554621cae6ef6f00334c2baf28316aff3e'
PROGRESS = ROOT/'control/qib-turbo-baselines-judge-ram-progress.json'


def write(path, value):
    temp = path.with_name(path.name+f'.partial-{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def stage():
    original = ROOT/'qib/judge'
    source = (original/NAME).resolve()
    assert str(source).startswith('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints/qjudger-')
    assert source.stat().st_size == SIZE
    mem_available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:'))
    assert mem_available > 128*1024**3, 'Keep enough system RAM for active training and judging'
    DEST.mkdir(mode=0o700, exist_ok=True)
    assert DEST.stat().st_uid == os.getuid() and not DEST.is_symlink()
    final = DEST/NAME
    started = time.time()
    if not final.exists():
        assert shutil.disk_usage(DEST).free > SIZE+4*1024**3
        temp = DEST/(NAME+f'.partial-{os.getpid()}')
        copied = 0
        reported = -1
        digest = hashlib.sha256()
        with source.open('rb') as inp, temp.open('xb') as out:
            for block in iter(lambda: inp.read(8*1024**2), b''):
                assert out.write(block) == len(block)
                digest.update(block)
                copied += len(block)
                if copied//(512*1024**2) != reported:
                    reported = copied//(512*1024**2)
                    write(PROGRESS, dict(phase='sequential_object_to_remote_tmpfs', bytes=copied,
                        total_bytes=SIZE, epoch=time.time(), source=str(source), destination=str(final)))
            out.flush()
            os.fsync(out.fileno())
        assert copied == SIZE and digest.hexdigest() == SHA
        temp.replace(final)
    assert not final.is_symlink() and final.stat().st_size == SIZE
    digest = hashlib.sha256()
    checked = 0
    with final.open('rb') as inp:
        for block in iter(lambda: inp.read(8*1024**2), b''):
            digest.update(block)
            checked += len(block)
    assert checked == SIZE and digest.hexdigest() == SHA
    for path in original.iterdir():
        if path.name in (NAME, '.cache'):
            continue
        link = DEST/path.name
        if link.exists() or link.is_symlink():
            assert link.is_symlink() and link.resolve() == path.resolve()
        else:
            link.symlink_to(path)
    receipt = dict(passed=True, model=str(DEST), persistent_source=str(source), bytes=SIZE,
        sha256=SHA, final_named_readback=True, only_remote_tmpfs=True, original_model_unchanged=True,
        seconds=time.time()-started, completed_epoch=time.time())
    write(ROOT/'control/qib-turbo-baselines-judge-ram-verified.json', receipt)
    write(PROGRESS, dict(phase='verified', bytes=SIZE, total_bytes=SIZE, epoch=time.time()))
    print(json.dumps(receipt), flush=True)
    return DEST


if __name__ == '__main__':
    stage()
