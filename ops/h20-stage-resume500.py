#!/usr/bin/env python3
"""Stage the archived step500 resume onto an ordinary remote filesystem before mmap.

No training starts. The existing FSDP snapshot's G-Adam-history limitation remains.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path('/data/rdm-workspace/rdm-sets')
C=ROOT/'control/storage-relocation-20261106'
receipt=json.loads((C/'files/resume-step500.json').read_text())
assert receipt['passed'] and receipt['final_named_SHA_readback']
source=Path(receipt['destination']);assert source.is_relative_to('/mnt/airs-business/rdm-experiments/checkpoints')
target=ROOT/'staging/resume-step500.pth';target.parent.mkdir(exist_ok=True)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

if not target.exists():
    assert shutil.disk_usage(target.parent).free>receipt['bytes']+64*1024**3, 'Keep64GiB node headroom'
    temporary=target.with_suffix('.partial');assert not temporary.is_symlink()
    h=hashlib.sha256()
    with source.open('rb') as inp,temporary.open('wb') as out:
        for b in iter(lambda:inp.read(8*1024**2),b''):
            assert out.write(b)==len(b);h.update(b)
        out.flush();os.fsync(out.fileno())
    assert temporary.stat().st_size==receipt['bytes'] and h.hexdigest()==receipt['sha256']
    temporary.replace(target)
assert not target.is_symlink() and target.stat().st_size==receipt['bytes'] and sha(target)==receipt['sha256']
print(json.dumps(dict(passed=True,local_remote_path=str(target),sha256=receipt['sha256'],bytes=receipt['bytes'],
    no_training_started=True,generator_Adam_history_not_saved_by_original_snapshot=True)))
