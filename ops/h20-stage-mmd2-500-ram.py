#!/usr/bin/env python3
"""Stage only the exact student500 into remote tmpfs for safe shared mmap inference."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path('/data/rdm-workspace/rdm-sets')
DEST=Path('/dev/shm/rdm-mmd2-student500')
SHA='7a189cf104b9c4b7f09ef7e27b090e1824ca520a35c9626ff734bba2d47bae32'

def write(p,v):
    t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)

def stage():
    proof=json.loads((ROOT/'control/dino-qwenvl-mmd2-from0-archive/checkpoint-archive-500.json').read_text())
    assert proof['passed'] and proof['final_named_sha_readback'] and proof['step']==500 and proof['sha256']==SHA
    source=Path(proof['destination'])
    assert source.name=='step_0000500.pth' and source.parent.name=='zimage-2step-1024-dino-qwenvl-mmd2-from0'
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    assert available>256*1024**3
    DEST.mkdir(mode=0o700,exist_ok=True);assert not DEST.is_symlink() and DEST.stat().st_uid==os.getuid()
    target=DEST/'step_0000500.pth';start=time.time()
    if not target.exists():
        assert shutil.disk_usage(DEST).free>proof['bytes']+64*1024**3
        temp=DEST/(target.name+'.partial-'+str(os.getpid()));h=hashlib.sha256();done=0;reported=-1
        with source.open('rb') as inp,temp.open('xb') as out:
            for b in iter(lambda:inp.read(8*1024**2),b''):
                assert out.write(b)==len(b);h.update(b);done+=len(b)
                if done//1024**3!=reported:
                    reported=done//1024**3
                    write(ROOT/'control/qib-mmd2-s500-student-ram-progress.json',dict(bytes=done,total=proof['bytes'],epoch=time.time()))
            out.flush();os.fsync(out.fileno())
        assert done==proof['bytes'] and h.hexdigest()==SHA;temp.replace(target)
    assert not target.is_symlink() and target.stat().st_size==proof['bytes']
    h=hashlib.sha256()
    with target.open('rb') as inp:
        for b in iter(lambda:inp.read(8*1024**2),b''):h.update(b)
    assert h.hexdigest()==SHA
    value=dict(passed=True,model=str(target),source=str(source),step=500,sha256=SHA,bytes=proof['bytes'],
        final_named_readback=True,remote_tmpfs_only=True,seconds=time.time()-start,epoch=time.time())
    write(ROOT/'control/qib-mmd2-s500-student-ram-verified.json',value)
    print(json.dumps(value),flush=True);return target

if __name__=='__main__':stage()
