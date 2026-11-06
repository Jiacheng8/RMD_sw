#!/usr/bin/env python3
"""User-authorized exact checkpoint list; protect second-training500 and all source/data/logs."""
import json
import os
from pathlib import Path
import time

ROOT=Path('/data/rdm-workspace/rdm-sets')
C=ROOT/'control'
EXP='zimage-2step-1024-dino-qwenvl-mmd2-from0'
ARCHIVE=Path('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints')/EXP
OBJECT=Path('/mnt/airs-business/rdm-experiments')
RECEIPT=C/'delete-prior-except-mmd2-500-20261106.json'
SHA500='7a189cf104b9c4b7f09ef7e27b090e1824ca520a35c9626ff734bba2d47bae32'
OLD=Path('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A/step_0000350.pth')


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.partial');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)


def main():
    if RECEIPT.exists():
        result=json.loads(RECEIPT.read_text());assert result['passed'],'Partial cleanup needs inspection'
        print(json.dumps(result));return
    final=json.loads((C/'dino-qwenvl-mmd2-from0-final-integrity.json').read_text())
    assert final['passed'] and final['actual_formal_exit']==0 and final['final_student_SHA']==SHA500
    keep={ARCHIVE/'step_0000500.pth':24619908172,
          OBJECT/'checkpoints'/EXP/'resume-step500.pth':31720122291}
    saved={}
    for p,size in keep.items():
        st=p.stat();assert st.st_size==size and not p.is_symlink()
        with p.open('rb') as f:assert f.read(2)==b'PK','Invalid Torch ZIP header'
        saved[str(p)]=[st.st_size,st.st_mtime_ns]
    alias=ROOT/'zimage/work_dirs'/EXP/'resume.pth'
    assert alias.is_symlink() and alias.resolve()==OBJECT/'checkpoints'/EXP/'resume-step500.pth'
    candidates=[]
    for step in [250,300,350,400,450]:
        p=ARCHIVE/f'step_{step:07d}.pth'
        proof=json.loads((C/f'dino-qwenvl-mmd2-from0-archive/checkpoint-archive-{step}.json').read_text())
        assert proof['passed'] and proof['step']==step and proof['destination']==str(p)
        assert proof['final_named_sha_readback'] and not p.is_symlink() and p.stat().st_size==proof['bytes']
        candidates.append(dict(path=str(p),step=step,bytes=proof['bytes'],sha256=proof['sha256']))
    assert not OLD.is_symlink() and OLD.stat().st_size==24619908172
    candidates.append(dict(path=str(OLD),step=350,bytes=OLD.stat().st_size,
        sha256='6428309c6fd2af6b1e1588428b44545524ee5d066c07bdaf1fdcb5dc87bb663b'))
    # Only guard actual open checkpoint files, not stale command text or history.
    wanted={r['path'] for r in candidates}
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name)==os.getpid():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:target=os.readlink(fd)
                except OSError:continue
                assert target not in wanted,f'Checkpoint still open by PID{proc.name}: {target}'
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
    result=dict(passed=False,authorization='User: delete prior training weights except second training500',
        source=str(Path(__file__).resolve()),epoch=time.time(),protected=saved,
        protected_student_sha256=SHA500,files=candidates,deleted=[])
    write(RECEIPT,result)
    for row in candidates:
        p=Path(row['path']);local=ROOT/'zimage/work_dirs'/p.parent.name/p.name
        if local.is_symlink():assert local.resolve()==p;local.unlink()
        p.unlink();assert not p.exists();result['deleted'].append(str(p));write(RECEIPT,result)
    for p,expected in saved.items():
        st=Path(p).stat();assert [st.st_size,st.st_mtime_ns]==expected
    result.update(passed=True,deleted_bytes=sum(r['bytes'] for r in candidates),completed_epoch=time.time())
    write(RECEIPT,result)
    write(OBJECT/'manifests/delete-prior-except-mmd2-500-20261106.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
