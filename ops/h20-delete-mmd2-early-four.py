#!/usr/bin/env python3
"""Delete only explicitly requested new-MMD2 steps50/100/150/200; retain all proofs."""
import json
import os
from pathlib import Path
import time

ROOT=Path('/data/rdm-workspace/rdm-sets')
EXP='zimage-2step-1024-dino-qwenvl-mmd2-from0'
ARCHIVE=Path('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints')/EXP
DELETE=[50,100,150,200]
KEEP=[250,300,350,400,450,500]
C=ROOT/'control'
RECEIPT=C/'user-mmd2-delete-50-100-150-200-20261106.json'


def write(p,v):
    t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)


def main():
    final=json.loads((C/'dino-qwenvl-mmd2-from0-final-integrity.json').read_text())
    assert final['passed'] and final['step']==500 and final['actual_formal_exit']==0
    assert final['final_student_SHA']=='7a189cf104b9c4b7f09ef7e27b090e1824ca520a35c9626ff734bba2d47bae32'
    assert not RECEIPT.exists(),'Cleanup already recorded; inspect instead of silently repeating'
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name)==os.getpid():continue
        try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert not ('mmd2' in cmd and ('complex-preview.py' in cmd or 'ten-stage-preview.py' in cmd)), 'A preview might still read early checkpoints'
    preserved={}
    paths=[ARCHIVE/f'step_{s:07d}.pth' for s in KEEP]
    paths += [ROOT/'zimage/work_dirs'/EXP/'resume.pth',Path('/mnt/airs-business/rdm-experiments/checkpoints')/EXP/'resume-step500.pth',
        Path('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A/step_0000350.pth')]
    for p in paths:st=p.stat();preserved[str(p)]=[st.st_size,st.st_mtime_ns]
    files=[]
    for s in DELETE:
        p=ARCHIVE/f'step_{s:07d}.pth';proof=json.loads((C/f'dino-qwenvl-mmd2-from0-archive/checkpoint-archive-{s}.json').read_text())
        assert proof['step']==s and proof['destination']==str(p) and proof['passed'] and proof['final_named_sha_readback']
        assert p.parent==ARCHIVE and not p.is_symlink() and p.stat().st_size==proof['bytes']
        local=ROOT/'zimage/work_dirs'/EXP/p.name
        assert not local.exists() or local.is_symlink() and local.resolve()==p
        files.append(dict(step=s,path=str(p),bytes=proof['bytes'],sha256=proof['sha256'],local_alias=str(local)))
    result=dict(authorization='User: delete new training checkpoints50,100,150,200; evaluate current500',
        experiment=EXP,epoch=time.time(),files=files,preserved=preserved,deleted=[],passed=False)
    write(RECEIPT,result)
    for row in files:
        p=Path(row['path']);local=Path(row['local_alias'])
        if local.is_symlink():local.unlink()
        p.unlink();assert not p.exists();result['deleted'].append(row['step']);write(RECEIPT,result)
    for p in paths:st=p.stat();assert [st.st_size,st.st_mtime_ns]==preserved[str(p)]
    result.update(passed=True,deleted_object_bytes=sum(r['bytes'] for r in files),retained_steps=KEEP,completed_epoch=time.time())
    write(RECEIPT,result)
    dest=Path('/mnt/airs-business/rdm-experiments/manifests/user-checkpoint-deletions-20261106.json')
    write(dest,result);assert json.loads(dest.read_text())==result
    remaining=[]
    for s in KEEP:
        p=C/f'dino-qwenvl-mmd2-from0-archive/checkpoint-archive-{s}.json';v=json.loads(p.read_text());remaining.append({k:v[k] for k in ['step','destination','bytes','sha256']})
    write(dest.with_name('students-retained-current.json'),dict(experiment=EXP,retained=remaining,deleted=DELETE,epoch=time.time()))
    print(json.dumps(result))

if __name__=='__main__':main()
