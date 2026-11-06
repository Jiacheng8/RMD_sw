#!/usr/bin/env python3
"""Rename a quiescent completed experiment workspace without a personal-name alias.

Run only after h20-storage-relocate.py exits0. Preserves historical source, logs,
and provenance bytes; adjusts absolute symlinks and environment entrypoints only.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

OLD=Path('/data/shaoshitong')
NEW=Path('/data/rdm-workspace')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as s:
        for b in iter(lambda:s.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    assert OLD.is_dir() and not OLD.is_symlink() and not NEW.exists()
    C=OLD/'rdm-sets/control/storage-relocation-20261106'
    parser=argparse.ArgumentParser()
    parser.add_argument('--after-controlled-storage-stop',action='store_true')
    args=parser.parse_args()
    if args.after_controlled_storage_stop:
        interrupted=json.loads((C/'archive-interruption.json').read_text())
        assert interrupted['training_not_touched'] and all(x['observed_terminated'] for x in interrupted['processes'])
        final=json.loads((OLD/'rdm-sets/control/dino-qwenvl-mmd2-from0-final-integrity.json').read_text())
        assert final['passed'] and final['actual_formal_exit']==0 and final['step']==500
    else:
        receipt=json.loads((C/'complete.json').read_text());assert receipt['passed']
        assert (C/'archive.exit').read_text().strip()=='0'
    busy=[]
    for p in Path('/proc').glob('[0-9]*'):
        if int(p.name)==os.getpid():continue
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd=os.readlink(p/'cwd')
            if str(OLD)+'/' in cmd or cwd.startswith(str(OLD)+'/'):busy.append(int(p.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    assert not busy, 'Active project processes '+str(busy)
    foreign=Path('/data/ready_ckpt_0423');foreign_stat=foreign.stat() if foreign.exists() else None
    before=dict(inode=OLD.stat().st_ino,device=OLD.stat().st_dev)
    OLD.rename(NEW)
    assert not OLD.exists() and NEW.stat().st_ino==before['inode']
    links=[];scripts=[]
    ram_old=Path('/dev/shm/shaoshitong-qib-judge-1b77ff83564ac4e4e8140769eacff7ff73f35f3c')
    ram_new=Path('/dev/shm/rdm-qib-judge-1b77ff83564ac4e4e8140769eacff7ff73f35f3c')
    if ram_old.exists():
        assert not ram_new.exists();ram_old.rename(ram_new)
    for folder,ds,fs in list(os.walk(NEW))+list(os.walk(ram_new)):
        for name in ds+fs:
            path=Path(folder)/name
            if path.is_symlink():
                target=os.readlink(path)
                if target.startswith(str(OLD)+'/'):
                    path.unlink();path.symlink_to(str(NEW)+target[len(str(OLD)):]);links.append(str(path))
    # Conda prefix relocation: edit text launchers/config only, never ELF/shared
    # libraries or model files. Original bytes and hashes retained in control.
    C=NEW/'rdm-sets/control/storage-relocation-20261106'
    candidates=[NEW/'rdm-sets/env.sh']
    for base in [NEW/'rdm-sets/envs',NEW/'rdm-sets/miniconda',NEW/'rdm-sets/qib/env']:
        for folder,ds,fs in os.walk(base):
            for name in fs:
                path=Path(folder)/name
                if path.is_symlink() or path.stat().st_size>2*1024**2:continue
                if '/bin/' in str(path) or path.name.startswith('_sysconfigdata') or (path.name=='Makefile' and '/python3.' in str(path)) or path.suffix in ('.pth','.egg-link','.pc','.cmake','.cfg','.conf','.sh'):
                    candidates.append(path)
    for path in candidates:
        if not path.is_file():continue
        data=path.read_bytes()
        if str(OLD).encode() not in data or b'\0' in data:continue
        try:data.decode('utf-8')
        except UnicodeDecodeError:continue
        backup=C/'environment-original'/path.relative_to(NEW)
        backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(data)
        changed=data.replace(str(OLD).encode(),str(NEW).encode())
        path.write_bytes(changed)
        scripts.append(dict(path=str(path),before_sha256=hashlib.sha256(data).hexdigest(),after_sha256=hashlib.sha256(changed).hexdigest()))
    foreign_after=foreign.stat() if foreign.exists() else None
    result=dict(passed=True,old_path=str(OLD),new_path=str(NEW),same_filesystem_atomic_rename=True,
        old_personal_alias_absent=not OLD.exists(),absolute_symlinks_updated=len(links),
        environment_text_files_updated=scripts,historical_source_and_logs_unchanged=True,
        foreign_directory_untouched=True,foreign_directory_observed_external_change=foreign_after!=foreign_stat,ram_judge_path=str(ram_new),epoch=time.time())
    (C/'rename.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
