#!/usr/bin/env python3
"""User-authorized removal of obsolete experiment weights, retaining old350/current run."""
import argparse
import datetime
import json
from pathlib import Path
import shutil

ROOT = Path('/data/rdm-workspace/rdm-sets')
OLD = 'zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A'
ACTIVE = 'zimage-2step-1024-dino-qwenvl-mmd2-from0'
ARCHIVE = Path('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints')
KEEP = ARCHIVE/OLD/'step_0000350.pth'
TARGETS = [
    (ROOT/'zimage/work_dirs/zimage-2step-1024-dlr1e-5-spatial-patchgan-softplus/resume.pth',31848888019),
    (ROOT/'zimage/work_dirs'/OLD/'resume.pth',31722878123),
    (ROOT/'control/spatial-A-user-visualization/step_0000200-model.pth',24619900326),
    (ROOT/'control/spatial-A-user-visualization/step_0000150-model.pth',24619900326),
    (ARCHIVE/'legacy-spatial-A/step_0000150-model.pth',24619900326),
] + [(ARCHIVE/OLD/f'step_{s:07d}.pth',24619908236 if s==250 else 24619908172)
     for s in [200,250,300,400]]


def write(path, value):
    tmp = path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def main(execute=False):
    assert KEEP.exists() and not KEEP.is_symlink() and KEEP.stat().st_size==24619908172
    # Fresh run still uses its own immutable source, checkpoint and optimizer.
    command=json.loads((ROOT/'control/dino-qwenvl-mmd2-from0-formal.command.json').read_text())
    assert command['source_commit']=='ad5dddc428a32591fedf9b034bf5f1639ced0514'
    cmdline=Path(f"/proc/{command['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
    assert command['source'] in cmdline and ACTIVE in ' '.join(command['command'])
    assert not (ROOT/'control/dino-qwenvl-mmd2-from0-formal.exit.json').exists()
    active_resume=ROOT/'zimage/work_dirs'/ACTIVE/'resume.pth'
    before_active=(active_resume.stat().st_size,active_resume.stat().st_mtime_ns)
    files=[]
    for path, size in TARGETS:
        assert ACTIVE not in str(path) and path!=KEEP and path.resolve()!=KEEP.resolve()
        assert path.exists() and path.stat().st_size==size, f'Unexpected target identity: {path}'
        files.append(dict(path=str(path),bytes=size,symlink=path.is_symlink(),resolved=str(path.resolve())))
    receipt=dict(authorization='User: previous training only350 retained; delete other stored oldweights',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execute=execute,
        preserved_old350=str(KEEP),preserved_active=str(active_resume),files=files,
        disk_free_before=shutil.disk_usage(ROOT).free,deleted=[])
    control=ROOT/'control/user-old-weights-cleanup-20261105.json'
    if execute:
        assert not control.exists(), 'Never silently repeat a prior cleanup'
        write(control,receipt)
        for entry in files:
            path=Path(entry['path'])
            path.unlink()
            assert not path.exists() and not path.is_symlink()
            receipt['deleted'].append(entry['path'])
            write(control,receipt)
        assert KEEP.exists() and KEEP.stat().st_size==24619908172
        assert (active_resume.stat().st_size,active_resume.stat().st_mtime_ns)==before_active
        assert Path(f"/proc/{command['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()==cmdline
        receipt.update(passed=True,disk_free_after=shutil.disk_usage(ROOT).free,
            removed_node_local_bytes=sum(e['bytes'] for e in files if not e['symlink'] and e['path'].startswith(str(ROOT))),
            removed_object_bytes=sum(e['bytes'] for e in files if e['path'].startswith(str(ARCHIVE))))
        write(control,receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    main(parser.parse_args().execute)
