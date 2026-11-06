#!/usr/bin/env python3
"""Read small receipts and actual transfer I/O, without scanning the object mount."""
import json
from pathlib import Path
import shutil
import time

BASE=Path('/data/rdm-workspace')
if not BASE.exists():BASE=Path('/data/shaoshitong')
C=BASE/'rdm-sets/control/storage-relocation-20261106'
def read(name):
    p=C/name
    return json.loads(p.read_text()) if p.exists() else None
pid=read('archive.pid')
P=Path('/proc')/str(pid)
print(json.dumps(dict(epoch=time.time(),workspace=str(BASE),progress=read('progress.json'),
    archive_complete=read('complete.json'),actual_exit=read('archive.exit'),rename=read('rename.json'),
    pid=pid,live_cmd=(P/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace') if P.exists() else None,
    io=(P/'io').read_text() if P.exists() else None,disk_free_bytes=shutil.disk_usage('/data').free,
    last30=(C/'archive.log').read_text().splitlines()[-30:])))
