#!/usr/bin/env python3
"""Log real exit and last30 on failure for the one-shot storage migration."""
import json
import os
from pathlib import Path
import runpy
import traceback

SOURCE=Path(__file__).resolve().parent/'h20-storage-relocate.py'
BASE=Path('/data/rdm-workspace')
if not BASE.exists():BASE=Path('/data/shaoshitong')
ROOT=BASE/'rdm-sets'
C=ROOT/'control/storage-relocation-20261106'
C.mkdir(parents=True,exist_ok=True)
code=1
try:
    runpy.run_path(str(SOURCE),run_name='__main__')
    code=0
except BaseException:
    traceback.print_exc()
finally:
    (C/'archive.exit').write_text(str(code)+'\n')
raise SystemExit(code)
