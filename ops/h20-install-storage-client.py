#!/usr/bin/env python3
"""Extract an authenticated distro CLI into project cache; no global installation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/data/rdm-workspace/rdm-sets')
D=ROOT/'.cache/storage-tools';D.mkdir(exist_ok=True)
version='1.53.3-4ubuntu1.22.04.5'
metadata=subprocess.check_output(['apt-cache','show','rclone='+version],text=True)
expected=[s.split(': ',1)[1] for s in metadata.splitlines() if s.startswith('SHA256: ')];assert len(set(expected))==1
subprocess.run(['apt-get','download','rclone='+version],cwd=D,check=True,timeout=90)
package=next(D.glob('rclone_'+version+'_amd64.deb'))
assert hashlib.sha256(package.read_bytes()).hexdigest()==expected[0]
subprocess.run(['dpkg-deb','-x',str(package),str(D/'extracted')],check=True)
binary=D/'rclone';shutil.copy2(D/'extracted/usr/bin/rclone',binary);binary.chmod(0o700)
receipt=dict(passed=True,version=version,binary=str(binary),authenticated_distro_package_sha256=expected[0],
    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),no_global_environment_change=True)
(D/'rclone-verified.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
