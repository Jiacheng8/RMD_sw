#!/usr/bin/env python3
"""Fetch only four verified, user-requested native PNG pages via the owned SSH helper.

No inference, resampling, weight transfer, or connection configuration changes.
"""
import argparse
import base64
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ssh-helper', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.verification.read_text())
    proof = source['proof']
    step = proof['actual_step']
    assert source['actual_exit'] == 0 and proof['passed']
    assert proof['native_tile_pixel_equality_verified']
    assert step in [150, 200, 250, 300, 350, 400, 450, 500]
    remote_root = Path('/data/rdm-workspace/rdm-sets/visualizations/twohead-lr10-A/qib-portraits-v1')
    pages = proof['native_comparisons']
    expected_names = {f'portrait-0-{step}-seed{seed}-page{page}.png'
                      for seed in (0, 1) for page in (1, 2)}
    assert len(pages) == 4 and {Path(p['path']).name for p in pages} == expected_names
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for item in pages:
        remote = Path(item['path'])
        assert remote.parent == remote_root/f'step_{step:07d}'
        assert 0 < item['bytes'] <= 20*1024**2 and len(item['sha256']) == 64
        target = args.output/remote.name
        started = time.monotonic()
        cached = target.exists()
        if cached:
            raw = target.read_bytes()
        else:
            command = (
                'import pathlib,json,hashlib,base64,sys\n'
                f'item=json.loads({json.dumps(item)!r})\n'
                "raw=pathlib.Path(item['path']).read_bytes()\n"
                "assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']\n"
                "assert raw.startswith(b'\\x89PNG\\r\\n\\x1a\\n')\n"
                "sys.stdout.write(base64.b64encode(raw).decode('ascii'))\n"
            )
            proc = subprocess.run([sys.executable, str(args.ssh_helper)], input=command,
                                  text=True, capture_output=True, timeout=120)
            if proc.returncode:
                failure = dict(source=str(remote), actual_exit=proc.returncode,
                               stderr_last30=proc.stderr.splitlines()[-30:],
                               completed_files=results, training_untouched=True)
                failure_path = args.output/'native-pages-download-failure.json'
                failure_path.write_text(json.dumps(failure, indent=2)+'\n')
                raise RuntimeError(f'Image transfer exited{proc.returncode}; evidence:{failure_path}')
            raw = base64.b64decode(proc.stdout, validate=True)
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            assert image.format == 'PNG' and image.size == (2048, 4416)
        if not cached:
            temporary = target.with_suffix('.partial')
            temporary.write_bytes(raw)
            temporary.replace(target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']
        results.append(dict(source=str(remote), path=str(target.resolve()), bytes=len(raw),
                            sha256=item['sha256'], pixels=[2048, 4416], reused=cached,
                            seconds=time.monotonic()-started))
        receipt = dict(passed=len(results) == 4, actual_step=step, files=results,
                       no_resampling=True, no_inference=True, no_weights_transferred=True,
                       verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (args.output/'native-pages-download-verified.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(dict(file=target.name, bytes=len(raw), verified=True, reused=cached)), flush=True)


if __name__ == '__main__':
    main()
