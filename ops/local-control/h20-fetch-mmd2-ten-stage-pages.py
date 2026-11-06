#!/usr/bin/env python3
"""Fetch only requested, lossless final comparison PNGs; no weights/fullbenchmark assets."""
import argparse
import base64
import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

REMOTE=Path('/data/rdm-workspace/rdm-sets/visualizations/zimage-2step-1024-dino-qwenvl-mmd2-from0/qib-portraits-v1/native-ten-stages')


def remote(helper, code):
    result=subprocess.run([sys.executable,str(helper)],input=code,text=True,capture_output=True,timeout=120)
    if result.returncode:
        raise RuntimeError(f'Owned SSH read exited{result.returncode}: '+repr(result.stderr.splitlines()[-10:]))
    return result.stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-helper',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    script=f"from pathlib import Path\nprint((Path({str(REMOTE)!r})/'ten-stages-verified.json').read_text())\n"
    proof=json.loads(remote(args.ssh_helper,script))
    assert proof['passed'] and proof['stages']==list(range(0,451,50))
    assert proof['pixel_identical_tiles']==160 and proof['case_page_pixel_identical_tiles']==80
    assert proof['resampled'] is False and proof['lossless_PNG'] is True
    files=proof['files']+proof['case_pages']
    assert len(proof['files'])==4 and len(proof['case_pages'])==8
    args.output.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(args.output).free>sum(x['bytes'] for x in files)+256*1024**2
    results=[]
    from PIL import Image
    for item in files:
        source=Path(item['path'])
        assert source.parent==REMOTE and source.suffix=='.png' and 0<item['bytes']<=80*1024**2
        target=args.output/source.name
        if target.exists():
            raw=target.read_bytes()
        else:
            script=('import pathlib,base64,hashlib,sys\n'
                f'p=pathlib.Path({str(source)!r})\n'
                'raw=p.read_bytes()\n'
                f"assert len(raw)=={item['bytes']} and hashlib.sha256(raw).hexdigest()=={item['sha256']!r}\n"
                'sys.stdout.write(base64.b64encode(raw).decode())\n')
            raw=base64.b64decode(remote(args.ssh_helper,script),validate=True)
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            assert image.format=='PNG' and list(image.size)==item['dimensions']
        temp=target.with_suffix('.partial')
        temp.write_bytes(raw);temp.replace(target)
        results.append(dict(source=str(source),path=str(target.resolve()),bytes=len(raw),
            sha256=item['sha256'],dimensions=item['dimensions']))
        receipt=dict(passed=len(results)==12,files=results,remote_verification=proof,
            no_weights_transferred=True,no_resampling=True,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (args.output/'download-verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(dict(file=source.name,verified=True,completed=len(results),total=12)),flush=True)


if __name__=='__main__':
    main()
