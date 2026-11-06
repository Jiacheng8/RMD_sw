#!/usr/bin/env python3
"""Readable native0/450 pairs from verified complex-case PNGs; no new inference."""
import argparse
import base64
import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path

ROOT=Path('/data/rdm-workspace/rdm-sets/visualizations/zimage-2step-1024-dino-qwenvl-mmd2-from0/complex-cases-v1')
spec=importlib.util.spec_from_file_location('complex_fetch',Path(__file__).with_name('h20-fetch-mmd2-complex-pages.py'))
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-helper',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from PIL import Image,ImageDraw,ImageFont,ImageChops
    from matplotlib.font_manager import findfont
    font=ImageFont.truetype(findfont('DejaVu Sans'),36)
    proofs={}
    for step in [0,450]:
        folder=ROOT/f'step_{step:07d}'
        p=json.loads(client.remote(args.ssh_helper,f'from pathlib import Path\nprint(Path({str(folder/"preview-verified.json")!r}).read_text())\n'))
        assert p['passed'] and p['actual_step']==p['checkpoint_step']==step
        assert p['case_suite']=='complex-cases-v1' and len(p['files'])==16
        assert p['training_commit']=='ad5dddc428a32591fedf9b034bf5f1639ced0514'
        proofs[step]=p
    for key in ['cases_sha256','sampling','image_size','seeds','batch_size','attention','sampler_autocast','decode_autocast','prompt_token_counts']:
        assert proofs[0][key]==proofs[450][key]
    files=[]
    for ident in range(9001,9009):
        canvas=Image.new('RGB',(2048,1120),'white');draw=ImageDraw.Draw(canvas)
        tiles=[]
        for col,step in enumerate([0,450]):
            path=ROOT/f'step_{step:07d}'/f'case{ident}-seed0.png'
            record=next(x for x in proofs[step]['files'] if x['path']==str(path))
            script=('from pathlib import Path\nimport base64,hashlib,sys\n'
                f'raw=Path({str(path)!r}).read_bytes()\n'
                f"assert len(raw)=={record['bytes']} and hashlib.sha256(raw).hexdigest()=={record['sha256']!r}\n"
                'sys.stdout.write(base64.b64encode(raw).decode())\n')
            raw=base64.b64decode(client.remote(args.ssh_helper,script),validate=True)
            assert hashlib.sha256(raw).hexdigest()==record['sha256']
            with Image.open(io.BytesIO(raw)) as image:
                image.load();assert image.size==(1024,1024) and image.mode=='RGB'
                canvas.paste(image,(col*1024,64))
                tiles.append((image.copy(),[col*1024,64,(col+1)*1024,1088]))
            draw.text((col*1024+16,12),f'Case {ident} | {step} iter | seed0',font=font,fill='black')
        target=args.output/f'case{ident}-0-vs-450-native.png';canvas.save(target,format='PNG',compress_level=1)
        with Image.open(target) as image:
            assert image.size==(2048,1120)
            for source,box in tiles:assert ImageChops.difference(image.crop(box),source).getbbox() is None
        files.append(dict(path=str(target.resolve()),bytes=target.stat().st_size,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),dimensions=[2048,1120],case=ident))
        print(json.dumps(dict(case=ident,completed=len(files),total=8,verified=True)),flush=True)
    result=dict(passed=True,files=files,steps=[0,450],seed=0,native_tile_size=[1024,1024],
        pixel_identical_tiles=16,no_resampling=True,new_inference=False,source_proofs=proofs,
        verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output/'endpoint-verified.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
