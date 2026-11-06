#!/usr/bin/env python3
"""CPU-only native comparison of the exact requested0/200/250/300/350/400 stages."""
import datetime
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path('/data/rdm-workspace/rdm-sets')
BASE = ROOT/'visualizations/twohead-lr10-A/qib-portraits-v1'
OUT = BASE/'native-0-200-250-300-350-400'
STEPS = [0, 200, 250, 300, 350, 400]
IDS = [5, 8, 47, 81, 172, 312, 393, 849]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'PNG export is CPU-only'
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'six-stages-verified.json').exists(), 'Already exported; inspect the retained result'
    spec = Path(__file__).with_name('qib-portrait-cases-v1.json')
    spec_sha = sha(spec)
    manifest = {}
    contract = None
    for step in STEPS:
        folder = BASE/f'step_{step:07d}'
        proof = json.loads((folder/'preview-verified.json').read_text())
        parent = 150 if step == 0 else step
        assert (ROOT/'control'/f'twohead-qib-portraits-{parent}.exit').read_text().strip() == '0'
        assert proof['passed'] and proof['actual_step'] == proof['checkpoint_step'] == step
        assert proof['cases_sha256'] == spec_sha
        current = {k: proof[k] for k in ['cases_sha256', 'sampling', 'image_size', 'seeds',
                   'batch_size', 'attention', 'decode_autocast', 'prompt_token_counts']}
        if contract is None:
            contract = current
        assert current == contract, 'Prompt/noise/sampling/backend contract differs by stage'
        assert current['image_size'] == 1024 and current['seeds'] == [0, 1]
        expected = {f'qib{i:04d}-seed{s}.png' for i in IDS for s in (0, 1)}
        records = {Path(f['path']).name: f for f in proof['files']}
        assert set(records) == expected and len(proof['files']) == 16
        for name, record in records.items():
            path = folder/name
            assert Path(record['path']) == path and path.stat().st_size == record['bytes']
            assert sha(path) == record['sha256'], 'Source PNG bytes changed'
            with Image.open(path) as image:
                image.load()
                assert image.size == (1024, 1024) and image.mode == 'RGB'
        manifest[step] = records
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 40)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 30)
    files = []
    for seed in (0, 1):
        for page in (0, 1):
            ids = IDS[page*4:page*4+4]
            canvas = Image.new('RGB', (6144, 4416), 'white')
            draw = ImageDraw.Draw(canvas)
            tiles = []
            for col, step in enumerate(STEPS):
                draw.text((col*1024+24, 22), f'{step} iter / seed{seed}', fill='black', font=font)
                for row, case in enumerate(ids):
                    name = f'qib{case:04d}-seed{seed}.png'
                    record = manifest[step][name]
                    source = Path(record['path'])
                    box = [col*1024, 96+row*1080, (col+1)*1024, 1120+row*1080]
                    with Image.open(source) as image:
                        image.load()
                        canvas.paste(image, (box[0], box[1]))
                    draw.text((box[0]+16, box[3]+12), f'QIB {case} | native1024 | 2-step', font=small, fill='black')
                    tiles.append(dict(step=step, case=case, seed=seed, source=str(source), box=box,
                                      source_sha256=record['sha256']))
            target = OUT/f'qib-0-200-250-300-350-400-seed{seed}-page{page+1}.png'
            assert not target.exists(), 'Never overwrite an exported page'
            canvas.save(target, format='PNG', compress_level=1)
            del canvas, draw
            with Image.open(target) as saved:
                saved.load()
                assert saved.size == (6144, 4416)
                for tile in tiles:
                    with Image.open(tile['source']) as original:
                        original.load()
                        cut = saved.crop(tile['box'])
                        assert ImageChops.difference(cut, original).getbbox() is None
                        tile['pixel_sha256'] = hashlib.sha256(cut.tobytes()).hexdigest()
            files.append(dict(path=str(target), bytes=target.stat().st_size, sha256=sha(target),
                              dimensions=[6144, 4416], tiles=tiles))
            print(json.dumps(dict(page=str(target), complete=len(files), pixel_identical_tiles=len(files)*24)), flush=True)
    proof = dict(passed=True, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 stages=STEPS, cases=IDS, seeds=[0, 1], matched_contract=contract, files=files,
                 native_tile_dimensions=[1024, 1024], pixel_identical_tiles=96,
                 resampled=False, lossless_PNG=True, CPU_only=True, new_inference=False)
    temporary = OUT/'six-stages-verified.json.partial'
    temporary.write_text(json.dumps(proof, indent=2)+'\n')
    temporary.replace(OUT/'six-stages-verified.json')
    print(json.dumps(dict(passed=True, files=len(files), stages=STEPS, pixel_identical_tiles=96)), flush=True)


if __name__ == '__main__':
    main()
