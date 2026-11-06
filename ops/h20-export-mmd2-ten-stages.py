#!/usr/bin/env python3
"""CPU-only native0,50..450 comparisons from the fresh DINO/Qwen MMD2 experiment."""
import datetime
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path('/data/rdm-workspace/rdm-sets')
BASE = ROOT/'visualizations/zimage-2step-1024-dino-qwenvl-mmd2-from0/qib-portraits-v1'
OUT = BASE/'native-ten-stages'
STEPS = list(range(0,451,50))
IDS = [5, 8, 47, 81, 172, 312, 393, 849]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'PNG export is CPU-only'
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'ten-stages-verified.json').exists(), 'Already exported; inspect the retained result'
    spec = Path(__file__).with_name('qib-portrait-cases-v1.json')
    spec_sha = sha(spec)
    manifest = {}
    contract = None
    for step in STEPS:
        folder = BASE/f'step_{step:07d}'
        proof = json.loads((folder/'preview-verified.json').read_text())
        assert proof['experiment']=='zimage-2step-1024-dino-qwenvl-mmd2-from0'
        assert proof['training_commit']=='ad5dddc428a32591fedf9b034bf5f1639ced0514'
        assert proof['passed'] and proof['actual_step'] == proof['checkpoint_step'] == step
        assert proof['cases_sha256'] == spec_sha
        current = {k: proof[k] for k in ['cases_sha256', 'sampling', 'image_size', 'seeds',
                   'batch_size', 'attention', 'sampler_autocast', 'decode_autocast', 'prompt_token_counts']}
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
            canvas = Image.new('RGB', (10240, 4416), 'white')
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
            target = OUT/f'qib-0-50-100-150-200-250-300-350-400-450-seed{seed}-page{page+1}.png'
            assert not target.exists(), 'Never overwrite an exported page'
            canvas.save(target, format='PNG', compress_level=1)
            del canvas, draw
            with Image.open(target) as saved:
                saved.load()
                assert saved.size == (10240, 4416)
                for tile in tiles:
                    with Image.open(tile['source']) as original:
                        original.load()
                        cut = saved.crop(tile['box'])
                        assert ImageChops.difference(cut, original).getbbox() is None
                        tile['pixel_sha256'] = hashlib.sha256(cut.tobytes()).hexdigest()
            files.append(dict(path=str(target), bytes=target.stat().st_size, sha256=sha(target),
                              dimensions=[10240, 4416], tiles=tiles))
            print(json.dumps(dict(page=str(target), complete=len(files), pixel_identical_tiles=len(files)*40)), flush=True)
    case_pages = []
    for case in IDS:
        canvas = Image.new('RGB', (5120, 2240), 'white')
        draw = ImageDraw.Draw(canvas)
        tiles = []
        for index, step in enumerate(STEPS):
            col, row = index%5, index//5
            source = BASE/f'step_{step:07d}'/f'qib{case:04d}-seed0.png'
            x, y = col*1024, row*1120+64
            draw.text((x+20,row*1120+12), f'{step} iter | QIB {case} | seed0', fill='black',font=small)
            with Image.open(source) as image: canvas.paste(image,(x,y))
            tiles.append(dict(step=step, source=str(source), box=[x,y,x+1024,y+1024]))
        target = OUT/f'qib{case:04d}-seed0-ten-stages-native.png'
        assert not target.exists()
        canvas.save(target,format='PNG',compress_level=1)
        del canvas, draw
        with Image.open(target) as saved:
            for tile in tiles:
                with Image.open(tile['source']) as original:
                    assert ImageChops.difference(saved.crop(tile['box']),original).getbbox() is None
        case_pages.append(dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target),
            dimensions=[5120,2240],case=case,seed=0,tiles=tiles))
    proof = dict(passed=True, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 stages=STEPS, cases=IDS, seeds=[0, 1], matched_contract=contract, files=files,
                 native_tile_dimensions=[1024, 1024], pixel_identical_tiles=160,
                 case_pages=case_pages, case_page_pixel_identical_tiles=80,
                 resampled=False, lossless_PNG=True, CPU_only=True, new_inference=False)
    temporary = OUT/'ten-stages-verified.json.partial'
    temporary.write_text(json.dumps(proof, indent=2)+'\n')
    temporary.replace(OUT/'ten-stages-verified.json')
    print(json.dumps(dict(passed=True, files=len(files), stages=STEPS, pixel_identical_tiles=160)), flush=True)


if __name__ == '__main__':
    main()
