#!/usr/bin/env python3
"""Five columns of native1024px PNGs; no resampling or JPEG roundtrip."""
import datetime, hashlib, json, os, pathlib
from PIL import Image, ImageDraw, ImageFont, ImageChops

ROOT = pathlib.Path('/data/rdm-workspace/rdm-sets')
BASE0 = ROOT/'visualizations/spatial-A/step_0000000'
STEP50 = ROOT/'visualizations/twohead-lr10-A/step_0000050'
STEP100 = ROOT/'visualizations/twohead-lr10-A/training-prompts-100-v1'
OLD150 = ROOT/'visualizations/twohead-lr10-A/comparison-0-50-100-150-v1'
SOURCE = ROOT/'visualizations/twohead-lr10-A/comparison-0-50-100-150-200-v1'
OUT = SOURCE/'native-five-columns'
TILE, HEADER, FOOTER = 1024, 96, 80


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def write(name, data):
    p = OUT/name; temporary = p.with_name(p.name+'.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, p)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU export only'
    original = json.loads((SOURCE/'comparison-verified.json').read_text())
    assert original['passed'] and original['checkpoint_step'] == 200
    old = json.loads((OLD150/'comparison-verified.json').read_text())
    assert old['passed'] and old['checkpoint_step'] == 150
    expected = {r['path']: r['sha256'] for r in old['source_pngs'] + original['source_pngs']}
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'native-export-verified.json').exists(), 'Native export already complete'
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 40)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 30)
    completed, verified_tiles = [], 0

    def export(name, examples, paths, group):
        nonlocal verified_tiles
        width, height = 5*TILE, HEADER+len(examples)*(TILE+FOOTER)
        sheet = Image.new('RGB', (width, height), 'white'); draw = ImageDraw.Draw(sheet)
        for col, stage in enumerate((0, 50, 100, 150, 200)):
            draw.text((col*TILE+24, 22), f'{stage} iter / original1024x1024', font=font, fill='black')
        tiles, missing = [], []
        for row, index in enumerate(examples):
            top = HEADER+row*(TILE+FOOTER)
            for col, path in enumerate(paths(index)):
                box = (col*TILE, top, (col+1)*TILE, top+TILE)
                if path is None:
                    draw.rectangle((box[0], box[1], box[2]-1, box[3]-1), fill=(238, 238, 238))
                    draw.text((box[0]+45, top+450), 'Unavailable for this prompt', font=font, fill=(70, 70, 70))
                    draw.text((box[0]+45, top+510), 'Preview not retained at this stage', font=small, fill=(90, 90, 90))
                    missing.append(dict(row=index, stage=(0, 50, 100, 150, 200)[col], box=list(box)))
                    continue
                assert expected[str(path)] == sha(path), 'Source PNG changed after verification'
                with Image.open(path) as image:
                    image.load(); assert image.mode == 'RGB' and image.size == (TILE, TILE)
                    sheet.paste(image, (box[0], box[1]))  # Direct pixels; no resize/crop/convert.
                tiles.append(dict(source=str(path), stage=(0, 50, 100, 150, 200)[col], row=index,
                                  box=list(box), source_file_sha256=expected[str(path)]))
            draw.text((24, top+TILE+20), f'{group} P{index+1:02d} | all available cells share the same prompt and noise | native pixels', font=small, fill='black')
        target = OUT/name
        sheet.save(target, format='PNG', compress_level=1)  # Lossless byte coding; original resolution unchanged.
        del sheet, draw
        with Image.open(target) as saved:
            saved.load(); assert saved.size == (width, height) and saved.mode == 'RGB'
            for record in tiles:
                tile = saved.crop(tuple(record['box']))
                with Image.open(record['source']) as original_tile:
                    original_tile.load()
                    assert ImageChops.difference(tile, original_tile).getbbox() is None, 'Tile pixels changed'
                    record['pixel_sha256'] = hashlib.sha256(tile.tobytes()).hexdigest()
                verified_tiles += 1
        completed.append(dict(file=name, width=width, height=height, bytes=target.stat().st_size,
            sha256=sha(target), native_tile_size=[TILE, TILE], tiles=tiles, missing=missing,
            every_available_tile_pixel_identical=True, image_resampled=False, lossless_PNG=True))
        write('progress.json', dict(completed_files=len(completed), pixel_identical_tiles=verified_tiles,
                                   files=[r['file'] for r in completed]))
        print(json.dumps(dict(file=name, dimensions=[width, height], bytes=target.stat().st_size,
                              completed_files=len(completed), pixel_identical_tiles=verified_tiles)), flush=True)

    for seed in (0, 1):
        export(f'native-0-50-100-150-200-handwritten-seed{seed}.png', [0, 1, 3, 6, 9, 11],
            lambda i: [BASE0/f'{i:05d}_s{seed}.png', STEP50/f'{i:05d}_s{seed}.png', None,
                       OLD150/'handwritten150'/f'{i:05d}_s{seed}.png',
                       SOURCE/'handwritten200'/f'{i:05d}_s{seed}.png'], f'Handwritten seed{seed}')
    for page in range(2):
        export(f'native-0-50-100-150-200-training-page{page+1}.png', list(range(page*4, page*4+4)),
            lambda i: [STEP100/f'{i:02d}-initial.png', None, STEP100/f'{i:02d}-iteration100.png',
                       OLD150/'training150'/f'{i:02d}-iteration150.png',
                       SOURCE/'training200'/f'{i:02d}-iteration200.png'], 'Actual training prompt')
    assert len(completed) == 4 and verified_tiles == 80
    result = dict(passed=True, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        CPU_only=True, no_inference_or_training_launched=True, columns=[0, 50, 100, 150, 200],
        tile_dimensions=[1024, 1024], canvas_width=5120, files=completed,
        available_tiles_verified_pixel_identical=80, unavailable_cells=20,
        unavailable_reason='50/100 snapshots were overwritten and their previews used different prompt sets',
        source_receipt=str(SOURCE/'comparison-verified.json'))
    write('native-export-verified.json', result)
    print(json.dumps(dict(passed=True, files=4, native_width=5120, pixel_identical_tiles=80)), flush=True)


if __name__ == '__main__':
    main()
