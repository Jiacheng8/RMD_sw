#!/usr/bin/env python3
"""Build faithful contact sheets and pixel-difference summaries; never change samples."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/data/rdm-workspace/rdm-sets/visualizations/spatial-A')
OUT=ROOT/'comparison-0-150-200'


def main():
    OUT.mkdir(exist_ok=True)
    metrics=[]
    for step in [0,150,200]:
        assert (ROOT/f'step_{step:07d}'/'preview-verified.json').exists()
    for seed in [0,1]:
        sheet=Image.new('RGB',(1536,4*284+40),'white');draw=ImageDraw.Draw(sheet)
        draw.text((10,12),f'Existing A run / 2-step / each pair: iter150 | iter200 / seed {seed}',fill='black')
        for i in range(12):
            arrays={}
            for step in [0,150,200]:
                p=ROOT/f'step_{step:07d}'/f'{i:05d}_s{seed}.png'
                with Image.open(p) as im:
                    assert im.size==(1024,1024)
                    arrays[step]=np.asarray(im.convert('RGB')).copy()
                    if step:
                        x=(i%3)*512+(0 if step==150 else 256);y=40+(i//3)*284
                        sheet.paste(im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS),(x,y))
                        draw.text((x+5,y+262),f'P{i+1:02d} / iter{step}',fill='black')
            for a,b in [(0,150),(0,200),(150,200)]:
                delta=np.abs(arrays[a].astype(np.int16)-arrays[b].astype(np.int16))
                metrics.append({'prompt_id':i+1,'seed':seed,'steps':[a,b],
                    'pixel_MAE_0_to255':float(delta.mean()),
                    'identical_RGB_pixel_fraction':float(np.all(arrays[a]==arrays[b],axis=-1).mean())})
        sheet.save(OUT/f'compare150-200-seed{seed}.jpg',quality=94)
        chosen=[0,1,3,6,9,11]
        ref=Image.new('RGB',(3*320,len(chosen)*348+40),'white');dr=ImageDraw.Draw(ref)
        dr.text((8,12),f'Same 2-step contract / columns: init0 | iter150 | iter200 / seed {seed}',fill='black')
        for row,i in enumerate(chosen):
            for col,step in enumerate([0,150,200]):
                with Image.open(ROOT/f'step_{step:07d}'/f'{i:05d}_s{seed}.png') as im:
                    ref.paste(im.convert('RGB').resize((320,320),Image.Resampling.LANCZOS),(col*320,row*348+40))
                dr.text((col*320+5,row*348+365),f'P{i+1:02d} / iter{step}',fill='black')
        ref.save(OUT/f'baseline0-150-200-seed{seed}.jpg',quality=94)
    summary={'prompts':12,'seeds':[0,1],'steps':[0,150,200],
             'missing_checkpoint_steps':[50,100],
             'pixel_differences_are_not_quality_scores':True,'metrics':metrics,
             'current_training_uses_legacy_three_GAN_heads':True,
             'corrected_two_GAN_candidate_not_used_for_these_training_weights':True}
    (OUT/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'output':str(OUT),'comparisons':len(metrics),'images_per_checkpoint':24}))


if __name__=='__main__':
    main()
