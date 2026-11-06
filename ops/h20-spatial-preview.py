#!/usr/bin/env python3
"""Pin a closed requested checkpoint and render on an independently idle GPU."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/rdm-workspace/rdm-sets')
RUN = ROOT/'zimage/work_dirs/zimage-2step-1024-dlr1e-5-spatial-patchgan-softplus'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True, choices=[0, 150, 200])
    parser.add_argument('--gpu', type=int, required=True, choices=[6, 7])
    args = parser.parse_args()
    assert sys.platform.startswith('linux') and os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
    repo = Path(__file__).resolve().parent.parent
    control = ROOT/'control/spatial-A-user-visualization'
    control.mkdir(exist_ok=True)
    frozen = control/f'step_{args.step:07d}-resume.pth'
    if args.step and not frozen.exists():
        deadline = time.monotonic()+3600
        while time.monotonic() < deadline:
            lines=(ROOT/'launch-logs/spatial-A-500.log').read_text(errors='replace').splitlines()
            if any(f'resume state (step {args.step})' in s for s in lines):
                os.link(RUN/'resume.pth', frozen)
                break
            print(f'waiting_for_saved_step={args.step} utc={dt.datetime.now(dt.timezone.utc).isoformat()}',flush=True)
            time.sleep(5)
        else:
            raise RuntimeError('Requested checkpoint did not finish within the bounded wait')
    sampling={'sigmas':[1.0,0.75],'renoise':True,'detach_between_steps':True}
    if args.step:
        import torch
        state=torch.load(frozen,map_location='cpu',weights_only=False,mmap=True)
        assert state['step']==args.step and state['num_steps']==2
        sampling=state['sampling_args'];assert len(sampling['sigmas'])==2 and sampling['renoise']
        del state
    out=ROOT/'visualizations/spatial-A'/f'step_{args.step:07d}'
    out.mkdir(parents=True,exist_ok=True)
    receipt={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'step':args.step,
             'checkpoint':str(frozen) if args.step else None,'physical_gpu':args.gpu,'sampling':sampling,
             'prompts':str(repo/'ops/spatial-preview-prompts.jsonl'),'seeds':[0,1],
             'image_size':1024,'training_gpus_untouched':[4,5]}
    (out/'preview-command.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with (ROOT.parent/f'rdm-preview-gpu{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for _ in range(2):
            values=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
                '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            memory,util=map(int,values.strip().split(','))
            assert memory<=32 and util==0, f'Preview GPU not idle: {memory} MiB, {util}%'
            time.sleep(2)
        command=[sys.executable,'-m','zimage_distill.sample',
            '--prompts',str(repo/'ops/spatial-preview-prompts.jsonl'),
            '--out',str(out),'--num-steps','2','--seeds','2','--batch','2','--img-size','1024']
        if args.step: command += ['--ckpt',str(frozen)]
        subprocess.run(command,check=True)
    expected=[out/f'{i:05d}_s{s}.png' for i in range(12) for s in [0,1]]
    assert all(p.is_file() and p.stat().st_size>1000 for p in expected)
    from PIL import Image,ImageDraw
    for seed in [0,1]:
        sheet=Image.new('RGB',(3*384,4*412+36),'white');draw=ImageDraw.Draw(sheet)
        draw.text((10,10),f'A step {args.step} / 2-step / seed {seed} / same prompts and noise across checkpoints',fill='black')
        for i in range(12):
            with Image.open(out/f'{i:05d}_s{seed}.png') as im:
                assert im.size==(1024,1024)
                thumb=im.convert('RGB').resize((384,384),Image.Resampling.LANCZOS)
            x=(i%3)*384;y=36+(i//3)*412
            sheet.paste(thumb,(x,y));draw.text((x+8,y+388),f'P{i+1:02d} / seed {seed}',fill='black')
        sheet.save(out/f'contact-seed{seed}.jpg',quality=92)
    receipt.update(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),images=24,passed=True)
    (out/'preview-verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
