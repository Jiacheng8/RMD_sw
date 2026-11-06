#!/usr/bin/env python3
"""Original pretrained Turbo Euler2/9, official1000 Chinese prompts, adaptive batch on6/7."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/rdm-workspace/rdm-sets')
SOURCE = Path(__file__).resolve().parents[1]
REVISION = 'f332072aa78be7aecdf3ee76d5c247082da564a6'
SNAPSHOT = ROOT/'hf/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots'/REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=[0, 1], required=True)
    parser.add_argument('--gpu', type=int, choices=[6, 7], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--teacher-steps', type=int, choices=[2, 9], required=True)
    args = parser.parse_args()
    assert args.gpu == args.rank+6 and os.environ['CUDA_VISIBLE_DEVICES'] == str(args.gpu)
    assert (SOURCE/'SOURCE_COMMIT').is_file()
    assert (SNAPSHOT/'model_index.json').is_file()
    sys.path[:0] = [str(SOURCE/'z-image'), str(SOURCE)]
    import torch
    from PIL import Image
    from eval_qib.bench import bench_prompts
    from zimage_distill.model import ZImageStudent, ZImageVAE, QwenPromptEncoder, noise_for, turbo_sigmas
    from rdm.utils.io import save_uint8_png
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.95)
    cap = torch.cuda.get_device_properties(0).total_memory*.95
    prompts = bench_prompts(ROOT/'qib/bench', 'zh')
    assert len(prompts) == 1000
    images = args.out/'images'
    images.mkdir(parents=True, exist_ok=True)
    mine = []
    for ident, prompt in prompts[args.rank::2]:
        p = images/f'{ident:04d}.png'
        if p.exists():
            with Image.open(p) as image:
                image.load()
                assert image.mode == 'RGB' and image.size == (1024, 1024)
        else:
            mine.append((ident, prompt))
    if not mine:
        print(json.dumps(dict(event='rank_images_already_complete', rank=args.rank)), flush=True)
        return
    # Same official online text encoding; contexts stay only in remote CPU RAM.
    text = QwenPromptEncoder(str(SNAPSHOT), device='cuda', cache_size=len(mine)+1)
    caps = {}
    for lo in range(0, len(mine), 16):
        chunk = mine[lo:lo+16]
        for (ident, _), value in zip(chunk, text([p for _, p in chunk])):
            caps[ident] = value.cpu()
        print(json.dumps(dict(event='contexts', rank=args.rank, encoded=min(lo+16,len(mine)), total=len(mine))), flush=True)
    del text
    gc.collect()
    torch.cuda.empty_cache()
    model = ZImageStudent(1024, repo=str(SNAPSHOT), param_dtype=torch.bfloat16, gradient_checkpointing=False)
    sampling = dict(sigmas=turbo_sigmas(args.teacher_steps), renoise=False, detach_between_steps=True)
    model = model.to('cuda').eval()
    vae = ZImageVAE(str(SNAPSHOT), device='cuda', gradient_checkpointing=False)
    done, batch, ceiling, safe_batch = 0, 32, 64, 0
    def render(sub):
        noise = torch.stack([torch.stack([noise_for(i, 1000*k, (16,128,128),'cuda')
            for k in range(args.teacher_steps)]) for i,_ in sub])
        latent = model(noise, [caps[i].to('cuda') for i,_ in sub], sampling)[-1]
        pixels = vae.decode(latent)  # Official evaluator: no autocast around sampler/decode.
        assert bool(torch.isfinite(pixels).all())
        return pixels
    with torch.no_grad():
        while done < len(mine):
            sub = mine[done:done+batch]
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            try:
                pixels = render(sub)
            except torch.cuda.OutOfMemoryError as error:
                print(json.dumps(dict(event='bounded_batch_OOM',rank=args.rank,batch=batch,error=str(error))),flush=True)
                error.__traceback__ = None
                assert batch > 1, 'Batch1 OOM is a real generation failure'
                ceiling = batch-1
                batch = safe_batch if 0 < safe_batch < batch else max(1,batch//2)
                gc.collect(); torch.cuda.empty_cache()
                continue
            peak = torch.cuda.max_memory_allocated()
            for (ident,_), img in zip(sub,pixels):
                temporary = images/f'.{ident:04d}.png'
                save_uint8_png(img,str(temporary))
                temporary.replace(images/f'{ident:04d}.png')
            del pixels
            done += len(sub)
            safe_batch = batch
            value = dict(event='actual_images_saved',rank=args.rank,completed=done,total=len(mine),batch=len(sub),
                seconds=time.monotonic()-started,peak_allocated_bytes=peak,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                allocator_cap_bytes=int(cap),official_model_revision=REVISION,teacher_steps=args.teacher_steps)
            (args.out/f'generate-rank{args.rank}-progress.json').write_text(json.dumps(value,indent=2)+'\n')
            print(json.dumps(value),flush=True)
            larger = min(ceiling,64,batch+32,(batch+ceiling+1)//2)
            if larger > batch and peak*larger/batch < .90*cap:
                batch = larger


if __name__ == '__main__':
    main()
