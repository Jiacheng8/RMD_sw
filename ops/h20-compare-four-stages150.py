#!/usr/bin/env python3
"""Compare existing actual 0/50/100 images with a new, strictly bound step150 preview.

The saved 50/100 previews use different prompt sets; unavailable intersections
are recorded explicitly. No weights are reconstructed, copied or hard-linked.
"""
import argparse, datetime, fcntl, gc, hashlib, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path('/data/rdm-workspace/rdm-sets')
SOURCE = pathlib.Path('/data/rdm-workspace/RDM-twohead-lr10-v1')
TRAIN_SHA = 'a25e4c42794b13d5c881d3f1994944929b019190'
EXPERIMENT = ROOT/'zimage/work_dirs/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A'
CKPT = EXPERIMENT/'resume.pth'
BASE0 = ROOT/'visualizations/spatial-A/step_0000000'
STEP50 = ROOT/'visualizations/twohead-lr10-A/step_0000050'
STEP100 = ROOT/'visualizations/twohead-lr10-A/training-prompts-100-v1'
OUT = ROOT/'visualizations/twohead-lr10-A/comparison-0-50-100-150-v1'


def write(name, value):
    p = OUT/name
    temporary = p.with_name(p.name+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, p)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=[6, 7], required=True)
    a = parser.parse_args()
    assert sys.platform.startswith('linux') and os.environ['CUDA_VISIBLE_DEVICES'] == str(a.gpu)
    assert (SOURCE/'SOURCE_COMMIT').read_text().strip() == TRAIN_SHA
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'comparison-verified.json').exists(), 'Comparison already complete'
    for folder in (BASE0, STEP50, STEP100):
        assert json.loads((folder/'preview-verified.json').read_text())['passed']
    selections = json.loads((STEP100/'selected-prompts.json').read_text())
    assert len(selections) == 8
    receipt = dict(training_commit=TRAIN_SHA, checkpoint=str(CKPT), checkpoint_step=150,
        physical_gpu=a.gpu, allocator_cap_gib=32, image_size=1024, training_gpus_untouched=[4, 5],
        availability={'handwritten_prompts': [0, 50, 150], 'actual_training_prompts': [0, 100, 150]},
        missing='50 and100 rolling snapshots were overwritten; their saved previews use different prompts',
        no_synthetic_missing_stages=True, no_checkpoint_copy_or_hardlink=True)
    with (ROOT.parent/f'rdm-preview-gpu{a.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checks = []
        for _ in range(2):
            used, total, util = map(int, subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True).strip().split(','))
            assert total-used >= 48*1024, 'Need32GiB allocator cap +16GiB headroom'
            checks.append(dict(used_mib=used, total_mib=total, utilization=util))
            time.sleep(2)
        import torch
        from PIL import Image, ImageDraw, ImageFont
        sys.path[:0] = [str(SOURCE/'z-image'), str(SOURCE)]
        from zimage_distill.model import ZImageStudent, ZImageVAE, QwenPromptEncoder, noise_for
        from zimage_distill.store import ZImageReferenceStore, read_prompts
        from zimage_distill.attention import pin_attention_kernel
        from rdm.sw_lmmd.sharding import row_noise
        from rdm.utils.io import save_uint8_png

        torch.set_num_threads(2)
        before = CKPT.stat()
        state = torch.load(CKPT, map_location='cpu', weights_only=False, mmap=True)
        assert state['step'] == 150 and state['schedule']['step'] == 150 and state['num_steps'] == 2
        assert all(c['contract']['head_order'] == ['dinov3_l', 'qwen3vl_4b'] for c in state['critics'])
        sampling = state['sampling_args']
        assert sampling == {'sigmas': [1., .75], 'renoise': True, 'detach_between_steps': True}
        receipt.update(preflight=checks, checkpoint_inode=before.st_ino, checkpoint_bytes=before.st_size,
                       started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), sampling=sampling)
        write('comparison-command.json', receipt)
        torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        model = ZImageStudent(1024, param_dtype=torch.bfloat16, gradient_checkpointing=False)
        model.load_state_dict(state['model'], strict=True)
        del state
        model = model.to('cuda').eval()
        vae = ZImageVAE(device='cuda', gradient_checkpointing=False)
        text = QwenPromptEncoder(device='cuda', cache_size=2)
        prompts = read_prompts(SOURCE/'ops/spatial-preview-prompts.jsonl')
        assert len(prompts) == 12
        original_prompt_path = pathlib.Path(json.loads((BASE0/'preview-verified.json').read_text())['prompts'])
        assert original_prompt_path.read_bytes() == (SOURCE/'ops/spatial-preview-prompts.jsonl').read_bytes()
        generated = []
        with torch.no_grad():
            jobs = [(p, s) for p in range(12) for s in (0, 1)]
            for lo in range(0, len(jobs), 2):
                sub = jobs[lo:lo+2]
                noise = torch.stack([torch.stack([noise_for(p, s, (16, 128, 128), 'cuda'),
                    noise_for(p, 1000+s, (16, 128, 128), 'cuda')]) for p, s in sub])
                cap = text([prompts[p] for p, _ in sub])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    latent = model(noise, cap, sampling)[-1]
                pixels = vae.decode(latent)  # Matches published 0/50 sampler: decode outside autocast.
                assert bool(torch.isfinite(pixels).all())
                for (p, s), img in zip(sub, pixels):
                    path = OUT/'handwritten150'/f'{p:05d}_s{s}.png'
                    save_uint8_png(img, str(path)); generated.append(path)
                print(json.dumps(dict(group='handwritten150', complete=lo+len(sub), total=24)), flush=True)
                write('progress.json', dict(group='handwritten150', complete=lo+len(sub), total=24))
            del text, cap, pixels, latent, noise
            gc.collect(); torch.cuda.empty_cache()

            pin_attention_kernel('_flash_3_varlen_hub')
            model.transformer.set_attention_backend('_flash_3_varlen_hub')
            store = ZImageReferenceStore(str(ROOT/'zimage/zimage-ref-v1/reference_store_qwen'),
                                        ['dinov3_l', 'siglip2', 'aimv2_huge'])
            for lo in range(0, len(selections), 2):
                chosen = selections[lo:lo+2]; ids = [r['row_id'] for r in chosen]
                assert store.prompts(ids) == [r['prompt'] for r in chosen]
                noise = row_noise(ids, (2, 16, 128, 128), seed=0, epoch=0, device='cuda', dtype=torch.float32)
                cap = store.generator_context(ids, device='cuda', dtype=torch.float32)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    latent = model(noise, cap, sampling)[-1]
                    pixels = vae.decode(latent)  # Matches actual step100 training-prompt preview.
                assert bool(torch.isfinite(pixels).all())
                for row, img in zip(chosen, pixels):
                    path = OUT/'training150'/f"{row['index']:02d}-iteration150.png"
                    save_uint8_png(img, str(path)); generated.append(path)
                print(json.dumps(dict(group='training150', complete=lo+len(chosen), total=8)), flush=True)
                write('progress.json', dict(group='training150', complete=lo+len(chosen), total=8))

        after = CKPT.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        assert len(generated) == len(set(generated)) == 32
        for path in generated:
            with Image.open(path) as img:
                assert img.size == (1024, 1024) and img.mode == 'RGB'
                img.load()
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
        used = set(generated)

        def sheet(name, columns, examples, paths):
            image = Image.new('RGB', (1152, 44+416*len(examples)), 'white')
            draw = ImageDraw.Draw(image)
            for col, label in enumerate(columns):
                draw.text((col*384+10, 10), label, font=font, fill='black')
            for row, i in enumerate(examples):
                for col, path in enumerate(paths(i)):
                    used.add(path)
                    with Image.open(path) as img:
                        assert img.size == (1024, 1024) and img.mode == 'RGB'
                        img.load()
                        image.paste(img.resize((384, 384), Image.Resampling.LANCZOS), (col*384, 44+row*416))
                draw.text((10, 44+row*416+390), f'P{i+1:02d} / same prompt and noise within this row', font=font, fill='black')
            image.save(OUT/name, quality=95)

        for seed in (0, 1):
            sheet(f'compare-0-50-150-seed{seed}.jpg', ['Initial0 / 2-step', 'Iteration50 / 2-step', 'Iteration150 / 2-step'],
                [0, 1, 3, 6, 9, 11], lambda i: [BASE0/f'{i:05d}_s{seed}.png', STEP50/f'{i:05d}_s{seed}.png', OUT/'handwritten150'/f'{i:05d}_s{seed}.png'])
        for page in range(2):
            sheet(f'compare-0-100-150-training-page{page+1}.jpg', ['Initial0 / 2-step', 'Iteration100 / 2-step', 'Iteration150 / 2-step'],
                list(range(page*4, page*4+4)), lambda i: [STEP100/f'{i:02d}-initial.png', STEP100/f'{i:02d}-iteration100.png', OUT/'training150'/f'{i:02d}-iteration150.png'])
        receipt.update(passed=True, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            generated_images=32, source_pngs=[dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(used)],
            handwritten_prompt_method='published noise_for; online Qwen; native attention; batch2; seeds0/1; matched0/50',
            training_prompt_method='actual selected row_noise seed0 epoch0; cached contexts; FA3; batch2; matched0/100',
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        write('comparison-verified.json', receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k != 'source_pngs'}), flush=True)


if __name__ == '__main__':
    main()
