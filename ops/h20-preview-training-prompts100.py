#!/usr/bin/env python3
"""Read-only train-row comparison: canonical teacher, initial two-step, iteration100.

No training hooks or checkpoint copies. Rows are fixed before looking at images.
The two student variants share the actual cached contexts and training row_noise.
The teacher's nine-step image has its own dataset seed, explicitly recorded.
"""
import argparse, datetime, fcntl, hashlib, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path('/data/rdm-workspace/rdm-sets')
SOURCE = pathlib.Path('/data/rdm-workspace/RDM-twohead-lr10-v1')
SHA = 'a25e4c42794b13d5c881d3f1994944929b019190'
EXPERIMENT = ROOT / 'zimage/work_dirs/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A'
CKPT = EXPERIMENT / 'resume.pth'
OUT = ROOT / 'visualizations/twohead-lr10-A/training-prompts-100-v1'
STEPS = [1, 15, 30, 45, 60, 75, 90, 100]


def write(name, data):
    target = OUT / name
    temporary = target.with_name(target.name + '.partial')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=[6, 7], required=True)
    args = parser.parse_args()
    assert sys.platform.startswith('linux') and os.environ['CUDA_VISIBLE_DEVICES'] == str(args.gpu)
    assert (SOURCE / 'SOURCE_COMMIT').read_text().strip() == SHA
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'preview-verified.json').exists(), 'Already complete; do not repeat'
    with (ROOT.parent / f'rdm-preview-gpu{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight = []
        for _ in range(2):
            used, total, util = map(int, subprocess.check_output([
                'nvidia-smi', '-i', str(args.gpu), '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True).strip().split(','))
            assert total - used >= 48 * 1024, 'Need 32GiB allocator cap plus 16GiB headroom'
            preflight.append(dict(used_mib=used, total_mib=total, utilization=util))
            time.sleep(2)

        import torch
        from PIL import Image, ImageDraw, ImageFont
        sys.path[:0] = [str(SOURCE / 'z-image'), str(SOURCE)]
        from zimage_distill.store import ZImageReferenceStore
        from zimage_distill.model import ZImageStudent, ZImageVAE
        from rdm.sw_lmmd.window_schedule import SlidingWindowSchedule
        from rdm.sw_lmmd.sharding import row_noise, row_seed
        from rdm.sw_lmmd.reference_images import ReferenceImageStore
        from rdm.utils.io import save_uint8_png

        torch.set_num_threads(2)
        before = CKPT.stat()
        state = torch.load(CKPT, map_location='cpu', weights_only=False, mmap=True)
        assert state['step'] == 100 and state['num_steps'] == 2
        assert all(c['contract']['head_order'] == ['dinov3_l', 'qwen3vl_4b'] for c in state['critics'])
        sampling = state['sampling_args']
        assert sampling['sigmas'] == [1.0, .75] and sampling['renoise']
        store = ZImageReferenceStore(str(ROOT / 'zimage/zimage-ref-v1/reference_store_qwen'),
                                    ['dinov3_l', 'siglip2', 'aimv2_huge'])
        assert store.has_context
        schedule = SlidingWindowSchedule(store.row_order(3407), window_size=128, stride=32)
        schedule.load_state_dict(state['schedule'])
        assert schedule.epoch == 0
        logs = [json.loads(line) for line in (EXPERIMENT / 'train_log.jsonl').read_text().splitlines() if line.strip()]
        by_step = {int(row['step']): row for row in logs if 'step' in row}
        assert all(step in by_step and not by_step[step].get('skipped', False) for step in STEPS)
        rows = [int(schedule._window_at(step - 1).active_ids[0]) for step in STEPS]
        assert len(set(rows)) == len(STEPS)
        real = ReferenceImageStore(ROOT / 'zimage/zimage-ref-v1/reference_images_patchgan')
        records, prompts = real.records(rows), store.prompts(rows)
        selections = []
        for i, (step, rid, prompt, record) in enumerate(zip(STEPS, rows, prompts, records)):
            meta = json.loads(record['teacher_metadata'])
            assert meta['prompt'] == prompt
            assert hashlib.sha256(prompt.encode()).hexdigest() == record['prompt_sha256']
            assert int(store.prompt_rows([rid])[0]) == record['prompt_id']
            assert meta['steps'] == 9 and meta['model'] == 'Tongyi-MAI/Z-Image-Turbo'
            data = real.read_pngs([rid])[0]
            (OUT / f'{i:02d}-reference.png').write_bytes(data)
            selections.append(dict(index=i, row_id=rid, prompt_id=record['prompt_id'],
                active_at_iteration=step, active_slot=0, prompt=prompt,
                prompt_sha256=record['prompt_sha256'], reference_png_sha256=record['png_sha256'],
                reference_metadata=meta, student_row_seed=row_seed(0, 0, rid),
                context_source='actual flat cached training context', reference_uses_student_noise=False))
        real.close()
        write('selected-prompts.json', selections)
        print(json.dumps({'selection': selections}, ensure_ascii=False), flush=True)

        torch.cuda.set_per_process_memory_fraction(32 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        model = ZImageStudent(1024, param_dtype=torch.bfloat16, gradient_checkpointing=False,
                              attention_backend='_flash_3_varlen_hub').to('cuda').eval()
        vae = ZImageVAE(device='cuda', gradient_checkpointing=False)
        started = time.monotonic()
        timings = {}
        with torch.no_grad():
            for tag in ('initial', 'iteration100'):
                if tag == 'iteration100':
                    model.load_state_dict(state['model'], strict=True)
                tick = time.monotonic()
                for lo in range(0, len(rows), 2):
                    ids = rows[lo:lo + 2]
                    noise = row_noise(ids, (2, 16, 128, 128), seed=0, epoch=0, device='cuda', dtype=torch.float32)
                    contexts = store.generator_context(ids, device='cuda', dtype=torch.float32)
                    assert all(c.shape[1] == 2560 and bool(torch.isfinite(c).all()) for c in contexts)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        latents = model(noise, contexts, sampling)
                        pixels = vae.decode(latents[-1])
                    assert bool(torch.isfinite(pixels).all())
                    for offset, img in enumerate(pixels):
                        save_uint8_png(img, str(OUT / f'{lo + offset:02d}-{tag}.png'))
                    torch.cuda.synchronize()
                    print(json.dumps(dict(variant=tag, completed=lo + len(ids), total=len(rows),
                                          seconds=time.monotonic() - tick)), flush=True)
                    write('progress.json', dict(variant=tag, completed=lo + len(ids), total=len(rows)))
                timings[tag] = time.monotonic() - tick

        after = CKPT.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        files = []
        for i in range(len(rows)):
            for tag in ('reference', 'initial', 'iteration100'):
                path = OUT / f'{i:02d}-{tag}.png'
                with Image.open(path) as img:
                    assert img.size == (1024, 1024) and img.mode == 'RGB'
                    img.load()
                files.append(dict(path=path.name, bytes=path.stat().st_size,
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
        for page in range(2):
            sheet = Image.new('RGB', (1152, 1744), 'white')
            draw = ImageDraw.Draw(sheet)
            for col, label in enumerate(('Dataset teacher / 9-step', 'Initial model / 2-step', 'Iteration100 / 2-step')):
                draw.text((col * 384 + 10, 12), label, fill='black', font=font)
            for row, i in enumerate(range(page * 4, page * 4 + 4)):
                y = 44 + row * 424
                for col, tag in enumerate(('reference', 'initial', 'iteration100')):
                    with Image.open(OUT / f'{i:02d}-{tag}.png') as img:
                        sheet.paste(img.resize((384, 384), Image.Resampling.LANCZOS), (col * 384, y))
                label = f"P{i + 1} | row {rows[i]} | active at iteration {STEPS[i]} | original prompt in selected-prompts.json"
                draw.text((10, y + 390), label, fill='black', font=font)
            sheet.save(OUT / f'training-prompts-page{page + 1}.jpg', quality=95)
        receipt = dict(passed=True, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            training_commit=SHA, checkpoint=str(CKPT), checkpoint_step=100, checkpoint_inode=before.st_ino,
            no_checkpoint_copied_or_pinned=True, training_gpus_untouched=[4, 5], physical_gpu=args.gpu,
            allocator_cap_gib=32, preflight=preflight, selection='active_slot0 of fixed iterations 1,15,30,45,60,75,90,100; chosen before rendering',
            sampling=sampling, student_noise='actual row_noise(seed=0,epoch=0,row_id), shape 2x16x128x128 FP32',
            student_context='actual cached training contexts, not re-encoded', teacher_seed_matches_student=False,
            teacher_steps=9, student_steps=2, image_count=24, generated_images=16, files=files,
            seconds=time.monotonic() - started, timings=timings,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        write('preview-verified.json', receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
