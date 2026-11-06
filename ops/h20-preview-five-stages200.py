#!/usr/bin/env python3
"""Add actual archived200 previews to existing0/50/100/150 native comparisons.

The saved 50/100 previews use different prompt sets; unavailable intersections
are recorded explicitly. No weights are reconstructed, copied or hard-linked.
"""
import argparse, datetime, fcntl, gc, hashlib, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path('/data/rdm-workspace/rdm-sets')
SOURCE = pathlib.Path('/data/rdm-workspace/RDM-twohead-lr10-v1')
TRAIN_SHA = 'a25e4c42794b13d5c881d3f1994944929b019190'
EXPERIMENT = ROOT/'zimage/work_dirs/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A'
from h20_portrait_checkpoint import archived_checkpoint, load_archived_state, validate_state
CKPT = None
BASE0 = ROOT/'visualizations/spatial-A/step_0000000'
STEP50 = ROOT/'visualizations/twohead-lr10-A/step_0000050'
STEP100 = ROOT/'visualizations/twohead-lr10-A/training-prompts-100-v1'
OUT = ROOT/'visualizations/twohead-lr10-A/comparison-0-50-100-150-200-v1'


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
    global CKPT
    bound = archived_checkpoint(200)
    assert bound is not None, 'Need final-named full-SHA receipt for student200'
    CKPT, archive_receipt = bound
    assert sys.platform.startswith('linux') and os.environ['CUDA_VISIBLE_DEVICES'] == str(a.gpu)
    assert (SOURCE/'SOURCE_COMMIT').read_text().strip() == TRAIN_SHA
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'comparison-verified.json').exists(), 'Comparison already complete'
    for folder in (BASE0, STEP50, STEP100):
        assert json.loads((folder/'preview-verified.json').read_text())['passed']
    selections = json.loads((STEP100/'selected-prompts.json').read_text())
    assert len(selections) == 8
    receipt = dict(training_commit=TRAIN_SHA, checkpoint=str(CKPT), checkpoint_step=200,
        physical_gpu=a.gpu, allocator_cap_gib=32, image_size=1024, training_gpus_untouched=[4, 5],
        availability={'handwritten_prompts': [0, 50, 150, 200], 'actual_training_prompts': [0, 100, 150, 200]},
        missing='50 and100 rolling snapshots were overwritten; their saved previews use different prompts',
        no_synthetic_missing_stages=True, no_checkpoint_copy_or_hardlink=True)
    with (ROOT.parent/f'rdm-preview-gpu{a.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
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
        def archive_progress(read, total):
            value = dict(group='checkpoint200-RAM-SHA', read_bytes=read, total_bytes=total)
            write('progress.json', value); print(json.dumps(value), flush=True)
        state, available = load_archived_state(CKPT, archive_receipt, archive_progress)
        validate_state(state, 200, archived=True)
        receipt.update(archive_receipt=archive_receipt, RAM_headroom_bytes=available,
                       checkpoint_loading='full-SHA-verified remote RAM; no disk staging')
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
                    path = OUT/'handwritten200'/f'{p:05d}_s{s}.png'
                    save_uint8_png(img, str(path)); generated.append(path)
                print(json.dumps(dict(group='handwritten200', complete=lo+len(sub), total=24)), flush=True)
                write('progress.json', dict(group='handwritten200', complete=lo+len(sub), total=24))
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
                    path = OUT/'training200'/f"{row['index']:02d}-iteration200.png"
                    save_uint8_png(img, str(path)); generated.append(path)
                print(json.dumps(dict(group='training200', complete=lo+len(chosen), total=8)), flush=True)
                write('progress.json', dict(group='training200', complete=lo+len(chosen), total=8))

        after = CKPT.stat()
        assert before.st_size == after.st_size == archive_receipt['bytes']
        assert len(generated) == len(set(generated)) == 32
        for path in generated:
            with Image.open(path) as img:
                assert img.size == (1024, 1024) and img.mode == 'RGB'
                img.load()
        used = set(generated)
        receipt.update(passed=True, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            generated_images=32, source_pngs=[dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(used)],
            handwritten_prompt_method='published noise_for; online Qwen; native attention; batch2; seeds0/1; matched0/50',
            training_prompt_method='actual selected row_noise seed0 epoch0; cached contexts; FA3; batch2; matched0/100',
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        write('comparison-verified.json', receipt)
        from h20_export_native_five_columns import main as export_native
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Export consumes verified PNG pixels only.
        export_native()
        print(json.dumps({k:v for k,v in receipt.items() if k != 'source_pngs'}), flush=True)


if __name__ == '__main__':
    main()
