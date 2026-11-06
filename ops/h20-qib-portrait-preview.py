#!/usr/bin/env python3
"""Fixed official QIB portrait cases, pretrained0 and strictly bound saved stages.

Runs only on an explicitly selected spare GPU; never changes training or saves weights.
"""
import argparse, datetime, fcntl, gc, hashlib, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path('/data/rdm-workspace/rdm-sets')
SOURCE = pathlib.Path(os.environ.get('H20_TRAIN_SOURCE','/data/rdm-workspace/RDM-twohead-lr10-v1'))
TRAIN_SHA = os.environ.get('H20_TRAIN_SHA','a25e4c42794b13d5c881d3f1994944929b019190')
EXPERIMENT = ROOT/'zimage/work_dirs/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A'
OUT = ROOT/'visualizations/twohead-lr10-A/qib-portraits-v1'
SAMPLING = dict(sigmas=[1., .75], renoise=True, detach_between_steps=True)


def atomic(path, value):
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step', type=int, choices=range(50, 501, 50), required=True)
    p.add_argument('--gpu', type=int, choices=[6, 7], required=True)
    p.add_argument('--include-initial', action='store_true')
    p.add_argument('--checkpoint', type=pathlib.Path,
                   help='Exact final-SHA-verified independent archived student for this step')
    a = p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(a.gpu)
    assert (SOURCE/'SOURCE_COMMIT').read_text().strip() == TRAIN_SHA
    OUT.mkdir(parents=True, exist_ok=True)
    cases_path = pathlib.Path(__file__).with_name('qib-portrait-cases-v1.json')
    spec = json.loads(cases_path.read_text())
    cases = spec['cases']
    assert len(cases) == 8 and len({c['ID'] for c in cases}) == 8
    assert spec['revision'] == 'd2493deb153b020cf169c7e3f57d15e4dd697038'
    folder = OUT/f'step_{a.step:07d}'
    folder.mkdir(exist_ok=True)
    assert not (folder/'preview-verified.json').exists(), 'Stage already complete'
    with (ROOT.parent/f'rdm-preview-gpu{a.gpu}.lock').open('a') as lock:
        # Other authorized fixed-case previews share GPU6. Queue behind their
        # lock instead of reporting contention as a failed inference run.
        # The outer wrapper's 1800s timeout also bounds this wait.
        print(json.dumps(dict(step=a.step, phase='waiting_preview_GPU_lock', gpu=a.gpu)), flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        checks = []
        for _ in range(2):
            used, total, util = map(int, subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True).strip().split(','))
            assert total-used >= 48*1024, 'Need32GiB cap plus16GiB free headroom'
            checks.append(dict(used_mib=used, total_mib=total, utilization=util))
            time.sleep(2)
        import torch
        from PIL import Image, ImageDraw, ImageFont
        sys.path[:0] = [str(SOURCE/'z-image'), str(SOURCE)]
        from zimage_distill.model import ZImageStudent, ZImageVAE, QwenPromptEncoder, noise_for
        from rdm.utils.io import save_uint8_png
        torch.set_num_threads(2)
        from h20_portrait_checkpoint import archived_checkpoint, validate_state, load_archived_state
        archive_proof = None
        checkpoint = EXPERIMENT/'resume.pth'
        if a.checkpoint is not None:
            verified = archived_checkpoint(a.step)
            assert verified is not None, 'Archived input has not completed final-object SHA verification'
            checkpoint, archive_proof = verified
            assert checkpoint == a.checkpoint.resolve(), 'Requested input differs from the verified milestone'
        before = checkpoint.stat()
        memory_available = None
        if archive_proof is not None:
            def loading_progress(read_bytes, total_bytes):
                event = dict(step=a.step, phase='archive_RAM_read', read_bytes=read_bytes,
                             total_bytes=total_bytes, images=0)
                atomic(folder/'progress.json', event)
                print(json.dumps(event), flush=True)
            state, memory_available = load_archived_state(checkpoint, archive_proof, loading_progress)
        else:
            state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        after = checkpoint.stat()
        if archive_proof is not None:
            assert before.st_size == after.st_size == archive_proof['bytes']
        else:
            assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        validate_state(state, a.step, archived=archive_proof is not None)
        receipt = dict(training_commit=TRAIN_SHA, checkpoint=str(checkpoint), checkpoint_step=a.step,
            checkpoint_inode=before.st_ino, checkpoint_bytes=before.st_size, physical_gpu=a.gpu,
            allocator_cap_gib=32, preflight=checks, cases_sha256=digest(cases_path),
            sampling=SAMPLING, image_size=1024, seeds=[0, 1], batch_size=2,
            attention='native', decode_autocast=False, no_checkpoint_copy_or_hardlink=True,
            checkpoint_kind='verified-independent-student' if archive_proof else 'rolling-resume',
            checkpoint_archive_SHA256=archive_proof['sha256'] if archive_proof else None,
            archive_load_mode='remote-process-RAM-no-disk-copy' if archive_proof else None,
            remote_memory_headroom_before_load=memory_available,
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic(folder/'preview-command.json', receipt)
        atomic(OUT/'cases.json', spec)
        used, total = map(int, subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
            '--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'], text=True).strip().split(','))
        assert total-used >= 48*1024, 'GPU headroom changed during archived input load'
        torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        text = QwenPromptEncoder(device='cuda', cache_size=8, batch=2)
        counts = []
        for c in cases:
            wrapped = text.tokenizer.apply_chat_template([dict(role='user', content=c['prompt_en'])],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            length = len(text.tokenizer(wrapped, truncation=False).input_ids)
            assert length <= 512, f"QIB{c['ID']} would truncate at512 tokens: {length}"
            counts.append(dict(ID=c['ID'], tokens=length))
        receipt['prompt_token_counts'] = counts
        cap_all = text([c['prompt_en'] for c in cases])
        del text
        gc.collect(); torch.cuda.empty_cache()
        model = ZImageStudent(1024, param_dtype=torch.bfloat16, gradient_checkpointing=False).to('cuda').eval()
        vae = ZImageVAE(device='cuda', gradient_checkpointing=False)
        stages = [0, a.step] if a.include_initial else [a.step]
        jobs = [(i, s) for i in range(8) for s in (0, 1)]
        generated = []
        with torch.no_grad():
            for stage in stages:
                destination = OUT/f'step_{stage:07d}'
                destination.mkdir(exist_ok=True)
                assert not (destination/'preview-verified.json').exists(), 'Stage already verified'
                if stage:
                    model.load_state_dict(state['model'], strict=True)
                    del state
                for lo in range(0, len(jobs), 2):
                    sub = jobs[lo:lo+2]
                    noise = torch.stack([torch.stack([
                        noise_for(cases[i]['ID'], seed, (16, 128, 128), 'cuda'),
                        noise_for(cases[i]['ID'], seed+1000, (16, 128, 128), 'cuda')]) for i, seed in sub])
                    cap = [cap_all[i] for i, seed in sub]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        latent = model(noise, cap, SAMPLING)[-1]
                    pixels = vae.decode(latent)
                    assert bool(torch.isfinite(pixels).all())
                    for (i, seed), img in zip(sub, pixels):
                        path = destination/f"qib{cases[i]['ID']:04d}-seed{seed}.png"
                        save_uint8_png(img, str(path)); generated.append(path)
                    progress = dict(step=stage, images=lo+len(sub), total=16,
                        utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    atomic(folder/'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                files = []
                for path in sorted(destination.glob('qib*.png')):
                    with Image.open(path) as im:
                        assert im.size == (1024, 1024) and im.mode == 'RGB'; im.load()
                    files.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path)))
                assert len(files) == 16
                proof = dict(receipt, passed=True, actual_step=stage, files=files,
                    completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
                if stage == 0:
                    proof['checkpoint'] = None; proof['checkpoint_step'] = 0
                    proof['weights'] = 'official pretrained Z-Image-Turbo, before load_state_dict'
                atomic(destination/'preview-verified.json', proof)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
        tile_checks = []
        for seed in (0, 1):
            for page in (0, 1):
                sheet = Image.new('RGB', (2048, 48+4*1092), 'white')
                draw = ImageDraw.Draw(sheet)
                draw.text((12, 10), 'Pretrained0 / 2-step / 1024px', font=font, fill='black')
                draw.text((1036, 10), f'Iteration{a.step} / 2-step / 1024px', font=font, fill='black')
                for row, case in enumerate(cases[page*4:page*4+4]):
                    for col, stage in enumerate((0, a.step)):
                        path = OUT/f'step_{stage:07d}'/f"qib{case['ID']:04d}-seed{seed}.png"
                        with Image.open(path) as im:
                            assert im.size == (1024, 1024)
                            box = (col*1024, 48+row*1092)
                            sheet.paste(im, box)
                            tile_checks.append((path, box))
                    draw.text((12, 48+row*1092+1034), f"QIB ID {case['ID']} | {case['label']} | seed{seed}", font=font, fill='black')
                out = folder/f'portrait-0-{a.step}-seed{seed}-page{page+1}.png'
                sheet.save(out)
                with Image.open(out) as reread:
                    for path, (x, y) in tile_checks[-8:]:
                        with Image.open(path) as original:
                            assert reread.crop((x, y, x+1024, y+1024)).tobytes() == original.tobytes()
        proof = json.loads((folder/'preview-verified.json').read_text())
        proof['native_comparisons'] = [dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p))
            for p in sorted(folder.glob('portrait-*.png'))]
        proof['native_tile_pixel_equality_verified'] = len(tile_checks)
        atomic(folder/'preview-verified.json', proof)
        print(json.dumps(dict(passed=True, step=a.step, generated=len(generated),
            peak_allocated_gib=proof['peak_allocated_gib'])), flush=True)


if __name__ == '__main__':
    main()
