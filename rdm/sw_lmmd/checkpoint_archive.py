"""Preserve full-precision milestone weights on remote object storage.

The local source is removed only after destination SHA readback and a durable receipt.
Resume checkpoints are never candidates for deletion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def inspect_student(path, step):
    import torch
    torch.set_num_threads(1)
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert state['step'] == state['schedule']['step'] == step
    assert state['world_size'] == 2 and state['num_steps'] == 2
    assert not any(k in state for k in ['optimizer', 'cache', 'spatial_cache', 'critics', 'critic'])
    model = state['model']
    assert len(model) == 521 and sum(v.numel() for v in model.values()) == 6154908736
    assert all(v.dtype == torch.float32 for v in model.values())
    return dict(step=step, model_tensors=len(model), model_numel=6154908736, model_dtype='float32')


def archive_checkpoint(source, archive_dir, receipt_path, validator=inspect_student, *, verify_final=False):
    source = Path(source); archive_dir = Path(archive_dir); receipt_path = Path(receipt_path)
    assert source.name.startswith('step_') and source.suffix == '.pth', 'Never archive/delete resume state'
    step = int(source.stem.split('_')[1]); metadata = validator(source, step)
    before = source.stat(); stamp = (before.st_ino, before.st_size, before.st_mtime_ns)
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir/source.name
    temporary = destination.with_name(destination.name+f'.partial-{os.getpid()}')
    if destination.exists():
        expected = sha256(source)
        assert destination.stat().st_size == before.st_size and sha256(destination) == expected, 'Existing archived weight differs; preserved both'
    else:
        expected_hash = hashlib.sha256()
        with source.open('rb') as incoming, temporary.open('xb') as outgoing:
            for block in iter(lambda: incoming.read(8*1024**2), b''):
                outgoing.write(block); expected_hash.update(block)
            outgoing.flush(); os.fsync(outgoing.fileno())
        expected = expected_hash.hexdigest()
        assert temporary.stat().st_size == before.st_size and sha256(temporary) == expected, 'Archive readback failed; local source preserved'
        os.replace(temporary, destination)
        assert destination.stat().st_size == before.st_size, 'Published archive size differs'
    if verify_final:
        assert destination.stat().st_size == before.st_size and sha256(destination) == expected, 'Final-named archive readback failed; source preserved'
    after = source.stat()
    assert stamp == (after.st_ino, after.st_size, after.st_mtime_ns), 'Local source changed during archive; preserved'
    receipt = dict(metadata, passed=True, source=str(source), destination=str(destination),
        bytes=before.st_size, sha256=expected, verified_epoch=time.time(), local_source_removed=False,
        final_named_sha_readback=bool(verify_final))
    atomic_json(receipt_path, receipt)
    atomic_json(destination.with_suffix('.verified.json'), receipt)
    # Both SHA readback and receipts must precede removing our completed staging file.
    source.unlink()
    receipt['local_source_removed'] = True
    atomic_json(receipt_path, receipt)
    atomic_json(destination.with_suffix('.verified.json'), receipt)
    return receipt


def wait_for_space(directory, required_bytes, worker_exit, timeout=1200):
    deadline = time.monotonic()+timeout
    while shutil.disk_usage(directory).free < required_bytes:
        path = Path(worker_exit)
        if path.exists():
            raise RuntimeError('Archive worker exited before sufficient staging space; preserve checkpoints and inspect archive log')
        if time.monotonic() >= deadline:
            raise TimeoutError('Checkpoint staging waits20min for archive space; no checkpoint overwritten')
        time.sleep(2)


def watch(source_dir, archive_dir, control_dir, *, first_step=200, last_step=500, interval=50, verify_final=False):
    source_dir, archive_dir, control_dir = map(Path, [source_dir, archive_dir, control_dir])
    assert 0 < first_step <= last_step and interval > 0 and (last_step-first_step) % interval == 0
    results = []
    deadline = time.monotonic()+36*3600
    for step in range(first_step, last_step+1, interval):
        receipt_path = control_dir/f'checkpoint-archive-{step}.json'
        source = source_dir/f'step_{step:07d}.pth'
        while not source.exists():
            if receipt_path.exists():
                saved = json.loads(receipt_path.read_text())
                if saved.get('passed'):
                    destination = Path(saved['destination'])
                    assert destination.stat().st_size == saved['bytes'] and sha256(destination) == saved['sha256']
                    results.append(saved); break
            assert time.monotonic() < deadline, f'Archive wait exceeded36h at step{step}'
            time.sleep(2)
        else:
            print(json.dumps(dict(event='archive_started', step=step, bytes=source.stat().st_size)), flush=True)
            results.append(archive_checkpoint(source, archive_dir, receipt_path, verify_final=verify_final))
            print(json.dumps(dict(event='archive_verified', **results[-1])), flush=True)
        atomic_json(control_dir/'checkpoint-archive-progress.json', dict(
            verified_steps=[r['step'] for r in results], pending_steps=list(range(step+interval, last_step+1, interval)),
            last_receipt=results[-1], updated_epoch=time.time()))
    atomic_json(control_dir/'checkpoint-archive-complete.json', dict(passed=True, weights=results))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', required=True)
    p.add_argument('--archive-dir', required=True)
    p.add_argument('--control-dir', required=True)
    p.add_argument('--first-step', type=int, default=200)
    p.add_argument('--last-step', type=int, default=500)
    p.add_argument('--interval', type=int, default=50)
    p.add_argument('--verify-final-named', action='store_true')
    args = p.parse_args()
    assert args.archive_dir.startswith('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints/')
    control = Path(args.control_dir); control.mkdir(parents=True, exist_ok=True)
    code = 0
    try:
        watch(args.source_dir, args.archive_dir, control, first_step=args.first_step,
              last_step=args.last_step, interval=args.interval, verify_final=args.verify_final_named)
    except BaseException:
        import traceback
        traceback.print_exc(); code = 1
    finally:
        (control/'checkpoint-archive-worker.exit').write_text(str(code)+'\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
