"""Bind portrait inference to the exact independently verified milestone object."""
import json
import hashlib
import io
from pathlib import Path
import re

ROOT = Path('/data/rdm-workspace/rdm-sets')
ARCHIVE = Path('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A')
HEADS = ['s1/dinov3_l', 's1/qwen3vl_4b', 's2/dinov3_l', 's2/qwen3vl_4b']


def require_remote_memory_headroom(required=96*1024**3):
    info = Path('/proc/meminfo')
    assert info.exists(), 'Archived model loading is remote Linux-only'
    fields = dict(line.split(':', 1) for line in info.read_text().splitlines())
    host = int(fields['MemAvailable'].split()[0])*1024
    available = host
    limit = Path('/sys/fs/cgroup/memory.max')
    current = Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and current.exists() and limit.read_text().strip() != 'max':
        available = min(available, int(limit.read_text())-int(current.read_text()))
    assert available >= required, 'Need96GiB host/cgroup headroom for a remote RAM-only model load'
    return available


def read_verified_bytes(path, receipt, progress=None):
    data = io.BytesIO(); digest = hashlib.sha256(); read = 0; reported = -1
    try:
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2), b''):
                data.write(block); digest.update(block); read += len(block)
                if progress is not None and read//1024**3 != reported:
                    reported = read//1024**3
                    progress(read, receipt['bytes'])
        assert read == receipt['bytes'] and digest.hexdigest() == receipt['sha256'], 'Archive RAM stream differs from verified object'
        data.seek(0)
        return data
    except BaseException:
        data.close()
        raise


def load_archived_state(path, receipt, progress=None):
    import torch
    available = require_remote_memory_headroom()
    data = read_verified_bytes(path, receipt, progress)
    try:
        state = torch.load(data, map_location='cpu', weights_only=False, mmap=False)
    finally:
        data.close()
    return state, available


def archived_checkpoint(step, control=None):
    control = Path(control) if control is not None else ROOT/'control/twohead-retain50-archive'
    proof = control/f'checkpoint-final-readback-{step}.json'
    if not proof.exists():
        return None
    receipt = json.loads(proof.read_text())
    assert receipt['passed'] and receipt['step'] == step, 'Archive receipt step mismatch'
    path = Path(receipt['path']).resolve()
    assert path == (ARCHIVE/f'step_{step:07d}.pth').resolve(), 'Wrong milestone/path in archive receipt'
    assert receipt['bytes'] > 24_000_000_000 and re.fullmatch('[0-9a-f]{64}', receipt['sha256']), 'Invalid complete-model receipt'
    assert path.stat().st_size == receipt['bytes'], 'Archived student size differs from full SHA receipt'
    return path, receipt


def validate_state(state, step, archived=False):
    assert state['step'] == state['schedule']['step'] == step, 'Loaded checkpoint is a different step'
    assert state['world_size'] == 2 and state['num_steps'] == 2
    assert state['sampling_args'] == dict(sigmas=[1., .75], renoise=True, detach_between_steps=True)
    if archived:
        model = state['model']
        assert len(model) == 521 and sum(v.numel() for v in model.values()) == 6154908736
        assert all(str(v.dtype) == 'torch.float32' for v in model.values())
        assert state['adaptive_d_lr']['head_keys'] == HEADS
        assert state['adaptive_d_lr']['last_completed_step'] == step
    else:
        assert len(state['critics']) == 2
        assert all(c['contract']['head_order'] == ['dinov3_l', 'qwen3vl_4b'] for c in state['critics'])
