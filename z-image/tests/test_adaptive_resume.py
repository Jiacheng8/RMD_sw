"""Migration safety without writing any model/tensor files or using a GPU."""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


SPEC = importlib.util.spec_from_file_location('freeze_adaptive',
    Path(__file__).resolve().parents[2]/'ops/h20-freeze-adaptive-resume.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def migration(tmp_path, monkeypatch):
    root = tmp_path/'rdm-sets'
    experiment = root/'experiment'
    experiment.mkdir(parents=True)
    (root/'control').mkdir()
    # Deliberately not a checkpoint: mocked load returns metadata, never tensors.
    (experiment/'resume.pth').write_bytes(b'closed-metadata-fixture')
    rows = [{'step': i, 'skipped': False, 's1/gan/d_skipped': False,
             's2/gan/d_skipped': False} for i in range(1, 53)]
    (experiment/'train_log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    approval = root/'control/approval.json'
    approval.write_text(json.dumps({'accept_generator_optimizer_reset': True,
        'approved_utc': 'test-only', 'user_instruction': 'test-only explicit migration'}))
    state = {'step': 50, 'num_steps': 2, 'model': {}, 'schedule': {}, 'cache': {},
             'spatial_cache': {}, 'critics': [{'contract': {'loss': 'ns', 'lr': 1e-5}} for _ in range(2)]}
    monkeypatch.setattr(MODULE, 'ROOT', root)
    monkeypatch.setattr(MODULE, 'EXPERIMENT', experiment)
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(load=lambda *a, **k: state))
    return root, experiment, approval, state, rows


def test_freeze_retains_closed_inode_and_preserves_entire_unsaved_suffix(migration):
    root, experiment, approval, state, rows = migration
    receipt = MODULE._prepare_locked(50, 2147483647, approval)
    frozen = Path(receipt['checkpoint'])
    assert frozen.is_relative_to(root/'control/spatial-adaptive-migrations')
    assert frozen.stat().st_ino == (experiment/'resume.pth').stat().st_ino
    assert receipt['checkpoint_sha256'] == hashlib.sha256(b'closed-metadata-fixture').hexdigest()
    assert receipt['preserved_uncheckpointed_suffix_steps'] == [51, 52]
    assert not receipt['generator_optimizer_in_checkpoint']
    assert receipt['accept_generator_optimizer_reset'] and receipt['no_training_started']
    assert [json.loads(s) for s in Path(receipt['original_jsonl']).read_text().splitlines()] == rows
    assert [json.loads(s)['step'] for s in (experiment/'train_log.jsonl').read_text().splitlines()] == list(range(1, 51))
    assert json.loads((frozen.parent/'frozen.json').read_text()) == receipt


@pytest.mark.parametrize('failure', ['approval', 'wrong_step', 'missing_cache', 'wrong_loss', 'skip', 'live_pid'])
def test_invalid_migration_cannot_freeze_or_rewrite_logs(migration, failure):
    root, experiment, approval, state, rows = migration
    pid = 2147483647
    if failure == 'approval':
        approval.write_text(json.dumps({'accept_generator_optimizer_reset': False}))
    elif failure == 'wrong_step': state['step'] = 49
    elif failure == 'missing_cache': del state['spatial_cache']
    elif failure == 'wrong_loss': state['critics'][0]['contract']['loss'] = 'hinge'
    elif failure == 'skip':
        rows[0]['s1/gan/d_skipped'] = True
        (experiment/'train_log.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif failure == 'live_pid': pid = 1
    before = (experiment/'train_log.jsonl').read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        MODULE._prepare_locked(50, pid, approval)
    assert not (root/'control/spatial-adaptive-migrations').exists()
    assert (experiment/'train_log.jsonl').read_bytes() == before


def test_active_run_lock_blocks_migration_even_if_wrapper_pid_is_absent(migration, monkeypatch):
    root, experiment, approval, state, rows = migration
    monkeypatch.setattr(MODULE.sys, 'platform', 'linux')
    with (root.parent/'rdm-run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            MODULE.prepare(50, 2147483647, approval)
    assert not (root/'control/spatial-adaptive-migrations').exists()


def test_migration_entry_refuses_mac_before_opening_remote_paths(monkeypatch):
    monkeypatch.setattr(MODULE.sys, 'platform', 'darwin')
    with pytest.raises(RuntimeError, match='Linux-only'):
        MODULE.prepare(50, 2147483647, '/not-read')
