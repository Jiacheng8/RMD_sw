#!/usr/bin/env python3
"""Freeze a closed A checkpoint after an explicitly approved legacy migration.

Never stops a process. Keeps the original JSONL, checkpoint and all suffix logs.
Large files stay on /data; the immutable input is a hard link, not a Mac copy.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path('/data/rdm-workspace/rdm-sets')
EXPERIMENT = ROOT/'zimage/work_dirs/zimage-2step-1024-dlr1e-5-spatial-patchgan-softplus'


def prepare(expected_step, launcher_pid, approval_file):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('H20 migration is Linux-only; no checkpoint may be staged on the Mac')
    with (ROOT.parent/'rdm-run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare_locked(expected_step, launcher_pid, approval_file)


def _prepare_locked(expected_step, launcher_pid, approval_file):
    import torch

    root = ROOT.resolve(strict=True)
    if not 0 < expected_step < 500 or expected_step % 50:
        raise ValueError('Migration must use a completed 50-step resume below the full500 target')
    if launcher_pid <= 1 or Path(f'/proc/{launcher_pid}').exists():
        raise RuntimeError('Original owned launcher must be actually terminal before migration')
    approval_file = Path(approval_file).resolve(strict=True)
    if not approval_file.is_relative_to(root):
        raise ValueError('Approval receipt must be on the remote RDM filesystem')
    approval = json.loads(approval_file.read_text())
    if approval.get('accept_generator_optimizer_reset') is not True or \
            not approval.get('user_instruction') or not approval.get('approved_utc'):
        raise ValueError('Explicit user acceptance of legacy G optimizer reset is required')
    source = EXPERIMENT/'resume.pth'
    state = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    if state.get('step') != expected_step or state.get('num_steps') != 2:
        raise ValueError('Checkpoint does not contain the expected actual step/two-step model')
    if not all(key in state for key in ('model', 'schedule', 'cache', 'spatial_cache', 'critics')):
        raise ValueError('Checkpoint lacks required model/window/critic state')
    if len(state['critics']) != 2 or any(c['contract'].get('loss') != 'ns' or
            c['contract'].get('lr') != 1e-5 for c in state['critics']):
        raise ValueError('Legacy critic contract is not the current softplus/Dbase1e-5 A run')
    jsonl = EXPERIMENT/'train_log.jsonl'
    lines = jsonl.read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    prefix = [row for row in rows if row['step'] <= expected_step]
    if [row['step'] for row in prefix] != list(range(1, expected_step + 1)):
        raise ValueError('Formal JSONL prefix is not exactly the saved step sequence')
    if any(row.get('skipped') or row.get('s1/gan/d_skipped') or
           row.get('s2/gan/d_skipped') for row in prefix):
        raise ValueError('Saved prefix has optimizer skips; investigate before migrating')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    history = ROOT/'control/spatial-adaptive-migrations'/stamp
    history.mkdir(parents=True)
    frozen = history/f'step_{expected_step:07d}.pth'
    os.link(source, frozen)
    if source.stat().st_ino != frozen.stat().st_ino:
        raise RuntimeError('Frozen checkpoint must retain exactly the saved inode')
    digest = hashlib.sha256()
    with frozen.open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(block)
    shutil.copy2(jsonl, history/'original-train_log.jsonl')
    shutil.copy2(approval_file, history/'user-approval.json')
    # The stopped run's uncheckpointed suffix remains in original-train_log.jsonl.
    # The resumed run appends from step+1 without duplicate step IDs.
    if len(rows) != len(prefix):
        temporary = jsonl.with_name('train_log.jsonl.migration-partial')
        with temporary.open('x') as handle:
            for row in prefix:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, jsonl)
    receipt = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'checkpoint': str(frozen),
               'checkpoint_sha256': digest.hexdigest(), 'checkpoint_bytes': frozen.stat().st_size,
               'step': expected_step, 'target_steps': 500, 'old_launcher_pid': launcher_pid,
               'original_jsonl': str(history/'original-train_log.jsonl'),
               'preserved_uncheckpointed_suffix_steps': [row['step'] for row in rows[len(prefix):]],
               'generator_optimizer_in_checkpoint': state.get('optimizer') is not None,
               'accept_generator_optimizer_reset': True, 'approval_file': str(history/'user-approval.json'),
               'controller_history_present': state.get('adaptive_d_lr') is not None,
               'no_training_started': True}
    (history/'frozen.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-step', required=True, type=int)
    parser.add_argument('--old-launcher-pid', required=True, type=int)
    parser.add_argument('--approval-file', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.expected_step, args.old_launcher_pid, args.approval_file)), flush=True)
