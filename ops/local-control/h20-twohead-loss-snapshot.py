#!/usr/bin/env python3
"""Print existing scalar losses with the verified resume200 lineage; read-only.

Execute through h20-recovery-ssh.py. No Torch, new training metrics, or tensors.
"""
import contextlib
import io
import json
import os
import runpy

SOURCE = '/data/rdm-workspace/RDM-twohead-resume200-v3'
COMMIT = '1e06bf49569b4ee7847a417bc1ff1445bb8ba51b'
INITIAL = 'a25e4c42794b13d5c881d3f1994944929b019190'
os.environ.update(H20_CONTROL_PREFIX='twohead-lr10-resume200-500-v3',
                  H20_TRAIN_SOURCE=SOURCE, H20_TRAIN_SHA=COMMIT)
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path(SOURCE+'/ops/h20-observe-twohead-a.py', run_name='__main__')
status, rows = ns['out'], ns['rows']
assert status['source_commit'] == COMMIT
assert status['all_steps_contiguous'] and len(rows) > 200
assert not status['G_skips'] and not status['D_skips']
assert not status['nonfinite_numeric_count'] and not status['errors']
command = status['command.json']
assert command['source_commit'] == COMMIT and command['prior_source_commit'] == INITIAL
assert command['resume_from_step'] == 200 and command['G_Adam_reset_disclosed']
assert status['preflight.json']['passed']
assert any('resumed ' in line and 'at step 200' in line and 'optimizer FRESH' in line
           for line in ns['lines'])
assert all(row[f's{k}/gan/g_active'] == (row['step'] > 50) for row in rows for k in (1, 2))
keys = ['step', 'skipped', 'grad_norm', 'seconds', 'lr', 'force', 'raw_mmd2',
        'loss_total', 'gan/d_lr_used', 'gan/d_lr_next']
for k in (1, 2):
    keys += [f's{k}/gan/g_active', f's{k}/gan/lambda']
    keys += [f's{k}/gan/{encoder}/{kind}_loss' for encoder in ('dinov3_l', 'qwen3vl_4b')
             for kind in ('d', 'g')]
    keys += [f's{k}/{encoder}/mmd2' for encoder in ('dinov3_l', 'siglip2', 'aimv2_huge')]
actions = [{key: row['gan/d_lr_control'][key] for key in
            ('completed_step', 'effective_from_step', 'old_lr', 'new_lr', 'action', 'reason')}
           for row in rows if row['gan/d_lr_control']['changed']]
print(json.dumps(dict(observed_utc=status['utc'], source_commit=COMMIT,
    experiment=ns['exp'].name, rows=[{key: row[key] for key in keys if key in row} for row in rows],
    D_lr_actions=actions, GAN_G_first_iteration=51,
    trajectory_sources=[dict(steps=[1, 200], commit=INITIAL),
                        dict(steps=[201, rows[-1]['step']], commit=COMMIT)],
    resume200=dict(step=200, G_Adam_reset=True,
                   evidence='Actual resume log, launch command, and passed preflight'))))
