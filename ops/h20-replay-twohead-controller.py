#!/usr/bin/env python3
"""CPU replay of every recorded shared D LR decision in the actual two-head run."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path('/data/rdm-workspace/rdm-sets')
SOURCE = Path('/data/rdm-workspace/RDM-twohead-resume200-v3')
INITIAL = Path('/data/rdm-workspace/RDM-twohead-lr10-v1')
SHA = '1e06bf49569b4ee7847a417bc1ff1445bb8ba51b'
PREFIX = 'twohead-lr10-resume200-500-v3'
HEADS = ['dinov3_l', 'qwen3vl_4b']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', action='store_true', help='Require actual clean500 exit')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import yaml
    assert (SOURCE/'SOURCE_COMMIT').read_text().strip() == SHA
    path = SOURCE/'rdm/sw_lmmd/adaptive_d_lr.py'
    original = INITIAL/'rdm/sw_lmmd/adaptive_d_lr.py'
    assert path.read_bytes() == original.read_bytes(), 'Controller differs across resume200 source lineage'
    name = 'h20_exact_recorded_controller'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    cfg = yaml.safe_load((SOURCE/'z-image/configs/zimage_2step_1024_twohead_glr10_h20_fa3.yaml').read_text())
    owner = module.AdaptiveDLRController(base_lr=1e-5,
        cfg=module.AdaptiveDLRConfig.from_mapping(cfg['adaptive_d_lr']), heads=HEADS)
    exp = ROOT/'zimage/work_dirs/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A'
    rows = []
    incomplete = 0
    for line in (exp/'train_log.jsonl').read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            incomplete += 1
            continue
        if 'step' in row:
            rows.append(row)
    assert incomplete == 0, 'Incomplete scalar log; re-observe before replay'
    assert rows and [r['step'] for r in rows] == list(range(1, len(rows)+1))
    exit_file = ROOT/'control'/f'{PREFIX}.exit'
    actual_exit = exit_file.read_text().strip() if exit_file.exists() else None
    if args.final:
        assert actual_exit == '0' and len(rows) == 500, 'Final replay requires actual clean500'
    actions = []
    for row in rows:
        active = all(bool(row[f's{k}/gan/g_active']) for k in (1, 2))
        valid = not row['skipped'] and not any(row[f's{k}/gan/d_skipped'] for k in (1, 2))
        assert valid, 'Replay acceptance requires complete, nonskipped G and D iterations'
        assert active == (row['step'] > 50), 'Unexpected GAN-G warmup phase'
        heads = {}
        for k in (1, 2):
            for head in HEADS:
                prefix = f's{k}/gan/{head}'
                heads[f's{k}/{head}'] = dict(d_loss=row[f'{prefix}/d_loss'],
                    g_loss=row.get(f'{prefix}/g_loss') if active else None,
                    g_proxy=row.get(f'{prefix}/g_proxy_pre_d_window'))
        assert row['gan/d_lr_used'] == owner.current_lr, f"Wrong used D LR at{row['step']}"
        decision = owner.observe(row['step'], heads, generator_active=active, valid=valid)
        assert decision == row['gan/d_lr_control'], f"Controller decision/statistics differ at{row['step']}"
        assert row['gan/d_lr_next'] == owner.current_lr
        if decision['changed']:
            actions.append({k: v for k, v in decision.items() if k != 'statistics'})
        if row['step'] == 200:
            # Exercise the same serialized controller history boundary as the real resume.
            saved = json.loads(json.dumps(owner.state_dict()))
            restored = module.AdaptiveDLRController(base_lr=1e-5,
                cfg=module.AdaptiveDLRConfig.from_mapping(cfg['adaptive_d_lr']), heads=HEADS)
            restored.load_state_dict(saved)
            assert restored.state_dict() == owner.state_dict()
            owner = restored
    print(json.dumps(dict(passed=True, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Every actual raw D/active G/proxy scalar, full controller decision/statistics and used/next LR',
        complete_iterations=len(rows), decisions_equal=len(rows), actual_formal_exit=actual_exit,
        final500=args.final, current_D_lr=owner.current_lr, actions=actions,
        last_reason=decision['reason'], last_statistics=decision['statistics'],
        controller_policy_hash=owner.policy_hash,
        controller_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        training_source_commit=SHA, serialized_resume200_history_verified=len(rows) >= 200,
        CPU_only=True, training_modified=False)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(passed=False, error_type=type(error).__name__, error=str(error))), flush=True)
        raise SystemExit(1)
