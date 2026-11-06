#!/usr/bin/env python3
"""Read checkpoint integrity on CPU. Metadata while running; full scans only after clean500."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path('/data/rdm-workspace/rdm-sets')
TRAIN_SOURCE = Path(os.environ.get('H20_TRAIN_SOURCE','/data/rdm-workspace/RDM-twohead-lr10-v1'))
TRAIN_SHA = os.environ.get('H20_TRAIN_SHA','a25e4c42794b13d5c881d3f1994944929b019190')
EXP = ROOT/'zimage/work_dirs/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A'
HEADS = ['dinov3_l', 'qwen3vl_4b']
PREFIX = os.environ.get('H20_CONTROL_PREFIX','twohead-lr10-A-500')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    temporary = ROOT/'.cache/tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(temporary)
    import numpy as np
    import torch
    import yaml
    torch.set_num_threads(1)
    sys.path.insert(0, str(TRAIN_SOURCE))
    from rdm.sw_lmmd.window_schedule import SlidingWindowSchedule, build_row_order
    from rdm.sw_lmmd.adaptive_d_lr import AdaptiveDLRConfig, AdaptiveDLRController
    started = time.monotonic()
    control = ROOT/'control'
    command = json.loads((control/(PREFIX+'.command.json')).read_text())
    check(command['source_commit'] == TRAIN_SHA, 'Training command source mismatch')
    check((TRAIN_SOURCE/'SOURCE_COMMIT').read_text().strip() == TRAIN_SHA, 'Training source marker mismatch')
    check(command['physical_gpus'] == [4, 5] and command['steps'] == 500, 'Training identity mismatch')
    exit_path = control/(PREFIX+'.exit')
    exit_code = exit_path.read_text().strip() if exit_path.exists() else None
    if args.full or args.student:
        check(args.step == 500 and exit_code == '0', 'Full scans/student audit require actual clean500 exit')
    path = (Path(args.file) if args.file else EXP/('step_0000500.pth' if args.student else 'resume.pth')).resolve(strict=True)
    archive_proof = None
    if not path.is_relative_to(ROOT.resolve()):
        archive_root = Path('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints')
        check(args.student and path.is_relative_to(archive_root), 'Checkpoint escapes authorized remote storage')
        archive_proof = json.loads((control/'twohead-retain50-archive'/f'checkpoint-archive-{args.step}.json').read_text())
        check(archive_proof['passed'] and Path(archive_proof['destination']).resolve()==path,
              'Archived student lacks its matching successful receipt')
        check(args.full, 'Archived student audit requires full SHA and tensor-value verification')
    before = path.stat()
    check(before.st_size > 24_000_000_000, 'Checkpoint is too small for the full student')
    check(before.st_mtime >= command['created_epoch'], 'Checkpoint predates the current formal run')
    log = (ROOT/'launch-logs'/(PREFIX+'.log')).read_text(errors='replace')
    marker = ('checkpoint -> '+str(EXP/path.name)) if args.student else ('resume state (step '+str(args.step)+')')
    check(marker in log, 'No actual save-completion log for this artifact/step')
    verified_stream_sha = None
    remote_memory_headroom = None
    if archive_proof is not None:
        from h20_portrait_checkpoint import load_archived_state
        # Sequential object-store reads have been validated on the real200 file;
        # direct mmap metadata loading did not finish within90s on this mount.
        state, remote_memory_headroom = load_archived_state(path, archive_proof)
        verified_stream_sha = archive_proof['sha256']  # The loader hashes and asserts all actual bytes.
    else:
        state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    check(state['step'] == args.step, 'Payload step differs from requested step')
    check(state['schedule']['step'] == args.step, 'Schedule step differs from payload')
    check(state['world_size'] == 2 and state['num_steps'] == 2, 'World/generation-step mismatch')
    check(state['sampling_args']['sigmas'] == [1.0, 0.75], 'Sampling sigmas mismatch')
    check(state['sampling_args']['renoise'] is True and state['sampling_args']['detach_between_steps'] is True,
          'Sampling noise/detach contract mismatch')
    model = state['model']
    check(len(model) == 521 and sum(v.numel() for v in model.values()) == 6154908736, 'Incomplete model schema')
    baseline_path = control/'spatial-A-user-visualization/step_0000150-model.pth'
    schema = lambda m: {k: {'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in m.items()}
    schema_path = control/'legacy-baseline150-schema.json'
    if schema_path.exists():
        saved_schema=json.loads(schema_path.read_text())
        offload=json.loads((control/'legacy-baseline-offload.json').read_text())
        check(saved_schema['passed'] and offload['passed'] and
              saved_schema['source_archive_sha256']==offload['sha256'], 'Unverified legacy reference schema')
        check(schema(model)==saved_schema['model_schema'], 'Model schema differs from verified archived reference')
    else:
        baseline = torch.load(baseline_path, map_location='cpu', weights_only=False, mmap=True)['model']
        check(schema(model) == schema(baseline), 'Model key/shape/dtype schema differs from verified student')
        del baseline
    check(all(v.device.type == 'cpu' for v in model.values()), 'Model must map to CPU')

    feature_contract = state['spatial_gan_contract']
    manifest = json.loads((ROOT/'zimage/zimage-ref-v1/reference_images_patchgan/manifest.json').read_text())
    check(feature_contract['reference'] == manifest['contract'], 'Reference representation contract mismatch')
    check(feature_contract['index_sha256'] == manifest['index']['sha256'], 'Reference row index mismatch')
    check(feature_contract['cache_dtype'] == 'bfloat16' and
          feature_contract['balance_space'] == 'decoded-image-pixels', 'Spatial representation mismatch')
    cfg = yaml.safe_load((TRAIN_SOURCE/'z-image/configs/zimage_2step_1024_twohead_glr10_h20_fa3.yaml').read_text())
    controller_state = state['adaptive_d_lr']
    controller = AdaptiveDLRController(base_lr=1e-5,
        cfg=AdaptiveDLRConfig.from_mapping(cfg['adaptive_d_lr']), heads=HEADS)
    controller.load_state_dict(controller_state)
    check(controller.last_completed_step == args.step, 'Controller step mismatch')

    critics = []
    if not args.student:
        check(len(state['critics']) == 2, 'Expected one critic per generation step')
        for item in state['critics']:
            contract = item['contract']
            check(item['updates'] == args.step, 'Critic update counter mismatch')
            check(contract['head_order'] == HEADS and contract['global'] == [] and
                  contract['encoder_weight'] == 0.5, 'Wrong GAN heads/normalization; legacyAIM is not allowed')
            check(contract['loss'] == 'ns' and contract['lr'] == 1e-5, 'Critic loss/baseLR mismatch')
            check(not any('.aimv2_huge.' in k or '.siglip2.' in k for k in item['critic']),
                  'Unexpected AIM/SigLIP GAN parameters')
            optimizer = item['optimizer']
            lrs = [g['lr'] for g in optimizer['param_groups']]
            check(lrs and all(float(v) == controller.current_lr for v in lrs), 'D optimizer/controller LR mismatch')
            steps = [int(float(s['step'])) for s in optimizer['state'].values()]
            check(steps and min(steps) == max(steps) == args.step, 'D Adam step mismatch')
            critics.append({'updates': item['updates'], 'heads': contract['head_order'],
                            'encoder_weight': contract['encoder_weight'], 'optimizer_lrs': lrs,
                            'Adam_states': len(steps), 'Adam_step_min': min(steps), 'Adam_step_max': max(steps)})

    caches = {}
    if not args.student:
        schedule = SlidingWindowSchedule(build_row_order(125808, seed=3407), window_size=128, stride=32)
        schedule.load_state_dict(state['schedule'])
        expected_ids = schedule._window_at(args.step-1).all_ids
        global_shapes = {f's{k}/{name}': (128, dim) for k in (1, 2)
                         for name, dim in [('dinov3_l', 3584), ('siglip2', 3712), ('aimv2_huge', 4096)]}
        spatial_shapes = {f's{k}/patch/{name}': (128, channels, grid, grid) for k in (1, 2)
                          for name, channels, grid in [('dinov3_l', 1024, 64), ('qwen3vl_4b', 2560, 32)]}
        common_ages = None
        for group, expected, dtype in [('cache', global_shapes, torch.float32),
                                       ('spatial_cache', spatial_shapes, torch.bfloat16)]:
            check(set(state[group]) == set(expected), group+' stream names differ')
            for name, entry in state[group].items():
                tensor = entry['features']
                check(tuple(tensor.shape) == expected[name] and tensor.dtype == dtype and
                      tensor.device.type == 'cpu', group+'/'+name+' feature schema differs')
                check(np.array_equal(entry['row_ids'], expected_ids), group+'/'+name+' row identity differs')
                ages = entry['generated_at_steps']
                check(tuple(ages.shape) == (128,) and int(ages.min()) >= 0 and
                      int(ages.max()) <= args.step, 'Invalid cache generation steps')
                if common_ages is None:
                    common_ages = ages
                check(torch.equal(ages, common_ages), 'Global/spatial cache ages differ')
                caches[group+'/'+name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                    'rows_match_schedule': True, 'age_min': int(ages.min()), 'age_max': int(ages.max())}

    digest = None
    finite_tensors = 0
    if args.full:
        def finite(value):
            nonlocal finite_tensors
            if isinstance(value, torch.Tensor):
                check(value.device.type == 'cpu', 'Full verification is CPU-only')
                if value.is_floating_point() or value.is_complex():
                    flat = value.reshape(-1)
                    for lo in range(0, flat.numel(), 1_048_576):
                        check(bool(torch.isfinite(flat[lo:lo+1_048_576]).all()), 'Nonfinite saved tensor')
                    finite_tensors += 1
            elif isinstance(value, dict):
                for child in value.values():
                    finite(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    finite(child)
            elif isinstance(value, float):
                check(math.isfinite(value), 'Nonfinite saved scalar')
        finite(state)
        if verified_stream_sha is not None:
            digest = verified_stream_sha
        else:
            sha = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''):
                    sha.update(block)
            digest = sha.hexdigest()
    after = path.stat()
    if archive_proof is not None:
        check(before.st_size==after.st_size==archive_proof['bytes'] and digest==archive_proof['sha256'],
              'Final archived bytes/SHA differ from the verified saved student')
    else:
        check((before.st_ino, before.st_size, before.st_mtime_ns) ==
              (after.st_ino, after.st_size, after.st_mtime_ns), 'File replaced during audit; re-observe same run')
    return {'passed': True, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope': 'final-full-integrity' if args.full else 'metadata-only-no-weight-value-scan',
            'path': str(path), 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns,
            'source_commit': TRAIN_SHA, 'step': state['step'], 'schedule_step': state['schedule']['step'],
            'world_size': state['world_size'], 'num_steps': state['num_steps'],
            'model_tensor_count': len(model), 'model_numel': sum(v.numel() for v in model.values()),
            'model_schema_matches_verified150': True, 'critics': critics, 'cache': caches,
            'adaptive_controller': {k: controller_state[k] for k in ['current_lr', 'last_completed_step',
                'valid_observations', 'invalid_observations', 'action_sequence', 'head_keys']},
            'generator_optimizer_in_checkpoint': 'optimizer' in state,
            'loading_mode': 'full-SHA-verified-remote-RAM-stream' if archive_proof else 'local-remote-ROOT-mmap',
            'remote_memory_headroom_before_load': remote_memory_headroom,
            'sha256': digest, 'finite_saved_tensors_checked': finite_tensors,
            'actual_formal_exit': exit_code, 'seconds': time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--file')
    parser.add_argument('--student', action='store_true')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args), ensure_ascii=False), flush=True)
    except Exception as error:
        print(json.dumps({'passed': False, 'error': str(error), 'error_type': type(error).__name__,
                          'requested_step': args.step, 'file': args.file}), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
