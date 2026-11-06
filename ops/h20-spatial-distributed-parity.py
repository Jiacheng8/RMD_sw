#!/usr/bin/env python3
"""Two-rank CPU Gloo parity of the real mixed PatchGAN runtime and pixel GradCache.

Run: torchrun --standalone --nproc-per-node=2 ops/h20-spatial-distributed-parity.py
No model/data downloads, GPU use, training launch, or tensor artifacts. Each rank
also computes an independent single-rank B4/K8 oracle with collectives disabled.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

from h20_spatial_parity_fixtures import load_modules, TinySpatialProblem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, help='small scalar-only acceptance receipt')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    if args.threads < 1 or args.timeout < 1:
        raise ValueError('Positive threads and timeout required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 2:
        raise ValueError('Launch exactly2 CPU processes with torchrun --standalone --nproc-per-node=2')
    torch.set_num_threads(args.threads)
    dist.init_process_group('gloo', timeout=timedelta(seconds=args.timeout))
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        modules = load_modules()
        oracle = TinySpatialProblem(modules)
        expected = oracle.run(reference=True)
        tested = TinySpatialProblem(modules)
        actual = tested.run(rank=rank, world=world)
        # Exactly the FSDP/ordinary-DDP parameter-gradient mean after all local
        # microchunks. There is no additional /A, /64, or division of the losses.
        for parameter in tested.generator.parameters():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad /= world
        actual['g_gradients'] = {name: p.grad.detach().clone()
                                 for name, p in tested.generator.named_parameters()}
        mean_loss = torch.tensor(actual['loss'], dtype=torch.float64)
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM); mean_loss /= world
        failures, metrics = [], {}

        def check(name, left, right, *, atol, rtol):
            left = torch.as_tensor(left, dtype=torch.float64)
            right = torch.as_tensor(right, dtype=torch.float64)
            difference = float((left.double() - right.double()).abs().max())
            metrics[name] = difference
            try:
                torch.testing.assert_close(left, right, atol=atol, rtol=rtol, check_dtype=False)
            except AssertionError as error:
                failures.append({'check': name, 'error': str(error)[:1500]})

        check('mean_loss', mean_loss, expected['loss'], atol=3e-6, rtol=3e-5)
        g_max, d_grad_max, d_state_max = 0.0, 0.0, 0.0
        for name in actual['g_gradients']:
            check('g/' + name, actual['g_gradients'][name], expected['g_gradients'][name], atol=5e-6, rtol=5e-5)
            g_max = max(g_max, metrics['g/' + name])
        for k in (1, 2):
            check(f's{k}/d_loss', actual['d_logs'][k]['gan/d_loss'],
                  expected['d_logs'][k]['gan/d_loss'], atol=1e-10, rtol=1e-9)
            check(f's{k}/lambda', actual['logs'][f's{k}/pixel_adv_lambda'],
                  expected['logs'][f's{k}/pixel_adv_lambda'], atol=3e-6, rtol=3e-5)
            for kind in ('mmd', 'adv'):
                key = f's{k}/pixel_{kind}_grad_norm'
                # Each rank's local mean uses B_local=B/R, so its pixel
                # cotangents are R times the full-B oracle's coordinates. SUM
                # norm squares thus yields R*global_norm; the common factor
                # cancels from adaptive lambda. It must not be removed twice.
                check(key + '/normalized_by_world', actual['logs'][key] / world,
                      expected['logs'][key], atol=3e-7, rtol=3e-5)
            for name in actual['d_gradients'][k]:
                key = f's{k}/d_grad/' + name
                check(key, actual['d_gradients'][k][name], expected['d_gradients'][k][name],
                      atol=1e-10, rtol=1e-8)
                d_grad_max = max(d_grad_max, metrics[key])
            for name in actual['d_states'][k]:
                key = f's{k}/d_state/' + name
                check(key, actual['d_states'][k][name], expected['d_states'][k][name],
                      atol=2e-8, rtol=1e-8)
                d_state_max = max(d_state_max, metrics[key])
        local = {'rank': rank, 'passed': not failures, 'failures': failures,
                 'max_g_gradient_abs': g_max, 'max_d_gradient_abs': d_grad_max,
                 'max_d_state_abs': d_state_max,
                 'mean_loss_abs': metrics['mean_loss'],
                 'lambda_abs': {str(k): metrics[f's{k}/lambda'] for k in (1, 2)},
                 'generator_replay_max_abs': actual['logs']['pixel_gc/generator_replay_max_abs']}
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
        receipt = {'passed': all(item['passed'] for item in gathered), 'backend': 'gloo',
                   'world_size': world, 'global_active_rows': 4, 'local_active_rows': 2,
                   'window_rows': 8, 'generation_steps': 2, 'step_weights': [1.0, 0.25],
                   'text_dim': 2560, 'critic_dtype': 'float64', 'pixel_dtype': 'float32',
                   'spatial_boundary': 'bfloat16_values_then_float32',
                   'norm_square_reduce': 'SUM_once_before_sqrt; local-mean cotangent scales byR',
                   'g_gradient_reduce': 'MEAN_once_after_all_local_microchunks',
                   'encoder_weight': '1/3; no extra/64/R/A', 'ranks': gathered}
        def finite_json(value):
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            if isinstance(value, dict):
                return {key: finite_json(item) for key, item in value.items()}
            if isinstance(value, list):
                return [finite_json(item) for item in value]
            return value
        receipt = finite_json(receipt)
        if rank == 0:
            if args.json is not None:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.json.with_name(args.json.name + f'.tmp-{os.getpid()}')
                with temporary.open('w') as output:
                    json.dump(receipt, output, sort_keys=True, indent=2, allow_nan=False)
                    output.write('\n'); output.flush(); os.fsync(output.fileno())
                os.replace(temporary, args.json)
            print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
        return 0 if receipt['passed'] else 1
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
