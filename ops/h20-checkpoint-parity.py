"""Compare no-checkpoint / per-block / two-block checkpointing under real two-rank FSDP.

CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=z-image:. torchrun --nproc_per_node=2 \
    --master_port=29541 ops/h20-checkpoint-parity.py --out <remote-control>/checkpoint-parity.json

Add --mixed-precision --micro-batches 4 --shared-conditioning --dropout 0 to exercise the
RDM precision/accumulation path and a condition reused by every block. Comparisons always
write their measured errors and the explicitly selected tolerances; a failed comparison
returns exit1 after writing the JSON rather than aborting before the evidence is saved.

Only a tiny model is constructed. No model weights, optimizer, formal run, or evaluation
are saved. Each configuration is destroyed before constructing the next one.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint

from zimage_distill.checkpointing import configure_checkpoint_groups


class TinyBlock(nn.Module):
    def __init__(self, width: int = 32, dropout: float = 0.17):
        super().__init__()
        self.up = nn.Linear(width, width * 2)
        self.down = nn.Linear(width * 2, width)
        self.dropout = nn.Dropout(dropout)
        self.calls = 0

    def forward(self, x, conditioning=None):
        self.calls += 1
        result = x + self.down(self.dropout(torch.nn.functional.gelu(self.up(x))))
        return result if conditioning is None else result + 0.2 * conditioning


class TinyTransformer(nn.Module):
    def __init__(self, dropout=0.17, shared_conditioning=False):
        super().__init__()
        self.noise_refiner = nn.ModuleList()
        self.context_refiner = nn.ModuleList()
        self.layers = nn.ModuleList(TinyBlock(dropout=dropout) for _ in range(5))
        self.condition = nn.Linear(32, 32) if shared_conditioning else None
        self.projection = nn.Linear(32, 16)
        self.gradient_checkpointing = True

    def forward(self, x, controlnet_block_samples=None):
        conditioning = None if self.condition is None else self.condition(x.mean(dim=1)).unsqueeze(1)
        for layer in self.layers:
            x = checkpoint(layer.__call__, x, conditioning, use_reentrant=False) \
                if torch.is_grad_enabled() and self.gradient_checkpointing else layer(x, conditioning)
        return self.projection(x)


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    a, b = actual.double(), expected.double()
    return {
        'max_absolute': float((a - b).abs().max()) if a.numel() else 0.0,
        'relative_l2': float((a - b).norm() / b.norm().clamp_min(1e-20)),
    }


def compare_tensor(actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> dict:
    result = difference(actual, expected)
    result['allclose'] = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol))
    return result


def gradient_energy(gradients: dict) -> float:
    return sum(float(g.double().square().sum()) for g in gradients.values() if g is not None)


def run_mode(mode: str, state: dict, keys: list[str], local_rank: int, rank: int, args) -> dict:
    toy = TinyTransformer(args.dropout, args.shared_conditioning)
    toy.load_state_dict(state, strict=True)
    toy.gradient_checkpointing = mode != 'none'
    if mode != 'none':
        configure_checkpoint_groups(toy, 1 if mode == 'block1' else 2)
    assert sorted(toy.state_dict()) == keys, 'Checkpoint grouping changed canonical state keys'
    original_blocks = tuple(toy.layers._modules.values())
    policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TinyBlock})
    model = FSDP(
        toy, auto_wrap_policy=policy, sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True, device_id=torch.device('cuda', local_rank),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
            cast_forward_inputs=False, cast_root_forward_inputs=False,
        ) if args.mixed_precision else None,
    )
    assert all(isinstance(block, FSDP) for block in toy.layers._modules.values()), \
        'Each actual TinyBlock must remain its own FSDP unit'
    # Different inputs on each rank exercise real gradient averaging. The same local input,
    # target and dropout RNG are recreated independently for each checkpoint configuration.
    g = torch.Generator().manual_seed(1201 + rank)
    torch.manual_seed(902 + rank)
    torch.cuda.manual_seed(902 + rank)
    dist.barrier()
    outputs, input_gradients, losses, norms_per_micro, fsdp_norms_per_micro = [], [], [], [], []
    input_dtype = torch.bfloat16 if args.mixed_precision else torch.float32
    for _ in range(args.micro_batches):
        x = torch.randn(2, 7, 32, generator=g).to(device=local_rank, dtype=input_dtype).requires_grad_()
        target = torch.randn(2, 7, 16, generator=g).to(local_rank)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.mixed_precision):
            output = model(x)
            # Like RDM's fp32 feature-space objective, cast the output back to fp32
            # for the scalar loss. Every micro backward reduces normally: no no_sync.
            loss = (output.float() - target).square().mean() / args.micro_batches
        loss.backward()
        torch.cuda.synchronize()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        outputs.append(output.detach().cpu().clone())
        input_gradients.append(x.grad.detach().cpu().clone())
        losses.append(float(loss.detach()))
        energy = sum(float(p.grad.detach().double().square().sum())
                     for p in model.parameters() if p.grad is not None)
        total_energy = torch.tensor(energy, dtype=torch.float64, device=local_rank)
        dist.all_reduce(total_energy)
        norms_per_micro.append(float(total_energy.sqrt()))
        # Infinite max_norm leaves gradients unchanged. Record the same FSDP norm API
        # as real training separately from the direct double-precision shard reduction.
        fsdp_norms_per_micro.append(float(model.clip_grad_norm_(float('inf'))))
    gradients = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        gradients[name] = None if gradient is None else gradient.detach().cpu().clone()
    assert any(g is not None and g.numel() for g in gradients.values())
    calls = [block.calls for block in original_blocks]
    assert calls == ([args.micro_batches] * 5 if mode == 'none' else [2 * args.micro_batches] * 5), \
        (mode, calls)
    result = {
        'mode': mode, 'output': torch.cat(outputs),
        'input_gradient': torch.cat(input_gradients), 'gradients': gradients,
        'block_calls': calls, 'loss': sum(losses), 'global_grad_norm_after_each_micro': norms_per_micro,
        'fsdp_reported_grad_norm_after_each_micro': fsdp_norms_per_micro,
        'local_gradient_energy': gradient_energy(gradients),
        'gradient_dtypes': sorted({str(g.dtype) for g in gradients.values() if g is not None}),
    }
    # Keep only CPU copies. Avoid holding multiple FSDP model instances at once.
    del loss, output, x, target, model, toy, original_blocks, parameter, gradient
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return result


def compare_results(actual, reference, args) -> dict:
    assert actual['gradients'].keys() == reference['gradients'].keys()
    parameter_differences, per_parameter, error_energy = [], {}, 0.0
    for name, gradient in actual['gradients'].items():
        other = reference['gradients'][name]
        assert (gradient is None) == (other is None), name
        if gradient is not None:
            result = compare_tensor(gradient, other, args.rtol, args.atol)
            parameter_differences.append(result)
            per_parameter[name] = result
            error_energy += float((gradient.double() - other.double()).square().sum())
    output = compare_tensor(actual['output'], reference['output'], args.rtol, args.atol)
    input_gradient = compare_tensor(actual['input_gradient'], reference['input_gradient'], args.rtol, args.atol)
    return {
        'output': output, 'input_gradient': input_gradient,
        'parameter_gradient_max_absolute': max(row['max_absolute'] for row in parameter_differences),
        'parameter_gradient_max_relative_l2': max(row['relative_l2'] for row in parameter_differences),
        'parameter_gradient_error_energy': error_energy,
        'reference_gradient_energy': reference['local_gradient_energy'],
        'actual_gradient_energy': actual['local_gradient_energy'],
        'per_parameter_gradient_differences': per_parameter,
        'local_parameter_tensors_with_grad': len(parameter_differences),
        'block_calls': actual['block_calls'], 'loss': actual['loss'],
        'gradient_dtypes': actual['gradient_dtypes'],
        'actual_global_grad_norm_after_each_micro': actual['global_grad_norm_after_each_micro'],
        'reference_global_grad_norm_after_each_micro': reference['global_grad_norm_after_each_micro'],
        'actual_fsdp_reported_grad_norm_after_each_micro': actual['fsdp_reported_grad_norm_after_each_micro'],
        'reference_fsdp_reported_grad_norm_after_each_micro': reference['fsdp_reported_grad_norm_after_each_micro'],
        'passed': output['allclose'] and input_gradient['allclose']
                  and all(row['allclose'] for row in parameter_differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mixed-precision', action='store_true',
                        help='FP32 masters, BF16 compute/inputs, FP32 reduce, RDM FSDP cast flags')
    parser.add_argument('--micro-batches', type=int, default=1,
                        help='Scaled-loss micro backwards with normal FSDP reduction on every micro')
    parser.add_argument('--shared-conditioning', action='store_true',
                        help='Reuse one differentiable BF16 condition across every checkpoint group')
    parser.add_argument('--dropout', type=float, default=0.17)
    parser.add_argument('--rtol', type=float, default=3e-5,
                        help='Explicit per-element tolerance; default is NOT widened automatically for BF16')
    parser.add_argument('--atol', type=float, default=2e-6)
    args = parser.parse_args()
    assert args.micro_batches >= 1 and 0 <= args.dropout < 1
    assert args.rtol >= 0 and args.atol >= 0
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '4,5', 'Only physical GPUs4/5 are authorized'
    assert int(os.environ['WORLD_SIZE']) == 2
    local_rank, rank = int(os.environ['LOCAL_RANK']), int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(71)
        template = TinyTransformer(args.dropout, args.shared_conditioning)
        state = {k: v.detach().cpu().clone() for k, v in template.state_dict().items()}
        keys = sorted(state)
        del template
        baseline = run_mode('none', state, keys, local_rank, rank, args)
        summary = {'rank': rank, 'baseline_loss': baseline['loss'], 'modes': {}}
        per_block = None
        for mode in ('block1', 'block2'):
            actual = run_mode(mode, state, keys, local_rank, rank, args)
            summary['modes'][mode + '_vs_none'] = compare_results(actual, baseline, args)
            if mode == 'block1':
                per_block = actual
            else:
                summary['modes']['block2_vs_block1'] = compare_results(actual, per_block, args)
                del actual, per_block
        ranks = [None] * 2
        dist.all_gather_object(ranks, summary)
        passed = all(item['passed'] for row in ranks for item in row['modes'].values())
        if rank == 0:
            global_comparisons = {}
            for name in summary['modes']:
                comparisons = [row['modes'][name] for row in ranks]
                reference_energy = sum(c['reference_gradient_energy'] for c in comparisons)
                actual_energy = sum(c['actual_gradient_energy'] for c in comparisons)
                error_energy = sum(c['parameter_gradient_error_energy'] for c in comparisons)
                global_comparisons[name] = {
                    'parameter_gradient_relative_l2': (error_energy / max(reference_energy, 1e-40))**0.5,
                    'reference_parameter_grad_norm': reference_energy**0.5,
                    'actual_parameter_grad_norm': actual_energy**0.5,
                    'parameter_grad_norm_ratio': (actual_energy / max(reference_energy, 1e-40))**0.5,
                    'passed_elementwise_tolerance': all(c['passed'] for c in comparisons),
                }
            record = {
                'passed': passed, 'world_size': 2, 'master_dtype': 'float32',
                'compute_dtype': 'bfloat16' if args.mixed_precision else 'float32',
                'input_dtype': 'bfloat16' if args.mixed_precision else 'float32',
                'reduce_dtype': 'float32', 'cast_forward_inputs': False,
                'cast_root_forward_inputs': False, 'no_sync': False,
                'micro_batches': args.micro_batches, 'loss_scale_per_micro': 1 / args.micro_batches,
                'torch': torch.__version__, 'dropout': args.dropout, 'layers': 5,
                'shared_conditioning': args.shared_conditioning,
                'elementwise_tolerances': {'rtol': args.rtol, 'atol': args.atol},
                'canonical_state_keys': len(keys),
                'canonical_state_keys_sha256': hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
                'fsdp_unit': 'individual TinyBlock', 'use_orig_params': True,
                'global_comparisons': global_comparisons, 'ranks': ranks,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.out.with_name(args.out.name + f'.tmp-{os.getpid()}')
            tmp.write_text(json.dumps(record, indent=2) + '\n')
            os.replace(tmp, args.out)
            print(json.dumps(record), flush=True)
        dist.barrier()
        if not passed:
            raise SystemExit('Parity differences exceeded the explicitly recorded tolerances; JSON written')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
