#!/usr/bin/env python3
"""Two-rank fixed-feature D diagnostic; never performs a student optimizer update.

Compare hinge/D-lr1e-4, softplus/D-lr1e-4 and softplus/D-lr1e-5 using the same
initial critic, empty Adam moments and fixed features. No initialization, MMD,
architecture or generator learning-rate changes are made.
Only scalar diagnostics are written; no weights/features/checkpoints are saved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
for directory in (REPO, REPO / 'z-image'):
    sys.path.insert(0, str(directory))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w') as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n'); output.flush(); os.fsync(output.fileno())
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def cpu_state(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return copy.deepcopy(value)


def matrix_power_sigma(weight, iterations, *, initial_u=None, initial_v=None):
    """Matrix-only estimate; it does not update any parametrization buffers."""
    import torch
    from torch.nn import functional as F
    with torch.no_grad():
        matrix = weight.detach().flatten(1)
        if not bool(torch.isfinite(matrix).all()):
            raise RuntimeError('Nonfinite matrix in spectral diagnostic')
        v = (torch.arange(matrix.shape[1], device=matrix.device, dtype=torch.float32).remainder(97) - 48
             if initial_v is None else initial_v.detach().clone().float())
        v = F.normalize(v, dim=0, eps=1e-12)
        for _ in range(iterations):
            u = F.normalize(matrix.float() @ v, dim=0, eps=1e-12)
            v = F.normalize(matrix.float().T @ u, dim=0, eps=1e-12)
        return float(torch.dot(u, matrix.float() @ v).abs())


def sn_diagnostics(runtime, iterations):
    import torch
    from torch.nn.utils.parametrizations import _SpectralNorm
    result = {}
    # Aggregator is the primary hypothesis; fusion/projection/readout are controls.
    suffixes = ('aggregator', 'fusion', 'blocks.0.qkv', 'blocks.0.projection', 'readout')
    for branch, head in runtime.critic['spatial'].heads.items():
        modules = dict(head.named_modules())
        for suffix in suffixes:
            module = modules[suffix]
            sn = module.parametrizations.weight[0]
            require(isinstance(sn, _SpectralNorm), 'Expected the actual Torch SN parametrization')
            with torch.no_grad():
                matrix = sn._reshape_weight_to_matrix(module.parametrizations.weight.original).detach()
                u, v = sn._u.detach().clone(), sn._v.detach().clone()
                stale = float(torch.dot(u, matrix @ v))
                refined = matrix_power_sigma(matrix, iterations, initial_u=u, initial_v=v)
                require(torch.equal(u, sn._u) and torch.equal(v, sn._v), 'SN diagnostic mutated buffers')
                result[branch + '/' + suffix] = {
                    'matrix_shape': list(matrix.shape), 'stale_signed_sigma': stale,
                    'power_estimate_sigma': refined, 'power_iterations': iterations,
                    'effective_norm_estimate': refined / max(abs(stale), 1e-30),
                    'raw_weight_frobenius': float(matrix.norm()),
                    'raw_weight_rms': float(matrix.square().mean().sqrt()), 'buffers_unchanged': True}
    return result


def feature_statistics(values, device):
    import torch
    result = {}
    with torch.no_grad():
        for name, value in values.items():
            value = value.detach().to(device=device, dtype=torch.float32)
            axes = (0, 2, 3) if value.ndim == 4 else (0,)
            mean = value.mean(dim=axes)
            rms = float(value.square().mean().sqrt())
            mean_rms = float(mean.square().mean().sqrt())
            result[name] = {'shape': list(value.shape), 'mean': float(value.mean()), 'rms': rms,
                            'std': float(value.std(unbiased=False)), 'channel_mean_rms': mean_rms,
                            'channel_mean_to_rms': mean_rms/max(rms, 1e-30),
                            'finite': bool(torch.isfinite(value).all())}
    return result


def static_scores(runtime, batch, *, with_logits=False):
    import torch
    from rdm.sw_lmmd.adversarial import critic_loss, adversarial_loss
    fake_s, real_s, fake_g, real_g, tau = batch
    count = tau.shape[0]
    runtime.critic.eval()
    with torch.no_grad():
        rs, rg, text = runtime._chunk(real_s, real_g, tau, 0, count, detach=True)
        fs, fg, _ = runtime._chunk(fake_s, fake_g, tau, 0, count, detach=True)
        values = runtime._score({name: torch.cat((rs[name], fs[name])) for name in rs},
            {name: torch.cat((rg[name], fg[name])) for name in rg}, torch.cat((text, text)))
        result, snapshots = {}, {}
        for name, logits in values.items():
            real, fake = logits[:count], logits[count:]
            loss = critic_loss(real, fake, runtime.cfg.loss)
            result[name] = {'D_loss_same_batch': float(loss),
                'G_loss_same_batch': float(adversarial_loss(fake, runtime.cfg.loss)),
                'real_mean': float(real.mean()), 'real_min': float(real.min()), 'real_max': float(real.max()),
                'fake_mean': float(fake.mean()), 'fake_min': float(fake.min()), 'fake_max': float(fake.max()),
                'real_positive_fraction': float((real > 0).float().mean()),
                'fake_negative_fraction': float((fake < 0).float().mean()), 'logit_shape': list(fake.shape)}
            if with_logits:
                snapshots[name] = logits.detach().clone()
    return (result, snapshots) if with_logits else result


def g_cotangents(runtime, fake_s, fake_g, tau):
    import torch
    leaves_s = {name: value.detach().to(device=runtime.device, dtype=runtime.dtype).clone().requires_grad_()
                for name, value in fake_s.items()}
    leaves_g = {name: value.detach().to(device=runtime.device, dtype=runtime.dtype).clone().requires_grad_()
                for name, value in fake_g.items()}
    buffers = {name: value.detach().clone() for name, value in runtime.critic.named_buffers()}
    inputs = {**{name: value for name, value in leaves_s.items()},
              **{name: value for name, value in leaves_g.items()}}
    loss, logs = runtime.generator_loss(leaves_s, leaves_g, tau)
    gradients = torch.autograd.grad(loss, list(inputs.values()))
    result = {}
    for (name, value), gradient in zip(inputs.items(), gradients):
        norm = float(gradient.float().norm())
        require(bool(torch.isfinite(gradient).all()) and norm > 0, 'Invalid G feature-input cotangent: ' + name)
        result[name] = {'norm': norm, 'max_abs': float(gradient.abs().max()),
                        'finite_nonzero': True, 'shape': list(gradient.shape)}
    require(all(parameter.grad is None for parameter in runtime.critic.parameters()), 'G wrote critic gradients')
    require(all(torch.equal(value, buffers[name]) for name, value in runtime.critic.named_buffers()), 'G changed SN buffers')
    return {'coordinate': 'unaggregated critic feature inputs; not image pixels',
            'loss': float(loss.detach()), 'branches': result, 'logs': logs, 'critic_grads_absent': True}


def weight_update_diagnostics(runtime, before, iterations):
    import torch
    result = {}
    with torch.no_grad():
        for name, head in runtime.critic['spatial'].heads.items():
            current = head.aggregator.parametrizations.weight.original.detach()
            original = before[name].to(device=current.device, dtype=current.dtype)
            delta = current - original
            delta_fro = float(delta.norm())
            sigma = matrix_power_sigma(delta, iterations)
            result[name] = {'before_raw_frobenius': float(original.norm()),
                'after_raw_frobenius': float(current.norm()), 'delta_frobenius': delta_fro,
                'relative_frobenius': delta_fro/max(float(original.norm()), 1e-30),
                'delta_entry_rms': float(delta.square().mean().sqrt()),
                'delta_spectral_power_estimate': sigma, 'delta_spectral_to_frobenius': sigma/max(delta_fro,1e-30),
                'power_iterations': iterations}
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/rdm-workspace/rdm-sets'))
    parser.add_argument('--config', default='z-image/configs/zimage_2step_1024_dlr1e-4_spatial_patchgan_h20_fa3.yaml')
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--first-row', type=int, default=0)
    parser.add_argument('--power-iterations', type=int, choices=(20, 30, 50), default=30)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    require(platform.system() != 'Darwin', 'Real model/feature diagnostic must run remotely')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '4,5', 'Require exactly physicalGPU4,5')
    require(args.root.is_dir() and args.root.is_absolute() and args.out.is_absolute()
            and args.out.is_relative_to(args.root/'control'), 'Output must be an absolute remoteROOT/control path')
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    import torch
    import torch.distributed as dist
    import numpy as np
    from rdm.utils.distributed import setup_distributed
    from rdm.utils.seed import fix_random_seeds
    from rdm.train.launch import load_config
    from zimage_distill.launch import build_trainer
    from zimage_distill.store import stream_name
    rank, world, local_rank = setup_distributed()
    require(world == 2, 'Exactly two ranks are required')
    device = torch.device('cuda', local_rank)
    report = {'passed': False, 'formal_training_started': False, 'generator_updates': 0,
              'bootstrap_called': False, 'trainer_step_called': False, 'static_D_updates_per_case': 3,
              'started_epoch': time.time(), 'config': args.config,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_commit': (REPO/'SOURCE_COMMIT').read_text().strip() if (REPO/'SOURCE_COMMIT').exists() else None}
    try:
        fix_random_seeds(0)
        cfg = load_config(args.config)
        cfg.reference_root = str(args.reference_root or args.root/'zimage/zimage-ref-v1/reference_store_qwen')
        require(cfg.gan['loss'] == 'ns' and float(cfg.gan['lr']) == 1e-5 and float(cfg.lr) == 2.83e-6,
                'Use latest configured softplus/D-lr1e-5/G-base-lr2.83e-6; cases are diagnostic overrides only')
        report['hyperparameters'] = {'loss': cfg.gan['loss'], 'D_lr': cfg.gan['lr'],
            'D_betas': cfg.gan['betas'], 'G_lr': cfg.lr, 'g_start_step': cfg.gan['g_start_step'],
            'G_loss_evaluation': 'manual diagnostic even during warmup; no G optimizer step'}
        if rank == 0:
            atomic_json(args.out, report)
            print(json.dumps({'event': 'build_trainer_no_bootstrap', 'out': str(args.out)}), flush=True)
        trainer = build_trainer(cfg, device=str(device))
        require(trainer.num_steps == 2 and len(trainer.step_critics) == 2, 'Expected actual two-step spatial trainer')
        runtime = trainer.step_critics[1]
        ids = np.arange(args.first_row + rank*2, args.first_row + rank*2 + 2, dtype=np.int64)
        require(ids.min() >= 0 and ids.max() < trainer.store.num_rows, 'Diagnostic row IDs out of range')
        with torch.no_grad():
            features = trainer._encode_images(trainer._generate_images(ids), ids)
            fake_s = {name: features[f's2/patch/{name}'].detach() for name in runtime.spatial_names}
            real_s = trainer.spatial_battery.reference_features(ids, trainer.reference_images)
            fake_g = {name: features[stream_name(2, name)].detach() for name in runtime.global_names}
            real_g = {name: trainer._reference_window(stream_name(2, name), ids).detach() for name in runtime.global_names}
            tau = trainer.store.text_features(ids, device='cpu', dtype=torch.float32)
        del features
        batch = fake_s, real_s, fake_g, real_g, tau
        initial = cpu_state(runtime.state_dict())
        require(not initial['optimizer']['state'], 'Expected pristine Adam moments before this diagnostic')
        reference_initial_scores, reference_initial_logits = static_scores(runtime, batch, with_logits=True)
        local = {'rank': rank, 'row_ids': ids.tolist(), 'feature_statistics': {
            'real_spatial': feature_statistics(real_s, device), 'fake_spatial': feature_statistics(fake_s, device)},
            'reference_initial_scores': reference_initial_scores, 'cases': {}}
        formal_cfg = runtime.cfg
        cases = [('prior_hinge_lr1e-4','hinge',1e-4), ('softplus_lr1e-4','ns',1e-4),
                 ('softplus_lr1e-5','ns',1e-5)]
        for candidate, loss_kind, d_lr in cases:
            runtime.load_state_dict(initial)
            runtime.cfg = copy.deepcopy(formal_cfg)
            runtime.cfg.loss, runtime.cfg.lr = loss_kind, d_lr
            for group in runtime.optimizer.param_groups:
                group['lr'] = d_lr
            initial_scores, initial_logits = static_scores(runtime, batch, with_logits=True)
            initial_max = max(float((initial_logits[name]-reference_initial_logits[name]).abs().max())
                              for name in runtime.names)
            require(all(torch.allclose(initial_logits[name],reference_initial_logits[name],atol=1e-5,rtol=1e-5)
                        for name in runtime.names), 'Candidate changed initial normalized forward beyond FP32 roundoff')
            record = {'loss': loss_kind, 'D_lr': d_lr, 'initialization_changed': False,
                      'initial_scores': initial_scores,
                      'initial_normalized_forward_max_difference': initial_max,
                      'same_optimizer_start': 'restored initial state; original empty Adam moments', 'updates': []}
            for update in range(1, 4):
                before = {name: head.aggregator.parametrizations.weight.original.detach().to('cpu',copy=True)
                          for name, head in runtime.critic['spatial'].heads.items()}
                pre = static_scores(runtime, batch)
                pre_sn = sn_diagnostics(runtime, args.power_iterations)
                logs = runtime.update(*batch)
                require(logs['gan/d_skipped'] is False and runtime.updates == update, 'D diagnostic update skipped')
                post = static_scores(runtime, batch)
                post_sn = sn_diagnostics(runtime, args.power_iterations)
                row = {'update': update, 'pre_same_inputs': pre, 'post_same_inputs': post,
                    'runtime_logs': logs, 'SN_pre': pre_sn, 'SN_post': post_sn,
                    'aggregator_raw_update': weight_update_diagnostics(runtime,before,args.power_iterations),
                    'G_input_cotangents': g_cotangents(runtime,fake_s,fake_g,tau)}
                record['updates'].append(row)
                del before
                if rank == 0:
                    print(json.dumps({'event':'fixed_batch_D_update','candidate':candidate,'update':update,
                                      'pre':pre,'post':post}),flush=True)
            local['cases'][candidate] = record
            runtime.cfg = formal_cfg
            runtime.load_state_dict(initial)
        require(runtime.updates == 0 and runtime.cfg.loss == 'ns'
                and all(float(group['lr']) == 1e-5 for group in runtime.optimizer.param_groups),
                'Restore the pristine formal configuration/state after diagnostics')
        local['initial_state_and_formal_loss_lr_restored'] = True
        require(trainer.step_idx == 0 and all(parameter.grad is None for parameter in trainer._gen_params),
                'Diagnostic must not update or differentiate the student')
        torch.cuda.synchronize()
        local['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        local['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        gathered = [None]*world
        dist.all_gather_object(gathered,local)
        report.update(passed=True, finished_epoch=time.time(), world_size=world, physical_gpus=[4,5],
                      ranks=gathered, note='Fixed-feature diagnostic evidence, not proof of training stability')
        if rank == 0:
            atomic_json(args.out,report)
            print(json.dumps({'event':'diagnostic_complete','out':str(args.out)}),flush=True)
        return 0
    except Exception as error:
        if rank == 0:
            report.update(passed=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
            atomic_json(args.out,report)
        traceback.print_exc()
        return 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
