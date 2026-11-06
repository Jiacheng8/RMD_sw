"""Actual two-process CPU/Gloo checks; no GPU, pretrained weights or tensor files.

The real mixed spatial/AIM runtime must reduce detached G telemetry by image
count while leaving its local differentiable objective unchanged. The real
trainer hook must broadcast rank0's controller state and change only the two D
learning rates, including after an in-memory checkpoint round trip.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
HEADS = ('dinov3_l', 'qwen3vl_4b', 'aimv2_huge')
BASE_D_LR, G_LR = 1e-5, 2.83e-6


def _load_fixture(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'z-image/tests' / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for x, y in zip(left, right):
            _assert_same(x, y)
    else:
        assert left == right


def _global_g_oracle(rank, fixture, *, two_head=False):
    case = fixture.SpatialRuntimeTests()
    case.setUp()
    case.cfg.loss = 'ns'
    runtime = (fixture.SpatialGANRuntime(case.cfg, case.store, case.base_names, 'cpu', torch.float64,
                critic_micro_batch=2, step_seed=19, global_remaining=(),
                spatial_branch_specs=case.specs, spatial_options={'heads': 2})
               if two_head else case.runtime(2))
    buffers = {name: value.clone() for name, value in runtime.critic.named_buffers()}
    largest_gradient_difference = 0.0
    local_distributions = []
    # Both unequal rows and unequal distributions matter: averaging rank means
    # would silently pass equal-sized ranks and must fail the first case here.
    for row_counts in ((1, 3), (2, 2)):
        rows = row_counts[rank]
        torch.manual_seed(701 + rank)
        encoders = nn.ModuleDict({
            'dinov3_l': nn.Conv2d(3, 3, 1),
            'qwen3vl_4b': nn.Conv2d(3, 4, 1),
            'aimv2_huge': nn.Linear(6, 11),
        }).double().eval().requires_grad_(False)
        raw = {
            'dinov3_l': (torch.randn(rows, 3, 8, 8, dtype=torch.float64) + rank * 3).requires_grad_(),
            'qwen3vl_4b': (torch.randn(rows, 3, 16, 16, dtype=torch.float64) - rank * 2).requires_grad_(),
            'aimv2_huge': (torch.randn(rows, 6, dtype=torch.float64) + rank * 5).requires_grad_(),
        }
        features = {name: encoders[name](value) for name, value in raw.items()}
        tau = F.normalize(torch.randn(rows, 2560, dtype=torch.float64), dim=-1).requires_grad_()
        with fixture.module.frozen_spatial_critic(runtime.critic):
            logits = runtime._score({name: features[name] for name in runtime.spatial_names},
                                    {'aimv2_huge': features['aimv2_huge'][:, :6]}, tau.detach())
        expected_losses = {name: F.softplus(-value).mean() for name, value in logits.items()}
        local_objective = sum(expected_losses.values()) / len(runtime.names)
        expected_gradients = torch.autograd.grad(local_objective, tuple(raw.values()), retain_graph=True,
                                                 allow_unused=True)
        total, logs = runtime.generator_loss(
            {name: features[name] for name in runtime.spatial_names},
            {'aimv2_huge': features['aimv2_huge']}, tau)
        torch.testing.assert_close(total, local_objective, rtol=1e-12, atol=1e-12)
        total.backward()
        for (name, value), expected in zip(raw.items(), expected_gradients):
            if two_head and name == 'aimv2_huge':
                assert expected is None and value.grad is None
                continue
            assert value.grad is not None and bool(torch.isfinite(value.grad).all())
            assert value.grad.abs().sum() > 0
            difference = float((value.grad - expected).abs().max())
            largest_gradient_difference = max(largest_gradient_difference, difference)
            torch.testing.assert_close(value.grad, expected, rtol=1e-10, atol=1e-12)
        assert all(parameter.grad is None for parameter in encoders.parameters())
        assert all(parameter.grad is None for parameter in runtime.critic.parameters())
        assert tau.grad is None
        for name, value in runtime.critic.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)

        # Independent explicit SUM of image totals/counts; do not reuse runtime's
        # mean-reduction helper or derive the oracle from its reported losses.
        summed = torch.tensor([number for name in runtime.names for number in
                               (float(expected_losses[name]) * rows,
                                float(logits[name].detach().mean()) * rows)] + [rows],
                              dtype=torch.float64)
        dist.all_reduce(summed, op=dist.ReduceOp.SUM)
        count = int(summed[-1])
        expected_entries = (summed[:-1] / count).reshape(len(runtime.names), 2)
        for index, name in enumerate(runtime.names):
            assert logs[f'gan/{name}/g_loss'] == pytest.approx(float(expected_entries[index, 0]), abs=1e-12)
            assert logs[f'gan/{name}/g_fake'] == pytest.approx(float(expected_entries[index, 1]), abs=1e-12)
        assert logs['gan/g_loss'] == pytest.approx(float(expected_entries[:, 0].mean()), abs=1e-12)
        assert logs['gan/g_rows_global'] == sum(row_counts)
        assert logs['gan/g_metric_scope'] == 'global_active_rows'
        records = [None, None]
        dist.all_gather_object(records, {'logs': logs, 'local_loss': float(total.detach()), 'rows': rows})
        _assert_same(records[0]['logs'], records[1]['logs'])
        assert abs(records[0]['local_loss'] - records[1]['local_loss']) > 1e-5
        if row_counts[0] != row_counts[1]:
            wrong_rank_mean = sum(item['local_loss'] for item in records) / 2
            assert abs(logs['gan/g_loss'] - wrong_rank_mean) > 1e-5
        local_distributions.append([item['local_loss'] for item in records])
    return {'max_local_gradient_abs': largest_gradient_difference,
            'local_losses_per_case': local_distributions, 'global_cases': 2}


def _controller_shell(rank, trainer_class, config_class, controller_class, *, saved=None, heads=HEADS):
    config = config_class.from_mapping({'enabled': True})
    controller = controller_class(base_lr=BASE_D_LR, cfg=config, heads=heads)
    critics = []
    for step in (1, 2):
        parameter = nn.Parameter(torch.tensor([float(step)], dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=BASE_D_LR, betas=(0.0, 0.99), weight_decay=0)
        parameter.grad = torch.full_like(parameter, 0.2)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        critics.append(SimpleNamespace(names=list(heads), optimizer=optimizer, parameter=parameter))
    g_parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    g_optimizer = torch.optim.AdamW([g_parameter], lr=G_LR)
    g_parameter.grad = torch.full_like(g_parameter, 0.3)
    g_optimizer.step(); g_optimizer.zero_grad(set_to_none=True)
    shell = SimpleNamespace(d_lr_controller=controller, step_critics=critics, num_steps=2,
                            optimizer=g_optimizer, g_parameter=g_parameter)
    shell._adapt_discriminator_lr = MethodType(trainer_class._adapt_discriminator_lr, shell)
    if saved is not None:
        controller.load_state_dict(saved['controller'])
        for runtime, state in zip(critics, saved['critics']):
            runtime.optimizer.load_state_dict(state)
        g_optimizer.load_state_dict(saved['generator_optimizer'])
    # A real non-rank0 observe would fail instead of quietly producing the same
    # decision. Thus the test exercises the actual broadcast/load_state route.
    if rank != 0:
        def forbidden_observe(*args, **kwargs):
            raise AssertionError('non-rank0 must load rank0 state, never observe')
        controller.observe = forbidden_observe
    return shell


def _logs(step, d, g, *, active=True, skipped=False, heads=HEADS):
    result = {'step': step, 'skipped': skipped}
    for index in (1, 2):
        result[f's{index}/gan/g_active'] = active
        result[f's{index}/gan/d_skipped'] = False
        for name in heads:
            prefix = f's{index}/gan/{name}'
            result[f'{prefix}/d_loss'] = d
            result[f'{prefix}/g_proxy_pre_d_window'] = g
            if active:
                result[f'{prefix}/g_loss'] = g
    return result


def _controller_oracle(rank, trainer_fixture, *, heads=HEADS):
    from rdm.sw_lmmd.adaptive_d_lr import AdaptiveDLRConfig, AdaptiveDLRController
    trainer_class = trainer_fixture.SpatialMultiStepTrainer
    live = _controller_shell(rank, trainer_class, AdaptiveDLRConfig, AdaptiveDLRController, heads=heads)
    original_g = copy.deepcopy(live.optimizer.state_dict())
    original_g_parameter = live.g_parameter.detach().clone()
    original_d_moments = [copy.deepcopy(runtime.optimizer.state_dict()['state']) for runtime in live.step_critics]
    resumed = None
    actions = []
    for step in range(1, 94):
        if step <= 50:
            logs = _logs(step, 2.0, 1.0, active=False, heads=heads)
        elif step <= 60 or 81 <= step <= 91:
            logs = _logs(step, 2.0, 1.0, heads=heads)
        else:
            logs = _logs(step, 1.0, 2.0, heads=heads)
        if step == 92:
            # Even a loss shock must not adjust LR if the G iteration was skipped.
            logs = _logs(step, 1e8, 1e-8, skipped=True, heads=heads)
        live_logs = copy.deepcopy(logs)
        live._adapt_discriminator_lr(live_logs)
        decision = live_logs['gan/d_lr_control']
        for runtime in live.step_critics:
            assert runtime.optimizer.param_groups[0]['lr'] == live.d_lr_controller.current_lr
        if step <= 50:
            assert decision['action'] == 'hold' and decision['new_lr'] == BASE_D_LR
        if step == 70:
            assert decision['action'] == 'decrease' and decision['new_lr'] == BASE_D_LR * 0.5
            assert decision['effective_from_step'] == 71
        if step == 91:
            assert decision['action'] == 'increase' and decision['new_lr'] == BASE_D_LR * 0.5 * 1.25
        if step == 92:
            assert decision['action'] == 'hold' and decision['reason'] == 'invalid_observation'
        if decision['changed']:
            actions.append((step, decision['action'], decision['new_lr']))
        if resumed is not None:
            resumed_logs = copy.deepcopy(logs)
            resumed._adapt_discriminator_lr(resumed_logs)
            _assert_same(live_logs, resumed_logs)
            _assert_same(live.d_lr_controller.state_dict(), resumed.d_lr_controller.state_dict())
            for first, second in zip(live.step_critics, resumed.step_critics):
                _assert_same(first.optimizer.state_dict(), second.optimizer.state_dict())
        if step == 65:
            # No tensor artifacts on disk: real serialization round-trip remains
            # in RAM. Controller additionally traverses JSON's persisted schema.
            saved = {'controller': json.loads(json.dumps(live.d_lr_controller.state_dict())),
                     'critics': [runtime.optimizer.state_dict() for runtime in live.step_critics],
                     'generator_optimizer': live.optimizer.state_dict()}
            stream = io.BytesIO()
            torch.save(saved, stream)
            stream.seek(0)
            restored = torch.load(stream, map_location='cpu', weights_only=False)
            resumed = _controller_shell(rank, trainer_class, AdaptiveDLRConfig, AdaptiveDLRController,
                                        saved=restored, heads=heads)
        for runtime, moments in zip(live.step_critics, original_d_moments):
            _assert_same(runtime.optimizer.state_dict()['state'], moments)
        _assert_same(live.optimizer.state_dict(), original_g)
        torch.testing.assert_close(live.g_parameter, original_g_parameter, rtol=0, atol=0)
        if step in (65, 70, 91, 92, 93):
            states = [None, None]
            dist.all_gather_object(states, live.d_lr_controller.state_dict())
            _assert_same(states[0], states[1])
    assert len(actions) == 2
    return {'actions': actions, 'effective_lr': live.d_lr_controller.current_lr,
            'checkpoint_roundtrip_same_trajectory': True, 'G_optimizer_unchanged': True,
            'D_moments_unchanged_by_control': True}


def _worker(rank, port):
    os.environ.update(RANK=str(rank), WORLD_SIZE='2', LOCAL_RANK=str(rank),
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES='')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=40))
    try:
        # Existing fixtures load the true implementation, not a copied loss or
        # controller. Their Py3.9 annotation loader changes no repository source.
        runtime_fixture = _load_fixture('test_spatial_critic_runtime.py', '_adaptive_real_runtime_fixture')
        trainer_fixture = _load_fixture('test_spatial_trainer.py', '_adaptive_real_trainer_fixture')
        report = {'rank': rank, 'backend': 'gloo', 'world_size': 2, 'CPU_only': True,
                  'g_metrics': _global_g_oracle(rank, runtime_fixture),
                  'controller': _controller_oracle(rank, trainer_fixture),
                  'two_head_g_metrics': _global_g_oracle(rank, runtime_fixture, two_head=True),
                  'two_head_controller': _controller_oracle(rank, trainer_fixture, heads=HEADS[:2])}
        print(json.dumps({'passed': True, **report}, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='CPU Gloo unavailable')
def test_two_rank_global_raw_g_local_gradient_and_trainer_controller_resume():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       PYTHONPATH=os.pathsep.join((str(ROOT / 'z-image'), str(ROOT), os.environ.get('PYTHONPATH', ''))))
    processes = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--gloo-worker',
                                   str(rank), str(port)], env=environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True) for rank in range(2)]
    outputs = []
    try:
        for rank, process in enumerate(processes):
            output, _ = process.communicate(timeout=65)
            outputs.append((rank, process.returncode, output))
        failures = [entry for entry in outputs if entry[1] != 0]
        assert not failures, '\n'.join(f'rank={rank} exit={code}\n{output}' for rank, code, output in outputs)
        reports = [json.loads(next(line for line in output.splitlines() if line.startswith('{"CPU_only"')))
                   for _, _, output in outputs]
        assert all(report['passed'] and report['backend'] == 'gloo' for report in reports)
        assert reports[0]['controller']['actions'] == reports[1]['controller']['actions']
        assert reports[0]['two_head_controller']['actions'] == reports[1]['two_head_controller']['actions']
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--gloo-worker':
        _worker(int(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit('Run with pytest; the worker CLI is test-internal')
