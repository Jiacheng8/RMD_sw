"""CPU integration of real mixed GAN/MMD runtime with tiny frozen image encoders.

The toy adapters use 8px RGB; the real adapters independently test their strict
1024px input contract. Store readers, MMD, caches, mixed critic, optimizer,
two-level GradCache, refresh and checkpoint/resume are the production code.
"""
from __future__ import annotations

import __future__
import copy
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class _AnnotationLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # No source changes or .pyc writes. Match the deferred annotations used by
        # the training Python while exercising system-Python3.9's installed torch.
        return compile(self.get_data(self.path), self.path, 'exec',
                       flags=__future__.annotations.compiler_flag, dont_inherit=True)


class _AnnotationFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not (fullname == 'rdm' or fullname.startswith('rdm.') or
                fullname == 'zimage_distill' or fullname.startswith('zimage_distill.')):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            spec.loader = _AnnotationLoader(fullname, spec.origin)
        return spec


_finder = _AnnotationFinder() if sys.version_info < (3, 10) else None
if _finder:
    sys.meta_path.insert(0, _finder)
try:
    from rdm.representation.spatial_battery import SpatialBattery
    from rdm.sw_lmmd import CacheConfig, GANConfig, SlidingWindowSchedule, build_row_order
    from rdm.sw_lmmd.cache import GeneratedWindowCache
    from rdm.sw_lmmd.reference_store import write_reference_store
    from rdm.sw_lmmd.refresh import probe_drift, refresh_retained
    from rdm.sw_lmmd.spatial_adversarial import SpatialBranchSpec
    from rdm.sw_lmmd.spatial_critic_runtime import SpatialGANRuntime
    from zimage_distill.spatial_trainer import SpatialMultiStepTrainer, patch_stream
    from zimage_distill.store import ZImageReferenceStore, stream_names
    from zimage_distill.trainer import MultiStepTrainer
finally:
    if _finder:
        sys.meta_path.remove(_finder)


NAMES = ('dinov3_l', 'siglip2', 'aimv2_huge')
DIMS = (4, 5, 3)
ROWS, K, B = 24, 8, 4
SPECS = {'dinov3_l': SpatialBranchSpec(3, 16, 2),
         'qwen3vl_4b': SpatialBranchSpec(2, 8, 1)}


@pytest.fixture(scope='module', autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def store_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('spatial_trainer_reference')
    rng = np.random.default_rng(3)
    features = {name: rng.normal(size=(ROWS, dim)).astype('float32') for name, dim in zip(NAMES, DIMS)}
    text = rng.normal(size=(ROWS, 2560)).astype('float32')
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    write_reference_store(str(root), encoder_features=features, text_features=text,
                          bandwidths={name: {'sigma': 1.4, 'beta': 0.7} for name in NAMES},
                          row_order=build_row_order(ROWS, seed=3), prompt_ids=np.arange(ROWS),
                          extra_metadata={'reference_type': 'toy-zimage',
                                          'prompt_hash': 'toy-prompts', 'render_hash': 'toy-render'})
    with (root / 'prompts.jsonl').open('w') as handle:
        for row in range(ROWS):
            handle.write(json.dumps({'prompt': f'p{row}'}) + '\n')
    return str(root)


class ToyGenerator:
    def __init__(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43)
            self.model = nn.Linear(4, 3 * 8 * 8)
        self.sampling_args = {'num_steps': 2, 'detach_between_steps': False}
        self.calls = []

    def sample_steps(self, noise, prompts=None, contexts=None):
        self.calls.append(torch.is_grad_enabled())
        # Row-wise linears make chunk-size invariance independent of BLAS GEMM choices.
        first = torch.stack([self.model(row) for row in noise[:, 0]])
        bias = torch.tensor([int(prompt[1:]) * 0.001 for prompt in prompts])[:, None]
        first = (0.5 + 0.2 * torch.tanh(first + bias)).reshape(-1, 3, 8, 8)
        second = torch.stack([self.model(row) for row in noise[:, 1]])
        second = 0.5 + 0.2 * torch.tanh(second + bias + 0.1 * first.flatten(1))
        return [first, second.reshape(-1, 3, 8, 8)]


class ToyGlobalBattery(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(5)
        self.weights = nn.ParameterDict({name: nn.Parameter(
            torch.randn(3 * 4 * 4, dim, generator=generator) / 10, requires_grad=False)
            for name, dim in zip(NAMES, DIMS)})

    def forward(self, images, only=None):
        return {name: images.flatten(1) @ weight for name, weight in self.weights.items()
                if only is None or name in only}


class ToyMapEncoder(nn.Module):
    def __init__(self, channels, grid):
        super().__init__()
        generator = torch.Generator().manual_seed(channels)
        self.weight = nn.Parameter(torch.randn(channels, 3, 1, 1, generator=generator) * 0.2,
                                   requires_grad=False)
        self.grid = grid

    def forward(self, pixels):
        pixels = F.interpolate(pixels, size=(self.grid, self.grid), mode='bilinear', align_corners=False)
        return F.conv2d(pixels, self.weight).tanh()

    def feature_contract(self):
        return {'toy_rgb': 8, 'channels': self.weight.shape[0], 'grid': self.grid}


class ToySpatialBattery(SpatialBattery):
    """Keep the real bounded reference cache and quantized feature boundary."""
    def __init__(self):
        super().__init__(device='cpu', encoders={name: ToyMapEncoder(spec.channels, spec.grid_size)
                         for name, spec in SPECS.items()}, max_real_rows=K, micro_batch=2)

    def forward(self, images):
        assert tuple(images.shape[1:]) == (3, 8, 8)
        return {name: model(images).to(torch.bfloat16).float() for name, model in self.encoders.items()}


class ToyReferenceImages:
    num_rows = ROWS
    manifest = {'contract': {'prompt_hash': 'toy-prompts', 'render_hash': 'toy-render'},
                'index': {'sha256': 'toy-row-image-index'}}

    def __init__(self):
        self.png = torch.randint(0, 256, (ROWS, 3, 8, 8), dtype=torch.uint8,
                                 generator=torch.Generator().manual_seed(9))

    def tensor(self, row_ids, *, device, dtype):
        return self.png[np.asarray(row_ids)].to(device=device, dtype=dtype) / 255


def build(root, *, spatial=True, micro=2, critic_micro=2, weight=1.5, warmup=0,
          lr=0.005, step_weights=(0.25, 1.0), adaptive_d_lr=None, gan_loss='hinge',
          d_lr=0.002, allow_legacy_adaptive=False, global_gan=('aimv2_huge',)):
    store = ZImageReferenceStore(root, NAMES)
    generator = ToyGenerator()
    kwargs = dict(encoders=list(NAMES), num_steps=2, step_weights=list(step_weights),
                  feature_pool=2, feature_quantize=True, noise_shape=(2, 4),
                  cache=GeneratedWindowCache(stream_names(2, NAMES), store_dtype=torch.float32),
                  micro_batch=micro, grad_clip=0.0, kernel_block=4, seed=7, joint=True,
                  cache_cfg=CacheConfig(store_dtype=torch.float32), device='cpu',
                  feature_dtype=torch.float32, context_dtype=torch.float32, monitor_every=1)
    optimizer = torch.optim.AdamW(generator.model.parameters(), lr=lr, weight_decay=0)
    battery = ToyGlobalBattery()
    schedule = SlidingWindowSchedule(store.row_order(), window_size=K, stride=B)
    if not spatial:
        return MultiStepTrainer(generator, battery, store, schedule, optimizer, **kwargs)
    critics = [SpatialGANRuntime(
        GANConfig(enabled=True, hidden=8, depth=1, spectral_norm=False, stats_rows=16,
                  g_start_step=warmup, weight=weight, lr=d_lr, loss=gan_loss, seed=k),
        store, list(NAMES), dtype=torch.float32, critic_micro_batch=critic_micro,
        spatial_branch_specs=SPECS, spatial_options={'heads': 2},
        global_remaining=global_gan) for k in range(2)]
    return SpatialMultiStepTrainer(generator, battery, store, schedule, optimizer,
                                   critics=critics, spatial_battery=ToySpatialBattery(),
                                   reference_images=ToyReferenceImages(), adaptive_d_lr=adaptive_d_lr,
                                   adaptive_d_lr_allow_legacy_resume=allow_legacy_adaptive, **kwargs)


def _assert_aligned(trainer):
    first = trainer.cache.entries[trainer.encoder_names[0]]
    for entry in [*trainer.cache.entries.values(), *trainer.spatial_cache.entries.values()]:
        assert np.array_equal(entry.row_ids, first.row_ids)
        assert torch.equal(entry.generated_at_steps, first.generated_at_steps)
    for entry in trainer.spatial_cache.entries.values():
        assert entry.features.device.type == 'cpu' and entry.features.dtype == torch.bfloat16


def test_weight_zero_preserves_original_global_mmd_gradients_and_rows(store_root):
    baseline, spatial = build(store_root, spatial=False, lr=0), build(store_root, weight=0, lr=0)
    for trainer in (baseline, spatial):
        trainer.optimizer.step = lambda: None
        trainer.bootstrap()
    for _ in range(2):
        expected, actual = baseline.step(), spatial.step()
        assert actual['force'] == pytest.approx(expected['force'], rel=1e-6, abs=1e-7)
        for p, q in zip(baseline.generator.model.parameters(), spatial.generator.model.parameters()):
            torch.testing.assert_close(q.grad, p.grad, rtol=2e-5, atol=2e-7)
        for name in baseline.encoder_names:
            old, new = baseline.cache.entries[name], spatial.cache.entries[name]
            assert np.array_equal(old.row_ids, new.row_ids)
            torch.testing.assert_close(old.features, new.features, rtol=0, atol=0)
        _assert_aligned(spatial)


def test_bootstrap_two_updates_loss_names_gradient_routes_probe_and_refresh(store_root):
    trainer = build(store_root)
    before_g = [p.detach().clone() for p in trainer.generator.model.parameters()]
    before_d = [{name: p.detach().clone() for name, p in runtime.critic.named_parameters()}
                for runtime in trainer.step_critics]
    trainer.bootstrap()
    _assert_aligned(trainer)
    for _ in range(2):
        logs = trainer.step()
        assert not logs['skipped'] and np.isfinite(logs['loss_total'])
        assert logs['pixel_gc/generator_replay_max_abs'] == 0
        for k in (1, 2):
            assert logs[f's{k}/pixel_mmd_grad_norm'] > 0 and logs[f's{k}/pixel_adv_grad_norm'] > 0
            assert logs[f's{k}/gan/grad_ratio'] == pytest.approx(1.5, rel=1e-6)
            for name in NAMES:
                assert f's{k}/{name}/mmd2' in logs
            for name in ('dinov3_l', 'qwen3vl_4b', 'aimv2_huge'):
                assert f's{k}/gan/{name}/d_loss' in logs and f's{k}/gan/{name}/g_loss' in logs
            assert f's{k}/gan/siglip2/g_loss' not in logs
        _assert_aligned(trainer)
    assert any(not torch.equal(old, new) for old, new in zip(before_g, trainer.generator.model.parameters()))
    for before, runtime in zip(before_d, trainer.step_critics):
        assert runtime.updates == 2
        after = dict(runtime.critic.named_parameters())
        for branch in SPECS:
            for suffix in ('aggregator.weight', 'fusion.weight', 'blocks.0.qkv.weight', 'readout.weight'):
                name = f'spatial.heads.{branch}.{suffix}'
                assert not torch.equal(before[name], after[name])
        assert any(not torch.equal(before[name], value) for name, value in after.items()
                   if name.startswith('global.heads.aimv2_huge.'))
        assert all(p.grad is None for p in runtime.critic.parameters())
    assert len(trainer.spatial_battery._real) <= K
    assert all(p.grad is None for p in trainer.battery.parameters())
    assert all(p.grad is None for p in trainer.spatial_battery.parameters())
    assert all(torch.isfinite(p.grad).all() for p in trainer.generator.model.parameters())

    window = trainer.schedule.peek()
    global_before = copy.deepcopy(trainer.cache.state_dict())
    spatial_before = copy.deepcopy(trainer.spatial_cache.state_dict())
    probe = probe_drift(trainer, window, n_rows=4)
    assert probe['drift_probe_rows'] == 4
    for name in trainer.spatial_stream_names:
        assert f'{name}/drift_mean' in probe
        assert torch.equal(trainer.spatial_cache.entries[name].features, spatial_before[name]['features'])
    for name in trainer.encoder_names:
        assert torch.equal(trainer.cache.entries[name].features, global_before[name]['features'])
    refresh = refresh_retained(trainer, window)
    assert refresh['refreshed_rows'] == refresh['spatial_refreshed_rows'] == K-B
    _assert_aligned(trainer)
    fresh = trainer.rollout_refresh(window.retained_ids)
    for name in trainer.encoder_names:
        torch.testing.assert_close(trainer.cache.retained(name, window.retained_ids), fresh[name], rtol=0, atol=0)
    for name in trainer.spatial_stream_names:
        assert torch.equal(trainer.spatial_cache.retained(name, window.retained_ids), fresh[name].bfloat16())


def test_checkpoint_load_preserves_both_windows_and_next_update(store_root):
    trainer = build(store_root)
    trainer.bootstrap()
    trainer.step(); trainer.step()
    snapshot = copy.deepcopy(trainer.state_dict())
    resumed = build(store_root)
    result = resumed.load_state_dict(snapshot)
    assert result['cache'] and result['spatial_cache'] and result['critic'] and result['optimizer']
    _assert_aligned(resumed)
    for name in trainer.spatial_stream_names:
        assert torch.equal(trainer.spatial_cache.entries[name].features, resumed.spatial_cache.entries[name].features)
    a, b = trainer.step(), resumed.step()
    assert a['loss_total'] == pytest.approx(b['loss_total'], rel=2e-6, abs=1e-7)
    for p, q in zip(trainer.generator.model.parameters(), resumed.generator.model.parameters()):
        torch.testing.assert_close(p, q, rtol=2e-5, atol=2e-7)
    for left, right in zip(trainer.step_critics, resumed.step_critics):
        for p, q in zip(left.critic.parameters(), right.critic.parameters()):
            torch.testing.assert_close(p, q, rtol=2e-5, atol=2e-7)
    _assert_aligned(resumed)
    for name in trainer.encoder_names:
        torch.testing.assert_close(trainer.cache.entries[name].features,
                                   resumed.cache.entries[name].features, rtol=0, atol=0)
    for name in trainer.spatial_stream_names:
        assert torch.equal(trainer.spatial_cache.entries[name].features,
                           resumed.spatial_cache.entries[name].features)
    with pytest.raises(RuntimeError, match='aligned spatial'):
        resumed.load_state_dict({key: value for key, value in snapshot.items() if key != 'spatial_cache'})
    changed = copy.deepcopy(snapshot)
    changed['spatial_gan_contract']['representations']['encoders']['qwen3vl_4b']['grid'] = 7
    with pytest.raises(RuntimeError, match='representation/reference'):
        resumed.load_state_dict(changed)


@pytest.mark.parametrize('global_gan', [('aimv2_huge',), ()])
def test_enabled_controller_uses_real_losses_and_preserves_mmd_generator_and_resume(store_root, global_gan):
    policy = {'enabled': True, 'window_size': 2, 'volatility_min_jumps': 1,
              'min_lr': 1e-6, 'max_lr': 0.01, 'volatility_ratio': 1e9,
              'volatility_abs_range': 1e9, 'min_abs_change': 1e9}
    controlled = build(store_root, gan_loss='ns', adaptive_d_lr=policy, global_gan=global_gan)
    unchanged = build(store_root, gan_loss='ns', global_gan=global_gan)
    for trainer in (controlled, unchanged):
        trainer.bootstrap()
    for index in range(1, 4):
        actual, reference = controlled.step(), unchanged.step()
        assert actual['gan/d_lr_used'] == actual['gan/d_lr_next'] == 0.002
        assert controlled.d_lr_controller.last_completed_step == index
        assert len(controlled.d_lr_controller.head_keys) == 2 * (2 + len(global_gan))
        assert set(controlled.battery.weights) == set(NAMES)
        for k in (1, 2):
            assert actual[f's{k}/gan/g_metric_scope'] == 'global_active_rows'
            assert actual[f's{k}/gan/g_rows_global'] == B
            assert actual[f's{k}/gan/qwen3vl_4b/g_loss'] >= 0
        assert actual['force'] == pytest.approx(reference['force'], rel=1e-7)
        assert actual['loss_total'] == pytest.approx(reference['loss_total'], rel=1e-7)
        for p, q in zip(controlled.generator.model.parameters(), unchanged.generator.model.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    snapshot = copy.deepcopy(controlled.state_dict())
    resumed = build(store_root, gan_loss='ns', adaptive_d_lr=policy, global_gan=global_gan)
    assert resumed.load_state_dict(snapshot)['adaptive_d_lr'] == 'restored'
    assert resumed.d_lr_controller.state_dict() == controlled.d_lr_controller.state_dict()
    left, right = controlled.step(), resumed.step()
    assert left['gan/d_lr_control'] == right['gan/d_lr_control']
    assert left['loss_total'] == pytest.approx(right['loss_total'], rel=2e-6)
    legacy = copy.deepcopy(snapshot)
    del legacy['adaptive_d_lr']
    with pytest.raises(RuntimeError, match='Legacy checkpoint'):
        build(store_root, gan_loss='ns', adaptive_d_lr=policy, global_gan=global_gan).load_state_dict(legacy)
    explicit = build(store_root, gan_loss='ns', adaptive_d_lr=policy,
                     allow_legacy_adaptive=True, global_gan=global_gan)
    assert explicit.load_state_dict(legacy)['adaptive_d_lr'] == 'explicit_legacy_bootstrap'
    assert explicit.d_lr_controller.last_completed_step == 0
    with pytest.raises(RuntimeError, match='silently disable'):
        build(store_root, gan_loss='ns', global_gan=global_gan).load_state_dict(snapshot)


def test_two_head_disc_encodes_both_real_and_generated_pixels_in_both_backbones(store_root):
    trainer = build(store_root, gan_loss='ns', global_gan=())
    trainer.bootstrap()
    seen = {name: [] for name in SPECS}
    handles = []
    reference_call = [False]
    original_reference = trainer.spatial_battery.reference_features
    def trace_reference(*args, **kwargs):
        reference_call[0] = True
        try: return original_reference(*args, **kwargs)
        finally: reference_call[0] = False
    trainer.spatial_battery.reference_features = trace_reference
    for name,encoder in trainer.spatial_battery.encoders.items():
        def capture(module, args, key=name):
            seen[key].append((reference_call[0], args[0].detach().clone()))
        handles.append(encoder.register_forward_pre_hook(capture))
    expected = trainer.reference_images.png[trainer._pending_window.all_ids].float()/255
    logs = trainer.step()
    for handle in handles: handle.remove()
    assert [c.names for c in trainer.step_critics] == [['dinov3_l','qwen3vl_4b']]*2
    assert trainer.base_names == list(NAMES)
    for name,records in seen.items():
        real = torch.cat([pixels for is_real,pixels in records if is_real])
        assert real.shape == expected.shape
        # Reference rows are the canonical PNG pixels, not stored global vectors.
        torch.testing.assert_close(real, expected, rtol=0, atol=0)
        assert any(not is_real for is_real,_ in records), name
    assert len(seen['dinov3_l']) == len(seen['qwen3vl_4b'])
    for left,right in zip(seen['dinov3_l'],seen['qwen3vl_4b']):
        assert left[0] == right[0]
        torch.testing.assert_close(left[1],right[1],rtol=0,atol=0)
    assert not any('/gan/aimv2_huge/' in key for key in logs)
    for k in [1,2]:
        for name in SPECS: assert logs[f's{k}/gan/{name}/d_grad_norm'] > 0
        for name in NAMES: assert f's{k}/{name}/mmd2' in logs


@pytest.mark.parametrize('micro,critic_micro', [(1, 1), (4, 3)])
def test_microbatch_and_critic_only_warmup(store_root, micro, critic_micro):
    trainer = build(store_root, micro=micro, critic_micro=critic_micro, warmup=1)
    reference = build(store_root, micro=2, critic_micro=2, warmup=1)
    for value in (trainer, reference):
        value.bootstrap()
    for step in range(2):
        a, b = trainer.step(), reference.step()
        if step == 0:
            for k in (1, 2):
                assert a[f's{k}/pixel_adv_lambda'] == a[f's{k}/pixel_adv_grad_norm'] == 0
                assert f's{k}/gan/qwen3vl_4b/g_loss' not in a
        else:
            assert a['s1/pixel_adv_lambda'] > 0
        assert a['loss_total'] == pytest.approx(b['loss_total'], rel=1e-4, abs=1e-6)
        for p, q in zip(trainer.generator.model.parameters(), reference.generator.model.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-4, atol=2e-6)
        _assert_aligned(trainer)


def test_fixed_gan_route_and_scheduled_d_lr_without_balance_norms(store_root):
    from rdm.sw_lmmd.config import LRSchedule
    trainer = build(store_root, weight=5e-5, gan_loss='ns', d_lr=1e-5, global_gan=())
    for runtime in trainer.step_critics:
        runtime.cfg.adaptive = False
        runtime.cfg.grad_clip = .01
    trainer.discriminator_lr_schedule = LRSchedule(name='cosine', total_steps=3,
        min_lr_ratio=.5, end_at_last_step=True)
    trainer.bootstrap()
    for iteration in range(3):
        logs = trainer.step()
        assert not logs['skipped']
        assert logs['pixel_gc/balance_mode'] == 'fixed'
        assert not any('grad_ratio' in key or 'pixel_adv_grad_norm' in key for key in logs)
        for k in [1, 2]:
            assert logs[f's{k}/gan/lambda'] == 5e-5
            assert logs[f's{k}/gan/d_lr'] == pytest.approx(
                1e-5 * trainer.discriminator_lr_schedule.factor(iteration))
            assert 0 < logs[f's{k}/gan/d_clip_coefficient'] <= 1
    state = trainer.state_dict()
    assert state['discriminator_lr_schedule']['end_at_last_step']
    wrong = build(store_root, gan_loss='ns', global_gan=())
    with pytest.raises(RuntimeError, match='Discriminator LR schedule'):
        wrong.load_state_dict(state)
