"""CPU mixed-runtime tests using the real legacy critic and tiny spatial maps."""
import copy
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

# Avoid optional package __init__ imports; all tested implementation files are
# loaded unchanged under a private package, including the real legacy builder.
ROOT = Path(__file__).resolve().parents[2]
for name, directory in [('runtime_fixture', ROOT / 'rdm'),
                        ('runtime_fixture.sw_lmmd', ROOT / 'rdm/sw_lmmd'),
                        ('runtime_fixture.utils', ROOT / 'rdm/utils')]:
    package = types.ModuleType(name); package.__path__ = [str(directory)]
    sys.modules[name] = package
module = importlib.import_module('runtime_fixture.sw_lmmd.spatial_critic_runtime')
GANConfig = importlib.import_module('runtime_fixture.sw_lmmd.config').GANConfig
SpatialGANRuntime = module.SpatialGANRuntime
SpatialBranchSpec = importlib.import_module('runtime_fixture.sw_lmmd.spatial_adversarial').SpatialBranchSpec


class FakeStore:
    num_rows = 12
    dims = {'dinov3_l': 5, 'siglip2': 7, 'aimv2_huge': 6}

    def __init__(self):
        generator = torch.Generator().manual_seed(77)
        self.text = F.normalize(torch.randn(12, 2560, generator=generator), dim=-1)
        self.images = {name: torch.randn(12, dim, generator=generator) for name, dim in self.dims.items()}

    def dim(self, name): return self.dims[name]
    def reference_image_features(self, name, rows): return self.images[name][rows]
    def text_features(self, rows): return self.text[rows]


class SpatialRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(123)
        self.store = FakeStore()
        self.base_names = ['dinov3_l', 'siglip2', 'aimv2_huge']
        self.cfg = GANConfig(seed=11, hidden=16, depth=1, stats_rows=8, spectral_norm=True)
        self.specs = {'dinov3_l': SpatialBranchSpec(3, 8, 1),
                      'qwen3vl_4b': SpatialBranchSpec(4, 16, 2)}

    def runtime(self, micro, *, seed=19):
        return SpatialGANRuntime(self.cfg, self.store, self.base_names, 'cpu', torch.float64,
            critic_micro_batch=micro, step_seed=seed, spatial_branch_specs=self.specs,
            spatial_options={'heads': 2})

    def inputs(self, requires_grad=False):
        fake = {'dinov3_l': torch.randn(5, 3, 8, 8, dtype=torch.float64, requires_grad=requires_grad),
                'qwen3vl_4b': torch.randn(5, 4, 16, 16, dtype=torch.float64, requires_grad=requires_grad)}
        real = {name: torch.randn_like(value) for name, value in fake.items()}
        fake_g = {'aimv2_huge': torch.randn(5, 11, dtype=torch.float64, requires_grad=requires_grad)}
        real_g = {'aimv2_huge': torch.randn(5, 11, dtype=torch.float64)}
        tau = self.store.text[:5].double().detach().requires_grad_(requires_grad)
        return fake, real, fake_g, real_g, tau

    def test_aim_initialization_stats_and_rng_are_legacy_identical(self):
        before = torch.get_rng_state().clone(); runtime = self.runtime(2)
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
        cfg = copy.deepcopy(self.cfg); cfg.seed = 19
        old = module.build_critic(cfg, self.store, self.base_names, device='cpu', dtype=torch.float64)
        actual = runtime.critic['global']
        self.assertEqual(actual.names, ['aimv2_huge'])
        for name, value in actual.heads['aimv2_huge'].state_dict().items():
            torch.testing.assert_close(value, old.critic.heads['aimv2_huge'].state_dict()[name], atol=0, rtol=0)
        for name, value in actual.image_norm['aimv2_huge'].state_dict().items():
            torch.testing.assert_close(value, old.critic.image_norm['aimv2_huge'].state_dict()[name], atol=0, rtol=0)
        for name, value in actual.text_norm.state_dict().items():
            torch.testing.assert_close(value, old.critic.text_norm.state_dict()[name], atol=0, rtol=0)

    def test_one_vs_uneven_chunks_parameter_update_and_sn_buffers_match(self):
        whole, chunks = self.runtime(5), self.runtime(2)
        inputs = self.inputs()
        full_logs, chunk_logs = whole.update(*inputs), chunks.update(*inputs)
        self.assertEqual(whole.updates, 1); self.assertEqual(chunks.updates, 1)
        for name, value in whole.critic.state_dict().items():
            torch.testing.assert_close(value, chunks.critic.state_dict()[name], atol=2e-9, rtol=1e-8)
        for name in full_logs:
            if isinstance(full_logs[name], (bool, str)): self.assertEqual(full_logs[name], chunk_logs[name])
            else: self.assertAlmostEqual(full_logs[name], chunk_logs[name], places=10)
        self.assertAlmostEqual(full_logs['gan/d_loss'], sum(full_logs[f'gan/{n}/d_loss']
                                                         for n in whole.names) / 3, places=12)

    def test_spectral_norm_training_forward_occurs_exactly_once_per_update(self):
        runtime = self.runtime(2); counts = {}; handles = []
        for name, child in runtime.critic.named_modules():
            if child.__class__.__name__ == '_SpectralNorm':
                counts[name] = 0
                def count(layer, args, key=name):
                    if layer.training: counts[key] += 1
                handles.append(child.register_forward_pre_hook(count))
        runtime.update(*self.inputs())
        self.assertTrue(counts); self.assertEqual(set(counts.values()), {1})
        fake, _, fake_g, _, tau = self.inputs(requires_grad=True)
        runtime.generator_loss(fake, fake_g, tau)[0].backward()
        self.assertEqual(set(counts.values()), {1})
        for handle in handles: handle.remove()

    def test_float32_gradients_and_updates_above_roundoff_agree(self):
        options = {'spatial_branch_specs': self.specs, 'spatial_options': {'heads': 2}}
        whole = SpatialGANRuntime(self.cfg, self.store, self.base_names,
                                  critic_micro_batch=5, step_seed=19, **options)
        chunks = SpatialGANRuntime(self.cfg, self.store, self.base_names,
                                   critic_micro_batch=2, step_seed=19, **options)
        gradients = []
        for runtime in (whole, chunks):
            original_step = runtime.optimizer.step
            def step(*args, rt=runtime, original=original_step, **kwargs):
                gradients.append({name: p.grad.clone() for name, p in rt.critic.named_parameters()})
                return original(*args, **kwargs)
            runtime.optimizer.step = step
        values = self.inputs(); whole.update(*values); chunks.update(*values)
        for name, value in whole.critic.named_parameters():
            other = dict(chunks.critic.named_parameters())[name]
            g, h = gradients[0][name], gradients[1][name]
            torch.testing.assert_close(g, h, atol=3e-6, rtol=1e-4)
            difference = (value.detach() - other.detach()).abs()
            # Adam can amplify FP32 cancellation noise near epsilon (e.g. a
            # balanced hinge readout bias whose exact gradient is zero). Require
            # all updates beyond numerical tolerance to be confined to that
            # near-zero-gradient regime, rather than claiming bitwise parity.
            sensitive = difference > 2e-6
            if sensitive.any():
                self.assertLess(torch.maximum(g.abs(), h.abs())[sensitive].max().item(), 3e-6)
            self.assertLessEqual(difference.max().item(), 2 * self.cfg.lr)
        for name, value in whole.critic.named_buffers():
            torch.testing.assert_close(value, dict(chunks.critic.named_buffers())[name], atol=0, rtol=0)

    def test_g_uses_updated_disc_caption_concat_weight_one_third_and_image_prefix_only(self):
        runtime = self.runtime(2); runtime.update(*self.inputs())
        fake, _, fake_g, _, tau = self.inputs(requires_grad=True)
        before = {n: b.clone() for n, b in runtime.critic.named_buffers()}
        captured = []
        handle = runtime.critic['spatial'].heads['dinov3_l'].fusion.register_forward_pre_hook(
            lambda layer, args: captured.append(args[0].detach()))
        loss, logs = runtime.generator_loss(fake, fake_g, tau); handle.remove()
        self.assertAlmostEqual(loss.item(), sum(logs[f'gan/{n}/g_loss'] for n in runtime.names) / 3, places=12)
        self.assertEqual(captured[0].shape, (2, 64, 16 + 2560))
        torch.testing.assert_close(captured[0][..., 16:], tau[:2, None].expand(-1, 64, -1))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in runtime.critic.parameters()))
        for value in [*fake.values(), fake_g['aimv2_huge']]:
            self.assertTrue(torch.isfinite(value.grad).all()); self.assertGreater(value.grad.abs().sum().item(), 0)
        torch.testing.assert_close(fake_g['aimv2_huge'].grad[:, 6:], torch.zeros(5, 5, dtype=torch.float64))
        self.assertIsNone(tau.grad)
        for name, value in runtime.critic.named_buffers():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_d_detaches_all_inputs_and_warmup_uses_existing_cfg(self):
        runtime = self.runtime(2); values = self.inputs(requires_grad=True)
        runtime.update(*values)
        for tensor in [*values[0].values(), *values[2].values(), values[4]]:
            self.assertIsNone(tensor.grad)
        self.assertFalse(runtime.generator_active(49)); self.assertTrue(runtime.generator_active(50))
        self.assertEqual(runtime.optimizer.param_groups[0]['lr'], self.cfg.lr)
        self.assertEqual(runtime.optimizer.param_groups[0]['betas'], self.cfg.betas)

    def test_checkpoint_restores_optimizer_and_allows_different_microbatch(self):
        first, resumed = self.runtime(2), self.runtime(1)
        first.update(*self.inputs()); resumed.load_state_dict(copy.deepcopy(first.state_dict()))
        values = self.inputs(); first.update(*values); resumed.update(*values)
        self.assertEqual(resumed.updates, 2)
        for name, value in first.critic.state_dict().items():
            torch.testing.assert_close(value, resumed.critic.state_dict()[name], atol=2e-9, rtol=1e-8)
        wrong_seed = self.runtime(1, seed=20)
        with self.assertRaises(ValueError): wrong_seed.load_state_dict(first.state_dict())

    def test_d_gradient_norms_are_after_reduce_before_zero_without_extra_collectives(self):
        runtime = self.runtime(2)
        expected = {}
        original_step = runtime.optimizer.step
        def capture():
            def norm(parameters):
                return torch.cat([p.grad.detach().reshape(-1) for p in parameters]).norm().item()
            expected['gan/d_grad_norm'] = norm(runtime.critic.parameters())
            for name in runtime.spatial_names:
                expected[f'gan/{name}/d_grad_norm'] = norm(runtime.critic['spatial'].heads[name].parameters())
            for name in runtime.global_names:
                expected[f'gan/{name}/d_grad_norm'] = norm(runtime.critic['global'].heads[name].parameters())
            original_step()
        runtime.optimizer.step = capture
        # Artificial mean reduction is observable: the telemetry must reflect
        # post-reduce gradients, not the unreduced local values. The second call
        # is the pre-existing metrics reduction; no norm collective is added.
        sizes = []
        def simulated_reduce(value):
            sizes.append(value.numel())
            if len(sizes) == 1: value.mul_(0.5)
            return value
        with mock.patch.object(module, 'all_reduce_mean_', side_effect=simulated_reduce):
            logs = runtime.update(*self.inputs())
        self.assertEqual(len(sizes), 2)
        self.assertGreater(sizes[0], sizes[1])
        self.assertEqual(logs['gan/d_updates'], 1)
        self.assertTrue(all(p.grad is None for p in runtime.critic.parameters()))
        for key, value in expected.items():
            self.assertGreater(value, 0)
            self.assertAlmostEqual(logs[key], value, places=12)
        self.assertAlmostEqual(logs['gan/d_grad_norm'] ** 2,
            sum(logs[f'gan/{name}/d_grad_norm'] ** 2 for name in runtime.names), places=12)
        self.assertEqual(runtime.update(*self.inputs())['gan/d_updates'], 2)

    def test_r1_and_unsupported_global_critic_are_explicitly_rejected(self):
        self.cfg.r1_gamma = 1.0
        with self.assertRaisesRegex(ValueError, 'R1'): self.runtime(2)
        self.cfg.r1_gamma = 0.0; self.cfg.critic = 'block'
        with self.assertRaisesRegex(ValueError, 'row'): self.runtime(2)

    def test_two_head_gan_excludes_aim_and_averages_two_spatial_losses(self):
        self.cfg.loss = 'ns'
        runtime = SpatialGANRuntime(self.cfg, self.store, self.base_names, 'cpu', torch.float64,
            critic_micro_batch=2, step_seed=19, global_remaining=(),
            spatial_branch_specs=self.specs, spatial_options={'heads': 2})
        self.assertEqual(runtime.names, ['dinov3_l', 'qwen3vl_4b'])
        self.assertEqual(runtime.encoder_weight, 0.5)
        self.assertEqual(list(runtime.critic['global'].parameters()), [])
        values = self.inputs(requires_grad=True)
        d = runtime.update(*values)
        self.assertNotIn('gan/aimv2_huge/d_loss', d)
        self.assertAlmostEqual(d['gan/d_loss'], sum(d[f'gan/{n}/d_loss'] for n in runtime.names)/2, places=12)
        fake, _, fake_global, _, tau = values
        loss, logs = runtime.generator_loss(fake, fake_global, tau)
        self.assertAlmostEqual(loss.item(), sum(logs[f'gan/{n}/g_loss'] for n in runtime.names)/2, places=12)
        loss.backward()
        self.assertIsNone(fake_global['aimv2_huge'].grad)
        self.assertTrue(all(torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in fake.values()))
        self.assertEqual(runtime.contract['global'], [])
        with self.assertRaisesRegex(ValueError, 'contract'):
            self.runtime(2).load_state_dict(runtime.state_dict())

    def test_new_two_MMD_store_runs_D_and_G_without_removed_features(self):
        class TwoStore(FakeStore):
            dims = {'dinov3_l': 5, 'qwen3vl_4b': 9}
        self.store = TwoStore(); self.cfg.loss = 'ns'
        names = ['dinov3_l', 'qwen3vl_4b']
        runtime = SpatialGANRuntime(self.cfg, self.store, names, 'cpu', torch.float64,
            critic_micro_batch=2, step_seed=19, global_remaining=(),
            spatial_branch_specs=self.specs, spatial_options={'heads': 2})
        self.assertEqual(runtime.names, names); self.assertEqual(runtime.encoder_weight, .5)
        self.assertEqual(runtime.contract['base_encoders'], names)
        self.assertEqual(list(runtime.critic['global'].parameters()), [])
        fake, real, _, _, tau = self.inputs(requires_grad=True)
        d = runtime.update(fake, real, {}, {}, tau)
        self.assertEqual(runtime.updates, 1); self.assertFalse(d['gan/d_skipped'])
        self.assertAlmostEqual(d['gan/d_loss'], sum(d[f'gan/{n}/d_loss'] for n in names)/2, places=12)
        loss, logs = runtime.generator_loss(fake, {}, tau)
        self.assertAlmostEqual(loss.item(), sum(logs[f'gan/{n}/g_loss'] for n in names)/2, places=12)
        loss.backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in fake.values()))
        self.assertIsNone(tau.grad)
        with self.assertRaisesRegex(ValueError, 'global GAN head'):
            SpatialGANRuntime(self.cfg, self.store, names, global_remaining=('aimv2_huge',))

    def test_ns_all_three_heads_match_softplus_objective_gradients_and_raw_logit_logs(self):
        self.cfg.loss = 'ns'; self.cfg.lr = 1e-5
        whole, chunks = self.runtime(5), self.runtime(2)
        values = self.inputs(requires_grad=True)
        fake, real, fake_g, real_g, tau = values
        expected_d, expected_proxy = {}, {}
        original_step = whole.optimizer.step
        def checked_step():
            # Same features and same SN vectors, before the actual optimizer step.
            with module.frozen_spatial_critic(whole.critic):
                logits = whole._score({n: torch.cat((real[n], fake[n])).detach() for n in whole.spatial_names},
                    {'aimv2_huge': torch.cat((real_g['aimv2_huge'][:, :6], fake_g['aimv2_huge'][:, :6])).detach()},
                    torch.cat((tau, tau)).detach())
            for name, output in logits.items():
                expected_d[name] = (F.softplus(-output[:5]).mean() + F.softplus(output[5:]).mean()).item()
                expected_proxy[name] = F.softplus(-output[5:]).mean().item()
            return original_step()
        whole.optimizer.step = checked_step
        full_logs, chunk_logs = whole.update(*values), chunks.update(*values)
        for name in whole.names:
            self.assertAlmostEqual(full_logs[f'gan/{name}/d_loss'], expected_d[name], places=12)
            self.assertAlmostEqual(full_logs[f'gan/{name}/g_proxy_pre_d_window'], expected_proxy[name], places=12)
            self.assertAlmostEqual(full_logs[f'gan/{name}/d_loss'], chunk_logs[f'gan/{name}/d_loss'], places=12)
        for name, value in whole.critic.state_dict().items():
            torch.testing.assert_close(value, chunks.critic.state_dict()[name], atol=2e-9, rtol=1e-8)
        with module.frozen_spatial_critic(whole.critic):
            logits = whole._score(fake, {'aimv2_huge': fake_g['aimv2_huge'][:, :6]}, tau.detach())
        expected = sum(F.softplus(-output).mean() for output in logits.values()) / 3
        loss, logs = whole.generator_loss(fake, fake_g, tau)
        torch.testing.assert_close(loss, expected)
        image_inputs = tuple(fake.values()) + (fake_g['aimv2_huge'],)
        expected_grads = torch.autograd.grad(expected, image_inputs)
        actual_grads = torch.autograd.grad(loss, image_inputs)
        for actual, oracle in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, oracle, atol=2e-12, rtol=1e-9)
            self.assertTrue(torch.isfinite(actual).all()); self.assertGreater(actual.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in whole.critic.parameters()))
        self.assertIsNone(tau.grad)
        self.assertAlmostEqual(logs['gan/g_fake'], sum(v.mean().item() for v in logits.values()) / 3, places=12)
        self.assertNotAlmostEqual(logs['gan/g_fake'], -logs['gan/g_loss'], places=6)
        for name in whole.names:
            self.assertAlmostEqual(logs[f'gan/{name}/g_fake'], logits[name].mean().item(), places=12)
        self.assertEqual(whole.contract['loss'], 'ns')
        self.cfg.loss = 'hinge'
        with self.assertRaisesRegex(ValueError, 'contract'):
            self.runtime(2).load_state_dict(whole.state_dict())


if __name__ == '__main__':
    unittest.main()


def test_clip_after_full_chunk_accumulation_and_rank_mean():
    fixture = SpatialRuntimeTests()
    fixture.setUp()
    fixture.cfg.grad_clip = 0.01
    runtime = fixture.runtime(2)
    observed = []
    original = runtime.optimizer.step
    def checked_step(*args, **kwargs):
        norm = torch.stack([p.grad.double().square().sum() for p in runtime.critic.parameters()]).sum().sqrt()
        observed.append(float(norm))
        assert float(norm) <= .01000001
        return original(*args, **kwargs)
    runtime.optimizer.step = checked_step
    # Simulate a remote rank contributing 3x the local accumulated gradient.
    # The resulting rank mean is 2x, and clipping must follow that mean.
    def rank_mean(flat):
        flat.mul_(2)
    with mock.patch.object(module, 'all_reduce_mean_', side_effect=rank_mean):
        logs = runtime.update(*fixture.inputs())
    assert len(observed) == 1 and not logs['gan/d_skipped']
    assert logs['gan/d_grad_norm'] > .01
    assert 0 < logs['gan/d_clip_coefficient'] < 1
