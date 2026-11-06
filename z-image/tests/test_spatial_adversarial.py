"""Small CPU tests for the independent spatial-GAN contract; no real weights."""
from pathlib import Path
import importlib.util
import sys
import unittest

import torch

source = Path(__file__).resolve().parents[2] / 'rdm/sw_lmmd/spatial_adversarial.py'
spec = importlib.util.spec_from_file_location('independent_spatial_adversarial', source)
spatial = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = spatial
spec.loader.exec_module(spatial)
for name in ('SpatialBranchSpec', 'SpatialFeatureCritic', 'SpatialPatchHead', 'RotaryPosition2D',
             'region_coordinates', 'patch_hinge_loss', 'patch_discriminator_loss', 'patch_generator_loss',
             'frozen_spatial_critic', 'spatial_discriminator_loss', 'spatial_generator_loss'):
    globals()[name] = getattr(spatial, name)


class SpatialAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(7)

    def critic(self, spectral_norm=False):
        return SpatialFeatureCritic({'dino': SpatialBranchSpec(4, 16, 2),
                                     'qwen': SpatialBranchSpec(6, 8, 1)},
            hidden=16, depth=2, heads=2, text_dim=9, spectral_norm=spectral_norm)

    def inputs(self, *, requires_grad=False):
        return {'dino': torch.randn(2, 4, 16, 16, requires_grad=requires_grad),
                'qwen': torch.randn(2, 6, 8, 8, requires_grad=requires_grad)}

    def test_spatial_shapes_and_explicit_caption_concat(self):
        critic = self.critic(); text = torch.randn(2, 9)
        regions = critic.aggregate(self.inputs())
        self.assertEqual({k: tuple(v.shape) for k, v in regions.items()},
                         {'dino': (2, 64, 16), 'qwen': (2, 64, 16)})
        captured = []
        handle = critic.heads['dino'].fusion.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        outputs = critic.score(regions, text); handle.remove()
        self.assertEqual(outputs['dino'].shape, (2, 64, 1))
        self.assertEqual(outputs['qwen'].shape, (2, 64, 1))
        self.assertEqual(captured[0].shape, (2, 64, 25))
        torch.testing.assert_close(captured[0][..., :16], regions['dino'])
        torch.testing.assert_close(captured[0][..., 16:], text[:, None].expand(-1, 64, -1))

    def test_hinge_then_mean_and_old_logit_repeat_invariance(self):
        real = torch.tensor([[-2.0], [3.0]])
        fake = torch.tensor([[2.0], [-3.0]])
        rp = real[:, None].expand(-1, 64, -1); fp = fake[:, None].expand(-1, 64, -1)
        expected = torch.relu(1-real).mean() + torch.relu(1+fake).mean()
        torch.testing.assert_close(patch_hinge_loss(rp, fp), expected)
        torch.testing.assert_close(patch_generator_loss(fp), -fake.mean())
        mixed = torch.tensor([-3.0, 3.0]).repeat(32).reshape(1, 64, 1)
        loss = patch_hinge_loss(mixed, mixed)
        wrong = torch.relu(1-mixed.mean()) + torch.relu(1+mixed.mean())
        self.assertNotEqual(loss.item(), wrong.item())

    def test_ns_softplus_matches_scalar_loss_and_gradient_after64_region_repeat(self):
        real = torch.tensor([[-30.0], [0.2], [30.0]], dtype=torch.float64, requires_grad=True)
        fake = torch.tensor([[30.0], [-0.4], [-30.0]], dtype=torch.float64, requires_grad=True)
        rp = real[:, None].expand(-1, 64, -1)
        fp = fake[:, None].expand(-1, 64, -1)
        d = patch_discriminator_loss(rp, fp, kind='ns')
        g = patch_generator_loss(fp, kind='ns')
        expected_d = torch.nn.functional.softplus(-real).mean() + torch.nn.functional.softplus(fake).mean()
        expected_g = torch.nn.functional.softplus(-fake).mean()
        torch.testing.assert_close(d, expected_d)
        torch.testing.assert_close(g, expected_g)
        actual_grad = torch.autograd.grad(d + 1.7 * g, (real, fake), retain_graph=True)
        expected_grad = torch.autograd.grad(expected_d + 1.7 * expected_g, (real, fake))
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.isfinite(actual).all())
        mixed = torch.tensor([-3., 3.], dtype=torch.float64).repeat(32).reshape(1, 64, 1)
        correct = patch_generator_loss(mixed, kind='ns')
        self.assertGreater(correct.item(), torch.nn.functional.softplus(-mixed.mean()).item())

    def test_unknown_patch_loss_is_rejected(self):
        logits = torch.zeros(1, 64, 1)
        with self.assertRaisesRegex(ValueError, 'unknown adversarial loss'):
            patch_discriminator_loss(logits, logits, kind='other')
        with self.assertRaisesRegex(ValueError, 'unknown adversarial loss'):
            patch_generator_loss(logits, kind='other')

    def test_no_cross_image_attention(self):
        critic = self.critic().eval(); phi = self.inputs(); text = torch.randn(2, 9)
        together = critic(phi, text)
        for row in range(2):
            alone = critic({k: v[row:row+1] for k, v in phi.items()}, text[row:row+1])
            for name in together:
                torch.testing.assert_close(together[name][row:row+1], alone[name], atol=5e-7, rtol=1e-5)

    def test_coordinates_and_tokens_permute_together(self):
        head = SpatialPatchHead(SpatialBranchSpec(4, 16, 2), hidden=16, depth=2,
                                heads=2, text_dim=9, spectral_norm=False).eval()
        tokens, text = torch.randn(2, 64, 16), torch.randn(2, 9)
        coordinates = region_coordinates(); order = torch.randperm(64)
        output = head.f(tokens, text, coordinates=coordinates)
        reordered = head.f(tokens[:, order], text, coordinates=coordinates[order])
        torch.testing.assert_close(reordered, output[:, order], atol=5e-7, rtol=1e-5)

    def test_rope_has_separate_row_and_column_channel_halves(self):
        rope = RotaryPosition2D(8); value = torch.ones(1, 1, 2, 8)
        base, _ = rope(value, value, torch.zeros(2, 2))
        row, _ = rope(value, value, torch.tensor([[1., 0.], [1., 0.]]))
        col, _ = rope(value, value, torch.tensor([[0., 1.], [0., 1.]]))
        torch.testing.assert_close(row[..., 4:], base[..., 4:])
        torch.testing.assert_close(col[..., :4], base[..., :4])
        self.assertFalse(torch.equal(row[..., :4], base[..., :4]))
        self.assertFalse(torch.equal(col[..., 4:], base[..., 4:]))

    def test_discriminator_detaches_encoder_and_updates_aggregator_and_f(self):
        critic = self.critic(); real, fake = self.inputs(requires_grad=True), self.inputs(requires_grad=True)
        text = torch.randn(2, 9, requires_grad=True)
        losses = spatial_discriminator_loss(critic, real, fake, text, kind='ns')
        sum(losses.values()).backward()
        for tensor in [*real.values(), *fake.values(), text]:
            self.assertIsNone(tensor.grad)
        for head in critic.heads.values():
            for weight in (head.aggregator.weight, head.fusion.weight, head.blocks[0].qkv.weight, head.readout.weight):
                self.assertIsNotNone(weight.grad); self.assertGreater(weight.grad.abs().sum().item(), 0)

    def test_generator_preserves_fake_gradient_without_critic_grads_or_sn_mutation(self):
        critic = self.critic(spectral_norm=True).train(); fake = self.inputs(requires_grad=True)
        before = {n: x.detach().clone() for n, x in critic.named_buffers()}
        losses = spatial_generator_loss(critic, fake, torch.randn(2, 9), kind='ns')
        sum(losses.values()).backward()
        self.assertTrue(critic.training)
        self.assertTrue(all(p.requires_grad for p in critic.parameters()))
        self.assertTrue(all(p.grad is None for p in critic.parameters()))
        for value in fake.values():
            self.assertTrue(torch.isfinite(value.grad).all()); self.assertGreater(value.grad.abs().sum().item(), 0)
        for name, buffer in critic.named_buffers():
            torch.testing.assert_close(buffer, before[name], atol=0, rtol=0)

    def test_freezing_restores_mixed_modes_and_flags_even_on_exception(self):
        critic = self.critic().train(); critic.heads['dino'].norm.eval()
        first = next(critic.parameters()); first.requires_grad_(False)
        flags = [p.requires_grad for p in critic.parameters()]
        modes = [m.training for m in critic.modules()]
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with frozen_spatial_critic(critic):
                self.assertTrue(all(not p.requires_grad for p in critic.parameters()))
                raise RuntimeError('fixture')
        self.assertEqual(flags, [p.requires_grad for p in critic.parameters()])
        self.assertEqual(modes, [m.training for m in critic.modules()])

    def test_input_gradgrad_supported_by_math_attention(self):
        critic = self.critic().double().eval()
        fake = {k: v.double().requires_grad_() for k, v in self.inputs().items()}
        with frozen_spatial_critic(critic):
            logits = critic(fake, torch.randn(2, 9, dtype=torch.float64))
        first = torch.autograd.grad(sum(v.square().mean() for v in logits.values()),
                                    tuple(fake.values()), create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in first), tuple(fake.values()))
        self.assertTrue(all(torch.isfinite(g).all() for g in second))

    def test_invalid_grid_or_head_dimensions_fail(self):
        with self.assertRaises(ValueError): SpatialBranchSpec(4, 64, 4)
        with self.assertRaises(ValueError): RotaryPosition2D(6)
        with self.assertRaises(ValueError):
            SpatialPatchHead(SpatialBranchSpec(4, 16, 2), hidden=18, heads=3)
        with self.assertRaises(ValueError):
            patch_generator_loss(torch.ones(2, 1))


if __name__ == '__main__':
    unittest.main()
