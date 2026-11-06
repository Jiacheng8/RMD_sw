"""CPU row/precision/bounded-cache and refresh-contract checks; no real weights."""
import copy
import importlib
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for name, directory in [('cache_fixture', ROOT / 'rdm'),
                        ('cache_fixture.sw_lmmd', ROOT / 'rdm/sw_lmmd'),
                        ('cache_fixture.representation', ROOT / 'rdm/representation'),
                        ('cache_fixture.utils', ROOT / 'rdm/utils')]:
    package = types.ModuleType(name); package.__path__ = [str(directory)]
    sys.modules[name] = package
SpatialBattery = importlib.import_module('cache_fixture.representation.spatial_battery').SpatialBattery
SpatialWindowCache = importlib.import_module('cache_fixture.sw_lmmd.spatial_cache').SpatialWindowCache
GeneratedWindowCache = importlib.import_module('cache_fixture.sw_lmmd.cache').GeneratedWindowCache
refresh = importlib.import_module('cache_fixture.sw_lmmd.refresh')


class TinyEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.scale = nn.Parameter(torch.tensor(1.0012345)); self.channels = channels

    def forward(self, images):
        return (images[:, :1, :2, :2] * self.scale).expand(-1, self.channels, -1, -1)

    def feature_contract(self): return {'fixture': True, 'channels': self.channels}


class PixelReader:
    def __init__(self): self.calls = []

    def tensor(self, row_ids, *, device, dtype):
        self.calls.append((list(row_ids), torch.device(device), dtype))
        raw = torch.tensor(row_ids, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 1024, 1024)
        return raw.to(device).to(torch.float32).div(255)


class SpatialCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def battery(self, capacity=3):
        return SpatialBattery(device='cpu', encoders={'dinov3_l': TinyEncoder(2),
                              'qwen3vl_4b': TinyEncoder(3)}, max_real_rows=capacity, micro_batch=1)

    def maps(self, rows, *, shift=0):
        return {'dino': (torch.arange(rows * 8).reshape(rows, 2, 2, 2).float() + shift) / 7,
                'qwen': (torch.arange(rows * 6).reshape(rows, 3, 1, 2).float() + shift) / 7}

    def test_real_lru_protects_all_current_requested_rows(self):
        battery, reader = self.battery(), PixelReader()
        battery.reference_features([0, 1, 2], reader)
        result = battery.reference_features([0, 3, 4], reader)
        self.assertEqual(set(battery._real), {0, 3, 4})
        self.assertEqual(battery.real_rows_encoded, 5)
        self.assertEqual(result['dinov3_l'].shape, (3, 2, 2, 2))
        count = len(reader.calls)
        repeated = battery.reference_features([4, 0, 4], reader)
        self.assertEqual(len(reader.calls), count)
        torch.testing.assert_close(repeated['dinov3_l'][0], repeated['dinov3_l'][2])
        battery.reference_features([5], reader)
        self.assertEqual(set(battery._real), {0, 4, 5})
        self.assertLessEqual(len(battery._real), 3)
        for row in battery._real.values():
            for value in row.values():
                self.assertEqual(value.device.type, 'cpu'); self.assertEqual(value.dtype, torch.bfloat16)
                self.assertFalse(value.requires_grad)

    def test_real_request_order_duplicates_and_capacity_error(self):
        battery, reader = self.battery(), PixelReader()
        ids = [8, 2, 8, 5]
        result = battery.reference_features(ids, reader)
        expected = ((torch.tensor(ids).float() / 255) * 1.0012345).bfloat16()
        torch.testing.assert_close(result['dinov3_l'][:, 0, 0, 0], expected)
        self.assertEqual(battery.real_rows_encoded, 3)
        before = (set(battery._real), len(reader.calls))
        with self.assertRaises(ValueError): battery.reference_features([0, 1, 2, 3], reader)
        self.assertEqual(before, (set(battery._real), len(reader.calls)))

    def test_real_fake_share_bf16_values_and_fake_pixel_gradient(self):
        battery, reader = self.battery(), PixelReader()
        images = reader.tensor([7], device='cpu', dtype=torch.float32).requires_grad_()
        fake = battery(images)
        real = battery.reference_features([7], reader)
        for name in battery.names:
            self.assertEqual(fake[name].dtype, torch.float32)
            torch.testing.assert_close(fake[name].detach(), real[name].float(), atol=0, rtol=0)
            torch.testing.assert_close(fake[name], fake[name].bfloat16().float(), atol=0, rtol=0)
        sum(value.sum() for value in fake.values()).backward()
        self.assertTrue(torch.isfinite(images.grad).all()); self.assertGreater(images.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in battery.parameters()))
        self.assertTrue(all(device.type == 'cpu' and dtype == torch.float32 for _, device, dtype in reader.calls))
        battery.train(True); self.assertFalse(battery.training)

    def test_4d_cache_is_detached_cpu_bf16_and_owns_input_bytes(self):
        cache = SpatialWindowCache(['dino', 'qwen']); values = self.maps(4)
        for value in values.values(): value.requires_grad_()
        cache.initialize([9, 2, 7, 4], values, step=-1)
        stored = {name: entry.features.clone() for name, entry in cache.entries.items()}
        with torch.no_grad():
            for value in values.values(): value.fill_(999)
        for name, entry in cache.entries.items():
            self.assertEqual(entry.features.dtype, torch.bfloat16)
            self.assertEqual(entry.features.device.type, 'cpu'); self.assertFalse(entry.features.requires_grad)
            torch.testing.assert_close(entry.features, stored[name], atol=0, rtol=0)

    def test_retention_advance_age_and_row_identity(self):
        cache = SpatialWindowCache(['dino', 'qwen']); initial = self.maps(4)
        cache.initialize([9, 2, 7, 4], initial, step=-1)
        torch.testing.assert_close(cache.retained('dino', [7, 4]), initial['dino'][-2:].bfloat16())
        with self.assertRaises(RuntimeError): cache.retained('dino', [4, 7])
        with self.assertRaises(RuntimeError): cache.advance([4, 7, 6], 2, self.maps(1), step=3)
        cache.advance([7, 4, 6], 2, self.maps(1, shift=100), step=3)
        for name, entry in cache.entries.items():
            np.testing.assert_array_equal(entry.row_ids, [7, 4, 6])
            self.assertEqual(entry.features.shape[0], 3)
            self.assertEqual(entry.generated_at_steps.tolist(), [-1, -1, 3])
            torch.testing.assert_close(entry.features[:2], initial[name][-2:].bfloat16())
        self.assertEqual(cache.age_buckets(4), {5: 2, 1: 1})

    def test_refresh_flattens_all_spatial_axes_and_preserves_requested_order(self):
        cache = SpatialWindowCache(['dino', 'qwen'])
        cache.initialize([3, 1], {'dino': torch.ones(2, 2, 2, 2), 'qwen': torch.ones(2, 3, 1, 2)})
        fresh = torch.stack([torch.ones(2, 2, 2) * 2, torch.ones(2, 2, 2) * 4])
        drift = cache.refresh_retained('dino', [1, 3], fresh, step=9)
        self.assertEqual(drift.shape, (2,))
        torch.testing.assert_close(drift, torch.tensor([0.5, 0.75]))
        torch.testing.assert_close(cache.entries['dino'].features[:, 0, 0, 0], torch.tensor([4., 2.]).bfloat16())
        self.assertEqual(cache.entries['dino'].generated_at_steps.tolist(), [9, 9])
        with self.assertRaises(ValueError): cache.refresh_retained('dino', [1], torch.ones(1, 8), step=9)

    def test_checkpoint_shape_row_alignment_and_cpu_contract(self):
        cache = SpatialWindowCache(['dino', 'qwen']); cache.initialize([2, 3], self.maps(2), step=4)
        state = copy.deepcopy(cache.state_dict())
        restored = SpatialWindowCache(['dino', 'qwen'])
        restored.load_state_dict(state, device='cuda')  # It must remain CPU without touching CUDA.
        for name, entry in restored.entries.items():
            self.assertEqual(entry.features.device.type, 'cpu'); self.assertEqual(entry.features.dtype, torch.bfloat16)
            torch.testing.assert_close(entry.features, cache.entries[name].features, atol=0, rtol=0)
        for mutation in ('flatten', 'row_order', 'ages', 'missing_stream'):
            bad = copy.deepcopy(state)
            if mutation == 'flatten': bad['dino']['features'] = bad['dino']['features'].flatten(1)
            if mutation == 'row_order': bad['qwen']['row_ids'] = np.array([3, 2])
            if mutation == 'ages': bad['qwen']['generated_at_steps'] = torch.zeros(1, dtype=torch.int64)
            if mutation == 'missing_stream': del bad['qwen']
            with self.assertRaises(ValueError): restored.load_state_dict(bad)
            np.testing.assert_array_equal(restored.entries['dino'].row_ids, [2, 3])
        restored.load_state_dict({}); self.assertFalse(restored.initialized)

    def test_refresh_hooks_keep_mmd_drift_trigger_and_synchronize_spatial_cache(self):
        ids = np.array([2, 5, 9, 7])
        global_cache = GeneratedWindowCache(['global'])
        global_values = torch.tensor([[1., 0.], [2., 0.], [3., 0.], [4., 0.]])
        global_cache.initialize(ids, {'global': global_values.clone()}, step=2)
        spatial_cache = SpatialWindowCache(['patch'])
        spatial_cache.initialize(ids, {'patch': torch.ones(4, 2, 2, 2)}, step=2)
        calls = []
        def rollout(rows):
            calls.append(rows.copy())
            positions = GeneratedWindowCache._positions_of(ids, rows)
            return {'global': global_values[positions] * 2, 'patch': torch.ones(len(rows), 2, 2, 2) * 100}
        def probe(rows, fresh):
            pos = spatial_cache._positions_of(ids, rows)
            d = (spatial_cache.entries['patch'].features[pos].float() - fresh['patch']).flatten(1).norm(dim=1)
            d /= fresh['patch'].flatten(1).norm(dim=1)
            return {'patch/drift_mean': float(d.mean())}
        def sync(rows, fresh, step):
            spatial_cache.refresh_retained('patch', rows, fresh['patch'], step)
            return {'spatial_refreshed_rows': len(rows)}
        trainer = types.SimpleNamespace(cache=global_cache, encoder_names=['global'], micro_batch=1,
            seed=0, rollout_refresh=rollout, probe_spatial_cache=probe, refresh_spatial_cache=sync)
        window = types.SimpleNamespace(retained_ids=ids, step=10)
        before_global = global_cache.entries['global'].features.clone()
        before_spatial = spatial_cache.entries['patch'].features.clone()
        logs = refresh.probe_drift(trainer, window, n_rows=4)
        self.assertEqual(len(calls), 1); self.assertAlmostEqual(logs['drift_mean'], 0.5)
        self.assertGreater(logs['patch/drift_mean'], 0.9)
        cfg = types.SimpleNamespace(drift_threshold=0.6, refresh_every_windows=0)
        self.assertFalse(refresh.should_refresh(trainer, window, logs['drift_mean'], cfg))
        torch.testing.assert_close(global_cache.entries['global'].features, before_global)
        torch.testing.assert_close(spatial_cache.entries['patch'].features, before_spatial)
        logs = refresh.refresh_retained(trainer, window)
        self.assertEqual(len(calls), 2); self.assertEqual(logs['spatial_refreshed_rows'], 4)
        self.assertAlmostEqual(logs['refresh_drift_mean'], 0.5)
        torch.testing.assert_close(global_cache.entries['global'].features, global_values * 2)
        torch.testing.assert_close(spatial_cache.entries['patch'].features, torch.full((4, 2, 2, 2), 100.).bfloat16())
        self.assertEqual(global_cache.entries['global'].generated_at_steps.tolist(), [10] * 4)
        self.assertEqual(spatial_cache.entries['patch'].generated_at_steps.tolist(), [10] * 4)
        np.testing.assert_array_equal(global_cache.entries['global'].row_ids, ids)
        np.testing.assert_array_equal(spatial_cache.entries['patch'].row_ids, ids)


if __name__ == '__main__':
    unittest.main()
