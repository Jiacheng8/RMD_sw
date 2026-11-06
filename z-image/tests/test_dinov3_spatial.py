"""CPU contract checks with a tiny injected model; no timm install or downloads."""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rdm.representation.backbones.dinov3_spatial import (
    DINOv3SpatialBackbone, _build_cached_model, MODEL_ID,
)


class FakeDINO(nn.Module):
    num_features = 1024
    num_prefix_tokens = 5  # CLS plus all four register tokens.
    pretrained_cfg = {'mean': (0.5, 0.5, 0.5), 'std': (0.25, 0.25, 0.25)}
    patch_embed = types.SimpleNamespace(patch_size=(16, 16))

    def __init__(self, truncated=False):
        super().__init__(); self.scale = nn.Parameter(torch.tensor(2.0)); self.truncated = truncated

    def forward_features(self, images):
        patch = F.avg_pool2d(images[:, :1], 16).flatten(2).transpose(1, 2)
        patch = (patch * self.scale).expand(-1, -1, 1024)
        prefix = patch.new_full((images.shape[0], 5, 1024), 10000.0)
        tokens = torch.cat((prefix, patch), dim=1)
        return tokens[:, :-1] if self.truncated else tokens


class DINOv3SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_full1024_grid_all_prefix_removed_and_image_gradients_preserved(self):
        model = FakeDINO(); encoder = DINOv3SpatialBackbone(device='cpu', model=model)
        image = torch.rand(1, 3, 1024, 1024, requires_grad=True)
        spatial = encoder(image)
        self.assertEqual(spatial.shape, (1, 1024, 64, 64))
        expected = F.avg_pool2d((image[:, :1] - 0.5) / 0.25, 16) * 2
        torch.testing.assert_close(spatial[:, :1], expected)
        self.assertLess(spatial.abs().max().item(), 100)
        spatial.mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all()); self.assertGreater(image.grad.abs().sum().item(), 0)
        self.assertIsNone(model.scale.grad); self.assertFalse(model.scale.requires_grad)
        encoder.train(True)
        self.assertFalse(encoder.training); self.assertFalse(encoder.model.training)
        self.assertEqual(encoder.feature_contract()['prefix_tokens_removed'], 5)

    def test_224_and_512_inputs_and_wrong_token_count_rejected(self):
        encoder = DINOv3SpatialBackbone(device='cpu', model=FakeDINO())
        for size in (224, 512):
            with self.assertRaises(ValueError): encoder(torch.rand(1, 3, size, size))
        truncated = DINOv3SpatialBackbone(device='cpu', model=FakeDINO(truncated=True))
        with self.assertRaises(RuntimeError): truncated(torch.zeros(1, 3, 1024, 1024))

    def test_local_weight_source_and_dynamic1024_model_arguments(self):
        captured = {}
        def create_model(name, **kwargs):
            captured.update(name=name, **kwargs); return FakeDINO()
        config = types.SimpleNamespace(to_dict=lambda: {'hf_hub_id': 'unused', 'mean': (0.5,) * 3})
        fake_timm = types.SimpleNamespace(create_model=create_model)
        fake_models = types.SimpleNamespace(get_pretrained_cfg=lambda _: config)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules,
                {'timm': fake_timm, 'timm.models': fake_models}):
            path = Path(tmp) / 'prepared.safetensors'; path.write_bytes(b'fixture')
            model, provenance = _build_cached_model(path)
        self.assertEqual(captured['name'], MODEL_ID)
        self.assertEqual(captured['img_size'], 1024)
        self.assertTrue(captured['dynamic_img_size']); self.assertFalse(captured['dynamic_img_pad'])
        self.assertIsNone(captured['pretrained_cfg_overlay']['hf_hub_id'])
        self.assertIsNone(captured['pretrained_cfg_overlay']['url'])
        self.assertEqual(provenance['checkpoint_bytes'], 7)


if __name__ == '__main__':
    unittest.main()
