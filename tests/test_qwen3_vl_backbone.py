"""Qwen3-VL adapter contracts without network, model weights, transformers, or GPUs."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rdm.representation.backbones import build_backbone
from rdm.representation.backbones import qwen3_vl_backbone as qv
from rdm.representation.registry import all_specs, by_name, training_specs


def test_patchify_matches_explicit_spatial_channel_temporal_order():
    # Two nonsquare images and distinguishable channels catch B/C/merge-order mistakes.
    image = torch.arange(2 * 3 * 8 * 12).reshape(2, 3, 8, 12).float()
    actual, grid = qv.patchify_images(image, patch_size=2, merge_size=2)
    expected = []
    for b in range(2):
        for group_h in range(2):
            for group_w in range(3):
                for within_h in range(2):
                    for within_w in range(2):
                        top, left = group_h * 4 + within_h * 2, group_w * 4 + within_w * 2
                        tile = image[b, :, top:top+2, left:left+2]
                        expected.append(torch.stack([tile, tile], dim=1).reshape(-1))
    assert torch.equal(actual, torch.stack(expected))
    assert torch.equal(grid, torch.tensor([[1, 4, 6], [1, 4, 6]]))


def test_patchify_temporal_repeat_preserves_pixel_gradient():
    image = torch.randn(2, 3, 8, 12, requires_grad=True)
    patches, _ = qv.patchify_images(image, patch_size=2, merge_size=2)
    patches.sum().backward()
    assert torch.equal(image.grad, torch.full_like(image, 2))


@pytest.mark.parametrize("shape", [(1, 3, 7, 12), (0, 3, 8, 12)])
def test_patchify_rejects_invalid_batch_or_grid(shape):
    with pytest.raises(ValueError):
        qv.patchify_images(torch.zeros(shape), patch_size=2)


def test_preprocess_normalizes_once_and_keeps_raw_pixel_gradient():
    image = torch.full((2, 3, 20, 28), 0.75, requires_grad=True)
    pixels, grid = qv.prepare_pixel_values(image)
    assert pixels.shape == (2 * 256, 3 * 2 * 16 * 16)
    assert torch.equal(grid, torch.tensor([[1, 16, 16]] * 2))
    assert pixels.dtype == torch.float32
    assert torch.allclose(pixels, torch.full_like(pixels, 0.5), atol=2e-6)
    pixels.square().mean().backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0


@pytest.mark.parametrize("image,exception", [(torch.zeros(1, 1, 8, 8), ValueError),
                                             (torch.zeros(1, 3, 8, 8, dtype=torch.uint8), TypeError)])
def test_preprocess_rejects_wrong_input_contract(image, exception):
    with pytest.raises(exception):
        qv.prepare_pixel_values(image)


class _ToyFrozenVision(nn.Module):
    """Trainable parameter lets the test check freezing without severing pixel autograd."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, pixels, grid_thw, return_dict):
        assert return_dict and not self.training
        # Average each group of four patches, like the spatial merger's token count.
        value = pixels.mean(-1).reshape(-1, 4).mean(-1) * self.scale
        return SimpleNamespace(pooler_output=value[:, None].expand(-1, qv.FEATURE_DIM))


def test_adapter_means_each_image_without_l2_and_preserves_frozen_input_grad(monkeypatch):
    def load(*args):
        return _ToyFrozenVision().eval().requires_grad_(False)
    monkeypatch.setattr(qv, "_load_vision", load)
    model = qv.Qwen3VLBackbone(device="cpu")
    model.train(True)
    image = torch.stack([torch.full((3, 16, 24), 0.75),
                         torch.full((3, 16, 24), 0.25)]).requires_grad_()
    features = model(image)
    assert features.shape == (2, 2560)
    assert torch.allclose(features[0], torch.ones(2560), atol=1e-5)
    assert torch.allclose(features[1], -torch.ones(2560), atol=1e-5)
    assert not model.training and not model.model.training
    features.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert image.grad[0].abs().sum() > 0 and image.grad[1].abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert model.feature_contract()["l2_normalize"] is False


def test_adapter_fails_on_changed_transformers_pooler_semantics(monkeypatch):
    class BadVision(_ToyFrozenVision):
        def forward(self, *args, **kwargs):
            return SimpleNamespace(pooler_output=torch.zeros(1, 2560))
    monkeypatch.setattr(qv, "_load_vision", lambda *args: BadVision())
    with pytest.raises(RuntimeError, match="final-merger tokens"):
        qv.Qwen3VLBackbone(device="cpu")(torch.zeros(1, 3, 256, 256))


def test_cached_files_missing_fails_without_network_or_model_import(tmp_path):
    with pytest.raises(FileNotFoundError, match="never downloads"):
        qv.Qwen3VLBackbone(device="cpu", snapshot_dir=tmp_path)


def test_shard_and_configuration_must_match_pinned_architecture(tmp_path):
    shard = tmp_path / qv.WEIGHT_FILE
    shard.write_bytes(b"not the official model")
    with pytest.raises(ValueError, match="size mismatch"):
        qv._verify_weight_file(shard)
    with pytest.raises(ValueError, match="vision_config.depth"):
        qv._check_vision_config({"vision_config": {"depth": 27}})
    with pytest.raises(ValueError, match="315 visual keys"):
        qv._visual_keys({"weight_map": {"model.language_model.foo": "first.safetensors"}})
    official = {**qv._EXPECTED_VISION, "model_type": "qwen3_vl"}
    assert qv._check_vision_config({"vision_config": official})["model_type"] == "qwen3_vl_vision"
    with pytest.raises(ValueError, match="unexpected overrides"):
        qv._check_vision_config({"vision_config": {**official, "rope_parameters": {"rope_theta": 7}}})


def test_optional_encoder_dispatch_leaves_original_panel_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr(qv, "Qwen3VLBackbone", lambda **kwargs: calls.append(kwargs) or "adapter")
    spec = by_name("qwen3vl_4b")
    assert (spec.dim, spec.input_res, spec.source) == (2560, 256, "qwen3_vl")
    assert len(all_specs()) == 14 and len(training_specs()) == 10
    assert build_backbone(spec, device="cpu") == "adapter"
    assert calls == [{"device": "cpu", "model_id": qv.MODEL_ID, "target_size": 256}]


def test_patchify_matches_official_transformers_image_processor():
    """Independent oracle from the processor used by the pinned model; no weights."""
    pytest.importorskip("transformers", minversion="5.17.0")
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    processor = Qwen2VLImageProcessor(patch_size=2, merge_size=2, temporal_patch_size=2)
    image = torch.arange(2 * 3 * 8 * 12).reshape(2, 3, 8, 12).float() / 100 - 1
    official = processor(images=list(image), do_resize=False, do_rescale=False,
                         do_normalize=False, return_tensors="pt")
    pixels, grid = qv.patchify_images(image, patch_size=2, merge_size=2)
    assert torch.equal(pixels, official["pixel_values"])
    assert torch.equal(grid, official["image_grid_thw"])


def test_256px_preprocess_matches_official_uint8_rgb_processor():
    """Verify [0,1] adapter input equals official uint8 rescale+0.5 normalization."""
    pytest.importorskip("transformers", minversion="5.17.0")
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    processor = Qwen2VLImageProcessor(
        size={"shortest_edge": 65536, "longest_edge": 16777216},
        patch_size=16, merge_size=2, temporal_patch_size=2,
        image_mean=[0.5] * 3, image_std=[0.5] * 3,
    )
    # Every uint8 value appears repeatedly, so this catches double or missing rescaling.
    image = torch.arange(2 * 3 * 256 * 256).remainder(256).reshape(2, 3, 256, 256).to(torch.uint8)
    official = processor(images=list(image), return_tensors="pt")
    pixels, grid = qv.prepare_pixel_values(image.float() / 255)
    torch.testing.assert_close(pixels, official["pixel_values"], rtol=0, atol=2e-7)
    assert torch.equal(grid, official["image_grid_thw"])


def test_official_tiny_vision_api_and_input_backward_without_weights():
    """Runs in the training environment; no checkpoint or GPU is needed.

    Local environments without transformers skip this integration, while the explicit
    patch-order and adapter/autograd tests above still run with torch alone.
    """
    pytest.importorskip("transformers", minversion="5.17.0")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = Qwen3VLVisionConfig(depth=2, hidden_size=16, intermediate_size=32,
                                num_heads=4, out_hidden_size=32, patch_size=2,
                                temporal_patch_size=2, spatial_merge_size=2,
                                num_position_embeddings=16, deepstack_visual_indexes=[0])
    config._attn_implementation = "sdpa"
    model = Qwen3VLVisionModel(config).eval().requires_grad_(False)
    image = torch.randn(2, 3, 8, 8, requires_grad=True)
    pixels, grid = qv.patchify_images(image, patch_size=2, merge_size=2)
    output = model(pixels, grid_thw=grid, return_dict=True)
    assert output.last_hidden_state.shape == (2 * 16, 16)
    assert output.pooler_output.shape == (2 * 4, 32)
    assert len(output.deepstack_features) == 1
    output.pooler_output.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
