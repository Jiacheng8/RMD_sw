"""Spatial Qwen contracts and optional native tiny-model oracle; no downloads or weights."""
import copy
import hashlib
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rdm.representation.backbones import qwen3_vl_spatial as qs
from rdm.representation.backbones.qwen3_vl_backbone import patchify_images


def test_fixed_instruction_matches_page_hash_and_keeps_linebreaks():
    assert hashlib.sha256(qs.FIXED_INSTRUCTION.encode()).hexdigest() == qs.INSTRUCTION_SHA256
    assert qs.INSTRUCTION_SHA256 == "31882d15af308a6c920cf57fc1f4c7ab4e1e591783bbe9a89797c9275998f839"
    assert qs.FIXED_INSTRUCTION.endswith("Base the analysis entirely on the image.\n")


def test_full_1024_pixels_are_not_resized_or_pooled_and_have_gradient():
    image = torch.full((1, 3, 1024, 1024), 0.75, requires_grad=True)
    pixels, grid = qs.prepare_spatial_pixels(image)
    assert pixels.shape == (4096, 1536)
    assert torch.equal(grid, torch.tensor([[1, 64, 64]]))
    assert torch.equal(pixels, torch.full_like(pixels, 0.5))
    pixels.mean().backward()
    assert torch.isfinite(image.grad).all() and torch.all(image.grad > 0)


@pytest.mark.parametrize("size", [224, 256, 512])
def test_old_pooled_input_sizes_are_rejected(size):
    with pytest.raises(ValueError, match="1024"):
        qs.prepare_spatial_pixels(torch.zeros(1, 3, size, size))


def test_gather_uses_image_positions_not_sequence_tail_and_restores_batch_coordinates():
    # Image tokens are interleaved with text/padding to catch a tail-slice implementation.
    ids = torch.tensor([[9, 7, 9, 3, 9, 9, 0], [9, 9, 4, 9, 5, 9, 0]])
    hidden = torch.arange(2 * 7 * 3).reshape(2, 7, 3).float().requires_grad_()
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    actual = qs.image_positions_to_spatial(hidden, ids, 9, grid, expected_grid=(2, 2))
    assert actual.shape == (2, 3, 2, 2)
    for batch in range(2):
        expected = hidden[batch][ids[batch] == 9].reshape(2, 2, 3).permute(2, 0, 1)
        assert torch.equal(actual[batch], expected)
    actual.sum().backward()
    assert torch.equal(hidden.grad.sum(-1) != 0, ids == 9)


def test_bad_grid_or_missing_image_positions_fails():
    hidden = torch.zeros(1, 5, 3)
    with pytest.raises(ValueError, match="image positions"):
        qs.image_positions_to_spatial(hidden, torch.zeros(1, 5, dtype=torch.long), 9,
                                     torch.tensor([[1, 4, 4]]), expected_grid=(2, 2))
    with pytest.raises(ValueError, match="grid"):
        qs.image_positions_to_spatial(hidden, torch.full((1, 5), 9), 9,
                                     torch.tensor([[1, 2, 2]]), expected_grid=(2, 2))


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        vocabulary = {"<image>": 9, "<start>": 8, "<end>": 7}
        parts = re.split(r"(<image>|<start>|<end>)", text)
        ids = [vocabulary[part] for part in parts if part in vocabulary]
        # Represent all instruction characters as text before the image region.
        ids = [3] * len(qs.FIXED_INSTRUCTION) + ids
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}


class _FakeProcessor:
    image_token, image_token_id = "<image>", 9
    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["content"] == [{"type": "text", "text": qs.FIXED_INSTRUCTION}, {"type": "image"}]
        assert kwargs == {"tokenize": False, "add_generation_prompt": False}
        return qs.FIXED_INSTRUCTION + "<start><image><end>"

    def replace_image_token(self, image_inputs, image_idx):
        return self.image_token * int(image_inputs["image_grid_thw"][image_idx].prod() // 4)


def test_fixed_sequence_uses_official_template_expansion_and_image_types():
    sequence, contract = qs.fixed_sequence_inputs(_FakeProcessor())
    assert int((sequence["input_ids"] == 9).sum()) == 1024
    assert torch.equal(sequence["mm_token_type_ids"].bool(), sequence["input_ids"] == 9)
    assert contract["instruction_before_image"] and contract["fixed_instruction"] == qs.FIXED_INSTRUCTION
    assert sequence["input_ids"][0, -1] == 7  # Vision end is retained, never mistaken for an image.


def test_prefix_loader_selects_all_visual_deepstack_but_no_discarded_blocks_or_lm_head():
    weight_map = {f"model.visual.weight{i}": "model-00002-of-00002.safetensors" for i in range(315)}
    weight_map.update({f"model.language_model.layers.{i}.weight": "model-00001-of-00002.safetensors"
                       for i in range(36)})
    weight_map.update({name: "model-00001-of-00002.safetensors" for name in
                       ("model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight")})
    selected = qs._prefix_weight_map({"weight_map": weight_map})
    assert len(selected) == 315 + 24 + 2
    assert "model.language_model.layers.23.weight" in selected
    assert "model.language_model.layers.24.weight" not in selected
    assert "lm_head.weight" not in selected
    del weight_map["model.language_model.layers.35.weight"]
    with pytest.raises(ValueError, match="original 36"):
        qs._prefix_weight_map({"weight_map": weight_map})


def test_truncation_drops_final_norm_and_unused_blocks():
    language = nn.Module()
    language.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(36)])
    language.config = SimpleNamespace(num_hidden_layers=36)
    language.norm = nn.LayerNorm(2)
    model = SimpleNamespace(language_model=language)
    qs.truncate_language_prefix(model, 24)
    assert len(language.layers) == 24 and language.config.num_hidden_layers == 24
    assert isinstance(language.norm, nn.Identity)


def test_missing_snapshot_fails_without_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing cached"):
        qs._snapshot_directory(tmp_path)


def test_cached_official_processor_matches_sequence_expansion_without_weights():
    """Independent official multimodal __call__ oracle; strictly cached tokenizer only."""
    pytest.importorskip("transformers", minversion="5.17.0")
    location = os.environ.get("RDM_QWEN3VL_SPATIAL_SNAPSHOT") or os.environ.get("RDM_QWEN3VL_SNAPSHOT")
    if not location:
        pytest.skip("Set the existing cached snapshot path to verify the official tokenizer/template")
    processor = qs._load_processor(Path(location))
    sequence, _ = qs.fixed_sequence_inputs(processor)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": qs.FIXED_INSTRUCTION}, {"type": "image"}]}]
    template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    official = processor(text=[template], images=[torch.zeros(3, 1024, 1024, dtype=torch.uint8)],
                         return_tensors="pt", add_special_tokens=False)
    for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
        assert torch.equal(sequence[name], official[name])
    assert torch.equal(official["image_grid_thw"], torch.tensor([[1, 64, 64]]))


def _tiny_native_model():
    pytest.importorskip("transformers", minversion="5.17.0")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    config = Qwen3VLConfig.from_dict({
        "vision_config": {"depth": 3, "hidden_size": 16, "intermediate_size": 32,
                          "num_heads": 4, "out_hidden_size": 32, "patch_size": 2,
                          "temporal_patch_size": 2, "spatial_merge_size": 2,
                          "num_position_embeddings": 16, "deepstack_visual_indexes": [0, 1, 2]},
        "text_config": {"vocab_size": 100, "hidden_size": 32, "intermediate_size": 64,
                        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                        "num_hidden_layers": 4, "rope_theta": 10000,
                        "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2],
                                         "mrope_interleaved": True}, "use_cache": False},
        "vision_start_token_id": 90, "vision_end_token_id": 91,
        "image_token_id": 92, "video_token_id": 93,
    })
    config._attn_implementation = "sdpa"
    model = Qwen3VLModel(config).eval().requires_grad_(False)
    ids = torch.tensor([[1, 2, 90, 92, 92, 92, 92, 91, 3]])
    image = torch.randn(1, 3, 8, 8, requires_grad=True)
    pixels, grid = patchify_images(image, patch_size=2, merge_size=2)
    args = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": ids.eq(92).long(), "pixel_values": pixels,
            "image_grid_thw": grid, "use_cache": False, "return_dict": True}
    return model, image, args


def test_native_tiny_earlystop_matches_intermediate_state_with_deepstack_and_input_gradient():
    full, image, args = _tiny_native_model()
    # Exercise the production loader's from_pretrained(None, config, state_dict) API
    # without checkpoints, constructing only 3 of this tiny model's 4 blocks.
    config = copy.deepcopy(full.config)
    config.text_config.num_hidden_layers = 3
    state = {name: value for name, value in full.state_dict().items()
             if not name.startswith("language_model.layers.3.")}
    expected_keys = set(state)
    prefix, loading = type(full).from_pretrained(
        None, config=config, state_dict=state, dtype=torch.float32,
        attn_implementation="sdpa", output_loading_info=True, local_files_only=True,
    )
    assert all(not loading.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    assert set(prefix.state_dict()) == expected_keys
    qs.truncate_language_prefix(prefix, 3).eval().requires_grad_(False)
    captured = []
    hook = full.language_model.layers[3].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone()))
    with torch.no_grad():
        full(**args)
    hook.remove()
    actual = prefix(**args).last_hidden_state
    torch.testing.assert_close(actual, captured[0], atol=1e-6, rtol=1e-6)
    assert len(prefix.language_model.layers) == 3 and isinstance(prefix.language_model.norm, nn.Identity)
    assert not hasattr(prefix, "lm_head")
    spatial = qs.image_positions_to_spatial(actual, args["input_ids"], 92,
                                            args["image_grid_thw"], expected_grid=(2, 2))
    spatial.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
    assert all(p.grad is None for p in prefix.parameters())


def test_native_tiny_causal_text_prefix_affects_images_but_future_text_does_not():
    model, _, args = _tiny_native_model()
    qs.truncate_language_prefix(model, 3)
    with torch.no_grad():
        baseline = model(**args).last_hidden_state[:, 3:7]
        future_ids = args["input_ids"].clone()
        future_ids[:, -1] = 4
        future = model(**{**args, "input_ids": future_ids}).last_hidden_state[:, 3:7]
        prefix_ids = args["input_ids"].clone()
        prefix_ids[:, 0] = 5
        changed = model(**{**args, "input_ids": prefix_ids}).last_hidden_state[:, 3:7]
    torch.testing.assert_close(future, baseline, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(changed, baseline, atol=1e-6, rtol=1e-6)
