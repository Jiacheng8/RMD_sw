"""Checkpoint grouping keeps model identity and exact eager/dropout gradients on CPU.

The toy forward uses the same per-entry checkpoint callback as Diffusers. These tests
exercise the iterator adapter, not a second checkpoint implementation.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from zimage_distill.checkpointing import configure_checkpoint_groups


class TinyBlock(nn.Module):
    def __init__(self, width=8, dropout=0.0):
        super().__init__()
        self.up = nn.Linear(width, width * 2)
        self.down = nn.Linear(width * 2, width)
        self.dropout = nn.Dropout(dropout)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + self.down(self.dropout(torch.nn.functional.gelu(self.up(x)))) * 0.1


class TinyTransformer(nn.Module):
    def __init__(self, dropout=0.0, with_siglip=False):
        super().__init__()
        self.gradient_checkpointing = True
        self.noise_refiner = nn.ModuleList([TinyBlock(dropout=dropout) for _ in range(2)])
        self.context_refiner = nn.ModuleList([TinyBlock(dropout=dropout) for _ in range(3)])
        self.layers = nn.ModuleList([TinyBlock(dropout=dropout) for _ in range(5)])
        if with_siglip:
            self.siglip_refiner = nn.ModuleList([TinyBlock(dropout=dropout) for _ in range(2)])
        self.checkpoint_calls = 0

    def forward(self, x, t=None, cap_feats=None, return_dict=False,
                controlnet_block_samples=None):
        for name in ("noise_refiner", "context_refiner", "siglip_refiner", "layers"):
            if not hasattr(self, name):
                continue
            for layer in getattr(self, name):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    self.checkpoint_calls += 1
                    x = checkpoint(layer.__call__, x, use_reentrant=False,
                                   preserve_rng_state=True)
                else:
                    x = layer(x)
        return x


def _raw_blocks(model):
    return [module for module in model.modules() if isinstance(module, TinyBlock)]


@pytest.mark.parametrize("with_siglip", [False, True])
def test_grouping_preserves_parameters_keys_and_odd_tail(with_siglip):
    torch.manual_seed(31)
    model = TinyTransformer(with_siglip=with_siglip).double()
    before_state = {name: value.clone() for name, value in model.state_dict().items()}
    before_params = {name: id(value) for name, value in model.named_parameters()}
    raw_lists = {name: tuple(getattr(model, name).children())
                 for name in ("layers", "noise_refiner", "context_refiner", "siglip_refiner")
                 if hasattr(model, name)}
    report = configure_checkpoint_groups(model, 2)
    assert isinstance(report, dict)
    assert {name: id(value) for name, value in model.named_parameters()} == before_params
    assert model.state_dict().keys() == before_state.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
    for name, blocks in raw_lists.items():
        view = getattr(model, name)
        assert tuple(view.children()) == blocks
        assert len(view) == len(blocks)
        assert len(list(view)) == math.ceil(len(blocks) / 2)
    output = model(torch.randn(2, 8, dtype=torch.float64, requires_grad=True))
    assert output.shape == (2, 8)
    assert all(block.calls == 1 for block in _raw_blocks(model))
    assert model.checkpoint_calls == sum(math.ceil(len(b) / 2) for b in raw_lists.values())


def test_no_grad_and_checkpoint_disabled_iterate_original_blocks():
    model = TinyTransformer().double()
    original = tuple(model.layers.children())
    configure_checkpoint_groups(model, 2)
    with torch.no_grad():
        assert tuple(model.layers) == original
        model(torch.randn(2, 8, dtype=torch.float64))
    assert model.checkpoint_calls == 0
    model.gradient_checkpointing = False
    assert tuple(model.layers) == original
    output = model(torch.randn(2, 8, dtype=torch.float64, requires_grad=True))
    output.sum().backward()
    assert model.checkpoint_calls == 0


def test_slice_returns_plain_module_list_of_original_blocks():
    model = TinyTransformer()
    original = tuple(model.layers.children())
    configure_checkpoint_groups(model, 2)
    selected = model.layers[1:4]
    assert type(selected) is nn.ModuleList
    assert tuple(selected) == original[1:4]
    assert model.layers[-1] is original[-1]


@pytest.mark.parametrize("group_size", [1, 2, 3])
def test_float64_forward_and_gradients_match_with_dropout_recomputation(group_size):
    torch.manual_seed(194)
    plain = TinyTransformer(dropout=0.2, with_siglip=True).double()
    grouped = copy.deepcopy(plain)
    plain.gradient_checkpointing = False
    configure_checkpoint_groups(grouped, group_size)
    x = torch.randn(3, 8, dtype=torch.float64)
    probe = torch.randn(3, 8, dtype=torch.float64)

    def run(model):
        inp = x.clone().requires_grad_()
        torch.manual_seed(713)
        output = model(inp)
        loss = (output * probe).sum() + output.square().mean()
        loss.backward()
        return output.detach(), inp.grad, {n: p.grad for n, p in model.named_parameters()}

    expected, expected_input_grad, expected_grads = run(plain)
    actual, actual_input_grad, actual_grads = run(grouped)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_input_grad, expected_input_grad, rtol=1e-12, atol=1e-12)
    assert actual_grads.keys() == expected_grads.keys()
    for name in expected_grads:
        assert expected_grads[name] is not None and actual_grads[name] is not None
        torch.testing.assert_close(actual_grads[name], expected_grads[name], rtol=1e-12, atol=1e-12)
    # A backward really recomputed the checkpointed blocks, instead of just comparing
    # two ordinary forwards with the same dropout seed.
    assert all(block.calls > 1 for block in _raw_blocks(grouped))


@pytest.mark.parametrize("positional", [False, True])
def test_grouped_controlnet_is_rejected_for_keyword_and_positional_args(positional):
    model = TinyTransformer().double()
    configure_checkpoint_groups(model, 2)
    x = torch.randn(2, 8, dtype=torch.float64)
    samples = {0: torch.zeros_like(x)}
    with pytest.raises((ValueError, NotImplementedError), match="[Cc]ontrol[Nn]et|controlnet"):
        if positional:
            model(x, None, None, False, samples)
        else:
            model(x, controlnet_block_samples=samples)


@pytest.mark.parametrize("group_size", [0, -1])
def test_invalid_group_size_is_rejected(group_size):
    with pytest.raises(ValueError):
        configure_checkpoint_groups(TinyTransformer(), group_size)
