"""Small exact forward/backward checks for the large-element nearest2D guard."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from zimage_distill.vae_upsample import (
    INT_MAX, install_safe_vae_upsampling, needs_safe_nearest2d, safe_nearest_upsample_2d,
)


def test_actual_large_batch_boundary_without_allocating_tensors():
    assert not needs_safe_nearest2d((4, 256, 512, 512))
    assert needs_safe_nearest2d((8, 256, 512, 512))
    assert needs_safe_nearest2d((16, 256, 512, 512))
    # The real limit is strict (< INT_MAX), not <= and not input_elements*scale.
    assert needs_safe_nearest2d((1, 1, 1, 1), (1, INT_MAX))
    assert not needs_safe_nearest2d((1, 1, 1, 1), (1, INT_MAX - 1))


@pytest.mark.parametrize('output_size', [None, (9, 11), (3, 4)])
@pytest.mark.parametrize('channels_last', [False, True])
def test_nearest_forward_backward_match_for_arbitrary_gradient_layout(output_size, channels_last):
    torch.manual_seed(11)
    x = torch.randn(2, 4, 5, 7, dtype=torch.float64)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    actual_input = x.detach().clone().requires_grad_()
    reference_input = x.detach().clone().requires_grad_()
    actual = safe_nearest_upsample_2d(actual_input, output_size)
    reference = F.interpolate(reference_input, scale_factor=2.0, mode='nearest') if output_size is None \
        else F.interpolate(reference_input, size=output_size, mode='nearest')
    probe = torch.randn_like(reference).contiguous(memory_format=torch.channels_last)
    actual.backward(probe)
    reference.backward(probe)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=0, atol=0)
    assert actual.is_contiguous()


def test_custom_boundary_supports_gradcheck_and_gradgradcheck():
    x = torch.randn(1, 4, 2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(safe_nearest_upsample_2d, (x,))
    assert torch.autograd.gradgradcheck(safe_nearest_upsample_2d, (x,))


@pytest.mark.parametrize('name,norm', [('conv', None), ('Conv2d_0', 'ln_norm')])
def test_installed_adapter_preserves_module_parameters_keys_outputs_and_gradients(name, norm):
    upsampling = pytest.importorskip('diffusers.models.upsampling')
    torch.manual_seed(31)
    original = nn.Sequential(upsampling.Upsample2D(4, use_conv=True, name=name, norm_type=norm,
                                                   elementwise_affine=True, eps=1e-5)).double()
    guarded = copy.deepcopy(original)
    identities = {k: id(v) for k, v in guarded.named_parameters()}
    keys = tuple(guarded.state_dict())
    report = install_safe_vae_upsampling(guarded, element_limit=1)
    assert report['logical_batch_unchanged'] and report['patched_modules'] == ['0']
    assert tuple(guarded.state_dict()) == keys
    assert {k: id(v) for k, v in guarded.named_parameters()} == identities
    data = torch.randn(2, 4, 5, 7, dtype=torch.float64).contiguous(memory_format=torch.channels_last)

    def run(model):
        x = data.detach().clone().requires_grad_()
        output = model(x)
        output.square().mean().backward()
        return output.detach(), x.grad, dict(model.named_parameters())

    expected, expected_grad, expected_parameters = run(original)
    actual, actual_grad, actual_parameters = run(guarded)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)
    for key in expected_parameters:
        torch.testing.assert_close(actual_parameters[key].grad, expected_parameters[key].grad,
                                   rtol=1e-12, atol=1e-12)


def test_small_module_path_is_left_unpatched_numerically():
    upsampling = pytest.importorskip('diffusers.models.upsampling')
    model = upsampling.Upsample2D(4, use_conv=True).double()
    data = torch.randn(2, 4, 3, 5, dtype=torch.float64)
    expected = model(data).detach()
    install_safe_vae_upsampling(model)
    torch.testing.assert_close(model(data), expected, rtol=0, atol=0)
