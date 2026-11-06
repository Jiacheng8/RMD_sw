"""Avoid Torch2.8 nearest-NHWC backward's INT_MAX indexing limit.

The logical VAE/DiT batch is unchanged. Only at-risk nearest2D interpolations
use the existing NCHW kernel, whose CUDA forward/backward support64-bit indices.
Both forward input and backward grad_output are made contiguous: changing only
the forward layout is insufficient because backward selects from grad_output.
"""
from __future__ import annotations

import math
import types

import torch
from torch import nn
from torch.nn import functional as F

INT_MAX = 2**31 - 1


def nearest_output_size(shape, output_size=None) -> tuple[int, int]:
    if len(shape) != 4:
        raise ValueError('nearest2D guard expects NCHW tensors')
    if output_size is None:
        return int(shape[-2]) * 2, int(shape[-1]) * 2
    if isinstance(output_size, int):
        return output_size, output_size
    if len(output_size) != 2:
        raise ValueError('nearest2D output_size must have two dimensions')
    return int(output_size[0]), int(output_size[1])


def needs_safe_nearest2d(shape, output_size=None, element_limit: int = INT_MAX) -> bool:
    if element_limit < 1:
        raise ValueError('element_limit must be positive')
    height, width = nearest_output_size(shape, output_size)
    return max(math.prod(shape), int(shape[0]) * int(shape[1]) * height * width) >= element_limit


class _ContiguousNearest2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, output_size):
        ctx.input_size = tuple(hidden_states.shape)
        ctx.output_size = tuple(output_size)
        return F.interpolate(hidden_states.contiguous(), size=ctx.output_size, mode='nearest')

    @staticmethod
    def backward(ctx, grad_output):
        # The contiguous grad_output also makes the backward meta allocation NCHW,
        # selecting the64-bit CUDA branch even if the following conv emitted NHWC.
        grad = torch.ops.aten.upsample_nearest2d_backward.default(
            grad_output.contiguous(), list(ctx.output_size), list(ctx.input_size), None, None)
        return grad, None


def safe_nearest_upsample_2d(hidden_states: torch.Tensor, output_size=None) -> torch.Tensor:
    return _ContiguousNearest2D.apply(hidden_states, nearest_output_size(hidden_states.shape, output_size))


def install_safe_vae_upsampling(vae: nn.Module, *, element_limit: int = INT_MAX) -> dict:
    """Patch only Diffusers Upsample2D instances, with no parameter/key changes.

    Default paths below the indexing limit call their original forward verbatim.
    ``element_limit`` can be lowered for small CPU/GPU parity tests. This adapter is
    paired with Diffusers0.40/Torch2.8, and does not alter convolution, normalization,
    VAE slicing, DiT micro-batch, or checkpoint granularity.
    """
    from diffusers.models.upsampling import Upsample2D

    if element_limit < 1:
        raise ValueError('element_limit must be positive')
    keys = tuple(vae.state_dict())
    patched = []
    for name, module in vae.named_modules():
        if not isinstance(module, Upsample2D) or not module.interpolate or module.use_conv_transpose:
            continue
        if getattr(module, '_rdm_safe_nearest_installed', False):
            raise ValueError(f'safe nearest2D guard is already installed: {name}')
        original = module.forward

        def guarded_forward(this, hidden_states, output_size=None, *args,
                            _original=original, **kwargs):
            if not needs_safe_nearest2d(hidden_states.shape, output_size, element_limit):
                return _original(hidden_states, output_size, *args, **kwargs)
            if args or kwargs.get('scale') is not None:
                from diffusers.utils import deprecate
                deprecate('scale', '1.0.0', 'The scale argument is deprecated and ignored by Upsample2D.')
            assert hidden_states.shape[1] == this.channels
            if this.norm is not None:
                hidden_states = this.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            hidden_states = safe_nearest_upsample_2d(hidden_states, output_size)
            if this.use_conv:
                hidden_states = this.conv(hidden_states) if this.name == 'conv' else this.Conv2d_0(hidden_states)
            return hidden_states

        module.forward = types.MethodType(guarded_forward, module)
        module._rdm_safe_nearest_installed = True
        patched.append(name)
    if tuple(vae.state_dict()) != keys:
        raise RuntimeError('safe VAE upsampling changed checkpoint keys')
    return {'element_limit': element_limit, 'patched_modules': patched,
            'state_dict_keys_preserved': True, 'logical_batch_unchanged': True}
