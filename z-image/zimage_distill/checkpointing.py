"""Checkpoint consecutive DiT blocks without changing their registration or FSDP units.

The Diffusers0.40 forward already checkpoints each item yielded by its block lists.
This eager-only iteration view yields a callable group in that path while retaining
the original numbered children for state_dict, parameter discovery and FSDP wrapping.
"""
from __future__ import annotations

import inspect
import types
import weakref

import torch
from torch import nn


class _BlockGroup:
    def __init__(self, blocks):
        self.blocks = tuple(blocks)

    def __call__(self, hidden, *args, **kwargs):
        for block in self.blocks:
            hidden = block(hidden, *args, **kwargs)
        return hidden


class CheckpointBlockList(nn.ModuleList):
    def __init__(self, modules, *, owner, group_size):
        super().__init__(modules)
        self._owner_ref = weakref.ref(owner)
        self._group_size = int(group_size)

    def __iter__(self):
        # Resolve after FSDP has replaced children; never cache pre-wrap blocks.
        children = tuple(self._modules.values())
        owner = self._owner_ref()
        active = owner is not None and bool(owner.gradient_checkpointing) and torch.is_grad_enabled()
        if not active:
            yield from children
        else:
            for start in range(0, len(children), self._group_size):
                yield _BlockGroup(children[start:start + self._group_size])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return nn.ModuleList(tuple(self._modules.values())[index])
        return super().__getitem__(index)


def configure_checkpoint_groups(transformer: nn.Module, group_size: int = 1) -> dict:
    """Keep original checkpointing for1; group larger regions for eager SW-LMMD.

    ControlNet residual indices refer to individual blocks and are deliberately
    rejected for grouped operation. No torch.compile compatibility is claimed.
    VAE checkpointing and the FSDP block wrapping policy are unchanged.
    """
    group_size = int(group_size)
    if group_size < 1:
        raise ValueError('checkpoint group_size must be >=1; disable checkpointing separately')
    if getattr(transformer, '_rdm_checkpoint_group_size', None) is not None:
        raise ValueError('checkpoint groups have already been configured')
    keys = tuple(transformer.state_dict())
    counts = {}
    if group_size > 1:
        for name in ('noise_refiner', 'context_refiner', 'siglip_refiner', 'layers'):
            modules = getattr(transformer, name, None)
            if isinstance(modules, nn.ModuleList):
                counts[name] = len(modules)
                setattr(transformer, name, CheckpointBlockList(
                    tuple(modules._modules.values()), owner=transformer, group_size=group_size))
        original_forward = transformer.forward
        signature = inspect.signature(original_forward)

        def guarded_forward(this, *args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            if bound.arguments.get('controlnet_block_samples') is not None:
                raise ValueError('grouped checkpointing does not support ControlNet residual indices')
            return original_forward(*args, **kwargs)

        transformer.forward = types.MethodType(guarded_forward, transformer)
    transformer._rdm_checkpoint_group_size = group_size
    if tuple(transformer.state_dict()) != keys:
        raise RuntimeError('checkpoint grouping changed state_dict keys')
    return {'group_size': group_size, 'blocks': counts, 'state_dict_keys_preserved': True}
