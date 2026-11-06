"""Independent caption-conditioned spatial PatchGAN heads.

Frozen encoders supply *unaggregated* spatial maps. The learned convolution,
caption fusion, bidirectional region Transformer and raw-logit readout all belong
to the discriminator. Each branch processes B independent 8x8 region sequences;
there is no attention between images and no global-logit pooling.

This module applies no encoder/step/GAN weights or distributed/accumulation
scaling. The caller retains those existing training conventions.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


REGION_GRID = 8
REGION_COUNT = REGION_GRID ** 2


@dataclass(frozen=True)
class SpatialBranchSpec:
    channels: int
    grid_size: int
    kernel_size: int

    def __post_init__(self):
        if min(self.channels, self.grid_size, self.kernel_size) <= 0 \
                or self.grid_size != REGION_GRID * self.kernel_size:
            raise ValueError('Spatial branch must aggregate exactly to an8x8 region grid')


DEFAULT_SPATIAL_BRANCHES = {
    'dinov3_l': SpatialBranchSpec(1024, 64, 8),
    'qwen3vl_4b': SpatialBranchSpec(2560, 32, 4),
}


def region_coordinates(*, device=None, dtype=torch.float32):
    """Row-major (row,column) centers in normalized [0,1] image coordinates."""
    center = (torch.arange(REGION_GRID, device=device, dtype=dtype) + 0.5) / REGION_GRID
    row, col = torch.meshgrid(center, center, indexing='ij')
    return torch.stack((row, col), dim=-1).reshape(REGION_COUNT, 2)


class RotaryPosition2D(nn.Module):
    """Rotate disjoint Q/K channel halves by row and column, respectively.

    Every head allocates half its channels to each spatial axis and uses paired
    real rotations within that half. Coordinates may be permuted along with the
    tokens; flattened token indices are never used as spatial positions.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 4 or base <= 0:
            raise ValueError('2D RoPE requires head_dim divisible by4 and positive base')
        self.head_dim = head_dim
        axis_dim = head_dim // 2
        inv = base ** (-torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim)
        self.register_buffer('inv_frequency', inv, persistent=False)

    def _axis(self, value, position):
        angle = position[..., None] * self.inv_frequency.to(device=value.device, dtype=position.dtype)
        cosine, sine = angle.cos().to(value.dtype), angle.sin().to(value.dtype)
        paired = value.reshape(*value.shape[:-1], -1, 2)
        x, y = paired.unbind(-1)
        return torch.stack((x * cosine - y * sine, x * sine + y * cosine), -1).flatten(-2)

    def forward(self, q, k, coordinates):
        if q.ndim != 4 or q.shape != k.shape or q.shape[-1] != self.head_dim:
            raise ValueError('2D RoPE expects matching(B,heads,P,head_dim) Q/K')
        if coordinates.shape == (q.shape[-2], 2):
            coordinates = coordinates.unsqueeze(0)
        if coordinates.ndim != 3 or coordinates.shape[0] not in (1, q.shape[0]) \
                or coordinates.shape[1:] != (q.shape[-2], 2):
            raise ValueError('Coordinates must have shape(P,2) or(B,P,2)')
        precision = torch.float64 if q.dtype == torch.float64 else torch.float32
        coordinates = coordinates.to(device=q.device, dtype=precision)
        row, col = coordinates[:, None, :, 0], coordinates[:, None, :, 1]
        half = self.head_dim // 2

        def rotate(value):
            return torch.cat((self._axis(value[..., :half], row),
                              self._axis(value[..., half:], col)), dim=-1)
        return rotate(q), rotate(k)


def _linear(in_features, out_features, spectral_norm, *, bias=True):
    layer = nn.Linear(in_features, out_features, bias=bias)
    return nn.utils.parametrizations.spectral_norm(layer) if spectral_norm else layer


class SpatialTransformerBlock(nn.Module):
    def __init__(self, width, heads, *, mlp_ratio=2.0, rope_base=10000.0, spectral_norm=True):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads or (width // heads) % 4:
            raise ValueError('heads must divide width and head_dim must be divisible by4')
        if mlp_ratio <= 0:
            raise ValueError('mlp_ratio must be positive')
        self.heads, self.head_dim = heads, width // heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = _linear(width, width * 3, spectral_norm)
        self.projection = _linear(width, width, spectral_norm)
        self.rope = RotaryPosition2D(self.head_dim, rope_base)
        inner = max(1, int(width * mlp_ratio))
        self.mlp = nn.Sequential(_linear(width, inner, spectral_norm), nn.GELU(),
                                 _linear(inner, width, spectral_norm))

    def forward(self, tokens, coordinates):
        batch, positions, width = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(batch, positions, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rope(q, k, coordinates)
        scores = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        # This small64-token attention is explicitly bidirectional. Math attention
        # also supports input gradgrad for optional feature-space R1 penalties.
        softmax_dtype = torch.float32 if scores.dtype in (torch.float16, torch.bfloat16) else scores.dtype
        weights = F.softmax(scores, dim=-1, dtype=softmax_dtype).to(v.dtype)
        attended = (weights @ v).transpose(1, 2).reshape(batch, positions, width)
        tokens = tokens + self.projection(attended)
        return tokens + self.mlp(self.norm2(tokens))


class SpatialPatchHead(nn.Module):
    """One branch: spatial map + unscaled Qwen3 caption vector ->(B,64,1)."""

    def __init__(self, spec: SpatialBranchSpec, *, hidden=1024, depth=2, heads=8,
                 text_dim=2560, mlp_ratio=2.0, rope_base=10000.0, spectral_norm=True):
        super().__init__()
        if hidden <= 0 or depth <= 0 or text_dim <= 0:
            raise ValueError('hidden/depth/text_dim must be positive')
        self.spec, self.hidden, self.text_dim = spec, hidden, text_dim
        conv = nn.Conv2d(spec.channels, hidden, spec.kernel_size,
                         stride=spec.kernel_size, padding=0)
        self.aggregator = nn.utils.parametrizations.spectral_norm(conv) if spectral_norm else conv
        # Caption conditioning is an explicit broadcast-concatenate-fuse operation,
        # not a projection-logit substitute and not an addition before concatenation.
        self.fusion = _linear(hidden + text_dim, hidden, spectral_norm)
        self.blocks = nn.ModuleList([SpatialTransformerBlock(hidden, heads, mlp_ratio=mlp_ratio,
            rope_base=rope_base, spectral_norm=spectral_norm) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.readout = _linear(hidden, 1, spectral_norm)
        self.register_buffer('coordinates', region_coordinates(), persistent=True)

    def aggregate(self, spatial):
        expected = (self.spec.channels, self.spec.grid_size, self.spec.grid_size)
        if spatial.ndim != 4 or spatial.shape[0] == 0 or tuple(spatial.shape[1:]) != expected:
            raise ValueError(f'Expected nonempty(B,{expected}) frozen spatial features')
        regions = self.aggregator(spatial)
        if regions.shape[-2:] != (REGION_GRID, REGION_GRID):
            raise RuntimeError('Learned aggregator did not produce8x8 regions')
        return regions.flatten(2).transpose(1, 2)

    def f(self, regions, text, *, coordinates=None):
        if regions.ndim != 3 or regions.shape[1:] != (REGION_COUNT, self.hidden):
            raise ValueError('Region discriminator input must have shape(B,64,D)')
        if text is None or text.shape != (regions.shape[0], self.text_dim):
            raise ValueError(f'Caption condition must have shape(B,{self.text_dim})')
        repeated_text = text[:, None, :].expand(-1, REGION_COUNT, -1)
        tokens = self.fusion(torch.cat((regions, repeated_text), dim=-1))
        positions = self.coordinates if coordinates is None else coordinates
        for block in self.blocks:
            tokens = block(tokens, positions)
        return self.readout(self.norm(tokens))

    def forward(self, spatial, text, *, coordinates=None):
        return self.f(self.aggregate(spatial), text, coordinates=coordinates)


class SpatialFeatureCritic(nn.Module):
    """Separate Qwen/DINO heads with one shared-over-regions f per branch.

    Input: ``{name:(B,C,H,W)}`` and raw caption tau ``(B,2560)``.
    Output: ``{name:(B,64,1)}``. No AIM or global MMD feature path is altered.
    """

    def __init__(self, branches: Mapping[str, SpatialBranchSpec] | None = None, **head_options):
        super().__init__()
        branches = DEFAULT_SPATIAL_BRANCHES if branches is None else dict(branches)
        if not branches:
            raise ValueError('At least one spatial critic branch is required')
        self.names = list(branches)
        self.heads = nn.ModuleDict({name: SpatialPatchHead(spec, **head_options)
                                    for name, spec in branches.items()})

    def aggregate(self, features):
        return {name: self.heads[name].aggregate(features[name]) for name in self.names}

    def score(self, regions, text, *, coordinates=None):
        return {name: self.heads[name].f(regions[name], text, coordinates=coordinates)
                for name in self.names}

    def forward(self, features, text, *, coordinates=None):
        return self.score(self.aggregate(features), text, coordinates=coordinates)


def _validate_patch_logits(logits):
    if logits.ndim != 3 or logits.shape[0] == 0 or logits.shape[1:] != (REGION_COUNT, 1):
        raise ValueError('Patch logits must have shape(nonempty B,64,1)')


def patch_hinge_loss(real_logits, fake_logits):
    """Mean each hinge over batch+regions, then add real/fake; no extra scaling."""
    return patch_discriminator_loss(real_logits, fake_logits, kind='hinge')


def patch_discriminator_loss(real_logits, fake_logits, *, kind='hinge'):
    """Mean each real/fake term over batch+regions, with no extra64 scaling."""
    _validate_patch_logits(real_logits); _validate_patch_logits(fake_logits)
    if kind == 'hinge':
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    if kind == 'ns':
        return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
    raise ValueError(f'unknown adversarial loss {kind!r}')


def patch_generator_loss(fake_logits, *, kind='hinge'):
    _validate_patch_logits(fake_logits)
    if kind == 'hinge':
        return -fake_logits.mean()
    if kind == 'ns':
        return F.softplus(-fake_logits).mean()
    raise ValueError(f'unknown adversarial loss {kind!r}')


@contextmanager
def frozen_spatial_critic(critic):
    """Freeze weights/buffer updates for G while retaining the input-gradient graph."""
    parameters = [(p, p.requires_grad) for p in critic.parameters()]
    modes = [(module, module.training) for module in critic.modules()]
    try:
        critic.eval()  # Also freezes spectral-normalization power-iteration buffers.
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        yield critic
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training


def spatial_discriminator_loss(critic, real_features, fake_features, text, *, kind='hinge'):
    """Detached encoder/text inputs; returned per-branch losses train all disc layers."""
    real = critic({n: real_features[n].detach() for n in critic.names}, text.detach())
    fake = critic({n: fake_features[n].detach() for n in critic.names}, text.detach())
    return {n: patch_discriminator_loss(real[n], fake[n], kind=kind) for n in critic.names}


def spatial_generator_loss(critic, fake_features, text, *, kind='hinge'):
    """Fresh G forward through frozen disc, preserving gradients to fake feature maps."""
    with frozen_spatial_critic(critic):
        logits = critic(fake_features, text.detach())
    return {n: patch_generator_loss(logits[n], kind=kind) for n in critic.names}
