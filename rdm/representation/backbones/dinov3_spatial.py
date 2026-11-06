"""Frozen cached timm DINOv3-L:1024 RGB ->last-layer(B,1024,64,64).

This is a separate spatial-GAN encoder. It never uses the old224px registry
preprocessing, CLS readout, global pooling, or an image-feature L2 normalization.
The caller supplies the shared1024px real/fake image contract; only pretrained
channel normalization is applied here. Frozen weights still permit image grads.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlsplit

import torch
from torch import nn


MODEL_ID = 'vit_large_patch16_dinov3.lvd1689m'
IMAGE_SIZE, PATCH_SIZE, GRID_SIZE, FEATURE_DIM = 1024, 16, 64, 1024


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_checkpoint(config, explicit=None):
    """Resolve cached bytes only; never call a download API or moving-main refresh."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'Missing prepared DINOv3 checkpoint:{path}')
        return path
    path = config.get('file')
    if path and Path(path).is_file():
        return Path(path)
    repo = config.get('hf_hub_id')
    if repo:
        from huggingface_hub import try_to_load_from_cache
        explicit_name = config.get('hf_hub_filename')
        names = [explicit_name] if explicit_name else ['model.safetensors', 'pytorch_model.bin']
        for name in names:
            cached = try_to_load_from_cache(repo, name)
            if isinstance(cached, str) and Path(cached).is_file():
                return Path(cached)
    url = config.get('url')
    if url:
        cached = Path(torch.hub.get_dir()) / 'checkpoints' / Path(urlsplit(url).path).name
        if cached.is_file():
            return cached
    raise FileNotFoundError('DINOv3-L weights are not cached; prepare them remotely or pass checkpoint_path')


def _build_cached_model(checkpoint_path):
    import timm
    from timm.models import get_pretrained_cfg
    config = get_pretrained_cfg(MODEL_ID)
    if config is None:
        raise ValueError('Installed timm does not define the required DINOv3-L model')
    config = config.to_dict()
    cached = _cached_checkpoint(config, checkpoint_path)
    # A local file is the only permitted pretrained source. The dynamic-size
    # implementation handles positional/RoPE resizing for the actual1024 grid.
    model = timm.create_model(MODEL_ID, pretrained=True, num_classes=0,
        img_size=IMAGE_SIZE, dynamic_img_size=True, dynamic_img_pad=False,
        pretrained_cfg_overlay={'file': str(cached), 'hf_hub_id': None, 'url': None})
    return model, {'checkpoint_sha256': _hash_file(cached), 'checkpoint_bytes': cached.stat().st_size}


class DINOv3SpatialBackbone(nn.Module):
    """Strict1024px spatial adapter; ``model`` injection is for offline CPU tests."""

    has_logits = False

    def __init__(self, device='cuda', *, checkpoint_path=None,
                 gradient_checkpointing=False, model=None):
        super().__init__()
        if model is None:
            model, provenance = _build_cached_model(checkpoint_path)
        else:
            if checkpoint_path is not None:
                raise ValueError('Do not combine injected model and checkpoint_path')
            provenance = {'injected_model': True, 'checkpoint_sha256': None}
        if int(getattr(model, 'num_features', -1)) != FEATURE_DIM:
            raise ValueError('DINOv3-L spatial encoder requires1024 output channels')
        patch = getattr(getattr(model, 'patch_embed', None), 'patch_size', None)
        if patch not in (PATCH_SIZE, (PATCH_SIZE, PATCH_SIZE)):
            raise ValueError('DINOv3-L spatial encoder requires16x16 patch embedding')
        prefix = getattr(model, 'num_prefix_tokens', None)
        if type(prefix) is not int or prefix < 1:
            raise ValueError('timm model must explicitly expose all CLS/register prefix tokens')
        self.num_prefix_tokens = prefix
        config = getattr(model, 'pretrained_cfg', {})
        mean, std = config.get('mean'), config.get('std')
        if mean is None or std is None or len(mean) != 3 or len(std) != 3 \
                or any(float(s) <= 0 for s in std):
            raise ValueError('Cached model must provide its RGB normalization contract')
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1))
        self.model = model.float().to(device).eval().requires_grad_(False)
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        self.feat_dim = FEATURE_DIM
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing:
            if not hasattr(self.model, 'set_grad_checkpointing'):
                raise ValueError('This timm model does not support requested gradient checkpointing')
            self.model.set_grad_checkpointing(True)
        self.provenance = provenance
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def feature_contract(self):
        return {'model_id': MODEL_ID, 'input_size': IMAGE_SIZE, 'patch_size': PATCH_SIZE,
                'grid_size': GRID_SIZE, 'channels': FEATURE_DIM, 'layer': 'last_forward_features',
                'prefix_tokens_removed': self.num_prefix_tokens,
                'spatial_order': 'row-major', 'preprocess': 'no_resize;pretrained_RGB_mean_std',
                'mean': self.mean.flatten().tolist(), 'std': self.std.flatten().tolist(),
                'l2_normalize': False, 'gradient_checkpointing': self.gradient_checkpointing,
                'weight_dtype': str(next(self.model.parameters()).dtype), **self.provenance}

    def forward(self, images):
        if images.ndim != 4 or images.shape[0] == 0 \
                or tuple(images.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError('DINO spatial encoder requires nonempty(B,3,1024,1024) RGB input')
        if not images.is_floating_point():
            raise TypeError('DINO spatial encoder expects floating-point[0,1] RGB images')
        normalized = (images.float() - self.mean) / self.std
        features = self.model.forward_features(normalized)
        expected = (images.shape[0], self.num_prefix_tokens + GRID_SIZE ** 2, FEATURE_DIM)
        if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected:
            raise RuntimeError(f'DINO last-layer tokens must have shape{expected}; no prefix/grid guessing')
        patches = features[:, self.num_prefix_tokens:, :]
        return patches.transpose(1, 2).reshape(images.shape[0], FEATURE_DIM, GRID_SIZE, GRID_SIZE)
