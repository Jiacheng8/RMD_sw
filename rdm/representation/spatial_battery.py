"""Frozen1024px spatial features with a bounded, remote-CPU real-window cache."""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class SpatialBattery(nn.Module):
    """Separate GAN features; the original global MMD battery remains unchanged.

    Both real and fake maps pass through the same BF16 feature boundary, returned
    as FP32 GradCache leaves. This keeps cache quantization identical on both
    sides, while BF16's exponent range avoids FP16 pixel-cotangent underflow.
    No learned region aggregation is stored here.
    """

    def __init__(self, device='cuda', *, qwen_snapshot=None, encoders=None,
                 max_real_rows=128, micro_batch=1):
        super().__init__()
        if max_real_rows < 1 or micro_batch < 1:
            raise ValueError('Spatial real-cache capacity and microbatch must be positive')
        if encoders is None:
            from .backbones.dinov3_spatial import DINOv3SpatialBackbone
            from .backbones.qwen3_vl_spatial import Qwen3VLSpatialEncoder
            encoders = {'dinov3_l': DINOv3SpatialBackbone(device=device),
                        'qwen3vl_4b': Qwen3VLSpatialEncoder(snapshot_dir=qwen_snapshot, device=device)}
        self.encoders = nn.ModuleDict(encoders).eval().requires_grad_(False)
        self.names = list(self.encoders)
        if set(self.names) != {'dinov3_l', 'qwen3vl_4b'}:
            raise ValueError('Spatial GAN battery requires DINOv3-L and Qwen3-VL-4B')
        self.device = torch.device(device)
        self.max_real_rows, self.micro_batch = int(max_real_rows), int(micro_batch)
        self._real = OrderedDict()
        self.real_rows_encoded = 0
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, images):
        if tuple(images.shape[1:]) != (3, 1024, 1024):
            raise ValueError('Spatial GAN must receive1024px RGB without ADM downsampling')
        with torch.autocast('cuda', enabled=images.is_cuda, dtype=torch.bfloat16):
            values = {name: model(images) for name, model in self.encoders.items()}
        return {name: value.to(torch.bfloat16).float() for name, value in values.items()}

    @torch.no_grad()
    def reference_features(self, row_ids, images):
        ids = [int(row) for row in row_ids]
        # Callers request one critic microbatch at a time if it exceeds capacity.
        if len(set(ids)) > self.max_real_rows:
            raise ValueError('Reference request exceeds bounded spatial cache capacity')
        missing = list(dict.fromkeys(row for row in ids if row not in self._real))
        for lo in range(0, len(missing), self.micro_batch):
            batch = missing[lo:lo+self.micro_batch]
            # Uint8 moves first, then /255 runs on the same device as fake PNG pixels.
            pixels = images.tensor(batch, device=self.device, dtype=torch.float32)
            values = self(pixels)
            for offset, row in enumerate(batch):
                self._real[row] = {name: value[offset].detach().to('cpu', dtype=torch.bfloat16)
                                   for name, value in values.items()}
                self.real_rows_encoded += 1
                self._real.move_to_end(row)
                # Never evict a row required by this request while filling it.
                while len(self._real) > self.max_real_rows:
                    victim = next((key for key in self._real if key not in ids), None)
                    if victim is None:
                        raise RuntimeError('Spatial real-cache capacity accounting failed')
                    del self._real[victim]
        for row in ids:
            self._real.move_to_end(row)
        return {name: torch.stack([self._real[row][name] for row in ids]) for name in self.names}

    def feature_contract(self):
        return {'version': 1, 'input': '1024px-PNG-uint8-device-fp32-div255',
                'feature_boundary': 'bfloat16-values-float32-autograd',
                'encoders': {name: model.feature_contract() for name, model in self.encoders.items()}}
