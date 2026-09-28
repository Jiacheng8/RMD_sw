"""Shared toy fixtures for the SW-LMMD trainer + parity tests (not collected by pytest).

A linear "generator" and fixed linear "encoders" stand in for the 4B MM-DiT and the frozen
battery: gradient still flows generator -> images -> features -> force, which is the only
property the trainer's contract depends on. Everything is float64 so the world-size parity
comparison is limited by the algorithm, not by fp32 accumulation order.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from rdm.compare.median_heuristic import median_bandwidth
from rdm.sw_lmmd import ReferenceFeatureStore, SlidingWindowSchedule, build_row_order
from rdm.sw_lmmd.reference_store import write_reference_store

NAMES = ("enc_a", "enc_b")
DIMS = (6, 4)
D_TXT = 3
CTX_DIM = 5
NOISE_DIM = 4
IMG_DIM = 7


class ToyGenerator:
    """Deterministic differentiable stand-in with the ``sample(noise, condition)`` contract."""

    def __init__(self, seed: int = 0, dtype=torch.float64):
        g = torch.Generator().manual_seed(seed)
        self.model = nn.Sequential(nn.Linear(NOISE_DIM + CTX_DIM, 16), nn.Tanh(),
                                   nn.Linear(16, IMG_DIM)).to(dtype)
        with torch.no_grad():
            for p in self.model.parameters():
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64).to(dtype) * 0.3)

    def sample(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.model(torch.cat([noise, condition], dim=-1)))


class MockBattery:
    """Fixed linear read-outs of the generated 'image' -> ``{name: (n, d)}``."""

    def __init__(self, seed: int = 1, dtype=torch.float64):
        g = torch.Generator().manual_seed(seed)
        self.W = {n: (torch.randn(IMG_DIM, d, generator=g, dtype=torch.float64).to(dtype))
                  for n, d in zip(NAMES, DIMS)}

    def __call__(self, images, only=None):
        return {n: images @ w for n, w in self.W.items() if only is None or n in only}


def build_toy_store(root: str, num_rows: int = 64, seed: int = 3) -> str:
    """Write a complete, row-aligned reference store (the real on-disk layout)."""
    rng = np.random.default_rng(seed)
    encoder_features = {n: rng.standard_normal((num_rows, d)).astype(np.float32)
                        for n, d in zip(NAMES, DIMS)}
    text = rng.standard_normal((num_rows, D_TXT)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)          # tau(c) is L2-normalized
    context = rng.standard_normal((num_rows, CTX_DIM)).astype(np.float32)

    # beta = sigma_img / s_txt, exactly as rdm.refprep.build_joint_one freezes it offline.
    s_txt = float(median_bandwidth(torch.from_numpy(text).float()))
    bandwidths = {}
    for name, feat in encoder_features.items():
        sigma = float(median_bandwidth(torch.from_numpy(feat).float()))
        bandwidths[name] = {"sigma": sigma, "beta": sigma / s_txt}

    return write_reference_store(root, encoder_features=encoder_features, text_features=text,
                                 context=context, bandwidths=bandwidths,
                                 row_order=build_row_order(num_rows, seed=seed),
                                 extra_metadata={"reference_type": "toy"})


def build_trainer(store_root: str, *, window: int = 16, stride: int = 4, micro_batch: int = 1,
                  lr: float = 0.0, grad_clip: float = 0.0, seed: int = 0, joint: bool = True,
                  monitor_every: int = 1, model_seed: int = 0, dtype=torch.float64):
    """Assemble a trainer over the toy store; ``lr=0`` keeps parameters fixed for grad checks."""
    from rdm.sw_lmmd import CacheConfig, SWLMMDTrainer

    store = ReferenceFeatureStore(store_root, NAMES)
    schedule = SlidingWindowSchedule(store.row_order(), window_size=window, stride=stride)
    generator = ToyGenerator(seed=model_seed, dtype=dtype)
    optimizer = torch.optim.AdamW(generator.model.parameters(), lr=lr)
    trainer = SWLMMDTrainer(generator, MockBattery(dtype=dtype), store, schedule, optimizer,
                            encoder_names=list(NAMES), noise_shape=(NOISE_DIM,),
                            micro_batch=micro_batch, grad_clip=grad_clip, kernel_block=8,
                            seed=seed, joint=joint, monitor_every=monitor_every,
                            cache_cfg=CacheConfig(store_dtype=dtype),
                            device="cpu", feature_dtype=dtype, context_dtype=dtype)
    return trainer
