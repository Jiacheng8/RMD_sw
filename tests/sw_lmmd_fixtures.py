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
                  monitor_every: int = 1, model_seed: int = 0, dtype=torch.float64,
                  gan: dict | None = None, lr_schedule=None):
    """Assemble a trainer over the toy store; ``lr=0`` keeps parameters fixed for grad checks.

    ``gan`` adds the adversarial critic: :class:`rdm.sw_lmmd.GANConfig` overrides on top of a
    toy-sized head (``hidden=16``) and no warm-up."""
    from rdm.sw_lmmd import CacheConfig, GANConfig, SWLMMDTrainer, build_critic

    store = ReferenceFeatureStore(store_root, NAMES)
    schedule = SlidingWindowSchedule(store.row_order(), window_size=window, stride=stride)
    generator = ToyGenerator(seed=model_seed, dtype=dtype)
    optimizer = torch.optim.AdamW(generator.model.parameters(), lr=lr)
    critic = None
    if gan is not None:
        cfg = GANConfig(**{"enabled": True, "hidden": 16, "g_start_step": 0, **gan})
        critic = build_critic(cfg, store, list(NAMES), joint=joint, device="cpu", dtype=dtype)
    trainer = SWLMMDTrainer(generator, MockBattery(dtype=dtype), store, schedule, optimizer,
                            encoder_names=list(NAMES), noise_shape=(NOISE_DIM,),
                            micro_batch=micro_batch, grad_clip=grad_clip, kernel_block=8,
                            seed=seed, joint=joint, monitor_every=monitor_every,
                            cache_cfg=CacheConfig(store_dtype=dtype),
                            device="cpu", feature_dtype=dtype, context_dtype=dtype,
                            critic=critic, lr_schedule=lr_schedule)
    return trainer


# ---------------------------------------------------------------------------------------------
# Tiny REAL FLUX.2 path, for the FSDP test: the actual ``Flux2AdapterModel._run_dit`` over a
# randomly initialized ``flux2.model.Flux2`` a few thousand parameters wide. Only the VAE and
# the encoder battery are stubbed. fp32, since FSDP's mixed precision is part of what is tested.
# ---------------------------------------------------------------------------------------------
FLUX_CTX_DIM = 16         # Qwen3 context width (7680 in klein-4B)
FLUX_CTX_LEN = 3          # context tokens (48 in the real pool)
FLUX_LATENT = 2           # latent side -> 2x2 = 4 image tokens (32 at 512 px)
FLUX_KEEP = 4             # reference rows per prompt, as in the real store


def tiny_flux_adapter(gradient_checkpointing: bool = True):
    """A real :class:`Flux2AdapterModel` around a tiny ``Flux2``; same init on every rank."""
    from rdm.representation.generators.flux_generator import (Flux2AdapterModel,
                                                              _ensure_flux2_importable)
    _ensure_flux2_importable(None)
    from flux2.model import Flux2, Klein4BParams

    torch.manual_seed(0)
    params = Klein4BParams(context_in_dim=FLUX_CTX_DIM, hidden_size=64, num_heads=2, depth=1,
                           depth_single_blocks=2, axes_dim=[8, 8, 8, 8])
    return Flux2AdapterModel(image_resolution=16 * FLUX_LATENT, model=Flux2(params),
                             gradient_checkpointing=gradient_checkpointing)


class LatentTokenizer:
    """Stands in for the VAE: a fixed differentiable map from latents to 'pixels'."""

    def detokenize(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(z)


class LatentBattery:
    """Fixed linear read-outs of the flattened 'pixels' -> ``{name: (n, d)}``."""

    def __init__(self, seed: int = 1):
        g = torch.Generator().manual_seed(seed)
        n_in = 128 * FLUX_LATENT * FLUX_LATENT
        self.W = {n: torch.randn(n_in, d, generator=g) / n_in ** 0.5 for n, d in zip(NAMES, DIMS)}

    def __call__(self, images, only=None):
        x = images.flatten(1).float()
        return {n: x @ w for n, w in self.W.items() if only is None or n in only}


def build_flux_toy_store(root: str, num_prompts: int = 16, seed: int = 3) -> str:
    """The real layout at toy size: ``FLUX_KEEP`` reference rows per prompt via ``prompt_ids``,
    a prompt-indexed ``(P, L, d)`` generator context and a prompt-indexed text table."""
    rng = np.random.default_rng(seed)
    num_rows = num_prompts * FLUX_KEEP
    encoder_features = {n: rng.standard_normal((num_rows, d)).astype(np.float32)
                        for n, d in zip(NAMES, DIMS)}
    text = rng.standard_normal((num_prompts, D_TXT)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    context = rng.standard_normal((num_prompts, FLUX_CTX_LEN, FLUX_CTX_DIM)).astype(np.float32)
    s_txt = float(median_bandwidth(torch.from_numpy(text).float()))
    bandwidths = {n: {"sigma": float(median_bandwidth(torch.from_numpy(f).float())),
                      "beta": float(median_bandwidth(torch.from_numpy(f).float())) / s_txt}
                  for n, f in encoder_features.items()}
    return write_reference_store(root, encoder_features=encoder_features, text_features=text,
                                 context=context, bandwidths=bandwidths,
                                 row_order=build_row_order(num_rows, seed=seed),
                                 prompt_ids=np.arange(num_rows) // FLUX_KEEP,
                                 extra_metadata={"reference_type": "toy-flux"})
