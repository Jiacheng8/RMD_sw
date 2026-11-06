"""Tiny deterministic CPU fixtures for the real mixed runtime and pixel GradCache."""
from __future__ import annotations

from contextlib import nullcontext
import importlib
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F


def load_modules():
    # Load unchanged source under a private package, avoiding optional eager
    # package imports and supporting the Mac's Python3.9+torch CPU smoke check.
    root = Path(__file__).resolve().parents[1] / 'rdm'
    for suffix in ('', '.sw_lmmd', '.representation', '.train', '.compare', '.utils'):
        name = 'spatial_parity_fixture' + suffix
        package = types.ModuleType(name)
        package.__path__ = [str(root / suffix.lstrip('.').replace('.', '/'))]
        sys.modules[name] = package
    prefix = 'spatial_parity_fixture.'
    runtime = importlib.import_module(prefix + 'sw_lmmd.spatial_critic_runtime')
    pixel = importlib.import_module(prefix + 'train.pixel_grad_caching')
    config = importlib.import_module(prefix + 'sw_lmmd.config')
    spatial = importlib.import_module(prefix + 'sw_lmmd.spatial_adversarial')
    mmd = importlib.import_module(prefix + 'sw_lmmd.local_mmd')
    return runtime, pixel, config, spatial, mmd


class TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(93)
            self.first = nn.Linear(3, 4)
            self.second = nn.Linear(4, 4)

    def forward(self, inputs):
        first = self.first(inputs).tanh()
        second = (self.second(first) + 0.2 * first).tanh()
        return [first.reshape(-1, 1, 2, 2), second.reshape(-1, 1, 2, 2)]


class TinyStore:
    num_rows = 12
    dims = {'dinov3_l': 3, 'siglip2': 5, 'aimv2_huge': 4}

    def __init__(self, images, text):
        self.images, self.text = images, text

    def dim(self, name): return self.dims[name]
    def reference_image_features(self, name, rows): return self.images[name][rows]
    def text_features(self, rows): return self.text[rows]


class TinySpatialProblem:
    """K8 window, active rows4..7, two step weights(1,.25), same-row captions."""

    base_names = ('dinov3_l', 'siglip2', 'aimv2_huge')
    step_weights = (1.0, 0.25)
    active_ids = torch.arange(4, 8)
    window_ids = torch.arange(8)

    def __init__(self, modules, *, critic_micro_batch=2):
        self.runtime_module, self.pixel, config, spatial, mmd = modules
        gen = torch.Generator().manual_seed(618)
        self.inputs = torch.randn(12, 3, generator=gen)
        self.text = F.normalize(torch.randn(12, 2560, generator=gen), dim=-1)
        self.projections = {name: torch.randn(4, dim, generator=gen) * 0.4
                            for name, dim in TinyStore.dims.items()}
        self.biases = {name: torch.randn(dim, generator=gen) * 0.1 for name, dim in TinyStore.dims.items()}
        self.beta = {'dinov3_l': 0.7, 'siglip2': 0.8, 'aimv2_huge': 0.9}
        self.sigma = {'dinov3_l': 1.2, 'siglip2': 1.4, 'aimv2_huge': 1.3}
        self.generator = TinyGenerator()
        self.mmd = mmd.ExactLocalMMD(block_size=4)
        self.cfg = config.GANConfig(seed=31, hidden=16, depth=1, stats_rows=8,
            spectral_norm=True, weight=1.5, adaptive=True, g_start_step=0)
        specs = {'dinov3_l': spatial.SpatialBranchSpec(3, 8, 1),
                 'qwen3vl_4b': spatial.SpatialBranchSpec(4, 16, 2)}
        # Static teacher pixels, distinct from the student's current images.
        teacher = torch.randn(12, 4, generator=gen).tanh().reshape(12, 1, 2, 2)
        self.real_global = self.global_encode(teacher)
        self.real_spatial = self.spatial_encode(teacher)
        self.store = TinyStore(self.real_global, self.text)
        self.runtimes = [self.runtime_module.SpatialGANRuntime(
            self.cfg, self.store, self.base_names, 'cpu', torch.float64,
            critic_micro_batch=critic_micro_batch, step_seed=self.cfg.seed + k,
            spatial_branch_specs=specs, spatial_options={'heads': 2}) for k in range(2)]
        self.context, self.fake_spatial_window = {}, {}
        with torch.no_grad():
            per_row = [self.encode(self.generate(torch.tensor([row])), torch.tensor([row]))
                       for row in self.window_ids.tolist()]
            for k in (1, 2):
                self.context[k] = {name: torch.cat([v[f's{k}/{name}'] for v in per_row])
                                   for name in self.base_names}
                self.fake_spatial_window[k] = {
                    name: torch.cat([v[f's{k}/patch/{name}'] for v in per_row])
                    for name in specs}
        self.d_gradients = {}
        for k, runtime in enumerate(self.runtimes, 1):
            original = runtime.optimizer.step
            def step(*args, index=k, owner=runtime, function=original, **kwargs):
                self.d_gradients[index] = {name: p.grad.detach().clone()
                                           for name, p in owner.critic.named_parameters()}
                return function(*args, **kwargs)
            runtime.optimizer.step = step

    def generate(self, ids):
        return self.generator(self.inputs[ids])

    def global_encode(self, image):
        value = image.flatten(1)
        return {name: (value @ self.projections[name] + self.biases[name]).sin()
                for name in self.base_names}

    @staticmethod
    def spatial_encode(image):
        dino = F.interpolate(image, size=(8, 8), mode='bilinear', align_corners=False)
        qwen = F.interpolate(image, size=(16, 16), mode='bilinear', align_corners=False)
        dino = torch.cat([torch.sin(dino * (i + 1) * 0.7 + i * 0.1) for i in range(3)], 1)
        qwen = torch.cat([torch.cos(qwen * (i + 1) * 0.4 + i * 0.2) for i in range(4)], 1)
        # Match the production spatial real/fake value boundary and differentiable cast.
        return {'dinov3_l': dino.bfloat16().float(), 'qwen3vl_4b': qwen.bfloat16().float()}

    def encode(self, images, ids):
        output = {}
        for k, image in enumerate(images, 1):
            for name, value in self.global_encode(image).items():
                output[f's{k}/{name}'] = torch.cat((value, self.beta[name] * self.text[ids]), -1)
            for name, value in self.spatial_encode(image).items():
                output[f's{k}/patch/{name}'] = value
        return output

    def reference_joint(self, name, ids):
        return torch.cat((self.real_global[name][ids], self.beta[name] * self.text[ids]), -1)

    def run(self, *, rank=0, world=1, reference=False):
        if 4 % world or 8 % world:
            raise ValueError('Fixture requires equal active/window row partitions')
        active = self.active_ids if reference else self.active_ids.chunk(world)[rank]
        d_ids = self.window_ids if reference else self.window_ids.chunk(world)[rank]
        d_logs = {}
        def terms(leaves):
            result = {}
            for k, weight in enumerate(self.step_weights, 1):
                runtime = self.runtimes[k-1]
                d_logs[k] = runtime.update(
                    {n: v[d_ids] for n, v in self.fake_spatial_window[k].items()},
                    {n: v[d_ids] for n, v in self.real_spatial.items()},
                    {n: self.context[k][n][d_ids] for n in self.base_names},
                    {n: self.reference_joint(n, d_ids) for n in self.base_names}, self.text[d_ids])
                force = None
                for name in self.base_names:
                    value, _ = self.mmd.active_force(leaves[f's{k}/{name}'], self.context[k][name],
                                                    self.reference_joint(name, self.window_ids), self.sigma[name])
                    value = value * (weight / 3)
                    force = value if force is None else force + value
                adv, _ = runtime.generator_loss(
                    {n: leaves[f's{k}/patch/{n}'] for n in self.real_spatial},
                    {n: leaves[f's{k}/{n}'] for n in self.base_names}, self.text[active])
                result[k] = (force, adv)
            return result, {}
        def balance(k, mmd_norm, adv_norm):
            lam = self.cfg.weight * mmd_norm / max(adv_norm, self.cfg.min_norm_ratio * mmd_norm, 1e-30)
            return lam, {}
        # Independent full-batch oracle on each rank; these two local calls must
        # never accidentally enter the distributed collectives of the tested run.
        disable_d_reduce = patch.object(self.runtime_module, 'all_reduce_mean_', lambda value: value) \
                           if reference else nullcontext()
        with disable_d_reduce:
            loss, logs = self.pixel.pixel_gradcache_backward(
                list(active.split(1)), self.generate, self.encode, terms, balance,
                norm_reduce_fn=(lambda squares: squares) if reference else None,
                replay_atol=0.0)
        return {'loss': loss, 'logs': logs, 'd_logs': d_logs,
                'g_gradients': {name: p.grad.detach().clone() for name, p in self.generator.named_parameters()},
                'd_gradients': self.d_gradients,
                'd_states': {k: {name: value.detach().clone() for name, value in runtime.critic.state_dict().items()}
                             for k, runtime in enumerate(self.runtimes, 1)}}
