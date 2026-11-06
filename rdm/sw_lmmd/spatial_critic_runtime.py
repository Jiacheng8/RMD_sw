"""One generation step's mixed spatial-DINO/Qwen and original global-AIM GAN.

Only the GAN heads change. The runtime does not touch global MMD features, encode
images, choose generation-step weights, or calculate an adaptive lambda. Returned
adversarial losses contain the original1/E encoder average; the caller applies
the existing GAN/step weights and performs adaptive balancing in image space.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict

import torch
from torch import nn

from .adversarial import build_critic, critic_loss, adversarial_loss
from .sharding import all_reduce_mean_, get_world_size
from .spatial_adversarial import (
    DEFAULT_SPATIAL_BRANCHES, SpatialFeatureCritic,
    frozen_spatial_critic, patch_discriminator_loss, patch_generator_loss,
)


class SpatialGANRuntime:
    """Replicated mixed critic+AdamW, with one instance per generator step.

    ``store`` supplies the configured MMD panel, either the original three-global
    encoders or the DINO/Qwen successor. Building its legacy global heads before
    pruning preserves old-panel initialization/statistics and never asks a new
    two-stream store for removed SigLIP/AIM features. ``fake_global``/``real_global`` may be joint vectors;
    only their image prefix is read and caption tau is supplied separately.

    Spatial inputs are unaggregated maps; learned aggregators are never cached.
    Large CPU/GPU feature windows are sliced *before* conversion to this device.
    """

    def __init__(self, cfg, store, base_encoder_names, device='cpu', dtype=torch.float32,
                 critic_micro_batch=1, step_seed=None, *, global_remaining=('aimv2_huge',),
                 spatial_branch_specs=None, spatial_options=None):
        cfg.validate()
        if cfg.r1_gamma != 0:
            raise ValueError('SpatialGANRuntime does not implement R1; require r1_gamma=0')
        if cfg.loss not in ('hinge', 'ns') or cfg.critic != 'row':
            raise ValueError('SpatialGANRuntime requires hinge/ns loss and row global critic')
        if not isinstance(critic_micro_batch, int) or critic_micro_batch <= 0:
            raise ValueError('critic_micro_batch must be a positive integer')
        self.cfg, self.device, self.dtype = cfg, torch.device(device), dtype
        self.critic_micro_batch = critic_micro_batch
        self.base_encoder_names = list(base_encoder_names)
        panels = ({'dinov3_l', 'siglip2', 'aimv2_huge'}, {'dinov3_l', 'qwen3vl_4b'})
        if set(self.base_encoder_names) not in panels or len(self.base_encoder_names) != len(set(self.base_encoder_names)):
            raise ValueError('Spatial GAN requires the original MMD panel or the DINO/Qwen MMD panel')
        if cfg.encoders is not None and list(cfg.encoders) != self.base_encoder_names:
            raise ValueError('Spatial GAN requires gan.encoders to preserve the configured MMD panel')
        self.global_names = list(global_remaining)
        if self.global_names not in ([], ['aimv2_huge']):
            raise ValueError('Global GAN heads must be empty or the explicit legacy AIM head')
        if not set(self.global_names).issubset(self.base_encoder_names):
            raise ValueError('A global GAN head must exist in the configured MMD store')
        self.spatial_specs = dict(DEFAULT_SPATIAL_BRANCHES if spatial_branch_specs is None else spatial_branch_specs)
        if set(self.spatial_specs) != {'dinov3_l', 'qwen3vl_4b'}:
            raise ValueError('Spatial GAN requires the DINO and Qwen spatial branches')
        self.spatial_names = list(self.spatial_specs)
        self.names = self.spatial_names + self.global_names
        # GAN head normalization is independent of the configured MMD streams.
        self.encoder_weight = 1.0 / len(self.names)
        self.seed = int(cfg.seed if step_seed is None else step_seed)
        self.text_dim = int(store.text.shape[1])
        if self.text_dim != 2560:
            raise ValueError('Spatial GAN caption condition must be the2560D Qwen3-4B text vector')

        old_cfg = copy.deepcopy(cfg)
        old_cfg.seed = self.seed
        # The legacy builder uses torch.manual_seed inside a CPU fork. Protect all
        # visible CUDA RNGs as well so critic construction cannot reset the student.
        rng_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=rng_devices):
            old = build_critic(old_cfg, store, self.base_encoder_names, joint=True,
                               device='cpu', dtype=dtype)
        global_critic = old.critic
        global_critic.heads = nn.ModuleDict({name: global_critic.heads[name] for name in self.global_names})
        global_critic.image_norm = nn.ModuleDict({name: global_critic.image_norm[name] for name in self.global_names})
        global_critic.names = list(self.global_names)
        global_critic.image_dims = {name: global_critic.image_dims[name] for name in self.global_names}
        self.global_image_dims = dict(global_critic.image_dims)
        del old  # Its discarded optimizer must not retain the removed heads.

        options = {'hidden': cfg.hidden, 'depth': cfg.depth, 'heads': 8,
                   'text_dim': self.text_dim, 'spectral_norm': cfg.spectral_norm}
        options.update(spatial_options or {})
        if options['text_dim'] != self.text_dim or options['spectral_norm'] != cfg.spectral_norm:
            raise ValueError('Spatial text dimension and spectral norm must match the saved GAN configuration')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            spatial = SpatialFeatureCritic(self.spatial_specs, **options)
        self.critic = nn.ModuleDict({'spatial': spatial, 'global': global_critic}).to(
            device=self.device, dtype=self.dtype).eval()
        self.optimizer = torch.optim.AdamW(self.critic.parameters(), lr=cfg.lr,
                                           betas=tuple(cfg.betas), weight_decay=0.0)
        self.updates = 0
        self._last_d_grad_norms = {}
        self.image_dims = {name: self.spatial_specs[name].channels for name in self.spatial_names}
        self.image_dims.update(self.global_image_dims)
        self.block = None
        self.contract = {'version': 1, 'base_encoders': self.base_encoder_names,
                         'head_order': self.names, 'dtype': str(self.dtype),
                         'spatial': {name: asdict(spec) for name, spec in self.spatial_specs.items()},
                         'spatial_options': options, 'global': self.global_names,
                         'seed': self.seed, 'encoder_weight': self.encoder_weight,
                         'global_conditional': bool(cfg.conditional),
                         'loss': cfg.loss, 'lr': float(cfg.lr), 'betas': list(cfg.betas)}
        if cfg.grad_clip > 0:
            self.contract['grad_clip'] = float(cfg.grad_clip)

    @property
    def conditional(self):
        # Spatial heads always use the explicitly requested caption concatenation.
        return True

    def generator_active(self, step_idx):
        return self.cfg.weight > 0 and step_idx >= self.cfg.g_start_step

    def _rows(self, spatial, global_features, text):
        if text is None or text.ndim != 2 or text.shape[1] != self.text_dim or text.shape[0] == 0:
            raise ValueError(f'Expected nonempty caption tau(B,{self.text_dim})')
        rows = text.shape[0]
        for name in self.spatial_names:
            spec = self.spatial_specs[name]
            if spatial[name].shape != (rows, spec.channels, spec.grid_size, spec.grid_size):
                raise ValueError(f'Invalid spatial window shape for{name}')
        for name in self.global_names:
            value = global_features[name]
            if value.ndim != 2 or value.shape[0] != rows or value.shape[1] < self.global_image_dims[name]:
                raise ValueError(f'Invalid image/joint global window shape for{name}')
        return rows

    def _chunk(self, spatial, global_features, text, lo, hi, *, detach):
        def transfer(value):
            value = value.detach() if detach else value
            return value.to(device=self.device, dtype=self.dtype)
        space = {name: transfer(spatial[name][lo:hi]) for name in self.spatial_names}
        glob = {name: transfer(global_features[name][lo:hi, :self.global_image_dims[name]])
                for name in self.global_names}
        return space, glob, text[lo:hi].detach().to(device=self.device, dtype=self.dtype)

    def _score(self, spatial, global_features, text):
        logits = self.critic['spatial'](spatial, text)
        if self.global_names:
            global_text = text if self.critic['global'].conditional else None
            logits.update(self.critic['global'](global_features, global_text))
        return logits

    def _reduce_grads(self):
        params = list(self.critic.parameters())
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in params])
        all_reduce_mean_(flat)  # Exactly one cross-rank mean of accumulated D gradients.
        offset = 0
        for parameter in params:
            parameter.grad = flat[offset:offset + parameter.numel()].view_as(parameter)
            offset += parameter.numel()
        # Observe the already averaged gradients before optimizer.step/zero_grad.
        # Reduce each parameter in its native dtype, then accumulate only scalar
        # norm-squares in fp64; never materialize a second full-size gradient copy.
        squares = {id(parameter): parameter.grad.detach().norm().double().square()
                   for parameter in params}
        def norm(parameters):
            terms = [squares[id(parameter)] for parameter in parameters]
            return float(torch.stack(terms).sum().sqrt()) if terms else 0.0
        self._last_d_grad_norms = {'gan/d_grad_norm': norm(params)}
        for name in self.spatial_names:
            self._last_d_grad_norms[f'gan/{name}/d_grad_norm'] = norm(
                self.critic['spatial'].heads[name].parameters())
        for name in self.global_names:
            self._last_d_grad_norms[f'gan/{name}/d_grad_norm'] = norm(
                self.critic['global'].heads[name].parameters())
        finite = bool(torch.isfinite(flat).all()) and math.isfinite(self._last_d_grad_norms['gan/d_grad_norm'])
        if self.cfg.grad_clip > 0:
            # All chunks and the cross-rank mean are complete. Reuse the logged
            # total norm; no extra backward or per-layer telemetry is required.
            norm_value = self._last_d_grad_norms['gan/d_grad_norm']
            coefficient = min(1.0, self.cfg.grad_clip / (norm_value + 1e-6)) if finite else 0.0
            if finite:
                flat.mul_(coefficient)  # parameter.grad views share this flat storage
            self._last_d_grad_norms['gan/d_clip_coefficient'] = coefficient
        return finite

    def update(self, fake_spatial, real_spatial, fake_global, real_global, tau):
        rows = self._rows(fake_spatial, fake_global, tau)
        if self._rows(real_spatial, real_global, tau) != rows:
            raise ValueError('D update requires matched real/fake window rows')
        metrics = {name: torch.zeros(5, device=self.device, dtype=torch.float64) for name in self.names}
        self.optimizer.zero_grad(set_to_none=True)
        self.critic.train()
        try:
            for lo in range(0, rows, self.critic_micro_batch):
                hi = min(rows, lo + self.critic_micro_batch)
                real_s, real_g, text = self._chunk(real_spatial, real_global, tau, lo, hi, detach=True)
                fake_s, fake_g, _ = self._chunk(fake_spatial, fake_global, tau, lo, hi, detach=True)
                spatial = {name: torch.cat((real_s[name], fake_s[name]), 0) for name in self.spatial_names}
                global_features = {name: torch.cat((real_g[name], fake_g[name]), 0) for name in self.global_names}
                logits = self._score(spatial, global_features, torch.cat((text, text), 0))
                # Each SN layer saw exactly one training forward, independent of
                # microbatch size. Hold those same power vectors for every chunk.
                self.critic.eval()
                size, fraction = hi - lo, (hi - lo) / rows
                losses = {}
                for name, values in logits.items():
                    real_l, fake_l = values[:size], values[size:]
                    loss = patch_discriminator_loss(real_l, fake_l, kind=self.cfg.loss) if name in self.spatial_names \
                           else critic_loss(real_l, fake_l, self.cfg.loss)
                    losses[name] = loss
                    with torch.no_grad():
                        accuracy = ((real_l > 0).double().mean() + (fake_l < 0).double().mean()) / 2
                        # Paired pre-update diagnostic, from these same logits;
                        # this is NOT the actual post-update active-row G loss.
                        proxy = patch_generator_loss(fake_l, kind=self.cfg.loss) if name in self.spatial_names \
                                else adversarial_loss(fake_l, self.cfg.loss)
                        metrics[name] += torch.stack((loss.detach().double(), real_l.detach().double().mean(),
                                                     fake_l.detach().double().mean(), accuracy,
                                                     proxy.detach().double())) * fraction
                loss = sum(losses.values()) * self.encoder_weight * fraction
                loss.backward()
            finite = self._reduce_grads()
            if finite:
                self.optimizer.step(); self.updates += 1
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.critic.eval()
        vector = torch.cat([metrics[name] for name in self.names])
        all_reduce_mean_(vector)  # Metrics only; no further parameter-gradient scaling.
        values = vector.reshape(len(self.names), 5).tolist()
        logs = {'gan/d_loss': sum(v[0] for v in values) * self.encoder_weight,
                'gan/d_real': sum(v[1] for v in values) * self.encoder_weight,
                'gan/d_fake': sum(v[2] for v in values) * self.encoder_weight,
                'gan/d_acc': sum(v[3] for v in values) * self.encoder_weight,
                'gan/d_skipped': not finite, 'gan/d_updates': self.updates,
                'gan/d_lr': float(self.optimizer.param_groups[0]['lr']),
                **self._last_d_grad_norms}
        logs['gan/d_phase'] = 'pre_update_window'
        for name, entry in zip(self.names, values):
            logs[f'gan/{name}/d_loss'], logs[f'gan/{name}/d_acc'] = entry[0], entry[3]
            logs[f'gan/{name}/d_real'] = entry[1]
            logs[f'gan/{name}/d_fake'] = entry[2]
            logs[f'gan/{name}/g_proxy_pre_d_window'] = entry[4]
        return logs

    def generator_loss(self, spatial, global_feats, tau):
        rows = self._rows(spatial, global_feats, tau)
        losses = {name: None for name in self.names}
        fake_means = {name: 0.0 for name in self.names}
        with frozen_spatial_critic(self.critic):
            for lo in range(0, rows, self.critic_micro_batch):
                hi = min(rows, lo + self.critic_micro_batch)
                space, glob, text = self._chunk(spatial, global_feats, tau, lo, hi, detach=False)
                logits = self._score(space, glob, text)
                for name, values in logits.items():
                    value = patch_generator_loss(values, kind=self.cfg.loss) if name in self.spatial_names \
                            else adversarial_loss(values, self.cfg.loss)
                    fraction = (hi - lo) / rows
                    fake_means[name] += float(values.detach().mean()) * fraction
                    value = value * fraction
                    losses[name] = value if losses[name] is None else losses[name] + value
        total = sum(losses.values()) * self.encoder_weight
        # Reduce detached telemetry only. Keep the local differentiable mean
        # above unchanged: FSDP already averages the actual parameter gradients.
        # Including row count also handles unequal per-rank active row counts.
        vector = torch.tensor([v for name in self.names for v in
                               (float(losses[name].detach()) * rows, fake_means[name] * rows)]
                              + [float(rows)], dtype=torch.float64, device=self.device)
        all_reduce_mean_(vector)
        denominator = float(vector[-1])
        entries = (vector[:-1] / denominator).reshape(len(self.names), 2).tolist()
        logs = {f'gan/{name}/g_loss': entry[0] for name, entry in zip(self.names, entries)}
        logs.update({f'gan/{name}/g_fake': entry[1] for name, entry in zip(self.names, entries)})
        logs.update({'gan/g_loss': sum(entry[0] for entry in entries) * self.encoder_weight,
                     'gan/g_fake': sum(entry[1] for entry in entries) * self.encoder_weight,
                     'gan/g_rows_global': int(round(denominator * get_world_size())),
                     'gan/g_metric_scope': 'global_active_rows',
                     'gan/g_phase': 'post_update_active'})
        return total, logs

    def state_dict(self):
        return {'critic': self.critic.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'updates': self.updates, 'contract': copy.deepcopy(self.contract)}

    def load_state_dict(self, state):
        if state.get('contract') != self.contract:
            raise ValueError('Incompatible mixed spatial GAN checkpoint contract')
        self.critic.load_state_dict(state['critic'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.updates = int(state.get('updates', 0))
        self.critic.eval()
