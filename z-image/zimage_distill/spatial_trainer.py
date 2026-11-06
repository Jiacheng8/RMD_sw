"""Independent spatial GAN integration; existing global MMD streams stay intact.

The two-level GradCache cuts at decoded pixels: frozen-encoder VJPs and student
backward use separate graphs. Adaptive GAN weighting compares gradients in their
shared pixel space, rather than comparing unrelated global/spatial feature axes.
"""
from __future__ import annotations

import numpy as np
import torch

from rdm.compare.grad_balance import clip_generator_grads
from rdm.representation.joint_feature import couple
from rdm.sw_lmmd.sharding import (
    all_gather_detached, all_reduce_mean_, partition_rows, reduce_scalar, row_noise,
)
from rdm.sw_lmmd.spatial_cache import SpatialWindowCache
from rdm.sw_lmmd.config import LRSchedule
from dataclasses import asdict
from rdm.sw_lmmd.adaptive_d_lr import AdaptiveDLRConfig, AdaptiveDLRController
from rdm.train.pixel_grad_caching import pixel_gradcache_backward
from rdm.utils.distributed import get_rank, is_dist

from .model import reference_pixels
from .store import split_stream, stream_name
from .trainer import MultiStepTrainer, _PRINTED_GAN_KEYS


def patch_stream(step, encoder):
    return f's{int(step)}/patch/{encoder}'


class SpatialMultiStepTrainer(MultiStepTrainer):
    def __init__(self, *args, spatial_battery, reference_images, adaptive_d_lr=None,
                 adaptive_d_lr_allow_legacy_resume=False, discriminator_lr_schedule=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.joint or not self.feature_quantize:
            raise ValueError('Spatial GAN requires Qwen text joint MMD and shared PNG quantization')
        if len(self.step_critics) != self.num_steps:
            raise ValueError('Spatial GAN requires one mixed critic per generation step')
        self.spatial_battery = spatial_battery
        self.reference_images = reference_images
        self.spatial_names = list(spatial_battery.names)
        self.spatial_stream_names = [patch_stream(k, e) for k in range(1, self.num_steps+1)
                                     for e in self.spatial_names]
        self.spatial_cache = SpatialWindowCache(self.spatial_stream_names)
        real_contract = reference_images.manifest['contract']
        if (reference_images.num_rows != self.store.num_rows or
                real_contract.get('prompt_hash') != self.store.metadata.get('prompt_hash') or
                real_contract.get('render_hash') != self.store.metadata.get('render_hash')):
            raise ValueError('Spatial real-image rows differ from the existing MMD reference')
        self.spatial_contract = {'version': 1, 'representations': spatial_battery.feature_contract(),
                                 'reference': real_contract,
                                 'index_sha256': reference_images.manifest['index']['sha256'],
                                 'cache_dtype': 'bfloat16', 'balance_space': 'decoded-image-pixels'}
        self.d_lr_config = AdaptiveDLRConfig.from_mapping(adaptive_d_lr or {})
        self.discriminator_lr_schedule = (LRSchedule(**discriminator_lr_schedule)
                                          if discriminator_lr_schedule else None)
        if self.discriminator_lr_schedule is not None:
            self.discriminator_lr_schedule.validate()
            if self.d_lr_config.enabled:
                raise ValueError('Choose scheduled D LR or adaptive D LR, not both')
        self.d_lr_controller = None
        if self.d_lr_config.enabled:
            if any(runtime.cfg.loss != 'ns' for runtime in self.step_critics):
                raise ValueError('Adaptive spatial D LR requires the NS softplus loss')
            base_lr = float(self.step_critics[0].cfg.lr)
            if any(float(runtime.cfg.lr) != base_lr for runtime in self.step_critics):
                raise ValueError('Adaptive D LR requires one shared base LR for every step')
            heads = tuple(self.step_critics[0].names)
            if any(tuple(runtime.names) != heads for runtime in self.step_critics):
                raise ValueError('Adaptive D LR requires the same GAN heads in both steps')
            self.d_lr_controller = AdaptiveDLRController(base_lr=base_lr, cfg=self.d_lr_config, heads=heads)
        if not isinstance(adaptive_d_lr_allow_legacy_resume, bool):
            raise ValueError('adaptive_d_lr_allow_legacy_resume must be a boolean')
        self.adaptive_d_lr_allow_legacy_resume = adaptive_d_lr_allow_legacy_resume

    def _adapt_discriminator_lr(self, logs):
        """Apply rank-0's audited decision for the NEXT iteration, never the G LR."""
        if self.d_lr_controller is None:
            return
        current = self.d_lr_controller.current_lr
        for runtime in self.step_critics:
            if any(float(group['lr']) != current for group in runtime.optimizer.param_groups):
                raise RuntimeError('D optimizer LR differs from the saved adaptive controller')
        active = all(bool(logs.get(f's{k}/gan/g_active', False))
                     for k in range(1, self.num_steps + 1))
        valid = not logs['skipped'] and not any(
            logs[f's{k}/gan/d_skipped'] for k in range(1, self.num_steps + 1))
        heads = {}
        for k, runtime in enumerate(self.step_critics, 1):
            for name in runtime.names:
                prefix = f's{k}/gan/{name}'
                heads[f's{k}/{name}'] = {
                    'd_loss': logs[f'{prefix}/d_loss'],
                    'g_loss': logs.get(f'{prefix}/g_loss') if active else None,
                    'g_proxy': logs.get(f'{prefix}/g_proxy_pre_d_window')}
        packet = [None]
        if get_rank() == 0:
            decision = self.d_lr_controller.observe(logs['step'], heads,
                                                    generator_active=active, valid=valid)
            packet[0] = (decision, self.d_lr_controller.state_dict())
        if is_dist():
            torch.distributed.broadcast_object_list(packet, src=0)
        decision, state = packet[0]
        if get_rank() != 0:
            self.d_lr_controller.load_state_dict(state)
        new_lr = float(decision['new_lr'])
        for runtime in self.step_critics:
            for group in runtime.optimizer.param_groups:
                group['lr'] = new_lr
        logs.update({'gan/d_lr_used': current, 'gan/d_lr_next': new_lr,
                     'gan/d_lr_control': decision})

    def _generate_images(self, row_ids):
        ids = np.asarray(row_ids, dtype=np.int64)
        noise = row_noise(ids, self.noise_shape, seed=self.seed, epoch=self.epoch,
                          device=self.device, dtype=self.feature_dtype)
        if self.use_store_context:
            images = self.generator.sample_steps(noise, contexts=self.store.generator_context(
                ids, device=self.device, dtype=self.context_dtype))
        else:
            images = self.generator.sample_steps(noise, prompts=self.store.prompts(ids))
        if len(images) != self.num_steps:
            raise RuntimeError('Generator step count differs from the spatial GAN contract')
        return images

    def _encode_images(self, images, row_ids):
        ids = np.asarray(row_ids, dtype=np.int64)
        tau = self.store.text_features(ids, device=self.device, dtype=self.feature_dtype)
        output = {}
        for k, image in enumerate(images, 1):
            global_pixels = reference_pixels(image, self.feature_pool, self.feature_quantize)
            pooled = self.battery(global_pixels, only=set(self.base_names))
            for encoder in self.base_names:
                key = stream_name(k, encoder)
                output[key] = couple(pooled[encoder], tau, self.betas[key])
            # Spatial branches see full1024 PNG pixels, without the MMD ADM512 reduction.
            spatial = self.spatial_battery(reference_pixels(image, factor=1, quantize=True))
            for encoder in self.spatial_names:
                output[patch_stream(k, encoder)] = spatial[encoder]
        return output

    def _encode(self, row_ids):
        return self._encode_images(self._generate_images(row_ids), row_ids)

    @torch.no_grad()
    def bootstrap(self):
        window = self.schedule.next()
        self._pending_window = window
        local = partition_rows(window.retained_ids)
        features = self._encode_no_grad(self._chunks(local))
        pooled = {name: all_gather_detached(features[name]) for name in self.encoder_names}
        spatial = {name: all_gather_detached(features[name]).to('cpu', dtype=torch.bfloat16)
                   for name in self.spatial_stream_names}
        self.cache.initialize(window.retained_ids, pooled, step=-1)
        self.spatial_cache.initialize(window.retained_ids, spatial, step=-1)
        return {'bootstrap_rows': int(window.retained_ids.size),
                'bootstrap_rows_per_rank': int(local.size)}

    @torch.no_grad()
    def rollout_refresh(self, row_ids):
        local = partition_rows(row_ids)
        features = self._encode_no_grad(self._chunks(local))
        return {name: all_gather_detached(features[name])
                for name in self.encoder_names+self.spatial_stream_names}

    def probe_spatial_cache(self, row_ids, fresh):
        logs = {}
        for name in self.spatial_stream_names:
            entry = self.spatial_cache.entries[name]
            positions = self.spatial_cache._positions_of(entry.row_ids, row_ids)
            old = entry.features[positions].float()
            new = fresh[name].detach().to('cpu', dtype=torch.float32)
            drift = (old-new).flatten(1).norm(dim=1) / new.flatten(1).norm(dim=1).clamp_min(1e-8)
            logs[f'{name}/drift_mean'] = float(drift.mean())
            logs[f'{name}/drift_max'] = float(drift.max())
        return logs

    def refresh_spatial_cache(self, row_ids, fresh, step):
        values = [self.spatial_cache.refresh_retained(name, row_ids, fresh[name], step=step)
                  for name in self.spatial_stream_names]
        return {'spatial_refreshed_rows': len(row_ids),
                'spatial_refresh_drift_mean': float(torch.stack([v.mean() for v in values]).mean())}

    def step(self):
        if not self.cache.initialized or not self.spatial_cache.initialized:
            raise RuntimeError('Bootstrap both global and spatial windows before training')
        lr = self._apply_lr_schedule()
        if self.discriminator_lr_schedule is not None:
            factor = self.discriminator_lr_schedule.factor(self.step_idx)
            for runtime in self.step_critics:
                for group in runtime.optimizer.param_groups:
                    group['lr'] = runtime.cfg.lr * factor
        window = self._pending_window or self.schedule.next()
        self._pending_window = None
        local_active = partition_rows(window.active_ids)
        chunks = self._chunks(local_active)
        gathered, gathered_spatial = {}, {}
        monitor = self.monitor_every > 0 and self.step_idx % self.monitor_every == 0
        steps = range(1, self.num_steps+1)

        def loss_terms(features):
            logs = {}
            forces = {k: None for k in steps}
            raw = {k: [] for k in steps}
            global_windows = {k: {} for k in steps}
            spatial_windows = {k: {} for k in steps}
            for name in self.encoder_names:
                k, encoder = split_stream(name)
                active = features[name]
                gathered[name] = all_gather_detached(active)
                context = torch.cat((self.cache.retained(name, window.retained_ids).to(active),
                                     gathered[name]), 0)
                reference = self._reference_window(name, window.all_ids)
                force, stats = self.mmd.active_force(active, context, reference, self.sigmas[name])
                weighted = self.encoder_weights[name]*force
                forces[k] = weighted if forces[k] is None else forces[k]+weighted
                global_windows[k][encoder] = (context, reference)
                logs.update({f'{name}/force': float(stats['force']),
                             f'{name}/repulsion': float(stats['active_repulsion']),
                             f'{name}/attraction': float(stats['active_attraction'])})
                if monitor:
                    value = self.mmd.monitor(context, reference, self.sigmas[name])
                    logs.update(value.as_log(prefix=f'{name}/'))
                    raw[k].append(float(value.mmd2))
            for k in steps:
                for encoder in self.spatial_names:
                    name = patch_stream(k, encoder)
                    gathered_spatial[name] = all_gather_detached(features[name]).to('cpu', dtype=torch.bfloat16)
                    spatial_windows[k][encoder] = torch.cat((
                        self.spatial_cache.retained(name, window.retained_ids), gathered_spatial[name]), 0)

            # The default window discriminator keeps exactly the old K/R row split.
            cfg = self.step_critics[0].cfg
            if cfg.critic_rows == 'active':
                positions = np.arange(window.all_ids.size-window.active_ids.size, window.all_ids.size)
            else:
                positions = partition_rows(np.arange(window.all_ids.size))
            lo, hi = int(positions[0]), int(positions[-1])+1
            real_ids = window.all_ids[lo:hi]
            real_maps = self.spatial_battery.reference_features(real_ids, self.reference_images)
            tau_d = self.store.text_features(real_ids, device='cpu', dtype=torch.float32)
            tau_g = self.store.text_features(local_active, device=self.device, dtype=torch.float32)
            terms = {}
            for k in steps:
                runtime = self.step_critics[k-1]
                dlogs = runtime.update(
                    {e: value[lo:hi] for e, value in spatial_windows[k].items()}, real_maps,
                    {e: pair[0][lo:hi] for e, pair in global_windows[k].items()},
                    {e: pair[1][lo:hi] for e, pair in global_windows[k].items()}, tau_d)
                logs.update({f's{k}/{name}': value for name, value in dlogs.items()})
                adv = None
                if runtime.generator_active(self.step_idx):
                    adv, glogs = runtime.generator_loss(
                        {e: features[patch_stream(k, e)] for e in self.spatial_names},
                        {e: features[stream_name(k, e)] for e in self.base_names}, tau_g)
                    logs.update({f's{k}/{name}': value for name, value in glogs.items()})
                terms[k] = (forces[k], adv)
                logs[f's{k}/force'] = float(forces[k].detach())
                if raw[k]:
                    logs[f's{k}/raw_mmd2'] = float(np.mean(raw[k]))
            logs['_force'] = sum(float(forces[k].detach()) for k in steps)
            if monitor:
                logs['raw_mmd2'] = logs[f's{self.num_steps}/raw_mmd2']
            return terms, logs

        def balance(k, mmd_norm, adv_norm):
            cfg = self.step_critics[k-1].cfg
            lam = (cfg.weight*mmd_norm/max(adv_norm, cfg.min_norm_ratio*mmd_norm, 1e-30)
                   if cfg.adaptive else cfg.weight)
            return lam, {'gan/lambda': lam,
                         'gan/grad_ratio': lam*adv_norm/max(mmd_norm, 1e-30),
                         'gan/grad_ratio_raw': adv_norm/max(mmd_norm, 1e-30)}

        self.optimizer.zero_grad(set_to_none=True)
        fixed = (None if any(runtime.cfg.adaptive for runtime in self.step_critics) else
                 {k: self.step_critics[k-1].cfg.weight for k in steps})
        value, logs = pixel_gradcache_backward(chunks, self._generate_images, self._encode_images,
                                               loss_terms, balance, fixed_weights=fixed,
                                               diagnose_replay=fixed is None)
        force_value = logs.pop('_force', value)
        if self.grad_reduce == 'mean':
            for parameter in self._gen_params:
                if parameter.grad is not None:
                    all_reduce_mean_(parameter.grad)
        grad_norm = clip_generator_grads(self._gen_params, self.grad_clip, module=self.clip_module) \
            if self.grad_clip > 0 else None
        finite = np.isfinite(value) and (grad_norm is None or bool(torch.isfinite(grad_norm)))
        if finite:
            self.optimizer.step()
        else:
            self.optimizer.zero_grad(set_to_none=True)
        self.cache.advance(window.all_ids, int(window.retained_ids.size), gathered, step=window.step)
        self.spatial_cache.advance(window.all_ids, int(window.retained_ids.size), gathered_spatial,
                                   step=window.step)
        self.step_idx += 1
        result = {'step': self.step_idx, 'window_step': window.step, 'epoch': window.epoch,
                  'force': reduce_scalar(force_value), 'skipped': not finite,
                  'grad_norm': float(grad_norm) if grad_norm is not None else None, 'lr': lr,
                  'loss_total': reduce_scalar(value), 'cache_max_age': self.cache.max_age(window.step),
                  'active_rows': int(window.active_ids.size),
                  'active_rows_per_rank': int(local_active.size),
                  'spatial_real_cache_rows': len(self.spatial_battery._real),
                  'spatial_real_rows_encoded': self.spatial_battery.real_rows_encoded, **logs}
        for key in _PRINTED_GAN_KEYS:
            if f's{self.num_steps}/{key}' in logs:
                result[key] = logs[f's{self.num_steps}/{key}']
        if self.grad_clip > 0 and grad_norm is not None:
            result['g_clip_coefficient'] = (min(1.0, self.grad_clip / (float(grad_norm) + 1e-6))
                                            if finite else 0.0)
        self._adapt_discriminator_lr(result)
        self._last_logs = result
        return result

    def state_dict(self, with_cache=True, with_optimizer=True, with_critic=True):
        state = super().state_dict(with_cache, with_optimizer, with_critic)
        state['spatial_gan_contract'] = self.spatial_contract
        if self.discriminator_lr_schedule is not None:
            state['discriminator_lr_schedule'] = asdict(self.discriminator_lr_schedule)
        if with_cache:
            state['spatial_cache'] = self.spatial_cache.state_dict()
        if self.d_lr_controller is not None:
            state['adaptive_d_lr'] = self.d_lr_controller.state_dict()
        return state

    def load_state_dict(self, state):
        expected = asdict(self.discriminator_lr_schedule) if self.discriminator_lr_schedule else None
        if state.get('discriminator_lr_schedule') != expected:
            raise RuntimeError('Discriminator LR schedule differs from checkpoint')
        if state.get('spatial_gan_contract') != self.spatial_contract:
            raise RuntimeError('Spatial GAN checkpoint representation/reference contract mismatch')
        if state.get('cache') is not None and state.get('spatial_cache') is None:
            raise RuntimeError('A global generated cache requires its aligned spatial window')
        controller_state = state.get('adaptive_d_lr')
        if controller_state is not None and self.d_lr_controller is None:
            raise RuntimeError('Cannot silently disable an active adaptive D LR on resume')
        if self.d_lr_controller is not None:
            if controller_state is None and not self.adaptive_d_lr_allow_legacy_resume:
                raise RuntimeError('Legacy checkpoint has no adaptive D LR history; migration must be explicit')
            if controller_state is not None:
                self.d_lr_controller.load_state_dict(controller_state)
        result = super().load_state_dict(state)
        if 'spatial_cache' in state:
            self.spatial_cache.load_state_dict(state['spatial_cache'])
        result['spatial_cache'] = 'spatial_cache' in state
        if self.d_lr_controller is not None:
            expected = self.d_lr_controller.current_lr
            if any(float(group['lr']) != expected for runtime in self.step_critics
                   for group in runtime.optimizer.param_groups):
                raise RuntimeError('Restored D optimizer LR disagrees with adaptive controller/base LR')
            result['adaptive_d_lr'] = 'restored' if controller_state is not None else 'explicit_legacy_bootstrap'
        return result
