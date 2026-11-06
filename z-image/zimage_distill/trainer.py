"""SW-LMMD for a few-step student: every step's output gets its own local MMD and its own critic.

For an ``S``-step student the loss is

    L = sum_k  w_k * ( force_k + lambda_k * adv_k ),        k = 1 .. S

where ``force_k`` is the SW-LMMD active force of step ``k``'s ``x0_hat`` against the window's
reference rows (the same references for every step), ``adv_k`` the generator term of step ``k``'s
own critic, and ``lambda_k`` that critic's adaptive weight against ``force_k`` alone.

How it maps onto :class:`rdm.sw_lmmd.trainer.SWLMMDTrainer`
    Each (step, encoder) pair is a feature *stream* ``s{k}/{enc}`` (:mod:`zimage_distill.store`).
    The parent's machinery is keyed by ``encoder_names``, so passing the streams there gives
    each step its own generated window, its own cache, its own refresh and drift numbers, with
    no change to the cache, the refresh, GradCache or the training loop. ``_encode`` rolls the
    student out once and returns every step's features; GradCache's pass 2 pushes all of them
    back in one backward. With ``detach_between_steps`` the student cuts the graph between
    steps, so step ``k``'s loss reaches only step ``k``'s forward.

    ``encoder_weights[s{k}/{enc}] = w_k / E``: uniform over encoders within a step, as in the
    1-step runs, times the step weight.

    The critics: one :class:`rdm.sw_lmmd.adversarial.AdversarialCritic` per step, each built over
    the plain encoder names. Step ``k``'s critic trains on step ``k``'s generated window against
    the references and scores step ``k``'s live rows; the parent's ``_adversarial`` does exactly
    that once it is handed step ``k``'s features under the plain names.
"""
from __future__ import annotations

import numpy as np
import torch

from rdm.compare.grad_balance import clip_generator_grads
from rdm.representation.joint_feature import couple
from rdm.sw_lmmd.sharding import all_gather_detached, all_reduce_mean_, partition_rows, \
    reduce_scalar, row_noise
from rdm.sw_lmmd.trainer import SWLMMDTrainer
from rdm.train.grad_caching import gradcache_backward
from rdm.utils.distributed import get_world_size

from .model import reference_pixels
from .store import StepStreamStore, split_stream, stream_name, stream_names

#: The console line of ``rdm.sw_lmmd.launch.train`` prints these unprefixed keys; they are
#: copied from the LAST step's critic (the image the student delivers).
_PRINTED_GAN_KEYS = ("gan/d_acc", "gan/d_real", "gan/d_fake", "gan/lambda")


class MultiStepTrainer(SWLMMDTrainer):
    """Sliding-window local MMD + one critic per student step."""

    def __init__(self, generator, battery, store, schedule, optimizer, *, encoders, num_steps: int,
                 step_weights=None, critics=None, feature_pool: int = 1,
                 feature_quantize: bool = False, use_store_context=None, **kw):
        if kw.get("critic") is not None:
            raise ValueError("pass one critic per step as critics=[...], not critic=")
        if kw.get("unbiased_siblings") or kw.get("sibling_weight", 1.0) != 1.0 \
                or getattr(schedule, "group_size", 1) != 1:
            raise ValueError("the multi-step trainer is ungrouped (as K128 · GAN)")
        self.base_names = list(encoders)
        self.num_steps = int(num_steps)
        weights = [1.0] * self.num_steps if step_weights is None else [float(w) for w in step_weights]
        if len(weights) != self.num_steps:
            raise ValueError(f"{len(weights)} step weights for {self.num_steps} steps")
        n = len(self.base_names)
        encoder_weights = {stream_name(k, e): weights[k - 1] / n
                           for k in range(1, self.num_steps + 1) for e in self.base_names}
        super().__init__(generator, battery, StepStreamStore(store), schedule, optimizer,
                         encoder_names=stream_names(self.num_steps, self.base_names),
                         encoder_weights=encoder_weights, critic=None, **kw)
        self.step_weights = weights
        # The references were encoded from 1024 px PNGs ADM-cropped to 512 (a 2x2 BOX reduce in
        # uint8): reference_pixels() reproduces that on the decoded images, so the encoders and
        # the critic see the two sides through one pipeline.
        self.feature_pool = int(feature_pool)
        self.feature_quantize = bool(feature_quantize)
        if use_store_context is None:
            use_store_context = bool(getattr(store, "has_context", False))
        self.use_store_context = bool(use_store_context)
        # the critics' balance() reads the image width by plain encoder name
        self.image_dims.update({e: store.dim(e) for e in self.base_names})

        self.step_critics = list(critics or [])
        if self.step_critics:
            if len(self.step_critics) != self.num_steps:
                raise ValueError(f"{len(self.step_critics)} critics for {self.num_steps} steps: "
                                 f"every step has its own")
            if any(getattr(c, "block", None) or getattr(c, "per_kind", False)
                   for c in self.step_critics):
                raise ValueError("only the row critic (gan.critic: row) is wired per step")
            if schedule.window_size % get_world_size():
                raise ValueError(f"window size {schedule.window_size} is not divisible by "
                                 f"world_size={get_world_size()}; the critic step shards the window")

    # ------------------------------------------------------------------ rollout
    def _encode(self, row_ids) -> dict:
        """rows -> ``{s{k}/{enc}: (n, d_joint)}`` for every step ``k``, from ONE rollout.

        The noise is ``(n, S, C, h, w)`` drawn per row id, so both GradCache passes and any
        refresh replay the same rollout, re-noising included.
        """
        ids = np.asarray(row_ids, dtype=np.int64)
        noise = row_noise(ids, self.noise_shape, seed=self.seed, epoch=self.epoch,
                          device=self.device, dtype=self.feature_dtype)
        if self.use_store_context:
            images = self.generator.sample_steps(noise, contexts=self.store.generator_context(
                ids, device=self.device, dtype=self.context_dtype))
        else:
            images = self.generator.sample_steps(noise, prompts=self.store.prompts(ids))
        if len(images) != self.num_steps:
            raise RuntimeError(f"the generator returned {len(images)} steps, expected "
                               f"{self.num_steps}")
        tau = self.store.text_features(ids, device=self.device, dtype=self.feature_dtype) \
            if self.joint else None
        out = {}
        for k, img in enumerate(images, 1):
            if self.feature_pool > 1 or self.feature_quantize:
                img = reference_pixels(img, self.feature_pool, self.feature_quantize)
            feats = self.battery(img, only=set(self.base_names))
            for e in self.base_names:
                name = stream_name(k, e)
                out[name] = couple(feats[e], tau, self.betas[name]) if self.joint else feats[e]
        return out

    # ------------------------------------------------------------------ per-step critic
    def _step_adversarial(self, k: int, force, local_feats: dict, window_feats: dict, window,
                          local_active, logs: dict):
        """``force_k + lambda_k * adv_k`` with step ``k``'s critic (the parent's ``_adversarial``
        on step ``k``'s features under the plain encoder names)."""
        local = {e: local_feats[stream_name(k, e)] for e in self.base_names}
        self.critic = self.step_critics[k - 1]
        try:
            out = self._adversarial(force, local, window_feats, window, local_active, logs)
        finally:
            self.critic = None
        logs.pop("_force", None)
        return out

    # ------------------------------------------------------------------ one step
    def step(self) -> dict:
        if not self.cache.initialized:
            raise RuntimeError("call bootstrap() before the first step -- the retained half of "
                               "the first window has to exist before it can be compared against")
        lr = self._apply_lr_schedule()
        window = self._pending_window or self.schedule.next()
        self._pending_window = None
        local_active = partition_rows(window.active_ids)
        chunks = self._chunks(local_active)

        gathered: dict = {}
        monitor_now = self.monitor_every > 0 and (self.step_idx % self.monitor_every == 0)
        steps = range(1, self.num_steps + 1)

        def loss_fn(local_feats: dict):
            logs, raws = {}, {k: [] for k in steps}
            forces = {k: None for k in steps}
            window_feats = {k: {} for k in steps}    # step k critic's (context, reference)
            for name in self.encoder_names:
                k, enc = split_stream(name)
                active = local_feats[name]
                glob = all_gather_detached(active)
                gathered[name] = glob
                retained = self.cache.retained(name, window.retained_ids)
                context = torch.cat([retained.to(device=glob.device, dtype=glob.dtype), glob], 0)
                reference = self._reference_window(name, window.all_ids)
                force, stats = self.mmd.active_force(active, context, reference, self.sigmas[name])
                weighted = self.encoder_weights[name] * force
                forces[k] = weighted if forces[k] is None else forces[k] + weighted
                logs[f"{name}/force"] = float(stats["force"])
                logs[f"{name}/repulsion"] = float(stats["active_repulsion"])
                logs[f"{name}/attraction"] = float(stats["active_attraction"])
                if monitor_now:
                    m = self.mmd.monitor(context, reference, self.sigmas[name])
                    logs.update(m.as_log(prefix=f"{name}/"))
                    raws[k].append(float(m.mmd2))
                if self.step_critics and enc in self.step_critics[k - 1].names:
                    window_feats[k][enc] = (context, reference)
                del reference, context

            total, force_sum = None, 0.0
            for k in steps:
                term = forces[k]
                force_sum += float(term.detach())
                logs[f"s{k}/force"] = float(term.detach())
                if raws[k]:
                    logs[f"s{k}/raw_mmd2"] = float(np.mean(raws[k]))
                if self.step_critics:
                    sub: dict = {}
                    term = self._step_adversarial(k, term, local_feats, window_feats[k], window,
                                                  local_active, sub)
                    logs.update({f"s{k}/{key}": v for key, v in sub.items()})
                total = term if total is None else total + term
            if monitor_now:
                logs["raw_mmd2"] = logs[f"s{self.num_steps}/raw_mmd2"]     # the delivered step
            logs["_force"] = force_sum
            return total, logs

        self.optimizer.zero_grad(set_to_none=True)
        loss_val, logs = gradcache_backward(chunks, self._encode, loss_fn,
                                            scale=1.0, gather=False)
        force_val = logs.pop("_force", loss_val)      # the forces alone, without adversarial

        if self.grad_reduce == "mean":
            for p in self._gen_params:                # no DDP/FSDP wrapper -> reduce here
                if p.grad is not None:
                    all_reduce_mean_(p.grad)
        grad_norm = clip_generator_grads(self._gen_params, self.grad_clip,
                                         module=self.clip_module) if self.grad_clip > 0 else None

        finite = np.isfinite(loss_val) and (grad_norm is None or torch.isfinite(grad_norm))
        if finite:
            self.optimizer.step()
        else:
            self.optimizer.zero_grad(set_to_none=True)      # skip a non-finite step

        self.cache.advance(window.all_ids, int(window.retained_ids.size), gathered,
                           step=window.step)
        self.step_idx += 1
        out = {"step": self.step_idx, "window_step": window.step, "epoch": window.epoch,
               "force": reduce_scalar(force_val), "skipped": not finite,
               "grad_norm": float(grad_norm) if grad_norm is not None else None, "lr": lr,
               "cache_max_age": self.cache.max_age(window.step),
               "active_rows": int(window.active_ids.size),
               "active_rows_per_rank": int(local_active.size), **logs}
        for key in _PRINTED_GAN_KEYS:
            if f"s{self.num_steps}/{key}" in logs:
                out[key] = logs[f"s{self.num_steps}/{key}"]
        if self.step_critics:
            out["loss_total"] = reduce_scalar(loss_val)     # sum_k forces + lambda_k * adv_k
        self._last_logs = out
        return out

    # ------------------------------------------------------------------ checkpointing
    def state_dict(self, with_cache: bool = True, with_optimizer: bool = True,
                   with_critic: bool = True) -> dict:
        state = super().state_dict(with_cache=with_cache, with_optimizer=with_optimizer,
                                   with_critic=False)
        state["num_steps"] = self.num_steps
        state["sampling_args"] = dict(getattr(self.generator, "sampling_args", {}) or {})
        if with_critic and self.step_critics:
            state["critics"] = [c.state_dict() for c in self.step_critics]
        return state

    def load_state_dict(self, state: dict) -> dict:
        saved_steps = int(state.get("num_steps", self.num_steps))
        if saved_steps != self.num_steps:
            raise RuntimeError(f"checkpoint is a {saved_steps}-step student, this config has "
                               f"{self.num_steps} steps")
        restored = super().load_state_dict(state)
        if self.step_critics:
            saved = state.get("critics")
            restored["critic"] = saved is not None
            if saved is not None:
                if len(saved) != len(self.step_critics):
                    raise RuntimeError(f"checkpoint has {len(saved)} critics, this run "
                                       f"{len(self.step_critics)}")
                for critic, s in zip(self.step_critics, saved):
                    critic.load_state_dict(s)
        return restored
