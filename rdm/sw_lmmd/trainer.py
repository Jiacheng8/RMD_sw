"""The SW-LMMD training loop: bootstrap the window, then slide it one stride per step.

Each step, in order:

1. take the next window; split its ``B`` active row ids contiguously across ranks, then into
   micro-batches within the rank;
2. **pass 1 (no grad)** -- roll the student out on those rows' prompts, embed through the
   frozen battery, couple ``tau(c)``, cache the detached features;
3. **middle** -- all-gather the active features (detached), concatenate them behind the
   retained cache to form the ``K``-row generated context, read the same rows' reference
   window, and back-propagate the per-encoder active force into the cached features;
4. **pass 2 (grad)** -- re-generate each micro-batch and push the cached feature gradient
   through the generator;
5. reduce, clip, step, absorb the active features into the cache, slide.

Steps 2-4 are :func:`rdm.train.grad_caching.gradcache_backward` unchanged -- the same two-pass
machinery the global iRDM loss uses, with two SW-LMMD-specific settings:

``gather=False``
    the active features are gathered *inside* the loss with no autograd edge, because the
    generated context is stop-gradient by construction. Gradient reaches the generator only
    through this rank's own live rows.

``scale=1.0``
    the middle backward already runs once over the rank's whole local active set, so there is
    no per-micro-batch loss to average. (The spec's alternative -- per-micro-batch kernel means
    normalized by ``b*K`` -- needs an extra ``1/A``; the two schemes must not be mixed.)

Combined with a cross-rank gradient **mean**, the per-rank ``1/(B_local * K)`` normalization
telescopes to the global ``2/(BK)``, so one GPU, four 4090s and thirty-two H200s produce the
same parameter gradient for the same rows. ``tests/test_sw_lmmd_parity.py`` is the proof.

Per-encoder weights are uniform. The global iRDM self-normalization and PID floors are
deliberately not reused: they are defined for an MMD^2 scalar, whereas the force surrogate is
signed and passes through zero, so dividing by its own magnitude is not a scale-free
reparametrization of the same objective.

With a ``critic`` (:mod:`rdm.sw_lmmd.adversarial`), the middle step also takes one critic step
on the window and adds ``lambda * adversarial`` to the force before the backward. Passes 1 and
2 are unchanged. The logged ``force`` stays the force alone, and ``loss_total`` is what was
back-propagated.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from ..compare.grad_balance import clip_generator_grads
from ..representation.joint_feature import couple
from ..train.grad_caching import gradcache_backward
from ..utils.distributed import get_world_size
from .cache import GeneratedWindowCache
from .config import CacheConfig
from .local_mmd import ExactLocalMMD, within_group_kernel_mean
from .sharding import all_gather_detached, all_reduce_mean_, partition_rows, reduce_scalar, row_noise

logger = logging.getLogger(__name__)


class SWLMMDTrainer:
    """Sliding-window prompt-aligned local MMD distillation."""

    def __init__(self, generator, battery, store, schedule, optimizer, *, encoder_names,
                 noise_shape, encoder_weights=None, cache: GeneratedWindowCache | None = None,
                 micro_batch: int = 1, grad_clip: float = 4.0, kernel_block: int = 256,
                 seed: int = 0, joint: bool = True, grad_reduce: str = "mean",
                 monitor_every: int = 1, cache_cfg: CacheConfig | None = None,
                 clip_module=None, device: str = "cuda",
                 feature_dtype: torch.dtype = torch.float32,
                 context_dtype: torch.dtype | None = None, critic=None, lr_schedule=None,
                 unbiased_siblings: bool = False, sibling_weight: float = 1.0):
        self.generator = generator
        self.battery = battery
        self.store = store
        self.schedule = schedule
        self.optimizer = optimizer
        self.encoder_names = list(encoder_names)
        self.noise_shape = tuple(noise_shape)
        self.cache_cfg = cache_cfg or CacheConfig()
        self.cache = cache or GeneratedWindowCache(self.encoder_names,
                                                   store_dtype=self.cache_cfg.store_dtype)
        self.micro_batch = int(micro_batch)
        self.grad_clip = float(grad_clip)
        self.seed = int(seed)
        self.joint = bool(joint)
        self.grad_reduce = grad_reduce
        # The FSDP-wrapped module, when there is one: gradient clipping must reduce the norm
        # across shards or every rank clips by its own partial norm.
        self.clip_module = clip_module
        self.monitor_every = int(monitor_every)
        self.device = device
        self.feature_dtype = feature_dtype
        self.context_dtype = context_dtype or feature_dtype
        self.mmd = ExactLocalMMD(kernel_block)

        n = len(self.encoder_names)
        self.encoder_weights = dict(encoder_weights) if encoder_weights else \
            {name: 1.0 / n for name in self.encoder_names}      # uniform (spec sec. 14.1)
        self.sigmas = {name: store.sigma(name) for name in self.encoder_names}
        self.betas = {name: store.beta(name) for name in self.encoder_names}
        self.image_dims = {name: store.dim(name) for name in self.encoder_names}

        # Optional adversarial term. Its critic step splits the K window rows over ranks, so K
        # must divide like B does (K = 1024 divides every world size B = 128 does).
        self.critic = critic
        if critic is not None and schedule.window_size % get_world_size():
            raise ValueError(f"window size {schedule.window_size} is not divisible by "
                             f"world_size={get_world_size()}; the critic step shards the window")

        # Weighted same-prompt repulsion: each sibling term weighs G/(G-1) (unbiased_siblings, the
        # unbiased per-prompt estimator) or sibling_weight. Siblings are found as blocks of G
        # consecutive local active rows, so every rank's share of B must be whole groups.
        self.unbiased_siblings = bool(unbiased_siblings)
        self.sibling_weight = float(sibling_weight)
        if self.unbiased_siblings and self.sibling_weight != 1.0:
            raise ValueError("pass unbiased_siblings OR sibling_weight, not both")
        if self.unbiased_siblings or self.sibling_weight != 1.0:
            group = int(getattr(schedule, "group_size", 1))
            if group < 2:
                raise ValueError("sibling weighting (unbiased_siblings / sibling_weight) needs a "
                                 "prompt-grouped schedule (group_size >= 2)")
            if (schedule.stride // get_world_size()) % group:
                raise ValueError(f"sibling weighting needs each rank's {schedule.stride} // "
                                 f"{get_world_size()} active rows to be whole groups of {group}")

        # The generator's lr schedule (config.LRSchedule): a function of step_idx, applied to the
        # base lr each group was built with, so resuming it is resuming the step counter.
        self.lr_schedule = lr_schedule
        self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]

        self.step_idx = 0
        self._pending_window = None
        self._last_logs: dict = {}

    # ------------------------------------------------------------------ helpers
    @property
    def _gen_params(self):
        return [p for p in self.generator.model.parameters() if p.requires_grad]

    @property
    def epoch(self) -> int:
        return self.schedule.epoch

    def _chunks(self, row_ids: np.ndarray) -> list:
        """Split this rank's rows into fixed-size micro-batches (pass 1 / pass 2 see the same)."""
        ids = np.asarray(row_ids, dtype=np.int64)
        if ids.size % self.micro_batch:
            raise ValueError(f"{ids.size} rows on this rank is not divisible by micro_batch="
                             f"{self.micro_batch}")
        return [ids[i:i + self.micro_batch] for i in range(0, ids.size, self.micro_batch)]

    def _encode(self, row_ids) -> dict:
        """rows -> ``{encoder: (n, d_joint)}``: roll out, embed, couple ``tau(c)``.

        Builds the generator graph, so it must NOT be called under ``no_grad`` on the pass that
        needs gradient. The noise is derived per row id, so both GradCache passes -- and any
        later refresh of these rows -- regenerate the same samples.
        """
        ids = np.asarray(row_ids, dtype=np.int64)
        noise = row_noise(ids, self.noise_shape, seed=self.seed, epoch=self.epoch,
                          device=self.device, dtype=self.feature_dtype)
        cond = self.store.generator_context(ids, device=self.device, dtype=self.context_dtype)
        images = self.generator.sample(noise, cond)
        feats = self.battery(images, only=set(self.encoder_names))
        if not self.joint:
            return {name: feats[name] for name in self.encoder_names}
        tau = self.store.text_features(ids, device=self.device, dtype=self.feature_dtype)
        return {name: couple(feats[name], tau, self.betas[name]) for name in self.encoder_names}

    @torch.no_grad()
    def _encode_no_grad(self, row_ids_list) -> dict:
        """Micro-batched, gradient-free encode of a list of row-id chunks (bootstrap / refresh)."""
        parts: dict = {}
        for chunk in row_ids_list:
            for name, value in self._encode(chunk).items():
                parts.setdefault(name, []).append(value.detach())
        return {name: torch.cat(v, 0) for name, v in parts.items()}

    def _reference_window(self, encoder_name: str, row_ids) -> torch.Tensor:
        """The window's reference joint features.

        Read fresh from the memory-mapped store each step. ``K - B`` of these rows were also
        read last step; mirroring the generated cache with a rolling reference window would cut
        the per-step read by ``1 - B/K``. Deliberately not done yet -- it adds a second row
        alignment to keep honest, and the read is not the bottleneck next to a 4B rollout.
        """
        return self.store.reference_joint_features(encoder_name, row_ids, device=self.device,
                                                   dtype=self.feature_dtype, joint=self.joint)

    def _adversarial(self, force, local_feats: dict, window_feats: dict, window, local_active,
                     logs: dict):
        """Critic step on the window, then ``force + lambda * adversarial`` on the live rows.

        ``window_feats`` holds each critic encoder's ``(generated context, reference)``, both
        ``K`` rows in window order and detached. This rank trains the critic on its contiguous
        ``K/R`` share of them. The generator side then uses only this rank's live active rows,
        exactly like the force.
        """
        critic = self.critic
        share = partition_rows(np.arange(window.all_ids.size))
        lo, hi = int(share[0]), int(share[-1]) + 1
        tau = self.store.text_features(window.all_ids[lo:hi], device=self.device,
                                       dtype=self.feature_dtype) if critic.conditional else None
        logs.update(critic.update({n: ctx[lo:hi] for n, (ctx, _) in window_feats.items()},
                                  {n: ref[lo:hi] for n, (_, ref) in window_feats.items()}, tau))
        if not critic.generator_active(self.step_idx):
            return force
        tau = self.store.text_features(local_active, device=self.device,
                                       dtype=self.feature_dtype) if critic.conditional else None
        adv, adv_logs = critic.generator_loss(local_feats, tau)
        lam, bal_logs = critic.balance(force, adv, local_feats, self.image_dims)
        logs.update(adv_logs)
        logs.update(bal_logs)
        logs["_force"] = float(force.detach())
        return force + lam * adv

    # ------------------------------------------------------------------ bootstrap
    def bootstrap(self) -> dict:
        """Fill the cache with the first window's retained rows under the current parameters.

        No optimizer step happens here: the first *training* step then back-propagates only its
        ``B`` active rows, exactly like every later step, so per-step cost is constant from the
        very first update. Old and new rows both come from the same initial parameters.
        """
        window = self.schedule.next()
        self._pending_window = window
        local_ids = partition_rows(window.retained_ids)
        feats = self._encode_no_grad(self._chunks(local_ids))
        gathered = {name: all_gather_detached(feats[name]) for name in self.encoder_names}
        self.cache.initialize(window.retained_ids, gathered, step=-1)
        return {"bootstrap_rows": int(window.retained_ids.size),
                "bootstrap_rows_per_rank": int(local_ids.size)}

    def start_epoch(self) -> dict:
        """Advance to a new permutation / offset and re-bootstrap.

        The overlap identity does not survive a re-ordering, so the cache is dropped rather
        than carried across the boundary -- reusing it would pair generated features with rows
        that are no longer their neighbours.
        """
        cfg = getattr(self.schedule, "_epoch_cfg", None)
        epoch = self.schedule.new_epoch(**(cfg or {}))
        self.cache.clear()
        logs = self.bootstrap()
        return {"epoch": epoch, **logs}

    # ------------------------------------------------------------------ one step
    def _apply_lr_schedule(self) -> float:
        """Set this step's lr on every param group; return the first group's."""
        if self.lr_schedule is not None:
            factor = self.lr_schedule.factor(self.step_idx)
            for group, base in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base * factor
        return float(self.optimizer.param_groups[0]["lr"])

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
        group = getattr(self.schedule, "group_size", 1)
        grouped_now = group > 1 and getattr(window, "active_grouped", True)
        sibling_weight = 1.0       # an ungrouped (U) step of a mixed schedule has no siblings
        if grouped_now:
            sibling_weight = group / (group - 1) if self.unbiased_siblings else self.sibling_weight

        def loss_fn(local_feats: dict):
            total, logs, raws = None, {}, []
            window_feats = {}                 # critic encoders' (context, reference), if any
            for name in self.encoder_names:
                active = local_feats[name]
                glob = all_gather_detached(active)
                gathered[name] = glob
                retained = self.cache.retained(name, window.retained_ids)
                context = torch.cat([retained.to(device=glob.device, dtype=glob.dtype), glob], 0)
                reference = self._reference_window(name, window.all_ids)
                force, stats = self.mmd.active_force(active, context, reference,
                                                     self.sigmas[name], group=group,
                                                     sibling_weight=sibling_weight)
                total = (self.encoder_weights[name] * force) if total is None else \
                    total + self.encoder_weights[name] * force
                logs[f"{name}/force"] = float(stats["force"])
                logs[f"{name}/repulsion"] = float(stats["active_repulsion"])
                logs[f"{name}/attraction"] = float(stats["active_attraction"])
                if "sibling_repulsion" in stats:
                    logs[f"{name}/sibling_repulsion"] = float(stats["sibling_repulsion"])
                if monitor_now:
                    m = self.mmd.monitor(context, reference, self.sigmas[name])
                    logs.update(m.as_log(prefix=f"{name}/"))
                    raws.append(float(m.mmd2))
                    if grouped_now:
                        # same-prompt similarity of this step's prompts (grouped steps only:
                        # an ungrouped block has no siblings to compare)
                        k_gen = within_group_kernel_mean(glob, group, self.sigmas[name])
                        k_ref = within_group_kernel_mean(reference[-glob.shape[0]:], group,
                                                         self.sigmas[name])
                        logs[f"{name}/group_k_gen"] = float(k_gen)
                        logs[f"{name}/group_k_ref"] = float(k_ref)
                if self.critic is not None and name in self.critic.names:
                    window_feats[name] = (context, reference)
                del reference, context
            if raws:
                logs["raw_mmd2"] = float(np.mean(raws))      # uniform mean, NOT the trained scalar
            if self.critic is not None:
                total = self._adversarial(total, local_feats, window_feats, window,
                                          local_active, logs)
            return total, logs

        self.optimizer.zero_grad(set_to_none=True)
        loss_val, logs = gradcache_backward(chunks, self._encode, loss_fn,
                                            scale=1.0, gather=False)
        force_val = logs.pop("_force", loss_val)      # the force alone, without adversarial

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
        if getattr(self.schedule, "group_pattern", None):           # mixed schedule
            out["grouped_step"] = int(window.active_grouped)
        if self.unbiased_siblings or self.sibling_weight != 1.0:
            out["sibling_weight"] = sibling_weight
        if self.critic is not None:
            out["loss_total"] = reduce_scalar(loss_val)     # force + lambda * adversarial
        self._last_logs = out
        return out

    def train(self, num_steps: int, log_fn=None) -> int:
        if not self.cache.initialized:
            self.bootstrap()
        for _ in range(num_steps):
            logs = self.step()
            if log_fn is not None:
                log_fn(logs)
        return self.step_idx

    # ------------------------------------------------------------------ checkpointing
    def state_dict(self, with_cache: bool = True, with_optimizer: bool = True,
                   with_critic: bool = True) -> dict:
        """Model + schedule (+ optimizer) (+ cache) (+ critic). Without the cache a resume MUST
        re-bootstrap: carrying features produced by different parameters, or by a different
        permutation, silently trains against the wrong context.

        The critic entry (weights, standardization, AdamW moments) is replicated rather than
        sharded, so unlike the generator's optimizer it can be resumed under FSDP too.

        Under FSDP the model entry is a collective gather, so every rank must call this; only
        rank 0 receives the full weights."""
        state = {"model": self.generator.model.state_dict(),
                 "step": self.step_idx,
                 "schedule": self.schedule.state_dict(),
                 "world_size": get_world_size()}
        if self.unbiased_siblings:              # absent = the plain biased force (old files)
            state["unbiased_siblings"] = True
        if self.sibling_weight != 1.0:
            state["sibling_weight"] = self.sibling_weight
        if with_optimizer:
            state["optimizer"] = self.optimizer.state_dict()
        if with_cache:
            state["cache"] = self.cache.state_dict()
        if with_critic and self.critic is not None:
            state["critic"] = self.critic.state_dict()
        return state

    def load_state_dict(self, state: dict) -> dict:
        """Resume from :meth:`state_dict`: step counter and schedule position, plus whatever else
        the checkpoint carries -- optimizer moments, generated cache, model weights.

        With the cache, the next :meth:`step` continues exactly where the saved run stopped;
        without it the caller must :meth:`bootstrap`, which re-encodes the next window's
        retained rows under the restored parameters. The model weights are loaded here only
        without FSDP (``clip_module is None``); under FSDP the generator must be built from the
        same file (``load_from``), before it is wrapped. Returns what was restored.
        """
        if int(state.get("world_size", get_world_size())) != get_world_size():
            logger.warning("[sw_lmmd] resuming a %s-rank checkpoint on %d ranks: the window and the "
                           "gradient are world-size independent, the throughput is not",
                           state.get("world_size"), get_world_size())
        saved = bool(state.get("unbiased_siblings", False))
        if saved != self.unbiased_siblings:
            raise RuntimeError(f"unbiased_siblings mismatch on resume: the checkpoint was written "
                               f"with {saved}, this config has {self.unbiased_siblings}. Resume "
                               f"with the run's own config.")
        saved_w = float(state.get("sibling_weight", 1.0))
        if saved_w != self.sibling_weight:
            raise RuntimeError(f"sibling_weight mismatch on resume: the checkpoint was written "
                               f"with {saved_w}, this config has {self.sibling_weight}. Resume "
                               f"with the run's own config.")
        restored = {"step": int(state["step"]), "model": False, "optimizer": False, "cache": False}
        if self.clip_module is None and state.get("model") is not None:
            self.generator.model.load_state_dict(state["model"])
            restored["model"] = True
        self.schedule.load_state_dict(state["schedule"])
        self.step_idx = int(state["step"])
        self._pending_window = None
        if state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
            restored["optimizer"] = True
        self.cache.clear()
        if state.get("cache"):
            self.cache.load_state_dict(state["cache"], device=self.device)
            restored["cache"] = True
        if self.critic is not None:                     # absent -> the critic starts fresh
            restored["critic"] = state.get("critic") is not None
            if restored["critic"]:
                self.critic.load_state_dict(state["critic"])
        return restored
