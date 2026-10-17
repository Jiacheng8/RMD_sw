"""``SWLMMDTrainer`` with a switchable attraction reference side (local / random / global).

``step()`` is COPIED FROM rdm/sw_lmmd/trainer.py, ``SWLMMDTrainer.step`` (commit d5b2e9a): the
attraction is computed inside its loss closure and there is no hook to replace it alone. Every line
is the original except the ones marked ``# [ref-ablation]``:

* the attraction's reference rows come from ``ref_mode`` (:meth:`_attraction_rows`); the force is
  :class:`RefAblationLoss` -- in local mode the original computation, bit for bit;
* per-encoder diagnostics are added to the logs (``ref/<encoder>/...``, averaged over ranks).

Unchanged in every mode: the generated window and its cache, the B live rows, the repulsion, the
monitored ``mmd2`` / ``group_k_*`` (still against the WINDOW's references, so they compare across
modes), the critic (it keeps training on the window's references -- the GAN is part of the base
config in all three modes), GradCache, the gradient reduction, clipping, the lr schedule.

Everything else is inherited: the trainer is built by :func:`rdm.sw_lmmd.launch.build_trainer` and
turned into this class by :meth:`RefAblationTrainer.from_trainer`, so the generator, battery,
store, schedule, optimizer and critic are constructed exactly as for a normal run.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import torch
import torch.distributed as dist

from rdm.compare.grad_balance import clip_generator_grads
from rdm.compare.kernels import gamma_from_sigma, gaussian_gram
from rdm.sw_lmmd.local_mmd import at_least_fp32, blockwise_mean_nograd, within_group_kernel_mean
from rdm.sw_lmmd.sharding import (all_gather_detached, all_reduce_mean_, partition_rows,
                                  reduce_scalar)
from rdm.sw_lmmd.trainer import SWLMMDTrainer
from rdm.train.grad_caching import gradcache_backward
from rdm.utils.distributed import get_world_size, is_dist, is_main_process

from .config import REF_MODES
from .random_reference import RandomReferenceSampler
from .ref_loss import RefAblationLoss


def _rows_digest(rows) -> str | None:
    if rows is None:
        return None
    return hashlib.sha1(np.ascontiguousarray(rows, dtype=np.int64).tobytes()).hexdigest()[:12]


class RefAblationTrainer(SWLMMDTrainer):
    """SW-LMMD whose attraction is taken against ``ref_mode``'s reference (see module docstring)."""

    @classmethod
    def from_trainer(cls, trainer: SWLMMDTrainer, *, ref_mode: str, ref_seed: int = 0,
                     nystrom=None, grad_norms: bool = True) -> "RefAblationTrainer":
        """Take over a trainer built by ``build_trainer`` (it must not be used afterwards).

        The new object shares every attribute -- generator, battery, store, schedule, optimizer,
        cache, critic -- so nothing is constructed twice and nothing differs from a normal run.
        """
        if type(trainer) is not SWLMMDTrainer:
            raise TypeError(f"expected a plain SWLMMDTrainer, got {type(trainer).__name__}")
        obj = cls.__new__(cls)
        obj.__dict__.update(trainer.__dict__)
        obj._setup_ref_ablation(ref_mode, ref_seed, nystrom, grad_norms)
        return obj

    def _setup_ref_ablation(self, mode: str, seed: int, nystrom, grad_norms: bool) -> None:
        if mode not in REF_MODES:
            raise ValueError(f"ref_mode must be one of {REF_MODES}, got {mode!r}")
        if nystrom is not None:
            missing = [n for n in self.encoder_names if n not in nystrom.names]
            if missing:
                raise ValueError(f"Nystrom bundles missing for encoders {missing}")
        self.ref_mode = mode
        self.ref_seed = int(seed)
        self.nystrom = nystrom
        self.ref_loss = RefAblationLoss(mode, block_size=self.mmd.block_size, nystrom=nystrom)
        self.ref_sampler = RandomReferenceSampler(self.store.num_rows, self.schedule.window_size,
                                                  seed=seed) if mode == "random" else None
        self.log_grad_norms = bool(grad_norms)
        self._last_attraction_rows = None

    # ------------------------------------------------------------------ the reference side
    def _attraction_rows(self, window):
        """This step's attraction rows: the window (local), a fresh draw (random), none (global).
        One draw per step, shared by every encoder."""
        if self.ref_mode == "local":
            rows = window.all_ids
        elif self.ref_mode == "random":
            rows = self.ref_sampler.draw()
        else:
            rows = None
        self._last_attraction_rows = rows
        return rows

    def _attraction_reference(self, name: str, rows, window_reference: torch.Tensor):
        """The ``(K, d)`` reference the attraction sees. Random rows go through the same store
        read as the window, so each carries its own caption's ``beta * tau(c)``."""
        if self.ref_mode == "local":
            return window_reference
        if self.ref_mode == "random":
            return self._reference_window(name, rows)
        return None

    # ------------------------------------------------------------------ per-step diagnostics
    def _collect_ref_diagnostics(self, name, active, force, repel, attract, reference,
                                 means: dict, sums: dict) -> None:
        """This rank's values; :meth:`_reduce_ref_diagnostics` combines the ranks."""
        gamma = gamma_from_sigma(self.sigmas[name])
        p = f"ref/{name}/"
        means[p + "attraction"] = attract.detach()
        means[p + "repulsion"] = repel.detach()
        means[p + "force"] = force.detach()
        means[p + "force_weighted"] = self.encoder_weights[name] * force.detach()   # what is summed
        with torch.no_grad():
            a = at_least_fp32(active).detach()
            # the window's own (local) attraction, whatever is trained on: the common yardstick
            means[p + "attraction_local"] = blockwise_mean_nograd(a, reference, gamma,
                                                                  self.ref_loss.block_size)
            if self.nystrom is not None:
                means[p + "attraction_global"] = self.nystrom.attraction(name, a)
        if self.log_grad_norms:
            # In force units (force = 2 repel - 2 attract), image columns only -- the text block
            # is a constant and passes no gradient to the generator. autograd.grad leaves .grad
            # untouched, so the training gradient is not affected.
            d = self.image_dims[name]
            g_att = 2.0 * torch.autograd.grad(attract, active, retain_graph=True)[0][:, :d].double()
            g_rep = 2.0 * torch.autograd.grad(repel, active, retain_graph=True)[0][:, :d].double()
            sums[p + "grad_attraction"] = g_att.pow(2).sum()
            sums[p + "grad_repulsion"] = g_rep.pow(2).sum()
            sums[p + "grad_force"] = (g_rep - g_att).pow(2).sum()
            sums[p + "grad_dot"] = (g_att * g_rep).sum()

    def _reduce_ref_diagnostics(self, means: dict, sums: dict) -> dict:
        """One all-reduce for every diagnostic: values are cross-rank means (each rank's is a mean
        over its B/R rows, so this is the mean over all B); gradient norms are global norms divided
        by the world size, which makes them world-size invariant (each rank's force is normalized
        by 1/(B_local K), i.e. R times the single-rank per-row gradient)."""
        keys_m, keys_s = list(means), list(sums)
        if not keys_m and not keys_s:
            return {}
        values = [means[k] for k in keys_m] + [sums[k] for k in keys_s]
        device = values[0].device
        vec = torch.stack([torch.as_tensor(v, device=device).to(torch.float64).reshape(())
                           for v in values])
        world = get_world_size()
        if is_dist():
            if vec.is_cuda:
                torch.cuda.current_stream().synchronize()
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        flat = vec.tolist()
        out = {k: flat[i] / world for i, k in enumerate(keys_m)}
        s = dict(zip(keys_s, flat[len(keys_m):]))
        for name in self.encoder_names:
            p = f"ref/{name}/"
            if p + "grad_attraction" not in s:
                continue
            n_att, n_rep = math.sqrt(s[p + "grad_attraction"]), math.sqrt(s[p + "grad_repulsion"])
            out[p + "grad_attraction"] = n_att / world
            out[p + "grad_repulsion"] = n_rep / world
            out[p + "grad_force"] = math.sqrt(s[p + "grad_force"]) / world
            out[p + "grad_cos_att_rep"] = s[p + "grad_dot"] / max(n_att * n_rep, 1e-300)
        return out

    # ------------------------------------------------------------------ one step (copied)
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
        attraction_rows = self._attraction_rows(window)                           # [ref-ablation]

        def loss_fn(local_feats: dict):
            total, logs, raws = None, {}, []
            window_feats = {}                 # critic encoders' (context, reference), if any
            ref_means, ref_sums = {}, {}                                          # [ref-ablation]
            for name in self.encoder_names:
                active = local_feats[name]
                glob = all_gather_detached(active)
                gathered[name] = glob
                retained = self.cache.retained(name, window.retained_ids)
                context = torch.cat([retained.to(device=glob.device, dtype=glob.dtype), glob], 0)
                reference = self._reference_window(name, window.all_ids)
                # [ref-ablation] the attraction's reference side; `reference` (the window's rows)
                # still feeds the monitor and the critic below, as in the original
                force, stats, repel, attract = self.ref_loss.active_force_parts(
                    name, active, context,
                    self._attraction_reference(name, attraction_rows, reference),
                    self.sigmas[name], group=group, sibling_weight=sibling_weight)
                total = (self.encoder_weights[name] * force) if total is None else \
                    total + self.encoder_weights[name] * force
                logs[f"{name}/force"] = float(stats["force"])
                logs[f"{name}/repulsion"] = float(stats["active_repulsion"])
                logs[f"{name}/attraction"] = float(stats["active_attraction"])
                if "sibling_repulsion" in stats:
                    logs[f"{name}/sibling_repulsion"] = float(stats["sibling_repulsion"])
                self._collect_ref_diagnostics(name, active, force, repel, attract,  # [ref-ablation]
                                              reference, ref_means, ref_sums)
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
            logs.update(self._reduce_ref_diagnostics(ref_means, ref_sums))        # [ref-ablation]
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
        out["ref_mode"] = self.ref_mode                                           # [ref-ablation]
        out["ref/rows_sha1"] = _rows_digest(attraction_rows)                      # [ref-ablation]
        self._last_logs = out
        return out

    # ------------------------------------------------------------------ before training
    def attraction_diagnostic(self, window=None, random_draws: int = 8, exact_global: bool = True,
                              exact_chunk: int = 16384) -> dict:
        """The three attractions on ONE window under the CURRENT parameters.

        The window's active rows are rolled out once more without gradient -- the same per-row
        noise, hence the same images the next training step generates -- and each attraction
        bracket is averaged over all ``B`` of them:

        * ``local``: against the window's ``K`` references, split into the same-prompt part and
          the rest (``local_same_prompt + local_cross_prompt == local``);
        * ``random``: against ``random_draws`` independent uniform draws of ``K`` rows (a separate
          stream: the training draws are not consumed); in random mode also this run's own
          first draw (``random_step1``);
        * ``global``: the Nystrom ``psi(x)^T mu_bar`` and, with ``exact_global``, the exact
          ``(1/N) sum_n k(x, y_n)`` over all ``N`` rows, their relative error, and the cosine
          between the two gradients (rank 0 streams the store once per encoder);
        * ``repulsion``: against the generated window (the same in every mode), for scale.

        Gradient-free with respect to the training state: no optimizer step, the cache, the
        schedule and the training sampler are untouched.
        """
        window = window if window is not None else (self._pending_window or self.schedule.peek())
        names = self.encoder_names
        with torch.no_grad():
            local_ids = partition_rows(window.active_ids)
            feats = self._encode_no_grad(self._chunks(local_ids))
            x_all = {n: all_gather_detached(feats[n]) for n in names}
        diag = RandomReferenceSampler(self.store.num_rows, self.schedule.window_size,
                                      seed=self.ref_seed, stream="diagnostic")
        draws = [diag.draw() for _ in range(int(random_draws))]
        step1 = copy.deepcopy(self.ref_sampler).draw() if self.ref_sampler is not None else None
        pid_active = self.store.prompt_rows(window.active_ids)
        pid_window = self.store.prompt_rows(window.all_ids)

        out = {"ref_mode": self.ref_mode, "train_step": int(self.step_idx),
               "window_step": int(window.step), "epoch": int(window.epoch),
               "B": int(window.active_ids.size), "K": int(window.all_ids.size),
               "N": int(self.store.num_rows), "random_draws": int(random_draws), "encoders": {}}
        for name in names:
            x = at_least_fp32(x_all[name]).detach()
            gamma = gamma_from_sigma(self.sigmas[name])
            rec = {}
            with torch.no_grad():
                ref_local = self._reference_window(name, window.all_ids)
                retained = self.cache.retained(name, window.retained_ids)
                context = torch.cat([retained.to(device=x.device, dtype=x.dtype), x], 0)
                rec["repulsion"] = float(blockwise_mean_nograd(x, context, gamma))
                k_local = gaussian_gram(x, at_least_fp32(ref_local).to(x.dtype), gamma)
                same = torch.as_tensor(pid_active[:, None] == pid_window[None, :], device=x.device)
                denom = float(k_local.numel())
                rec["local"] = float(k_local.sum() / denom)
                rec["local_same_prompt"] = float((k_local * same).sum() / denom)
                rec["local_cross_prompt"] = float((k_local * ~same).sum() / denom)
                vals = [float(blockwise_mean_nograd(x, self._reference_window(name, r), gamma))
                        for r in draws]
                rec["random_mean"] = float(np.mean(vals))
                rec["random_std"] = float(np.std(vals))
                if step1 is not None:
                    rec["random_step1"] = float(blockwise_mean_nograd(
                        x, self._reference_window(name, step1), gamma))
                if self.nystrom is not None:
                    rec["global_nystrom"] = float(self.nystrom.attraction(name, x))
            if exact_global and is_main_process():
                rec.update(self._exact_global(name, x, gamma, exact_chunk))
            out["encoders"][name] = rec
        if is_dist():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.barrier()
        return out

    def _exact_global(self, name: str, x: torch.Tensor, gamma: float, chunk: int) -> dict:
        """Exact ``(1/N) sum_n k(x_i, y_n)`` over every reference row, and how the Nystrom
        attraction compares with it (value and gradient)."""
        n_rows, n_x = int(self.store.num_rows), x.shape[0]
        xg = x.detach().clone().requires_grad_(True)
        total = torch.zeros(n_x, dtype=torch.float64, device=x.device)
        with torch.enable_grad():
            for lo in range(0, n_rows, int(chunk)):
                ids = np.arange(lo, min(lo + int(chunk), n_rows), dtype=np.int64)
                y = self._reference_window(name, ids).to(x.dtype)
                k = gaussian_gram(xg, y, gamma).sum(1)
                total += k.detach().double()
                (k.sum() / float(n_x * n_rows)).backward()
        exact_rows = total / n_rows
        rec = {"global_exact": float(exact_rows.mean())}
        if self.nystrom is not None:
            d = self.image_dims[name]
            xn = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                rows = self.nystrom.attraction_rows(name, xn)
                g_nys = torch.autograd.grad(rows.mean(), xn)[0][:, :d].double()
            g_exact = xg.grad[:, :d].double()
            rel = (rows.detach().double() - exact_rows).abs() / exact_rows.clamp_min(1e-300)
            cos_rows = (g_nys * g_exact).sum(1) / (g_nys.norm(dim=1) * g_exact.norm(dim=1)).clamp_min(1e-300)
            rec.update({
                "global_rel_err_of_mean": abs(float(rows.detach().double().mean()) -
                                              float(exact_rows.mean())) / float(exact_rows.mean()),
                "global_rel_err_rows_median": float(rel.median()),
                "global_rel_err_rows_max": float(rel.max()),
                "global_grad_cos": float((g_nys * g_exact).sum() /
                                         (g_nys.norm() * g_exact.norm()).clamp_min(1e-300)),
                "global_grad_cos_rows_mean": float(cos_rows.mean()),
                "global_grad_norm_ratio": float(g_nys.norm() / g_exact.norm().clamp_min(1e-300)),
            })
        return rec

    # ------------------------------------------------------------------ checkpointing
    def state_dict(self, with_cache: bool = True, with_optimizer: bool = True,
                   with_critic: bool = True) -> dict:
        state = super().state_dict(with_cache=with_cache, with_optimizer=with_optimizer,
                                   with_critic=with_critic)
        state["ref_ablation"] = {
            "ref_mode": self.ref_mode, "ref_seed": self.ref_seed,
            "sampler": None if self.ref_sampler is None else self.ref_sampler.state_dict(),
            "nystrom": None if self.nystrom is None else self.nystrom.fingerprints()}
        return state

    def load_state_dict(self, state: dict) -> dict:
        saved = state.get("ref_ablation")
        if saved is None:
            raise RuntimeError("this checkpoint was not written by the reference-ablation trainer "
                               "(no 'ref_ablation' entry); resume it with rdm.sw_lmmd.launch")
        if saved["ref_mode"] != self.ref_mode or int(saved["ref_seed"]) != self.ref_seed:
            raise RuntimeError(f"ref_ablation mismatch on resume: the checkpoint was written with "
                               f"ref_mode={saved['ref_mode']} ref_seed={saved['ref_seed']}, this "
                               f"config has ref_mode={self.ref_mode} ref_seed={self.ref_seed}")
        if self.ref_mode == "global" and saved.get("nystrom") != self.nystrom.fingerprints():
            raise RuntimeError("the Nystrom bundles changed since this checkpoint was written "
                               f"({saved.get('nystrom')} -> {self.nystrom.fingerprints()}); the "
                               "global target would silently move")
        restored = super().load_state_dict(state)
        if self.ref_sampler is not None:
            self.ref_sampler.load_state_dict(saved["sampler"])
            restored["ref_sampler"] = True
        return restored
