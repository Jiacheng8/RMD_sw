#!/usr/bin/env python
"""Baseline: DMD2 (Yin et al., "Improved Distribution Matching Distillation", NeurIPS 2024) on the
one-step FLUX.2 [klein] 4B student, after the iRDM paper's re-implementation (arXiv 2607.02375,
sec. 4.2 and appendix E.2: GenEval 0.804 / PickScore 22.36 at its 500-step peak).

    torchrun --nproc_per_node=4 -m rdm.baselines.dmd2_flux \
        --reference-root <root>/sw_lmmd/reference_store --output-dir <root>/sw_lmmd/work_dirs
    (or: bash scripts/train_dmd2_flux.sh --root <root>)

Three networks, all the klein-4B MM-DiT:

* the **student** -- klein-4B sampled in one Euler step from t = 1 (``x = noise - v``), the
  untrained 1-step model of the paper's Table 2 (GenEval 0.474), as SW-LMMD starts from;
* the **real score** -- klein-4B itself, frozen: the 4-step teacher. It is guidance-distilled,
  so there is no classifier-free guidance (``x0 = x_t - t v``);
* the **fake score** -- a trainable copy of klein-4B, fit by flow matching to the student's
  current samples.

One iteration (two time-scale update rule: the fake score every iteration, the student every
``--ratio`` 5th):

1. ``--batch`` 128 prompts, drawn from the reference store's COCO captions (their stored Qwen3
   contexts, ctx_len 48 -- the prompts SW-LMMD trains on); the student renders x;
2. student turn: ``x_t = (1-t) x + t eps``; ``grad = (x0_fake - x0_real) / mean|x - x0_real|``
   (both scores without gradient), loss ``0.5 ||x - sg(x - grad)||^2``;
3. fake turn: the flow-matching loss ``||v_fake(x_t) - (eps - sg(x))||^2``.

Timesteps (appendix E.2): a uniform draw u ~ U[0.02, 0.98] mapped through klein's SNR shift
``t = e^mu / (e^mu + 1/u - 1)`` with the teacher's 4-step mu (2.03 at 512 px), the spacing of
its own sampling nodes. Uniform t puts too many draws at low noise, where the teacher returns a
mode-averaged posterior mean.

Recipe (appendix E.2): global batch 128 at 512 px, AdamW, lr 5e-7 for both trained networks
with a short linear warm-up, no EMA, no GAN term (the paper's GAN variant was metric-neutral and
not reported), no regression loss. AdamW (0.9, 0.999), weight decay 0.01 and gradient clipping
10 are DMD2's official SDXL values (the appendix does not give them).

Two differences from the paper, both deliberate: the prompts are COCO's (the paper's best run
drew LAION captions; it reports COCO-only prompts lower at every checkpoint), and the step
count is iterations (``--iterations`` 1000, checkpoints at 250/500/750/1000 -- its Table 8).

Checkpoints ``step_NNNNNNN.pth`` hold the student only, as ``{"model": ...}``, exactly what
``scripts/eval_checkpoint.sh`` evaluates (1 step). ``resume.pth`` also carries the fake score;
the optimizers restart (under FSDP their state is sharded per rank).
Memory: two trained 3.9 B models (fp32 masters) + one frozen, FSDP across all ranks --
4 x 80 GB with fp32 AdamW (~33 GB of state per card), or --optimizer adamw8bit on fewer cards.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdm.sw_lmmd.time_budget import TimeBudget
from rdm.compare.grad_balance import clip_generator_grads
from rdm.sw_lmmd.sharding import all_reduce_mean_, partition_rows, row_noise
from rdm.utils.distributed import get_rank, get_world_size, is_main_process

logger = logging.getLogger("rdm")


# ------------------------------------------------------------------------------ the score models
class ScoreModel(nn.Module):
    """A klein-4B MM-DiT as a score model: ``forward(x_t (B,128,h,w), t (B,), ctx) -> v`` with
    flux2's convention ``x_t = (1-t) x0 + t eps``, ``v = eps - x0``, so ``x0 = x_t - t v``."""

    def __init__(self, adapter: nn.Module):
        super().__init__()
        self.adapter = adapter

    def fsdp_unit_types(self) -> set:
        return self.adapter.fsdp_unit_types()

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        from einops import rearrange
        a = self.adapter
        h, w = x_t.shape[-2:]
        x, x_ids = a._batched_prc_img(x_t)
        c, c_ids = a._batched_prc_txt(ctx.to(device=x_t.device, dtype=x_t.dtype))
        v = a._run_dit(x, x_ids, c, c_ids, t.to(device=x.device, dtype=x.dtype))
        return rearrange(v, "b (h w) c -> b c h w", h=h, w=w)


def teacher_mu(image_resolution: int, teacher_steps: int = 4) -> float:
    """klein's SNR-shift mu for its own ``teacher_steps``-step schedule at this resolution."""
    from flux2.sampling import compute_empirical_mu
    seq_len = (image_resolution // 16) ** 2
    return compute_empirical_mu(seq_len, teacher_steps)


def shift_timesteps(u: torch.Tensor, mu: float) -> torch.Tensor:
    """flux2's ``generalized_time_snr_shift`` (sigma 1): ``e^mu / (e^mu + 1/u - 1)``."""
    return math.exp(mu) / (math.exp(mu) + (1.0 / u - 1.0))


def dmd_loss(x: torch.Tensor, x0_real: torch.Tensor, x0_fake: torch.Tensor) -> torch.Tensor:
    """DMD2's distribution-matching loss: its gradient w.r.t. ``x`` is
    ``(x0_fake - x0_real) / mean|x - x0_real|`` (per sample), averaged over elements."""
    p_real = (x - x0_real).detach()
    grad = (x0_fake - x0_real).detach() / p_real.abs().mean(dim=(1, 2, 3), keepdim=True)
    grad = torch.nan_to_num(grad)
    return 0.5 * F.mse_loss(x.float(), (x - grad).detach().float())


# ------------------------------------------------------------------------------ the trainer
class DMD2FluxTrainer:
    def __init__(self, student, real, fake, gen_opt, fake_opt, store, *, latent_shape, mu: float,
                 batch: int = 128, ratio: int = 5, lr: float = 5e-7, fake_lr: float = 5e-7,
                 warmup: int = 20, min_u: float = 0.02, max_u: float = 0.98, micro_batch: int = 4,
                 grad_clip: float = 10.0, student_clip_module=None, fake_clip_module=None,
                 seed: int = 0, device="cuda", amp: bool = True):
        world = get_world_size()
        if batch % (world * micro_batch):
            raise ValueError(f"--batch {batch} must split into {world} ranks x micro_batch {micro_batch}")
        self.student, self.real, self.fake = student, real, fake
        self.gen_opt, self.fake_opt, self.store = gen_opt, fake_opt, store
        self.latent_shape, self.mu = tuple(latent_shape), float(mu)
        self.batch, self.ratio, self.warmup = int(batch), int(ratio), int(warmup)
        self.lr, self.fake_lr = float(lr), float(fake_lr)
        self.min_u, self.max_u = float(min_u), float(max_u)
        self.micro_batch, self.grad_clip, self.seed = int(micro_batch), float(grad_clip), int(seed)
        self.student_clip_module, self.fake_clip_module = student_clip_module, fake_clip_module
        self.device, self.amp = device, bool(amp)
        self.iteration = 0

    # ---------------------------------------------------------------- helpers
    def _rng(self, *tag) -> np.random.Generator:
        return np.random.default_rng([self.seed, 104729, self.iteration, *tag])

    def _ctx(self, rows) -> torch.Tensor:
        return self.store.generator_context(rows, device=self.device, dtype=torch.float32)

    def _t(self, n: int, *tag) -> torch.Tensor:
        u = self.min_u + (self.max_u - self.min_u) * self._rng(*tag).random(n)
        return shift_timesteps(torch.from_numpy(u).to(self.device, torch.float32), self.mu)

    def _noise_like(self, x: torch.Tensor, *tag) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(int(self._rng(*tag).integers(2 ** 62)))
        return torch.randn(x.shape, generator=g, dtype=torch.float32).to(x.device, x.dtype)

    def _autocast(self):
        return torch.autocast("cuda" if str(self.device).startswith("cuda") else "cpu",
                              dtype=torch.bfloat16, enabled=self.amp)

    def _x0(self, model, x_t, t, ctx) -> torch.Tensor:
        with self._autocast():
            v = model(x_t, t, ctx)
        return x_t - t[:, None, None, None] * v.float()

    def _render(self, rows, with_grad: bool) -> torch.Tensor:
        """The student's one-step sample for these rows (per-row noise: rank/chunking-free)."""
        noise = row_noise(rows, self.latent_shape, seed=self.seed, epoch=self.iteration,
                          device=self.device, dtype=torch.float32)
        with torch.set_grad_enabled(with_grad), self._autocast():
            return self.student(noise, self._ctx(rows), {"num_steps": 1}).float()

    def _set_lr(self) -> float:
        f = min(1.0, (self.iteration + 1) / self.warmup) if self.warmup > 0 else 1.0
        for g in self.gen_opt.param_groups:
            g["lr"] = self.lr * f
        for g in self.fake_opt.param_groups:
            g["lr"] = self.fake_lr * f
        return f

    @staticmethod
    def _reduce_grads(module, clip_module):
        if clip_module is None and get_world_size() > 1:   # no FSDP: average the ranks here
            for p in module.parameters():
                if p.grad is not None:
                    all_reduce_mean_(p.grad)

    # ---------------------------------------------------------------- one iteration
    def step(self) -> dict:
        rows = np.sort(self._rng(0).choice(self.store.num_rows, size=self.batch, replace=False))
        local = partition_rows(rows)
        chunks = [local[i:i + self.micro_batch] for i in range(0, local.size, self.micro_batch)]
        gen_turn = self.iteration % self.ratio == 0
        warm = self._set_lr()
        logs, xs = {"iteration": self.iteration + 1, "student_step": gen_turn,
                    "lr_scale": warm}, []

        # ---- student turn
        if gen_turn:
            self.gen_opt.zero_grad(set_to_none=True)
            acc = 0.0
            for c, ids in enumerate(chunks):
                x = self._render(ids, with_grad=True)
                n, ctx = x.shape[0], self._ctx(ids)
                t = self._t(n, get_rank(), c, 2)
                x_t = (1 - t)[:, None, None, None] * x.detach() + t[:, None, None, None] * \
                    self._noise_like(x, get_rank(), c, 3)
                with torch.no_grad():
                    x0_real = self._x0(self.real, x_t, t, ctx)
                    x0_fake = self._x0(self.fake, x_t, t, ctx)
                loss = dmd_loss(x, x0_real, x0_fake)
                (loss / len(chunks)).backward()
                acc += float(loss.detach()) / len(chunks)
                xs.append(x.detach())
            self._reduce_grads(self.student, self.student_clip_module)
            gnorm = clip_generator_grads([p for p in self.student.parameters() if p.requires_grad],
                                         self.grad_clip, module=self.student_clip_module)
            stepped = math.isfinite(acc) and bool(torch.isfinite(gnorm))
            if stepped:
                self.gen_opt.step()
            logs.update({"loss_dmd": acc, "grad_norm_student": float(gnorm),
                         "student_skipped": not stepped})
        else:
            for ids in chunks:
                xs.append(self._render(ids, with_grad=False))

        # ---- fake turn: flow matching on the student's current samples
        self.fake_opt.zero_grad(set_to_none=True)
        acc = 0.0
        for c, (ids, x) in enumerate(zip(chunks, xs)):
            n = x.shape[0]
            t = self._t(n, get_rank(), c, 6)
            eps = self._noise_like(x, get_rank(), c, 7)
            x_t = (1 - t)[:, None, None, None] * x + t[:, None, None, None] * eps
            with self._autocast():
                v = self.fake(x_t, t, self._ctx(ids))
            loss = F.mse_loss(v.float(), eps - x)
            (loss / len(chunks)).backward()
            acc += float(loss.detach()) / len(chunks)
        self._reduce_grads(self.fake, self.fake_clip_module)
        fnorm = clip_generator_grads([p for p in self.fake.parameters() if p.requires_grad],
                                     self.grad_clip, module=self.fake_clip_module)
        if math.isfinite(acc) and bool(torch.isfinite(fnorm)):
            self.fake_opt.step()
        self.iteration += 1
        logs.update({"loss_fake": acc, "grad_norm_fake": float(fnorm)})
        return logs


# ------------------------------------------------------------------------------ build + run
def _optimizer(params, lr: float, kind: str):
    kw = dict(lr=lr, betas=(0.9, 0.999), weight_decay=0.01)    # DMD2's AdamW
    if kind == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, **kw)
    return torch.optim.AdamW(params, **kw)


def build(a, device):
    from rdm.representation.generators.flux_generator import Flux2AdapterModel
    from rdm.sw_lmmd.config import MemoryPolicy
    from rdm.sw_lmmd.launch import wrap_fsdp
    from rdm.sw_lmmd.reference_store import ReferenceFeatureStore
    from rdm.train.launch import load_generator_weights

    store = ReferenceFeatureStore(a.reference_root, [], require_context=True)
    fsdp = a.shard == "fsdp"
    policy = MemoryPolicy(shard=a.shard, optimizer=a.optimizer, param_dtype=torch.float32,
                          compute_dtype=torch.bfloat16 if fsdp else None,
                          micro_batch=a.micro_batch, grad_reduce="none" if fsdp else "mean")
    policy.validate()

    def shard(module, frozen: bool):
        if not fsdp:
            return module.to(device), None
        p = dataclasses.replace(policy, param_dtype=torch.bfloat16) if frozen else policy
        wrapped = wrap_fsdp(module, p, device)
        return wrapped, wrapped

    def klein(dtype, ckpt: bool):
        return Flux2AdapterModel(image_resolution=a.img_size, param_dtype=dtype,
                                 gradient_checkpointing=ckpt)

    resume = torch.load(a.resume, map_location="cpu", weights_only=False, mmap=True) \
        if a.resume else None
    # one model at a time, sharded before the next loads: a whole fp32 klein is 15.5 GB
    student_m = klein(torch.float32, ckpt=True)
    if resume is not None:
        load_generator_weights(student_m, a.resume, "flux", strict=True)
    latent_shape = (student_m.in_channels, student_m.input_size, student_m.input_size)
    student, s_clip = shard(student_m, frozen=False)

    real_m = ScoreModel(klein(torch.bfloat16, ckpt=False)).requires_grad_(False).eval()
    real, _ = shard(real_m, frozen=True)

    fake_m = ScoreModel(klein(torch.float32, ckpt=True))
    if resume is not None:
        fake_m.load_state_dict(resume["fake"], strict=True)
    fake, f_clip = shard(fake_m, frozen=False)

    mu = teacher_mu(a.img_size, a.teacher_steps)
    trainer = DMD2FluxTrainer(
        student, real, fake,
        _optimizer([p for p in student.parameters() if p.requires_grad], a.lr, a.optimizer),
        _optimizer([p for p in fake.parameters() if p.requires_grad], a.fake_lr, a.optimizer),
        store, latent_shape=latent_shape, mu=mu, batch=a.batch, ratio=a.ratio, lr=a.lr,
        fake_lr=a.fake_lr, warmup=a.warmup, micro_batch=a.micro_batch, grad_clip=a.grad_clip,
        student_clip_module=s_clip, fake_clip_module=f_clip, seed=a.seed, device=device)
    if resume is not None:
        trainer.iteration = int(resume["iteration"])
        del resume
    logger.info("[dmd2-flux] student klein-4B 1-step | real score klein-4B (frozen, no CFG) | fake "
                "score klein-4B copy | %d prompts x %d rows from %s | batch %d, ratio %d, lr %g / "
                "%g, warm-up %d | t = shift(U[0.02, 0.98]), mu %.3f | %s, %s", store.num_prompts,
                store.num_rows, a.reference_root, a.batch, a.ratio, a.lr, a.fake_lr, a.warmup, mu,
                a.shard, a.optimizer)
    return trainer


def checkpoint_iterations(total: int, every: int) -> set:
    return set(range(every, total + 1, every)) | {total}


def main() -> int:
    from rdm.utils.distributed import setup_distributed
    from rdm.utils.logging import setup_logging
    from rdm.utils.seed import fix_random_seeds

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--reference-root", required=True, help="SW-LMMD reference store (its COCO contexts)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--exp-name", default="dmd2-flux-klein")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--micro-batch", type=int, default=4, help="rows per forward, per rank")
    ap.add_argument("--ratio", type=int, default=5, help="fake updates per student update")
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--fake-lr", type=float, default=5e-7)
    ap.add_argument("--warmup", type=int, default=20, help="linear lr warm-up, iterations")
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--teacher-steps", type=int, default=4, help="the schedule mu is taken from")
    ap.add_argument("--shard", choices=["fsdp", "none"], default="fsdp")
    ap.add_argument("--optimizer", choices=["adamw", "adamw8bit"], default="adamw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="<run dir>/resume.pth")
    ap.add_argument("--budget-gpu-hours", type=float, default=0,
                    help="stop after this much GPU time; --iterations is then only an upper bound")
    ap.add_argument("--budget-calib", type=int, default=20, help="iterations timed to plan the budget")
    a = ap.parse_args()

    rank, world, local_rank = setup_distributed()
    setup_logging(rank=rank)
    fix_random_seeds(a.seed)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    run = Path(a.output_dir) / a.exp_name
    run.mkdir(parents=True, exist_ok=True)
    trainer = build(a, device)
    if is_main_process():
        (run / "args.json").write_text(json.dumps({**vars(a), "world": world, "mu": trainer.mu},
                                                  indent=2))

    def save(path, full: bool):
        # every rank builds the state: under FSDP the full state dict is a collective gather
        state = {"model": trainer.student.state_dict(), "step": trainer.iteration,
                 "iteration": trainer.iteration, "objective": "dmd2",
                 "sampling_args": {"num_steps": 1}}
        if full:
            state["fake"] = trainer.fake.state_dict()
        if is_main_process():
            tmp = str(path) + ".partial"
            torch.save(state, tmp)
            os.replace(tmp, path)
            logger.info("[dmd2-flux] checkpoint -> %s", path)

    saves = checkpoint_iterations(a.iterations, a.save_every)
    budget = TimeBudget(a.budget_gpu_hours, str(run), a.budget_calib, device=device) if a.budget_gpu_hours else None
    if budget is not None and budget.total:
        saves = checkpoint_iterations(budget.total, budget.checkpoint_every())
    log = open(run / "train_log.jsonl", "a") if is_main_process() else None
    t0 = time.time()
    while trainer.iteration < a.iterations:
        st = time.time()
        logs = trainer.step()
        logs["seconds"] = time.time() - st
        logs["gpus"] = world
        if is_main_process():
            log.write(json.dumps(logs) + "\n")
            log.flush()
            logger.info("it %d/%d%s | fake %.4f%s | %.1f s", logs["iteration"], a.iterations,
                        " [student]" if logs["student_step"] else "", logs["loss_fake"],
                        f" | dmd {logs['loss_dmd']:.3g} gnorm {logs['grad_norm_student']:.3g}"
                        if logs["student_step"] else "", logs["seconds"])
        stop = False
        if budget is not None:
            planned = budget.total
            stop = budget.after_step(trainer.iteration)
            if planned is None and budget.total:
                saves = checkpoint_iterations(budget.total, budget.checkpoint_every())
        if trainer.iteration in saves or stop:
            save(run / f"step_{trainer.iteration:07d}.pth", full=False)
            save(run / "resume.pth", full=True)
        if stop:
            break
    if log is not None:
        log.close()
    logger.info("[dmd2-flux] done: %d iterations in %.2f h on %d GPUs -> %s", trainer.iteration,
                (time.time() - t0) / 3600, world, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
