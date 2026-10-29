#!/usr/bin/env python
"""Baseline: DMD2 (Yin et al., "Improved Distribution Matching Distillation", NeurIPS 2024) on the same
2-step Z-Image student, with the recipe of the official SDXL run.

    PYTHONPATH=z-image:. torchrun --nproc_per_node=8 -m zimage_distill.dmd2 \
        z-image/configs/zimage_2step_1024_h100.yaml --latents <dir> \
        --match-run <root>/zimage/work_dirs/zimage-2step-1024          # same GPU time as ours

Three networks, all the Z-Image DiT:

* the **student** -- ours, initialized from Z-Image-Turbo, 2 steps (sigmas [1, 0.75], re-noise);
* the **real score** -- Z-Image (the undistilled base), frozen, with classifier-free guidance
  (``pos + g (pos - neg)``, the pipeline's form; ``--guidance`` 4, Z-Image's recommended 3-5; the
  empty prompt is the negative);
* the **fake score** -- a trainable copy of the base, fit to the student's current samples, plus a
  GAN head on its middle block's image tokens (4x4 stride-2 conv + GroupNorm + SiLU blocks, pooled,
  linear -- DMD2's head, moved from the UNet bottleneck to the DiT's middle layer).

One iteration (DMD2's two time-scale update rule: the student every ``--ratio`` 5 iterations):

1. ``--batch`` 128 prompts; each micro-batch is rolled out to a random student step k (backward
   simulation: the steps before k without gradient, as at inference) -> x;
2. student turn: ``x_t = (1-s) x + s eps``, s ~ U[0.02, 0.98];
   ``grad = (x0_fake - x0_real) / mean|x - x0_real|`` (both scores without gradient), DMD loss
   ``0.5 ||x - sg(x - grad)||^2``; plus ``--gen-gan`` 5e-3 x softplus(-D(fake features of x noised
   at s ~ U[0, 1))) -- that term back-propagates through the fake score into x;
3. fake turn (every iteration): the flow-matching loss of the fake score on sg(x) noised at
   s ~ U[0.02, 0.98], plus ``--guidance-gan`` 1e-2 x the discriminator loss, real = Z-Image-Turbo
   reference renders (``dmd2_latents``, the data ours trains with), both noised at s ~ U[0, 1).

Recipe (DMD2's SDXL script): lr 5e-7 for the student and the fake score, AdamW (0.9, 0.999),
weight decay 0.01, constant lr, gradient clipping 10, batch 128, ratio 5, GAN weights 5e-3 / 1e-2,
diffusion GAN over the whole noise range. The SDXL run's real-score guidance 8 is SDXL's scale;
Z-Image's is 3-5. No regression loss, no ODE pairs (DMD2 drops them).

Same GPU time as ours, as rdm_global: the first iterations are timed (whole update-ratio cycles)
and the count is set to the budget; ``plan.json`` keeps it. Checkpoints are ours' format (the
student only) for eval_qib; ``resume.pth`` also carries the fake score and the GAN head.
Memory: three 6 B models, two of them trained -- 4 x 80 GB at least (FSDP), 8 recommended.
"""
from __future__ import annotations

import argparse
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

from rdm.compare.grad_balance import clip_generator_grads
from rdm.sw_lmmd.sharding import all_reduce_mean_, partition_rows, row_noise
from rdm.utils.distributed import get_rank, get_world_size, is_dist, is_main_process

from .rdm_global import checkpoint_steps, gpu_seconds_of_run

logger = logging.getLogger("rdm")
BASE_REPO = "Tongyi-MAI/Z-Image"


# ------------------------------------------------------------------------------ the score models
class ScoreModel(nn.Module):
    """A Z-Image DiT as a score model: ``forward(x_t, sigma (B,), caps) -> out`` with ``out = -v``,
    so ``x0_hat = x_t + sigma * out``. With ``feature_layer``, the output of that transformer block
    is kept (``self.features``: its first ``h*w/4`` tokens are the image's, row-major)."""

    def __init__(self, transformer: nn.Module, gradient_checkpointing: bool = False,
                 feature_layer: int | None = None):
        super().__init__()
        self.transformer = transformer
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.features = None
        if feature_layer is not None:
            self.transformer.layers[feature_layer].register_forward_hook(self._keep)

    def _keep(self, module, inputs, output):
        self.features = output[0] if isinstance(output, tuple) else output

    def fsdp_unit_types(self) -> set:
        return {type(self.transformer.layers[0])}

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, cap_feats: list,
                return_features: bool = False):
        """``out``; with ``return_features``, ``(out, features)``. A loss on the features must
        take them from here, not from ``self.features``: under FSDP the root unit (embedders)
        enters its backward through the tensors its forward returned, and features that reach
        the loss by a side door would bring gradient to the root's parameters while it is idle."""
        ts = sigma.to(torch.float32) * 1000                          # the pipeline's t, per sample
        t = (1000 - ts) / 1000
        out = self.transformer(list(x.unsqueeze(2).unbind(0)), t, cap_feats, return_dict=False)[0]
        out = torch.stack(out, 0).squeeze(2).to(x.dtype)
        return (out, self.features) if return_features else out


class GANHead(nn.Module):
    """DMD2's classification branch: image tokens (B, h*w, D) -> conv blocks -> pool -> one logit."""

    def __init__(self, dim: int, width: int = 256, blocks: int = 4):
        super().__init__()
        layers, c = [], dim
        for _ in range(blocks):
            layers += [nn.Conv2d(c, width, 4, 2, 1), nn.GroupNorm(32, width), nn.SiLU()]
            c = width
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(width, 1)

    def forward(self, tokens: torch.Tensor, side: int) -> torch.Tensor:
        b, _, d = tokens.shape
        x = tokens[:, :side * side].float().transpose(1, 2).reshape(b, d, side, side)
        return self.out(self.net(x).mean(dim=(2, 3))).squeeze(1)


def dmd_loss(x: torch.Tensor, x0_real: torch.Tensor, x0_fake: torch.Tensor) -> torch.Tensor:
    """DMD2's distribution-matching loss: its gradient w.r.t. ``x`` is
    ``(x0_fake - x0_real) / mean|x - x0_real|`` (per sample), averaged over elements."""
    p_real = (x - x0_real).detach()
    grad = (x0_fake - x0_real).detach() / p_real.abs().mean(dim=(1, 2, 3), keepdim=True)
    grad = torch.nan_to_num(grad)
    return 0.5 * F.mse_loss(x.float(), (x - grad).detach().float())


# ------------------------------------------------------------------------------ the trainer
class DMD2Trainer:
    def __init__(self, student, real, fake, head, gen_opt, fake_opt, store, latents, neg_ctx, *,
                 sigmas, renoise: bool = True, batch: int = 128, ratio: int = 5,
                 guidance: float = 4.0, gen_gan: float = 5e-3, guidance_gan: float = 1e-2,
                 min_sigma: float = 0.02, max_sigma: float = 0.98, micro_batch: int = 1,
                 grad_clip: float = 10.0, student_clip_module=None, fake_clip_module=None,
                 seed: int = 0, device="cuda", amp: bool = True):
        world = get_world_size()
        if batch % (world * micro_batch):
            raise ValueError(f"--batch {batch} must split into {world} ranks x micro_batch {micro_batch}")
        self.student, self.real, self.fake, self.head = student, real, fake, head
        self.gen_opt, self.fake_opt = gen_opt, fake_opt
        self.store, self.latents, self.neg_ctx = store, latents, neg_ctx
        # the real latents are indexed by prompt id, the contexts by reference row: prompt -> row
        prompt_of_row = np.asarray(store.prompt_rows(np.arange(store.num_rows)), dtype=np.int64)
        self._row_of_prompt = np.full(int(prompt_of_row.max()) + 1, -1, dtype=np.int64)
        self._row_of_prompt[prompt_of_row] = np.arange(prompt_of_row.size)
        self.sigmas, self.renoise = [float(s) for s in sigmas], bool(renoise)
        self.batch, self.ratio, self.guidance = int(batch), int(ratio), float(guidance)
        self.gen_gan, self.guidance_gan = float(gen_gan), float(guidance_gan)
        self.min_sigma, self.max_sigma = float(min_sigma), float(max_sigma)
        self.micro_batch, self.grad_clip, self.seed = int(micro_batch), float(grad_clip), int(seed)
        self.student_clip_module, self.fake_clip_module = student_clip_module, fake_clip_module
        self.device, self.amp = device, bool(amp)
        self.iteration = 0

    # ---------------------------------------------------------------- helpers
    def _rng(self, *tag) -> np.random.Generator:
        return np.random.default_rng([self.seed, 104729, self.iteration, *tag])

    def _ctx(self, rows) -> list:
        """The rows' stored Qwen3 contexts, as fp32 (lossless from the stored bf16), as ours does."""
        return self.store.generator_context(rows, device=self.device, dtype=torch.float32)

    def _sig(self, n: int, lo: float, hi: float, *tag) -> torch.Tensor:
        u = torch.from_numpy(self._rng(*tag).random(n)).to(self.device, torch.float32)
        return lo + (hi - lo) * u

    def _noise_like(self, x: torch.Tensor, *tag) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(int(self._rng(*tag).integers(2 ** 62)))
        return torch.randn(x.shape, generator=g, dtype=torch.float32).to(x.device, x.dtype)

    def _autocast(self):
        return torch.autocast("cuda" if str(self.device).startswith("cuda") else "cpu",
                              dtype=torch.bfloat16, enabled=self.amp)

    def _x0(self, model, x_t, sigma, caps) -> torch.Tensor:
        with self._autocast():
            out = model(x_t, sigma, caps)
        return x_t + sigma[:, None, None, None] * out.float()

    def _logit(self, x: torch.Tensor, sigma: torch.Tensor, caps: list, eps: torch.Tensor):
        x_t = (1 - sigma)[:, None, None, None] * x + sigma[:, None, None, None] * eps
        with self._autocast():
            _, feats = self.fake(x_t, sigma, caps, return_features=True)
        return self.head(feats, side=x.shape[-1] // 2)

    def _steps_for(self, chunk: int) -> int:
        """Backward simulation's random step k for a chunk: the same on every rank (no rank in
        the seed) -- under FSDP every student step is a collective, so the ranks must agree."""
        return 1 + int(self._rng(chunk, 1).integers(len(self.sigmas)))

    def _rollout(self, rows, chunk: int, with_grad: bool) -> torch.Tensor:
        """Backward simulation: the student to a random step k of its schedule (the earlier ones
        without gradient); returns that step's x0_hat."""
        k = self._steps_for(chunk)
        noise = row_noise(rows, (len(self.sigmas), *self.latents.shape), seed=self.seed,
                          epoch=self.iteration, device=self.device, dtype=torch.float32)
        args = {"sigmas": self.sigmas[:k], "renoise": self.renoise, "detach_between_steps": True}
        with torch.set_grad_enabled(with_grad), self._autocast():
            return self.student(noise, self._ctx(rows), args)[-1].float()

    # ---------------------------------------------------------------- one iteration
    def step(self) -> dict:
        world = get_world_size()
        rows = np.sort(self._rng(0).choice(self.store.num_rows, size=self.batch, replace=False))
        local = partition_rows(rows)
        chunks = [local[i:i + self.micro_batch] for i in range(0, local.size, self.micro_batch)]
        gen_turn = self.iteration % self.ratio == 0
        logs, xs = {"iteration": self.iteration + 1, "student_step": gen_turn}, []

        # ---- student turn
        if gen_turn:
            self.gen_opt.zero_grad(set_to_none=True)
            acc = {"dmd": 0.0, "gan_g": 0.0}
            for c, ids in enumerate(chunks):
                x = self._rollout(ids, c, with_grad=True)
                n = x.shape[0]
                caps = self._ctx(ids)
                s = self._sig(n, self.min_sigma, self.max_sigma, get_rank(), c, 2)
                x_t = (1 - s)[:, None, None, None] * x + s[:, None, None, None] * \
                    self._noise_like(x, get_rank(), c, 3)
                with torch.no_grad():
                    xd = x_t.detach()
                    pos = self._x0(self.real, xd, s, caps)
                    neg = self._x0(self.real, xd, s, [self.neg_ctx] * n)
                    x0_real = pos + self.guidance * (pos - neg)
                    x0_fake = self._x0(self.fake, xd, s, caps)
                loss_dm = dmd_loss(x, x0_real, x0_fake)
                sg = self._sig(n, 0.0, 1.0, get_rank(), c, 4)
                loss_g = F.softplus(-self._logit(x, sg, caps, self._noise_like(x, get_rank(), c, 5))).mean()
                ((loss_dm + self.gen_gan * loss_g) / len(chunks)).backward()
                acc["dmd"] += float(loss_dm.detach()) / len(chunks)
                acc["gan_g"] += float(loss_g.detach()) / len(chunks)
                xs.append(x.detach())
            if self.student_clip_module is None:            # no FSDP: average the ranks here
                for p in self.student.parameters():
                    if p.grad is not None:
                        all_reduce_mean_(p.grad)
            gnorm = clip_generator_grads([p for p in self.student.parameters() if p.requires_grad],
                                         self.grad_clip, module=self.student_clip_module)
            if math.isfinite(acc["dmd"]) and bool(torch.isfinite(gnorm)):
                self.gen_opt.step()
            logs.update({"loss_dmd": acc["dmd"], "loss_gan_g": acc["gan_g"],
                         "grad_norm_student": float(gnorm)})
        else:
            for c, ids in enumerate(chunks):
                xs.append(self._rollout(ids, c, with_grad=False))

        # ---- fake turn (the student turn's GAN term reached the fake score too: drop it)
        self.fake_opt.zero_grad(set_to_none=True)
        acc = {"fake": 0.0, "gan_d": 0.0, "d_real": 0.0, "d_fake": 0.0}
        avail = self.latents.available
        for c, (ids, x) in enumerate(zip(chunks, xs)):
            n, caps = x.shape[0], self._ctx(ids)
            s = self._sig(n, self.min_sigma, self.max_sigma, get_rank(), c, 6)
            eps = self._noise_like(x, get_rank(), c, 7)
            x_t = (1 - s)[:, None, None, None] * x + s[:, None, None, None] * eps
            with self._autocast():
                out = self.fake(x_t, s, caps)
            loss_fake = F.mse_loss(out.float(), x - eps)       # flow matching: out = -v = x - eps
            real_ids = self._rng(get_rank(), c, 8).choice(avail, size=n)
            real = self.latents.get(real_ids).to(self.device, torch.float32)
            lr_ = self._logit(real, self._sig(n, 0.0, 1.0, get_rank(), c, 9),
                              self._ctx(self._row_of_prompt[real_ids]),
                              self._noise_like(real, get_rank(), c, 10))
            lf = self._logit(x, self._sig(n, 0.0, 1.0, get_rank(), c, 11), caps,
                             self._noise_like(x, get_rank(), c, 12))
            loss_d = (F.softplus(-lr_) + F.softplus(lf)).mean()
            ((loss_fake + self.guidance_gan * loss_d) / len(chunks)).backward()
            acc["fake"] += float(loss_fake.detach()) / len(chunks)
            acc["gan_d"] += float(loss_d.detach()) / len(chunks)
            acc["d_real"] += float(lr_.detach().mean()) / len(chunks)
            acc["d_fake"] += float(lf.detach().mean()) / len(chunks)
        for p in self.head.parameters():                     # replicated head: average the ranks
            if p.grad is not None and world > 1:
                all_reduce_mean_(p.grad)
        if self.fake_clip_module is None:
            for p in self.fake.parameters():
                if p.grad is not None:
                    all_reduce_mean_(p.grad)
        fnorm = clip_generator_grads([p for p in self.fake.parameters() if p.requires_grad],
                                     self.grad_clip, module=self.fake_clip_module)
        hnorm = torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.grad_clip)
        if math.isfinite(acc["fake"]) and bool(torch.isfinite(fnorm)):
            self.fake_opt.step()
        self.iteration += 1
        logs.update({"loss_fake": acc["fake"], "loss_gan_d": acc["gan_d"], "d_real": acc["d_real"],
                     "d_fake": acc["d_fake"], "grad_norm_fake": float(fnorm),
                     "grad_norm_head": float(hnorm)})
        return logs


# ------------------------------------------------------------------------------ GPU-time budget
def plan_iterations(budget_gpu_s: float, world: int, seconds: list, ratio: int, done: int) -> int:
    """Iterations that spend the budget at the mean cost of whole update-ratio cycles (a student
    iteration costs more than a fake-only one), the first cycle left out once there are others."""
    usable = seconds[ratio:] if len(seconds) >= 2 * ratio else seconds
    usable = usable[:len(usable) // ratio * ratio] or usable
    return max(done, 1, int(budget_gpu_s / (world * float(np.mean(usable)))))


# ------------------------------------------------------------------------------ build + run
def _optimizer(params, lr: float, kind: str):
    kw = dict(lr=lr, betas=(0.9, 0.999), weight_decay=0.01)    # DMD2's AdamW
    if kind == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, **kw)
    return torch.optim.AdamW(params, **kw)


def build(cfg, a, device):
    import dataclasses

    from diffusers import ZImageTransformer2DModel

    from rdm.sw_lmmd.launch import _block, memory_from_config, wrap_fsdp

    from .dmd2_latents import Latents
    from .launch import check_render_config, load_student_weights, student_from_config
    from .model import ZIMAGE_REPO, ZImageStudent, resolve_snapshot
    from .store import ZImageReferenceStore

    policy = memory_from_config(cfg)
    if a.micro_batch:
        policy.micro_batch = int(a.micro_batch)
    student_cfg = student_from_config(cfg)
    zcfg = _block(cfg, "zimage")
    store = ZImageReferenceStore(cfg.reference_root, list(cfg.encoders))
    if not store.has_context:
        raise SystemExit(f"{cfg.reference_root}: no precomputed contexts (rebuild them first)")
    if store.num_rows != store.num_prompts:
        raise SystemExit("DMD2 baseline expects one reference row per prompt (reference_v1)")
    img_size = int(getattr(cfg, "img_size", 1024))
    check_render_config(store, img_size)
    latents = Latents(Path(a.latents) / "latents")
    neg_ctx = torch.load(Path(a.latents) / "negative_context.pt", map_location="cpu",
                         weights_only=True).to(device, torch.float32)

    def shard(module, frozen: bool):
        if policy.shard != "fsdp":
            return module.to(device), None
        p = dataclasses.replace(policy, param_dtype=torch.bfloat16) if frozen else policy
        wrapped = wrap_fsdp(module, p, device)
        return wrapped, wrapped

    # the student: ours (Turbo init); FSDP wraps it whole, as in ours' launch
    student = ZImageStudent(img_size, repo=zcfg.get("repo", ZIMAGE_REPO), param_dtype=policy.param_dtype,
                            gradient_checkpointing=True)
    if a.resume:
        load_student_weights(student, a.resume)
    student, s_clip = shard(student, frozen=False)

    base = os.path.join(resolve_snapshot(a.base_repo), "transformer")
    real_t = ZImageTransformer2DModel.from_pretrained(base, torch_dtype=torch.bfloat16)
    real_t.requires_grad_(False)
    real, _ = shard(ScoreModel(real_t).eval(), frozen=True)

    fake_t = ZImageTransformer2DModel.from_pretrained(base, torch_dtype=policy.param_dtype)
    mid = len(fake_t.layers) // 2
    fake_m = ScoreModel(fake_t, gradient_checkpointing=True, feature_layer=mid)
    head = GANHead(fake_t.config.dim).to(device)
    if a.resume:
        state = torch.load(a.resume, map_location="cpu", weights_only=False, mmap=True)
        fake_m.load_state_dict(state["fake"], strict=True)
        head.load_state_dict(state["head"])
        del state
    fake, f_clip = shard(fake_m, frozen=False)  # the feature hook stays on fake_m's middle block
    trainer = DMD2Trainer(
        student, real, _FeatureView(fake, fake_m), head,
        _optimizer([p for p in student.parameters() if p.requires_grad], a.lr, policy.optimizer),
        _optimizer([p for p in fake.parameters() if p.requires_grad] + list(head.parameters()),
                   a.fake_lr, policy.optimizer),
        store, latents, neg_ctx, sigmas=student_cfg["sampling_args"]["sigmas"],
        renoise=student_cfg["sampling_args"]["renoise"], batch=a.batch, ratio=a.ratio,
        guidance=a.guidance, gen_gan=a.gen_gan, guidance_gan=a.guidance_gan,
        micro_batch=policy.micro_batch, grad_clip=a.grad_clip, student_clip_module=s_clip,
        fake_clip_module=f_clip, seed=getattr(cfg, "seed", 0), device=device)
    logger.info("[dmd2] student (Turbo init, %d steps) | real score: %s, guidance %g | fake score "
                "+ GAN head on layer %d | batch %d, ratio %d, lr %g / %g, GAN %g / %g | %d real "
                "latents", len(trainer.sigmas), a.base_repo, a.guidance, mid, a.batch, a.ratio, a.lr,
                a.fake_lr, a.gen_gan, a.guidance_gan, latents.available.size)
    return trainer


class _FeatureView(nn.Module):
    """The (possibly FSDP-wrapped) fake score, exposing the inner module's kept ``features``."""

    def __init__(self, wrapped, inner):
        super().__init__()
        self.wrapped = wrapped
        self._inner = [inner]                    # a list: not registered as a second submodule

    @property
    def features(self):
        return self._inner[0].features

    def forward(self, *args, **kw):
        return self.wrapped(*args, **kw)

    def parameters(self, recurse: bool = True):
        return self.wrapped.parameters(recurse)

    def state_dict(self, *args, **kw):
        return self.wrapped.state_dict(*args, **kw)


def main() -> int:
    from rdm.sw_lmmd.launch import apply_override
    from rdm.train.launch import load_config
    from rdm.utils.distributed import setup_distributed
    from rdm.utils.logging import setup_logging
    from rdm.utils.seed import fix_random_seeds

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("config", help="ours' config (student, memory policy, store)")
    budget = ap.add_mutually_exclusive_group(required=True)
    budget.add_argument("--match-run", help="ours' run dir: spend the same GPU time")
    budget.add_argument("--match-gpu-hours", type=float)
    budget.add_argument("--iterations", type=int)
    ap.add_argument("--latents", required=True, help="dmd2_latents output dir (real data + neg. ctx)")
    ap.add_argument("--base-repo", default=BASE_REPO, help="the real / fake score init")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ratio", type=int, default=5, help="fake updates per student update")
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--fake-lr", type=float, default=5e-7)
    ap.add_argument("--gen-gan", type=float, default=5e-3)
    ap.add_argument("--guidance-gan", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--micro-batch", type=int, default=0)
    ap.add_argument("--calib-cycles", type=int, default=2, help="timed update-ratio cycles")
    ap.add_argument("--checkpoints", type=int, default=4)
    ap.add_argument("--exp-name", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = load_config(a.config)
    for o in a.set:
        apply_override(cfg, o)
    rank, world, local_rank = setup_distributed()
    setup_logging(rank=rank)
    fix_random_seeds(getattr(cfg, "seed", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    run = Path(getattr(cfg, "output_dir", "./work_dirs")) / (
        a.exp_name or f"{getattr(cfg, 'exp_name', 'zimage')}-dmd2")
    run.mkdir(parents=True, exist_ok=True)
    trainer = build(cfg, a, device)
    sharded = trainer.student_clip_module is not None

    plan_path = run / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    if a.resume:
        trainer.iteration = int(torch.load(a.resume, map_location="cpu", weights_only=False,
                                           mmap=True)["iteration"])
    if a.iterations:
        plan = {"iterations": int(a.iterations), "source": "--iterations"}
    elif not plan.get("iterations"):
        if a.match_run:
            s, w, n = gpu_seconds_of_run(a.match_run)
            plan = {"budget_gpu_hours": s / 3600, "source": str(a.match_run), "ref_gpus": w,
                    "ref_steps": n}
        else:
            plan = {"budget_gpu_hours": float(a.match_gpu_hours), "source": "--match-gpu-hours"}

    def save(path, full: bool):
        state = {"model": trainer.student.state_dict(), "step": trainer.iteration // a.ratio,
                 "iteration": trainer.iteration, "num_steps": len(trainer.sigmas),
                 "objective": "dmd2", "sampling_args": {"sigmas": trainer.sigmas,
                                                        "renoise": trainer.renoise,
                                                        "detach_between_steps": True}}
        if full:
            state["fake"] = trainer.fake.state_dict()
            state["head"] = trainer.head.state_dict()
        if is_main_process():
            tmp = str(path) + ".partial"
            torch.save(state, tmp)
            os.replace(tmp, path)
            logger.info("[dmd2] checkpoint -> %s", path)

    log = open(run / "train_log.jsonl", "a") if is_main_process() else None
    seconds, t0 = [], time.time()
    while True:
        total = plan.get("iterations")
        if total is not None and trainer.iteration >= total:
            break
        st = time.time()
        logs = trainer.step()
        logs["seconds"] = time.time() - st
        seconds.append(logs["seconds"])
        budget_s = plan.get("budget_gpu_hours", 0) * 3600
        if total is None and (len(seconds) >= a.calib_cycles * a.ratio
                              or sum(seconds) * world >= 0.1 * budget_s):
            total = plan_iterations(budget_s, world, seconds, a.ratio, trainer.iteration)
            plan.update(iterations=total, student_updates=math.ceil(total / a.ratio),
                        measured_s_per_iteration=float(np.mean(seconds)), gpus=world)
            if is_main_process():
                plan_path.write_text(json.dumps(plan, indent=2))
            logger.info("[dmd2] plan: %.2f GPU-h on %d GPUs -> %d iterations (%d student updates)",
                        plan["budget_gpu_hours"], world, total, plan["student_updates"])
        if is_main_process():
            log.write(json.dumps(logs) + "\n")
            log.flush()
            logger.info("it %d%s | fake %.4f | D %.3f (real %+.2f fake %+.2f)%s | %.1f s",
                        logs["iteration"], " [student]" if logs["student_step"] else "",
                        logs["loss_fake"], logs["loss_gan_d"], logs["d_real"], logs["d_fake"],
                        f" | dmd {logs['loss_dmd']:.4f} gan_g {logs['loss_gan_g']:.3f}"
                        if logs["student_step"] else "", logs["seconds"])
        if plan.get("iterations"):
            if is_main_process() and not plan_path.exists():
                plan_path.write_text(json.dumps(plan, indent=2))
            at_ckpt = trainer.iteration in checkpoint_steps(plan["iterations"], a.checkpoints)
            if at_ckpt:
                save(run / f"step_{trainer.iteration:07d}.pth", full=False)
            if at_ckpt or (logs["seconds"] > 600 and trainer.iteration % a.ratio == 0):
                save(run / "resume.pth", full=True)
    if log is not None:
        log.close()
    logger.info("[dmd2] done: %d iterations in %.2f h -> %s", trainer.iteration,
                (time.time() - t0) / 3600, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
