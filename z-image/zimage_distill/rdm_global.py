#!/usr/bin/env python
"""Baseline: the original RDM objective (local-to-global matching) on the same 2-step Z-Image student.

    PYTHONPATH=z-image:. torchrun --nproc_per_node=8 -m zimage_distill.rdm_global \
        z-image/configs/zimage_2step_1024_h100.yaml \
        --match-run <root>/zimage/work_dirs/zimage-2step-1024      # same GPU time as ours
    (or --match-gpu-hours H, or --steps N)

What changes against ours (``zimage_distill.trainer.MultiStepTrainer``), and only that:

* **local-to-global matching.** Every step draws ``--rows`` N fresh rows (random prompts of the
  reference store, each rendered from fresh noise) and scores them, per encoder, with the RDM loss
  ``mmd_nystrom_loss``: the biased MMD^2 between the N generated joint features ``[phi | beta tau]``
  and the WHOLE reference set -- exact within-batch repulsion, attraction to the reference's
  Nystrom kernel mean embedding (4096 k-means landmarks over all reference rows), constant
  ``k_rr``. A sample is not paired with its own prompt's reference: every sample is pulled toward
  the whole reference distribution. Each encoder's term is self-normalized, as in iRDM.
* **no cache, no window, no GAN critic** (the original RDM has none of them).

What stays the same: the 2-step student (sigmas [1, 0.75], re-noise, cut between the steps), the
loss on both steps' ``x0_hat`` (step weights w_k), the encoders and their store bandwidths, the
references' pixel pipeline (``reference_pixels``), the precomputed contexts, the memory policy
(FSDP etc.), the lr and the shape of its schedule (cosine, 5 % warm-up, floor 0.1).

Same GPU time: with ``--match-run`` (our run's dir: its ``train_log.jsonl`` holds the seconds of
every step, its rows-per-rank give its GPU count) or ``--match-gpu-hours``, the first
``--calib-steps`` steps are timed and the total is set to ``budget / (GPUs x median s/step)``;
the lr schedule then decays over that total. The plan is written to ``<run>/plan.json``, and a
resume keeps it. ``--match-run`` counts ours' optimizer steps only, which is what is comparable
step for step; ours' staleness probe and cache refresh (every 10 / 20 steps, ~5-8 % more) are
outside them -- pass ``--match-gpu-hours`` (GPUs x the wall time ours printed) to include them.

Checkpoints are ``{"model", "step", "num_steps", "sampling_args"}`` like ours, so
``eval_qib/run_eval.sh --ckpt`` evaluates them unchanged. The Nystrom bundles are built once from
the store (rank 0, a few minutes on one GPU) and cached in ``<store>/../nystrom_global/``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from rdm.compare.grad_balance import clip_generator_grads, self_normalize
from rdm.compare.mmd_nystrom import mmd_nystrom_loss
from rdm.representation.joint_feature import couple
from rdm.sw_lmmd.sharding import all_reduce_mean_, partition_rows, row_noise
from rdm.train.grad_caching import gradcache_backward
from rdm.utils.distributed import get_world_size, is_dist, is_main_process

from .model import reference_pixels
from .store import stream_name

logger = logging.getLogger("rdm")


# ------------------------------------------------------------------------------ the reference
def build_global_bundles(store, encoders, out_dir, n_landmarks: int = 4096, seed: int = 0,
                         chunk: int = 16384) -> dict:
    """Per encoder: the Nystrom reference over ALL store rows' joint ``[phi | beta tau]``, at the
    store's bandwidth. Cached as ``<out_dir>/<enc>_M<m>.pt``; returns ``{enc: path}``."""
    from rdm.compare.reference_math import build_nystrom_reference
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for enc in encoders:
        path = out_dir / f"{enc}_M{n_landmarks}.pt"
        paths[enc] = str(path)
        if path.exists():
            continue
        t0 = time.time()
        rows = np.arange(store.num_rows)
        joint = torch.cat([store.reference_joint_features(enc, rows[lo:lo + chunk])
                           for lo in range(0, rows.size, chunk)])
        bundle = build_nystrom_reference(joint, store.sigma(enc), n_landmarks=n_landmarks,
                                         seed=seed)
        bundle.update(encoder=enc, beta=store.beta(enc), num_rows=int(store.num_rows),
                      prompt_hash=store.metadata.get("prompt_hash"))
        tmp = path.with_suffix(".partial")
        torch.save(bundle, tmp)
        os.replace(tmp, path)
        logger.info("[rdm-global] %s: Nystrom reference over %d rows x %d -> %d landmarks, "
                    "k_rr %.4g (%.0f s) -> %s", enc, joint.shape[0], joint.shape[1], n_landmarks,
                    bundle["k_rr"], time.time() - t0, path)
    return paths


def load_bundles(paths: dict, store, device) -> dict:
    """``{enc: dict(Z, Z2, alpha, sigma, k_rr)}`` on ``device``, checked against the store."""
    packs = {}
    for enc, path in paths.items():
        b = torch.load(path, map_location="cpu", weights_only=False)
        if b.get("num_rows") != store.num_rows or abs(b["sigma"] - store.sigma(enc)) > 1e-6:
            raise RuntimeError(f"{path} was built for another store; delete it to rebuild")
        Z = b["Z"].float().to(device)
        packs[enc] = {"Z": Z, "Z2": (Z * Z).sum(1), "alpha": b["alpha"].float().to(device),
                      "sigma": float(b["sigma"]), "k_rr": float(b["k_rr"])}
    return packs


# ------------------------------------------------------------------------------ the trainer
class GlobalRDMTrainer:
    """Fresh rows every step, RDM loss against the global reference, on every student step."""

    def __init__(self, generator, battery, store, packs: dict, optimizer, *, encoders,
                 num_steps: int, rows_per_step: int, noise_shape, step_weights=None,
                 micro_batch: int = 1, feature_pool: int = 1, feature_quantize: bool = False,
                 use_store_context: bool = True, grad_clip: float = 4.0, seed: int = 0,
                 joint: bool = True, clip_module=None, grad_reduce: str = "mean",
                 device="cuda", feature_dtype=torch.float32, context_dtype=None,
                 lr_schedule=None, gen_chunk: int = 8192):
        world = get_world_size()
        if rows_per_step % (world * micro_batch):
            raise ValueError(f"--rows {rows_per_step} must split into {world} ranks x micro_batch "
                             f"{micro_batch}")
        self.generator, self.battery, self.store, self.packs = generator, battery, store, packs
        self.optimizer = optimizer
        self.encoders = list(encoders)
        self.num_steps = int(num_steps)
        self.rows_per_step = int(rows_per_step)
        self.noise_shape = tuple(noise_shape)
        self.step_weights = [float(w) for w in (step_weights or [1.0] * self.num_steps)]
        self.micro_batch = int(micro_batch)
        self.feature_pool, self.feature_quantize = int(feature_pool), bool(feature_quantize)
        self.use_store_context = bool(use_store_context)
        self.grad_clip, self.seed, self.joint = float(grad_clip), int(seed), bool(joint)
        self.clip_module, self.grad_reduce, self.device = clip_module, grad_reduce, device
        self.feature_dtype = feature_dtype
        self.context_dtype = context_dtype or feature_dtype
        self.lr_schedule = lr_schedule
        self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        self.gen_chunk = int(gen_chunk)
        self.step_idx = 0

    @property
    def _gen_params(self):
        return [p for p in self.generator.model.parameters() if p.requires_grad]

    def step_rows(self, step: int) -> np.ndarray:
        """The step's N global rows: uniform without replacement, a pure function of the step."""
        rng = np.random.default_rng([self.seed, 7919, int(step)])
        return np.sort(rng.choice(self.store.num_rows, size=self.rows_per_step, replace=False))

    def _encode(self, row_ids) -> dict:
        """rows -> ``{s{k}/{enc}: (n, d_joint)}`` from one rollout (noise drawn per row and step)."""
        ids = np.asarray(row_ids, dtype=np.int64)
        noise = row_noise(ids, self.noise_shape, seed=self.seed, epoch=self.step_idx,
                          device=self.device, dtype=self.feature_dtype)
        if self.use_store_context:
            images = self.generator.sample_steps(noise, contexts=self.store.generator_context(
                ids, device=self.device, dtype=self.context_dtype))
        else:
            images = self.generator.sample_steps(noise, prompts=self.store.prompts(ids))
        tau = self.store.text_features(ids, device=self.device, dtype=self.feature_dtype) \
            if self.joint else None
        out = {}
        for k, img in enumerate(images, 1):
            if self.feature_pool > 1 or self.feature_quantize:
                img = reference_pixels(img, self.feature_pool, self.feature_quantize)
            feats = self.battery(img, only=set(self.encoders))
            for e in self.encoders:
                out[stream_name(k, e)] = couple(feats[e], tau, self.store.beta(e)) \
                    if self.joint else feats[e]
        return out

    def _loss(self, glob: dict):
        """sum_k w_k / E sum_e self_norm(MMD^2_nystrom(step k's N features, global reference))."""
        total, logs, n_enc = None, {}, len(self.encoders)
        for k in range(1, self.num_steps + 1):
            for e in self.encoders:
                p = self.packs[e]
                raw = mmd_nystrom_loss(glob[stream_name(k, e)], p["Z"], p["Z2"], p["alpha"],
                                       p["sigma"], p["k_rr"], gen_chunk=self.gen_chunk)
                logs[f"s{k}/{e}/mmd2"] = float(raw.detach())
                term = (self.step_weights[k - 1] / n_enc) * self_normalize(raw)
                total = term if total is None else total + term
            logs[f"s{k}/mmd2"] = float(np.mean([logs[f"s{k}/{e}/mmd2"] for e in self.encoders]))
        return total, logs

    def step(self) -> dict:
        lr = float(self.optimizer.param_groups[0]["lr"])
        if self.lr_schedule is not None:
            factor = self.lr_schedule.factor(self.step_idx)
            for g, base in zip(self.optimizer.param_groups, self.base_lrs):
                g["lr"] = base * factor
            lr = float(self.optimizer.param_groups[0]["lr"])
        local = partition_rows(self.step_rows(self.step_idx))
        chunks = [local[i:i + self.micro_batch] for i in range(0, local.size, self.micro_batch)]
        self.optimizer.zero_grad(set_to_none=True)
        # The full-batch loss on the features gathered from every rank (with autograd: each rank's
        # backward reaches its own rows), times the world size, so that the cross-rank MEAN (FSDP's
        # reduce-scatter, or here) is exactly the full-batch gradient at any GPU count and
        # micro-batch -- ours has the same property. (iRDM scaled by 1/grad_accum instead, a
        # convention for its lr, which makes the gradient -- and the clipping -- depend on both.)
        loss_val, logs = gradcache_backward(chunks, self._encode, self._loss,
                                            scale=float(get_world_size()), gather=is_dist())
        if self.grad_reduce == "mean":
            for p in self._gen_params:
                if p.grad is not None:
                    all_reduce_mean_(p.grad)
        grad_norm = clip_generator_grads(self._gen_params, self.grad_clip,
                                         module=self.clip_module) if self.grad_clip > 0 else None
        finite = math.isfinite(loss_val) and (grad_norm is None or bool(torch.isfinite(grad_norm)))
        if finite:
            self.optimizer.step()
        else:
            self.optimizer.zero_grad(set_to_none=True)
        self.step_idx += 1
        return {"step": self.step_idx, "loss": loss_val, "skipped": not finite, "lr": lr,
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
                "rows": self.rows_per_step, "rows_per_rank": int(local.size), **logs}

    def oom_probe(self, real_chunks: int = 2) -> dict:
        """The memory of one full-size step in minutes instead of hours: what a step's size changes
        is the gathered features and the full-batch loss (the per-row rollout is the micro-batch's,
        whatever the size). So every buffer of the step is allocated at full size -- every chunk's
        feature leaves, the gathered N x d features, the N x N repulsion and N x M attraction
        kernels with their backward, one optimizer step -- but only the first ``real_chunks``
        micro-batches are rolled out (pass 1 and pass 2); the others reuse their features. The
        weights it changes are thrown away (nothing is saved)."""
        local = partition_rows(self.step_rows(self.step_idx))
        chunks = [local[i:i + self.micro_batch] for i in range(0, local.size, self.micro_batch)]
        real_chunks = max(1, min(int(real_chunks), len(chunks)))
        calls, last = [0], {}

        def encode(ci):
            pos, calls[0] = calls[0] % len(chunks), calls[0] + 1
            if pos < real_chunks:
                out = self._encode(ci)
                last.update({k: v.detach() for k, v in out.items()})
                return out
            # a stand-in of the real size: in pass 2 a leaf, so the surrogate backward is a no-op
            return {k: v.clone().requires_grad_(torch.is_grad_enabled()) for k, v in last.items()}

        cuda = torch.device(self.device).type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        loss_val, _ = gradcache_backward(chunks, encode, self._loss,
                                         scale=float(get_world_size()), gather=is_dist())
        if self.grad_reduce == "mean":
            for p in self._gen_params:
                if p.grad is not None:
                    all_reduce_mean_(p.grad)
        if self.grad_clip > 0:
            clip_generator_grads(self._gen_params, self.grad_clip, module=self.clip_module)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        out = {"rows": self.rows_per_step, "rows_per_rank": int(local.size),
               "real_chunks": real_chunks, "loss": loss_val, "seconds": time.time() - t0}
        if cuda:
            out.update(peak_allocated_gib=torch.cuda.max_memory_allocated(self.device) / 2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved(self.device) / 2**30,
                       device_gib=torch.cuda.get_device_properties(self.device).total_memory / 2**30)
        return out

    def state_dict(self, with_optimizer: bool) -> dict:
        state = {"model": self.generator.model.state_dict(), "step": self.step_idx,
                 "num_steps": self.num_steps, "objective": "rdm_global",
                 "sampling_args": dict(self.generator.sampling_args)}
        if with_optimizer:
            state["optimizer"] = self.optimizer.state_dict()
        return state


# ------------------------------------------------------------------------------ GPU-time budget
def gpu_seconds_of_run(run_dir) -> tuple[float, int, int]:
    """(GPU-seconds, GPUs, steps) of a finished run of ours, from its ``train_log.jsonl``."""
    rows = [json.loads(line) for line in open(Path(run_dir) / "train_log.jsonl") if line.strip()]
    rows = [r for r in rows if "seconds" in r and "step" in r]
    if not rows:
        raise SystemExit(f"{run_dir}/train_log.jsonl holds no timed steps")
    last = {}
    for r in rows:                      # a resumed run re-logs steps: keep each step once
        last[int(r["step"])] = r
    world = int(round(rows[-1]["active_rows"] / rows[-1]["active_rows_per_rank"]))
    return sum(r["seconds"] for r in last.values()) * world, world, len(last)


def measured_s_per_step(step_seconds: list) -> float:
    """Median s/step, the first step (warm-up) left out once there are others."""
    return statistics.median(step_seconds[1:] if len(step_seconds) > 1 else step_seconds)


def plan_steps(budget_gpu_s: float, world: int, step_seconds: list, done: int) -> int:
    """Steps that spend the budget at the measured s/step, rounded to the nearest (never fewer
    than already done). Rounded, not truncated: at 10240 rows one step is ~16 % of ours' budget,
    so truncating could leave the baseline most of a step short of ours' GPU time."""
    return max(done, 1, round(budget_gpu_s / (world * measured_s_per_step(step_seconds))))


def plan_ready(budget_gpu_s: float, world: int, step_seconds: list, calib: int) -> bool:
    """Plan after ``calib`` timed steps -- or as soon as 10 % of the budget is spent: at 10240
    rows one step is ~1/7 of ours' whole budget, so ten timed steps would overshoot it."""
    return len(step_seconds) >= calib or sum(step_seconds) * world >= 0.1 * budget_gpu_s


def checkpoint_steps(total: int, n: int) -> set:
    """``n`` evenly spaced steps ending at ``total`` (fewer if ``total < n``)."""
    return {math.ceil(i * total / n) for i in range(1, n + 1)}


# ------------------------------------------------------------------------------ build + run
def build(cfg, args, device):
    from rdm.representation.battery import Battery
    from rdm.representation.registry import by_name
    from rdm.sw_lmmd.launch import (_block, build_optimizer, lr_schedule_from_config,
                                    memory_from_config, shard_generator)

    from .launch import (check_render_config, load_student_weights, resolve_feature_pool,
                         student_from_config)
    from .model import ZIMAGE_REPO, QwenPromptEncoder, ZImageGenerator, ZImageStudent, ZImageVAE
    from .store import ZImageReferenceStore

    policy = memory_from_config(cfg)
    if args.micro_batch:
        policy.micro_batch = int(args.micro_batch)
    student = student_from_config(cfg)
    S = student["num_steps"]
    zcfg = _block(cfg, "zimage")
    repo = zcfg.get("repo", ZIMAGE_REPO)
    names = list(cfg.encoders)
    store = ZImageReferenceStore(cfg.reference_root, names)
    img_size = int(getattr(cfg, "img_size", 1024))
    feature_pool = resolve_feature_pool(store, img_size, zcfg.get("feature_pool", "auto"))
    check_render_config(store, img_size)
    use_ctx = store.has_context and str(zcfg.get("context", "auto")) != "online"

    # the global reference: built on rank 0, read by every rank
    nys_dir = args.nystrom_dir or str(Path(cfg.reference_root).parent / "nystrom_global")
    if is_main_process():
        build_global_bundles(store, names, nys_dir, n_landmarks=args.landmarks)
    if is_dist():
        dist.barrier()
    packs = load_bundles({e: str(Path(nys_dir) / f"{e}_M{args.landmarks}.pt") for e in names},
                         store, device)

    model = ZImageStudent(image_resolution=img_size, repo=repo, param_dtype=policy.param_dtype,
                          gradient_checkpointing=bool(getattr(cfg, "grad_checkpointing", True)))
    if args.resume:
        load_student_weights(model, args.resume)
    elif getattr(cfg, "load_from", None):
        load_student_weights(model, cfg.load_from)
    if policy.shard != "fsdp":
        model = model.to(device)
    generator = ZImageGenerator(model, vae=None, text=None, sampling_args=student["sampling_args"])
    clip_module = shard_generator(generator, model, policy, device)
    generator.vae = ZImageVAE(repo, device=device,
                              gradient_checkpointing=bool(zcfg.get("vae_checkpointing", True)))
    if not use_ctx:
        generator.text = QwenPromptEncoder(repo, device=device)
    battery = Battery([by_name(n) for n in names], device=device)
    cfg.lr_schedule = dict(_block(cfg, "lr_schedule"), total_steps=1)   # set by the plan
    trainer = GlobalRDMTrainer(
        generator, battery, store, packs,
        build_optimizer([p for p in model.parameters() if p.requires_grad], cfg, policy),
        encoders=names, num_steps=S, rows_per_step=args.rows,
        noise_shape=(S, model.in_channels, model.input_size, model.input_size),
        step_weights=student["step_weights"], micro_batch=policy.micro_batch,
        feature_pool=feature_pool, feature_quantize=bool(zcfg.get("feature_quantize", True)),
        use_store_context=use_ctx, grad_clip=getattr(cfg, "grad_clip", 4.0),
        seed=getattr(cfg, "seed", 0), joint=getattr(cfg, "joint", True), clip_module=clip_module,
        grad_reduce=policy.grad_reduce, device=device,
        lr_schedule=lr_schedule_from_config(cfg))
    logger.info("[rdm-global] %d-step student | %d fresh rows per step (%d per rank, micro %d) | "
                "global reference: %d Nystrom landmarks per encoder | no cache, no critic",
                S, args.rows, args.rows // get_world_size(), policy.micro_batch, args.landmarks)
    return trainer


def set_horizon(trainer, cfg, total: int) -> None:
    """The run's lr schedule over ``total`` steps, with the config's shape (warm-up as a fraction:
    ours warms up 100 of 2000 steps)."""
    from rdm.sw_lmmd.config import LRSchedule
    raw = dict(getattr(cfg, "lr_schedule", None) or {})
    ref_total = int(getattr(cfg, "steps", 2000)) or 2000
    frac = float(raw.get("warmup_steps", 0)) / ref_total
    sched = LRSchedule(name=raw.get("name", "constant"), warmup_steps=max(0, round(frac * total)),
                       min_lr_ratio=float(raw.get("min_lr_ratio", 0.0)), total_steps=int(total))
    sched.validate()
    trainer.lr_schedule = sched


def oom_probe_main(trainer, real_chunks: int) -> int:
    """Run ``trainer.oom_probe`` and print one verdict line per rank (``oom-probe: PASS|FAIL``)."""
    rank = dist.get_rank() if is_dist() else 0
    try:
        r = trainer.oom_probe(real_chunks)
    except torch.cuda.OutOfMemoryError as e:
        print(f"[rank {rank}] oom-probe: FAIL -- out of memory at {trainer.rows_per_step} rows: "
              f"{str(e).splitlines()[0]}", flush=True)
        return 1
    print(f"[rank {rank}] oom-probe: PASS -- {r['rows']} rows ({r['rows_per_rank']}/rank, "
          f"{r['real_chunks']} rolled out) | peak allocated {r.get('peak_allocated_gib', 0):.1f} GiB, "
          f"reserved {r.get('peak_reserved_gib', 0):.1f} of {r.get('device_gib', 0):.1f} GiB | "
          f"loss {r['loss']:.4f} | {r['seconds']:.0f} s", flush=True)
    return 0


def main() -> int:
    from rdm.sw_lmmd.launch import apply_override
    from rdm.train.launch import load_config
    from rdm.utils.distributed import setup_distributed
    from rdm.utils.logging import setup_logging
    from rdm.utils.seed import fix_random_seeds

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("config", help="ours' config (model, memory, lr, encoders, store)")
    budget = ap.add_mutually_exclusive_group()
    budget.add_argument("--match-run", help="ours' run dir: spend the same GPU time")
    budget.add_argument("--match-gpu-hours", type=float, help="spend this many GPU-hours")
    budget.add_argument("--steps", type=int, help="a fixed number of steps")
    ap.add_argument("--rows", type=int, default=10240,
                    help="fresh rows N per step (default 10240, iRDM's rollout size)")
    ap.add_argument("--calib-steps", type=int, default=10,
                    help="timed steps before the plan (fewer once 10 %% of the budget is spent)")
    ap.add_argument("--landmarks", type=int, default=4096)
    ap.add_argument("--nystrom-dir", default=None)
    ap.add_argument("--micro-batch", type=int, default=0)
    ap.add_argument("--exp-name", default=None, help="default: <config exp_name>-rdm-global-N<rows>")
    ap.add_argument("--checkpoints", type=int, default=4, help="evenly spaced step checkpoints")
    ap.add_argument("--resume", default=None, help="<run>/resume.pth")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--oom-probe", type=int, nargs="?", const=2, default=0, metavar="CHUNKS",
                    help="only test whether a --rows step fits: every buffer at full size, CHUNKS "
                         "(default 2) micro-batches really rolled out; minutes, saves nothing")
    a = ap.parse_args()
    if not (a.oom_probe or a.match_run or a.match_gpu_hours or a.steps):
        ap.error("one of --match-run / --match-gpu-hours / --steps is required")
    cfg = load_config(a.config)
    for o in a.set:
        apply_override(cfg, o)

    rank, world, local_rank = setup_distributed()
    setup_logging(rank=rank)
    fix_random_seeds(getattr(cfg, "seed", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    run = Path(getattr(cfg, "output_dir", "./work_dirs")) / (
        a.exp_name or f"{getattr(cfg, 'exp_name', 'zimage')}-rdm-global-N{a.rows}")
    if not a.oom_probe:
        run.mkdir(parents=True, exist_ok=True)
    trainer = build(cfg, a, device)
    sharded = trainer.clip_module is not None
    if a.oom_probe:
        return oom_probe_main(trainer, a.oom_probe)

    plan_path = run / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    if a.resume:
        state = torch.load(a.resume, map_location="cpu", weights_only=False, mmap=True)
        trainer.step_idx = int(state["step"])
        if state.get("optimizer") is not None:
            trainer.optimizer.load_state_dict(state["optimizer"])
        del state
        logger.info("[rdm-global] resumed at step %d (plan %s)", trainer.step_idx, plan)
    if a.steps:
        plan = {"steps": int(a.steps), "source": "--steps"}
    elif not plan.get("steps"):
        if a.match_run:
            budget_s, ref_world, ref_steps = gpu_seconds_of_run(a.match_run)
            plan = {"budget_gpu_hours": budget_s / 3600, "source": str(a.match_run),
                    "ref_gpus": ref_world, "ref_steps": ref_steps}
        else:
            plan = {"budget_gpu_hours": float(a.match_gpu_hours), "source": "--match-gpu-hours"}
    # provisional horizon until the plan (only the first steps' lr depends on it): the cost of a
    # step grows with its rows, ours does 2000 steps of 32 fresh rows
    provisional = plan.get("steps") or max(1, round(2000 * 32 / a.rows))
    set_horizon(trainer, cfg, provisional)

    log = open(run / "train_log.jsonl", "a") if is_main_process() else None
    seconds, t0 = [], time.time()

    def save(path, with_opt):
        state = trainer.state_dict(with_optimizer=with_opt)
        if is_main_process():
            tmp = str(path) + ".partial"
            torch.save(state, tmp)
            os.replace(tmp, path)
            logger.info("[rdm-global] checkpoint -> %s", path)

    while True:
        total = plan.get("steps")
        if total is not None and trainer.step_idx >= total:
            break
        st = time.time()
        logs = trainer.step()
        logs["seconds"] = time.time() - st
        seconds.append(logs["seconds"])
        budget_s = plan.get("budget_gpu_hours", 0) * 3600
        if total is None and plan_ready(budget_s, world, seconds, a.calib_steps):
            total = plan_steps(budget_s, world, seconds, trainer.step_idx)
            plan.update(steps=total, measured_s_per_step=measured_s_per_step(seconds),
                        gpus=world, rows=a.rows)
            set_horizon(trainer, cfg, total)
            if is_main_process():
                plan_path.write_text(json.dumps(plan, indent=2))
            logger.info("[rdm-global] plan: %.2f GPU-h at %.1f s/step on %d GPUs -> %d steps",
                        plan["budget_gpu_hours"], plan["measured_s_per_step"], world, total)
        if is_main_process():
            log.write(json.dumps(logs) + "\n")
            log.flush()
            logger.info("step %d | loss %.4f | mmd2 %s | grad %s | lr %.3g | %.1f s", logs["step"],
                        logs["loss"], " ".join(f"s{k}={logs[f's{k}/mmd2']:.4g}"
                                              for k in range(1, trainer.num_steps + 1)),
                        logs["grad_norm"], logs["lr"], logs["seconds"])
        if plan.get("steps"):
            if is_main_process() and not plan_path.exists():
                plan_path.write_text(json.dumps(plan, indent=2))
            at_ckpt = trainer.step_idx in checkpoint_steps(plan["steps"], a.checkpoints)
            if at_ckpt:
                save(run / f"step_{trainer.step_idx:07d}.pth", with_opt=False)
            # a long step (10240 rows: hours) is worth a resume point every time
            if at_ckpt or logs["seconds"] > 600:
                save(run / "resume.pth", with_opt=not sharded)
    if log is not None:
        log.close()
    logger.info("[rdm-global] done: %d steps in %.2f h -> %s", trainer.step_idx,
                (time.time() - t0) / 3600, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
