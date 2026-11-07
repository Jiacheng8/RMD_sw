#!/usr/bin/env python
"""E1: how stale is the SW-LMMD sliding-window cache, and does the staleness change the gradient?

Measurement only. Nothing here changes how training behaves: the trainer is built exactly as
``rdm.sw_lmmd.launch`` builds it, this script calls ``SWLMMDTrainer.step()`` itself (so the
probe and the refresh never run and no checkpoint is written), and it measures between steps.
How to run it end to end: tools/Cache-Drift-README.md.

    # per checkpoint (2 x H100, memory.shard: none)
    torchrun --nproc_per_node=2 tools/measure_cache_drift.py measure \
        configs/sw_lmmd_train_h100_2gpu_gan.yaml --ckpt <step_0000200.pth | resume.pth> \
        --out outputs/cache_drift/step_0200.json --set reference_root=<store>
    # age curve 1..31 (step 2000 only)
    torchrun ... measure <config> --ckpt resume.pth --long-ages 31 \
        --out outputs/cache_drift/age_curve_step2000.json
    # summary.md / summary.csv / age_curve_step2000.png / notes.md (CPU)
    python tools/measure_cache_drift.py summarize --dir outputs/cache_drift --train-log <train_log.jsonl>

Checkpoints
    resume.pth (step 2000)  weights, AdamW8bit state, critic and cache all restored; warm-up and
                            measured updates are the training step as it was (critic on).
    step_*.pth (200, 1000)  weights + step + window position only. AdamW8bit is new, the lr
                            schedule continues at the checkpoint's step, the cache is filled
                            under theta_s (bootstrap), and the critic is OFF for every update this
                            script makes (L_MMD only): a fresh critic would push random gradients.
                            A new AdamW's first steps are close to lr * sign(g), larger than in
                            steady state, so drift comes out larger (conservative).

Protocol: ``--warmup`` updates (3), then a check that the next window's 96 retained rows are
exactly the rows those 3 updates generated (ages 1/2/3, 32 each, in the order they were made).
Then ``--windows`` measured windows (8). For each, before its update:

  F_R_old   the 96 cached rows (made in training pass 1: 32-row active blocks)
  F_R_now   the same rows re-rolled under the current weights, chunked as the refresh does
  drift     per encoder and age, IMAGE block: ||phi_old - phi_now|| / sigma_e (sigma_e from
            bandwidths.json), ||dphi|| / ||phi_now||, and the log's joint definition
            ||x_old - x_now|| / ||x_now|| on [phi | beta tau] (= ``drift_age_k``).
  floor     the same weights, more re-rolls:
      a  the 96 rows, refresh chunking twice (identical batching): is the rollout deterministic?
      b  the 96 rows, training-pass chunking (each original 32-row active block on its own, as
         pass 1 ran it) vs refresh chunking: the batching difference between F_R_old and F_R_now
      c  the 32 active rows: the trainer's real pass-1 features (read from the cache after the
         update) vs this script's re-roll of them with pass 1's batching
      floor = max(b, c) per encoder.
  grad      featgrad(F_R, F_A, Y) = dL_MMD/dF_A (image columns; the trainer's
            ExactLocalMMD.active_force and encoder weights; context [F_R ; sg(F_A)]):
      cache_vs_now     featgrad(F_R_old) vs featgrad(F_R_now)
      floor            featgrad(F_R_now with training chunking) vs featgrad(F_R_now)
      resample_vs_now  96 reference rows R2 outside the window, re-rolled, references Y[R2 u A]
      relative change ||g - g_now|| / ||g_now|| and cosine, per encoder and concatenated.
  grad_total (step 2000 only) the same with L_MMD + lambda L_adv: the critic's window step sees
            the cached rows as fakes. Each variant runs on a snapshot of the critic, restored after.
then the window's ordinary update.

``--long-ages L`` (step 2000): warm up L updates instead and keep each update's 32 pass-1
feature rows; then re-roll all of them under the final weights: drift for ages 1..L, with
floors a (refresh chunking twice) and b (training vs refresh chunking) on the same rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time

import numpy as np
import torch

logger = logging.getLogger("rdm")
METRICS = ("abs_sigma", "rel_image", "rel_joint")
COMPARISONS = ("cache_vs_now", "floor", "resample_vs_now")

# How this script departs from the protocol it implements (written into every json and notes.md).
DEVIATIONS = [
    "step 200 / 1000 (weights-only files): the critic is OFF for every update the script makes "
    "(the 3 warm-up updates AND the 8 measured ones), so they are L_MMD-only; a fresh critic past "
    "g_start_step would add random gradients. No L_MMD + L_adv gradient (grad_total) for them.",
    "step 200 / 1000: AdamW8bit is new; its first updates are close to lr * sign(g), larger than in "
    "steady state, so their drift is an upper bound (conservative). The lr follows the schedule "
    "at the checkpoint's step.",
    "Feature gradients are taken w.r.t. the IMAGE columns of F_A only (the beta*tau block has no "
    "path to the generator), encoder-weighted as in training; 'all' concatenates the 3 encoders.",
    "Floor (c) needs the trainer's real pass 1, so it is measured on the window's 32 active rows; "
    "(a) and (b) on the 96 retained rows. floor = per encoder max(median b, median c).",
    "Age curve: floors (a) and (b) only; (c) cannot be reproduced for rows made under older weights.",
    "The code that runs is the repo HEAD (meta.commit), not 71e96e9: cache, refresh, sharding, "
    "configs, GradCache, generator, encoders and kernels are unchanged since 71e96e9, and the "
    "trainer's later features are off by default (the script refuses grouped windows, block / "
    "per-kind critics, sibling weighting, FSDP and battery_bf16).",
    "Resampled window R2: 96 reference rows drawn outside the window (seed meta.resample_seed + "
    "window step), re-rolled with refresh chunking; the active rows stay.",
]


# ----------------------------------------------------------------------------- small helpers
def _stats(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std()),
            "n": int(v.size)}


def _ms(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std()), "n": int(v.size)} if v.size else {"n": 0}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / ||b||."""
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(),
                                                       dim=0))


def _clone_opt_state(sd):
    """Deep copy of an optimizer state dict (tensors cloned)."""
    if torch.is_tensor(sd):
        return sd.detach().clone()
    if isinstance(sd, dict):
        return {k: _clone_opt_state(v) for k, v in sd.items()}
    if isinstance(sd, list):
        return [_clone_opt_state(v) for v in sd]
    return sd


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True, cwd=os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))).strip()
    except Exception:
        return None


# ----------------------------------------------------------------------------- the measurement
class DriftMeter:
    def __init__(self, trainer):
        self.t = trainer
        self.names = list(trainer.encoder_names)
        self.dims = dict(trainer.image_dims)
        self.sigmas = dict(trainer.sigmas)

    # -- per-row drift ---------------------------------------------------------------------------
    def row_metrics(self, old: dict, new: dict) -> dict:
        """{enc: {metric: (n,) cpu tensor}}: image-block abs/sigma and relative, joint relative."""
        out = {}
        for n in self.names:
            o, f = old[n].float(), new[n].float().to(old[n].device)
            d = self.dims[n]
            dphi = (o[:, :d] - f[:, :d]).norm(dim=1)
            out[n] = {"abs_sigma": (dphi / self.sigmas[n]).cpu(),
                      "rel_image": (dphi / (f[:, :d].norm(dim=1) + 1e-8)).cpu(),
                      "rel_joint": ((o - f).norm(dim=1) / (f.norm(dim=1) + 1e-8)).cpu(),
                      "max_abs": float((o - f).abs().max())}
        return out

    def by_age(self, rows: dict, ages) -> dict:
        """{enc|all: {age: {metric: {mean, median}, rows}}}; 'all' pools (encoder, row) pairs."""
        ages = np.asarray(ages)
        out = {n: {} for n in self.names + ["all"]}
        for a in sorted(set(ages.tolist())):
            sel = torch.from_numpy(ages == a)
            for n in out:
                encs = self.names if n == "all" else [n]
                rec = {m: _stats(torch.cat([rows[e][m][sel] for e in encs]).numpy())
                       for m in METRICS}
                out[n][str(int(a))] = {**{m: {"mean": rec[m]["mean"], "median": rec[m]["median"]}
                                          for m in METRICS}, "rows": int(sel.sum())}
        return out

    def pooled(self, rows: dict, metric: str = "abs_sigma") -> dict:
        """{enc|all: {mean, median}} of one metric."""
        out = {}
        for n in self.names + ["all"]:
            encs = self.names if n == "all" else [n]
            s = _stats(torch.cat([rows[e][metric] for e in encs]).numpy())
            out[n] = {"mean": s["mean"], "median": s["median"]}
        return out

    def enc_median_mean(self, rows: dict, mask=None) -> float:
        """Mean over encoders of the per-encoder median |dphi|/sigma (the summary-table unit)."""
        vals = []
        for n in self.names:
            v = rows[n]["abs_sigma"]
            vals.append(float(v[mask].median() if mask is not None else v.median()))
        return float(np.mean(vals))

    # -- feature gradients -----------------------------------------------------------------------
    def _force(self, leaves: dict, context_rows: dict, reference: dict):
        total = None
        for n in self.names:
            ctx = torch.cat([context_rows[n].to(leaves[n].dtype), leaves[n].detach()], 0)
            force, _ = self.t.mmd.active_force(leaves[n], ctx, reference[n], self.sigmas[n])
            term = self.t.encoder_weights[n] * force
            total = term if total is None else total + term
        return total

    def _image_grads(self, loss, leaves: dict) -> dict:
        grads = torch.autograd.grad(loss, [leaves[n] for n in self.names])
        return {n: g[:, :self.dims[n]].detach() for n, g in zip(self.names, grads)}

    def featgrad(self, F_R: dict, F_A: dict, Y: dict) -> dict:
        """dL_MMD / dF_A, image columns, per encoder."""
        with torch.enable_grad():
            leaves = {n: F_A[n].detach().clone().requires_grad_(True) for n in self.names}
            return self._image_grads(self._force(leaves, F_R, Y), leaves)

    def featgrad_total(self, critic, F_R, F_A, Y, tau_window, tau_active):
        """d(L_MMD + lambda L_adv)/dF_A after the critic's step on [F_R ; F_A] vs Y; the critic's
        exact state (weights, spectral-norm vectors, AdamW, update count) is restored after."""
        snap = {"critic": {k: v.detach().clone() for k, v in critic.critic.state_dict().items()},
                "optimizer": _clone_opt_state(critic.optimizer.state_dict()),
                "updates": int(critic.updates)}
        try:
            critic.update({n: torch.cat([F_R[n], F_A[n]], 0).detach() for n in critic.names},
                          {n: Y[n].detach() for n in critic.names}, tau_window)
            with torch.enable_grad():
                leaves = {n: F_A[n].detach().clone().requires_grad_(True) for n in self.names}
                force = self._force(leaves, F_R, Y)
                adv, _ = critic.generator_loss({n: leaves[n] for n in critic.names}, tau_active)
                lam, _ = critic.balance(force, adv, leaves, self.dims)
                grads = self._image_grads(force + lam * adv, leaves)
        finally:
            critic.critic.load_state_dict(snap["critic"])
            critic.optimizer.load_state_dict(_clone_opt_state(snap["optimizer"]))
            critic.updates = snap["updates"]
        return grads, float(lam)

    def compare(self, g: dict, g_ref: dict) -> dict:
        cat = lambda d: torch.cat([d[n].flatten() for n in self.names])
        out = {"all": {"rel": _rel(cat(g), cat(g_ref)), "cos": _cos(cat(g), cat(g_ref))}}
        for n in self.names:
            out[n] = {"rel": _rel(g[n], g_ref[n]), "cos": _cos(g[n], g_ref[n])}
        return out


def _cat_rows(parts: list) -> dict:
    """Concatenate a list of row_metrics dicts (same encoders)."""
    names = list(parts[0])
    return {n: {m: torch.cat([p[n][m] for p in parts]) for m in METRICS} for n in names}


# ----------------------------------------------------------------------------- rollouts
def reroll_refresh(trainer, row_ids) -> dict:
    """The refresh's batching: the rows split contiguously over ranks, then micro-batches."""
    from rdm.sw_lmmd.refresh import _rollout_rows
    return _rollout_rows(trainer, np.asarray(row_ids, dtype=np.int64))


def reroll_training(trainer, blocks: list) -> dict:
    """Training pass 1's batching: each B-row active block on its own (split over ranks, then
    micro-batches, exactly as SWLMMDTrainer.step chunks window.active_ids), concatenated."""
    parts = [reroll_refresh(trainer, b) for b in blocks]
    return {n: torch.cat([p[n] for p in parts], 0) for n in parts[0]}


def retained_blocks(trainer, window, expected_births=None) -> tuple:
    """The window's retained rows as their original active blocks, oldest first.

    Checks that every block is one update's whole B-row active set, in the order the cache
    holds it, and (if given) that the births are exactly ``expected_births``.
    """
    names = trainer.encoder_names
    entry = trainer.cache.entries[names[0]]
    B, n_ret = trainer.schedule.stride, window.retained_ids.size
    births = entry.generated_at_steps[-n_ret:].numpy()
    ids = entry.row_ids[-n_ret:]
    if not np.array_equal(ids, window.retained_ids):
        raise RuntimeError("cache rows are not the window's retained rows")
    uniq = list(dict.fromkeys(births.tolist()))
    blocks = [window.retained_ids[births == s] for s in uniq]
    ok = all(b.size == B for b in blocks) and np.array_equal(np.concatenate(blocks),
                                                             window.retained_ids)
    if expected_births is not None:
        ok = ok and sorted(uniq) == sorted(int(s) for s in expected_births)
    if not ok:
        raise RuntimeError(f"retained rows are not {n_ret // B} whole active blocks of {B} from "
                           f"the expected updates: births {uniq} (expected {expected_births}), "
                           f"block sizes {[b.size for b in blocks]}")
    ages = (window.step - births).astype(int)
    return blocks, ages, uniq


# ----------------------------------------------------------------------------- driver
def check_main_config(trainer, cfg) -> None:
    problems = []
    if trainer.critic is None:
        problems.append("gan is not enabled")
    else:
        c = trainer.critic
        if getattr(c, "block", None):
            problems.append("block critic")
        if getattr(c, "per_kind", False):
            problems.append("per-kind critics")
        if getattr(c.cfg, "critic_rows", "window") != "window":
            problems.append(f"gan.critic_rows={c.cfg.critic_rows}")
    if getattr(trainer.schedule, "group_size", 1) != 1:
        problems.append("prompt-grouped window")
    if getattr(trainer, "unbiased_siblings", False) or getattr(trainer, "sibling_weight", 1.0) != 1.0:
        problems.append("sibling weighting")
    if trainer.clip_module is not None:
        problems.append("FSDP (this script needs memory.shard: none, i.e. 80 GB cards)")
    if dict(getattr(cfg, "memory", {}) or {}).get("battery_bf16", False):
        problems.append("memory.battery_bf16 is true (the main GAN run used fp32 encoder weights)")
    if problems:
        raise SystemExit("not the main-run setup: " + "; ".join(problems))


def measure(argv) -> None:
    from rdm.sw_lmmd.launch import apply_override, build_trainer
    from rdm.train.launch import load_config
    from rdm.utils.distributed import get_world_size, is_main_process, setup_distributed
    from rdm.utils.logging import setup_logging
    from rdm.utils.seed import fix_random_seeds

    ap = argparse.ArgumentParser(prog="measure_cache_drift.py measure")
    ap.add_argument("config")
    ap.add_argument("--ckpt", required=True, help="resume.pth (full state) or step_NNNNNNN.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--windows", type=int, default=None, help="default 8 (0 with --long-ages)")
    ap.add_argument("--long-ages", type=int, default=0)
    ap.add_argument("--resample-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="cuda (cuda:<local rank>) or cpu (tests only)")
    args = ap.parse_args(argv)
    n_windows = args.windows if args.windows is not None else (0 if args.long_ages else 8)

    cfg = load_config(args.config)
    for o in args.set:
        apply_override(cfg, o)
    if getattr(cfg, "method", None) != "sw_lmmd":
        raise SystemExit(f"{args.config} is not a SW-LMMD config")
    rank, world, local_rank = setup_distributed()
    setup_logging(rank=rank)
    fix_random_seeds(getattr(cfg, "seed", 0))
    device = f"cuda:{local_rank}" if args.device == "cuda" else args.device
    t0 = time.time()

    trainer = build_trainer(cfg, device=device)
    check_main_config(trainer, cfg)
    meter = DriftMeter(trainer)
    names, B, K = meter.names, trainer.schedule.stride, trainer.schedule.window_size

    # ---------------------------------------------------------------- load
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    ckpt_keys = sorted(state.keys())
    restored = trainer.load_state_dict(state)
    del state
    step0 = trainer.step_idx
    full_state = bool(restored.get("optimizer") and restored.get("cache") and restored.get("critic"))
    critic = trainer.critic
    if not full_state:
        trainer.critic = None            # L_MMD-only updates with a weights-only checkpoint
    bootstrapped = False
    if not trainer.cache.initialized:
        trainer.bootstrap()              # fills the cache under theta_s
        bootstrapped = True
    lr_at_start = trainer.lr_schedule.factor(step0) * trainer.base_lrs[0] \
        if trainer.lr_schedule is not None else trainer.base_lrs[0]
    logger.info("[E1] %s: step %d | restored %s | critic %s | lr %.3g", args.ckpt, step0, restored,
                "on" if trainer.critic is not None else "OFF", lr_at_start)

    def train_step(log_list):
        logs = trainer.step()
        log_list.append({k: logs.get(k) for k in ("step", "window_step", "lr", "force", "grad_norm",
                                                  "cache_max_age", "gan/lambda", "gan/d_acc")})

    # ---------------------------------------------------------------- warm-up
    n_warm = max(args.warmup, args.long_ages)
    warm_logs, warm_births, history = [], [], []
    for _ in range(n_warm):
        train_step(warm_logs)
        e = trainer.cache.entries
        warm_births.append(int(warm_logs[-1]["window_step"]))
        if args.long_ages:               # this update's pass-1 features of its 32 active rows
            history.append((warm_births[-1], e[names[0]].row_ids[-B:].copy(),
                            {n: e[n].features[-B:].detach().to("cpu", copy=True) for n in names}))
    w = trainer.schedule.peek()
    _, ages0, births0 = retained_blocks(trainer, w, expected_births=warm_births[-(K // B - 1):])
    age_counts = {str(a): int((ages0 == a).sum()) for a in sorted(set(ages0.tolist()))}
    sanity = {"after_warmup": {"retained_rows": int(w.retained_ids.size), "age_counts": age_counts,
                               "births": [int(s) for s in births0],
                               "warmup_window_steps": warm_births,
                               "expected": {str(a): B for a in range(1, K // B)},
                               "passed": age_counts == {str(a): B for a in range(1, K // B)}}}

    meta = {
        "checkpoint": os.path.basename(args.ckpt), "checkpoint_path": os.path.abspath(args.ckpt),
        "checkpoint_keys": ckpt_keys, "step": step0, "lr": lr_at_start,
        "optimizer": "restored" if restored.get("optimizer") else "new (AdamW8bit)",
        "critic": "restored" if restored.get("critic") else "new, never used (all updates L_MMD only)",
        "cache": "restored" if restored.get("cache") else "new (bootstrap under theta_s)",
        "critic_on_during_warmup": trainer.critic is not None,
        "critic_on_during_measured_updates": trainer.critic is not None,
        "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else device,
        "gpu_count": get_world_size(),
        "commit": _git("rev-parse", "HEAD"),
        "uncommitted_changes": bool(_git("status", "--porcelain", "--", "rdm", "tools", "configs")),
        "sigma": meter.sigmas, "beta": dict(trainer.betas), "image_dims": meter.dims,
        "encoder_weights": dict(trainer.encoder_weights),
        "config": os.path.abspath(args.config), "overrides": args.set,
        "K": K, "B": B, "warmup_updates": n_warm, "windows": n_windows,
        "micro_batch": trainer.micro_batch, "resample_seed": args.resample_seed,
        "probe_and_refresh": "never called", "deviations": DEVIATIONS,
    }

    # ---------------------------------------------------------------- age curve (step 2000)
    if args.long_ages:
        nxt = trainer.schedule.peek()
        hist = [(born, ids, f) for born, ids, f in history if 1 <= nxt.step - born <= args.long_ages]
        blocks = [ids for _, ids, _ in hist]
        ages = np.concatenate([np.full(ids.size, nxt.step - born) for born, ids, _ in hist])
        old = {n: torch.cat([f[n] for _, _, f in hist]).to(device) for n in names}
        rows = np.concatenate(blocks)
        now = reroll_refresh(trainer, rows)
        now2 = reroll_refresh(trainer, rows)
        now_train = reroll_training(trainer, blocks)
        drift = meter.row_metrics(old, now)
        fa, fb = meter.row_metrics(now2, now), meter.row_metrics(now_train, now)
        age_list = sorted(set(ages.tolist()))
        fl = {n: max(float(fa[n]["abs_sigma"].median()), float(fb[n]["abs_sigma"].median()))
              for n in names}
        result = {
            "meta": meta,
            "ages": age_list, "rows_per_age": B,
            "drift_median": {n: [float(drift[n]["abs_sigma"][torch.from_numpy(ages == a)].median())
                                 for a in age_list] for n in names},
            "drift_median_enc_mean": [meter.enc_median_mean(drift, torch.from_numpy(ages == a))
                                      for a in age_list],
            "drift": meter.by_age(drift, ages),
            "floor": {"a": meter.pooled(fa), "b": meter.pooled(fb),
                      "used_median": fl, "line": float(np.mean(list(fl.values())))},
            "sanity": {**sanity, "floor_a_bitwise_identical": all(fa[n]["max_abs"] == 0.0 for n in names),
                       "floor_a_max_abs_diff": max(fa[n]["max_abs"] for n in names)},
            "train_logs": {"warmup": warm_logs}, "seconds": time.time() - t0,
        }
        _write(result, args.out, is_main_process())
        return

    # ---------------------------------------------------------------- measured windows
    per_window, measure_logs = [], []
    pooled = {"drift": [], "ages": [], "a": [], "b": [], "c": []}
    floor_a_max = 0.0
    for j in range(n_windows):
        assert trainer._pending_window is None
        w = trainer.schedule.peek()
        blocks, ages, births = retained_blocks(trainer, w)
        F_R_old = {n: trainer.cache.retained(n, w.retained_ids).float() for n in names}
        F_R_now = reroll_refresh(trainer, w.retained_ids)
        F_R_now2 = reroll_refresh(trainer, w.retained_ids)
        F_R_train = reroll_training(trainer, blocks)
        F_A = reroll_training(trainer, [w.active_ids])          # exactly pass 1's batching
        Y_W = {n: trainer._reference_window(n, w.all_ids) for n in names}

        rng = np.random.default_rng([args.resample_seed, w.step])
        pool = np.setdiff1d(np.arange(trainer.store.num_rows), w.all_ids)
        R2 = np.sort(rng.choice(pool, size=w.retained_ids.size, replace=False))
        F_R2 = reroll_refresh(trainer, R2)
        ids_R2A = np.concatenate([R2, w.active_ids])
        Y_R2A = {n: trainer._reference_window(n, ids_R2A) for n in names}

        d_rows = meter.row_metrics(F_R_old, F_R_now)
        a_rows = meter.row_metrics(F_R_now2, F_R_now)
        b_rows = meter.row_metrics(F_R_train, F_R_now)
        floor_a_max = max(floor_a_max, *(a_rows[n]["max_abs"] for n in names))

        g_now = meter.featgrad(F_R_now, F_A, Y_W)
        grads = {"cache_vs_now": meter.compare(meter.featgrad(F_R_old, F_A, Y_W), g_now),
                 "floor": meter.compare(meter.featgrad(F_R_train, F_A, Y_W), g_now),
                 "resample_vs_now": meter.compare(meter.featgrad(F_R2, F_A, Y_R2A), g_now)}
        rec = {"window_step": int(w.step), "births": [int(s) for s in births],
               "age_counts": {str(a): int((ages == a).sum()) for a in sorted(set(ages.tolist()))},
               "grad": grads}

        if trainer.critic is not None:   # step 2000: the full trained loss as well
            txt = lambda ids: trainer.store.text_features(ids, device=device,
                                                          dtype=trainer.feature_dtype) \
                if trainer.critic.conditional else None
            tau_A, tau_W = txt(w.active_ids), txt(w.all_ids)
            t_now, l_now = meter.featgrad_total(trainer.critic, F_R_now, F_A, Y_W, tau_W, tau_A)
            t_cache, l_cache = meter.featgrad_total(trainer.critic, F_R_old, F_A, Y_W, tau_W, tau_A)
            t_train, l_train = meter.featgrad_total(trainer.critic, F_R_train, F_A, Y_W, tau_W, tau_A)
            t_res, l_res = meter.featgrad_total(trainer.critic, F_R2, F_A, Y_R2A, txt(ids_R2A), tau_A)
            rec["grad_total"] = {"cache_vs_now": meter.compare(t_cache, t_now),
                                 "floor": meter.compare(t_train, t_now),
                                 "resample_vs_now": meter.compare(t_res, t_now),
                                 "lambda": {"now": l_now, "cache": l_cache,
                                            "train_chunking": l_train, "resample": l_res}}

        train_step(measure_logs)          # the ordinary update of this window
        pass1 = {n: trainer.cache.entries[n].features[-B:] for n in names}
        c_rows = meter.row_metrics(pass1, F_A)

        # the summary-table numbers of this window
        floor_b_med = {n: float(b_rows[n]["abs_sigma"].median()) for n in names}
        floor_c_med = {n: float(c_rows[n]["abs_sigma"].median()) for n in names}
        rec["table"] = {
            **{f"drift_age{a}": meter.enc_median_mean(d_rows, torch.from_numpy(ages == a))
               for a in sorted(set(ages.tolist()))},
            "floor": float(np.mean([max(floor_b_med[n], floor_c_med[n]) for n in names])),
            "floor_by_encoder": {n: max(floor_b_med[n], floor_c_med[n]) for n in names},
            "grad_cache_pct": 100 * grads["cache_vs_now"]["all"]["rel"],
            "grad_floor_pct": 100 * grads["floor"]["all"]["rel"],
            "grad_resample_pct": 100 * grads["resample_vs_now"]["all"]["rel"],
            "cos_cache": grads["cache_vs_now"]["all"]["cos"],
            "cos_resample": grads["resample_vs_now"]["all"]["cos"],
            "log_drift_age": {str(a): float(torch.cat([d_rows[n]["rel_joint"][torch.from_numpy(ages == a)]
                                                       for n in names]).mean())
                              for a in sorted(set(ages.tolist()))},
        }
        per_window.append(rec)
        for k, v in (("drift", d_rows), ("a", a_rows), ("b", b_rows), ("c", c_rows)):
            pooled[k].append(v)
        pooled["ages"].append(ages)
        t = rec["table"]
        logger.info("[E1] window %d/%d (step %d) | drift age1/2/3 %.4g/%.4g/%.4g sigma | floor %.3g "
                    "| grad %% cache %.2f floor %.2f resample %.2f", j + 1, n_windows, w.step,
                    t.get("drift_age1", float("nan")), t.get("drift_age2", float("nan")),
                    t.get("drift_age3", float("nan")), t["floor"], t["grad_cache_pct"],
                    t["grad_floor_pct"], t["grad_resample_pct"])
        del F_R_old, F_R_now, F_R_now2, F_R_train, F_A, F_R2, Y_W, Y_R2A

    # ---------------------------------------------------------------- assemble
    encs = names + ["all"]

    def grad_block(key):
        if key not in per_window[0]:
            return None
        out = {}
        for c in COMPARISONS:
            out[c] = {}
            for n in encs:
                rel = [r[key][c][n]["rel"] for r in per_window]
                cos = [r[key][c][n]["cos"] for r in per_window]
                out[c][n] = {"rel": _ms(rel), "cos": _ms(cos),
                             "per_window": {"rel": rel, "cos": cos}}
        if key == "grad_total":
            out["lambda"] = {k: _ms([r[key]["lambda"][k] for r in per_window])
                             for k in ("now", "cache", "train_chunking", "resample")}
        return out

    table_keys = [k for k in per_window[0]["table"] if not isinstance(per_window[0]["table"][k], dict)]
    all_ages = np.concatenate(pooled["ages"])
    result = {
        "meta": meta,
        "drift": meter.by_age(_cat_rows(pooled["drift"]), all_ages),
        "floor": {k: meter.pooled(_cat_rows(pooled[k])) for k in ("a", "b", "c")},
        "grad": grad_block("grad"),
        "grad_total": grad_block("grad_total"),
        "table": {k: _ms([r["table"][k] for r in per_window]) for k in table_keys},
        "log_drift_age": {a: _ms([r["table"]["log_drift_age"][a] for r in per_window
                                  if a in r["table"]["log_drift_age"]])
                          for a in sorted({a for r in per_window for a in r["table"]["log_drift_age"]})},
        "sanity": {**sanity,
                   "windows_age_counts": [r["age_counts"] for r in per_window],
                   "floor_a_bitwise_identical": floor_a_max == 0.0,
                   "floor_a_max_abs_diff": floor_a_max},
        "per_window": per_window,
        "train_logs": {"warmup": warm_logs, "measured": measure_logs},
        "seconds": time.time() - t0,
    }
    _write(result, args.out, is_main_process())


def _write(result: dict, path: str, main: bool) -> None:
    if not main:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".partial", "w") as f:
        json.dump(result, f, indent=2)
    os.replace(path + ".partial", path)
    logger.info("[E1] -> %s (%.1f min)", path, result["seconds"] / 60)


# ----------------------------------------------------------------------------- outputs
TABLE_COLS = [("drift_age1", "drift age 1 (σ)", 1), ("drift_age2", "drift age 2 (σ)", 1),
              ("drift_age3", "drift age 3 (σ)", 1), ("floor", "floor (σ)", 1),
              ("grad_cache_pct", "grad change cache vs now (%)", 1),
              ("grad_floor_pct", "grad change floor (%)", 1),
              ("grad_resample_pct", "grad change resampled (%)", 1),
              ("cos_cache", "cos(g_cache, g_now)", 1), ("cos_resample", "cos(g_resamp, g_now)", 1)]


def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    return f"{x:.4f}" if abs(x) >= 1e-3 else f"{x:.2e}"


def summarize(argv) -> None:
    ap = argparse.ArgumentParser(prog="measure_cache_drift.py summarize")
    ap.add_argument("--dir", required=True, help="outputs/cache_drift: step_*.json, age_curve_*.json")
    ap.add_argument("--train-log", help="the main run's train_log.jsonl (HF), for the drift_age_k check")
    ap.add_argument("--log-window", type=int, default=30, help="probes within +- this many steps")
    args = ap.parse_args(argv)
    d = args.dir
    steps = sorted(f for f in os.listdir(d) if f.startswith("step_") and f.endswith(".json"))
    runs = [(f, json.load(open(os.path.join(d, f)))) for f in steps]
    if not runs:
        raise SystemExit(f"no step_*.json in {d}")

    # ---- summary.md / summary.csv
    head = "| checkpoint | " + " | ".join(c[1] for c in TABLE_COLS) + " |"
    lines = ["# Cache drift (E1): main run K128 · GAN", "",
             "Each cell: mean ± std over the measured windows. drift = |phi_old - phi_now| / sigma_e "
             "(image block), per-encoder median, averaged over the 3 encoders; floor = per encoder "
             "max(median (b), median (c)) in the same unit, averaged over encoders; grad change = "
             "|g - g_now| / |g_now| of dL_MMD/dF_A, the 3 encoders concatenated. The last row averages "
             "the checkpoint rows (± = std of the three means).", "",
             head, "|" + "---|" * (len(TABLE_COLS) + 1)]
    csv_rows, means = [], {k: [] for k, _, _ in TABLE_COLS}
    for f, j in runs:
        cells, row = [], {"checkpoint": f"step {j['meta']['step']}"}
        for k, _, _ in TABLE_COLS:
            t = j["table"].get(k)
            if t and t.get("n"):
                cells.append(f"{_fmt(t['mean'])} ± {_fmt(t['std'])}")
                means[k].append(t["mean"])
                row[k + "_mean"], row[k + "_std"] = t["mean"], t["std"]
            else:
                cells.append("-")
        state = "" if j["meta"]["optimizer"] == "restored" else " †"
        lines.append(f"| step {j['meta']['step']}{state} | " + " | ".join(cells) + " |")
        csv_rows.append(row)
    avg = [f"{_fmt(np.mean(means[k]))} ± {_fmt(np.std(means[k]))}" if means[k] else "-"
           for k, _, _ in TABLE_COLS]
    lines.append("| mean of checkpoints | " + " | ".join(avg) + " |")
    lines += ["", "† weights-only checkpoint: new AdamW8bit, critic off, every update L_MMD only "
                  "(see notes.md)."]
    open(os.path.join(d, "summary.md"), "w").write("\n".join(lines) + "\n")
    import csv
    keys = ["checkpoint"] + [k + s for k, _, _ in TABLE_COLS for s in ("_mean", "_std")]
    with open(os.path.join(d, "summary.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(csv_rows)
    print("\n".join(lines))

    # ---- age curve png
    curves = sorted(f for f in os.listdir(d) if f.startswith("age_curve_") and f.endswith(".json"))
    for f in curves:
        c = json.load(open(os.path.join(d, f)))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for n, ys in c["drift_median"].items():
            ax.plot(c["ages"], ys, marker="o", ms=3, lw=1.5, label=n)
        ax.axhline(c["floor"]["line"], color="0.4", ls="--", lw=1,
                   label=f"floor (max of a, b; mean over encoders) = {c['floor']['line']:.2g}")
        ax.set_xlabel("row age (updates since the row was generated)")
        ax.set_ylabel(r"median $\|\phi_{old}-\phi_{now}\|\,/\,\sigma_e$")
        ax.set_title(f"Cache drift by age, step {c['meta']['step']}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(d, f.replace(".json", ".png")), dpi=200)
        plt.close(fig)

    # ---- notes.md (facts the agent then reviews; its own remarks go in the last section)
    notes = ["# Cache drift (E1): notes", "",
             "## Differences from the requested protocol", ""]
    notes += [f"- {x}" for x in DEVIATIONS]
    notes += ["", "## Per-checkpoint state", ""]
    for f, j in runs:
        m = j["meta"]
        notes.append(f"- step {m['step']} ({m['checkpoint']}): lr {m['lr']:.3g}; optimizer "
                     f"{m['optimizer']}; critic {m['critic']}; cache {m['cache']}; critic on in "
                     f"warm-up: {m['critic_on_during_warmup']}; {m['gpu_count']} x {m['gpu']}; commit "
                     f"{m['commit']}{' (uncommitted changes)' if m['uncommitted_changes'] else ''}")
    notes += ["", "## Automatic checks", ""]
    for f, j in runs:
        s = j["sanity"]
        bad_win = [i for i, a in enumerate(s["windows_age_counts"]) if a != s["after_warmup"]["expected"]]
        notes.append(f"- step {j['meta']['step']}: ages after warm-up {s['after_warmup']['age_counts']} "
                     f"({'OK' if s['after_warmup']['passed'] else 'FAILED'}); measured windows with other "
                     f"age counts: {bad_win or 'none'}; floor (a) bitwise identical: "
                     f"{s['floor_a_bitwise_identical']} (max |diff| {s['floor_a_max_abs_diff']:.3g})")
        per_enc = {n: j["drift"][n]["1"]["abs_sigma"]["median"] for n in j["meta"]["sigma"]
                   if "1" in j["drift"].get(n, {})}
        if per_enc:
            lo, hi = min(per_enc.values()), max(per_enc.values())
            flag = " <- one encoder > 2x another" if lo > 0 and hi / lo > 2 else ""
            notes.append(f"  - age-1 drift median by encoder (σ): "
                         + ", ".join(f"{n} {_fmt(v)}" for n, v in per_enc.items()) + flag)
        fl = j["table"].get("floor", {}).get("mean")
        a1 = j["table"].get("drift_age1", {}).get("mean")
        if fl is not None and a1:
            notes.append(f"  - floor / age-1 drift = {fl / a1:.3g}"
                         + (" <- the floor is not small next to the drift" if fl / a1 > 0.2 else ""))
    for f in curves:
        c = json.load(open(os.path.join(d, f)))
        s = c["sanity"]
        y = c["drift_median_enc_mean"]
        ratio = f"x{y[-1] / y[0]:.3g}" if y[0] else "-"
        notes.append(f"- {f}: ages {c['ages'][0]}..{c['ages'][-1]}, drift (enc-mean of medians) "
                     f"age 1 {_fmt(y[0])}, age {c['ages'][-1]} {_fmt(y[-1])} ({ratio}); "
                     f"floor line {_fmt(c['floor']['line'])}; floor (a) bitwise identical: "
                     f"{s['floor_a_bitwise_identical']}")

    notes += ["", "## Joint-definition drift vs the training log's drift_age_k", ""]
    if args.train_log:
        log = [json.loads(x) for x in open(args.train_log) if x.strip()]
        probes = [x for x in log if "drift_mean" in x]
        notes += ["drift_age_k in the log is the mean of ||x_cached - x_new|| / ||x_new|| over 64 "
                  "probed rows x 3 encoders (joint features), measured after the update on the next "
                  "window; here: the same definition on all 96 rows, mean over the measured windows. "
                  "The log's run had its trained critic and Adam state and refreshed the cache "
                  "(191 of 200 probes); a weights-only checkpoint here has neither.", "",
                  "| step | age | ours (mean ± std) | log (mean of probes) | probes used | ratio |",
                  "|---|---|---|---|---|---|"]
        for f, j in runs:
            s0 = j["meta"]["step"]
            near = [p for p in probes if abs(p["step"] - s0) <= args.log_window]
            for a, v in j["log_drift_age"].items():
                lv = [p[f"drift_age_{a}"] for p in near if f"drift_age_{a}" in p]
                lm = float(np.mean(lv)) if lv else float("nan")
                notes.append(f"| {s0} | {a} | {_fmt(v['mean'])} ± {_fmt(v['std'])} | {_fmt(lm)} | "
                             f"{[p['step'] for p in near]} | {v['mean'] / lm:.3g} |")
    else:
        notes.append("(not compared: pass --train-log)")
    notes += ["", "## Agent remarks", "", "<to be written by whoever ran this: anything unusual "
              "in the logs, reruns, and whether the log comparison above agrees>"]
    open(os.path.join(d, "notes.md"), "w").write("\n".join(notes) + "\n")
    print(f"\nwrote {d}/summary.md, summary.csv, notes.md" +
          "".join(f", {c.replace('.json', '.png')}" for c in curves))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "measure":
        measure(sys.argv[2:])
    else:
        raise SystemExit("usage: measure_cache_drift.py {measure|summarize} ... (see the docstring)")
