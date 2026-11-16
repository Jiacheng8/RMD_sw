"""Step 0: the own-row share of the L_MMD feature gradient on main-run checkpoints (no training run).

    torchrun --nproc_per_node=2 -m experiments.mechanism_ablation.probe \
        experiments/mechanism_ablation/configs/main_probe.yaml --ckpt step_0000200.pth \
        --out step0_share_step0200.json [--set reference_root=...]
    python -m experiments.mechanism_ablation.probe --table step0_share_step*.json     # markdown table

The tools/measure_cache_drift.py (Sec. "Cache Drift Measurement") protocol for a weights-only
checkpoint: the trainer is built exactly as for training (``build_trainer``, main-run config), the
checkpoint's weights, step and window position are restored, the critic is OFF (a fresh critic
would push random gradients; the share concerns L_MMD alone), the cache is bootstrapped under the
checkpoint's weights, then ``--warmup`` (3) ordinary L_MMD-only updates and ``--windows`` (8)
consecutive measured windows. AdamW8bit is new (the checkpoint holds no optimizer), so the few
updates the probe makes are slightly larger than in steady state.

For each measured window the trainer's own step splits ``dL_MMD/du`` over the active IMAGE features
(concatenated over encoders, each weighted by its lambda_e = 1/|E|) into
``g_own = -(2/(B K)) d/du_i k(X_i, Y_i)`` and ``g_rest`` = everything else
(``MechanismTrainer``, variant ``main``: the force itself is the method's, unchanged), and reports
``|g_own| / |g_own + g_rest|``, ``cos(g_own, g_own + g_rest)`` and ``cos(g_rest, g_own + g_rest)``,
per encoder and concatenated; the json holds every window and the mean over windows.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np
import torch

logger = logging.getLogger("rdm")
QUANTITIES = ("share", "cos_own_total", "cos_rest_total", "grad_own", "grad_rest", "grad_total")
DEVIATIONS = [
    "weights-only checkpoint: AdamW8bit is new, the critic is OFF for the probe's updates (L_MMD "
    "only), and the cache is bootstrapped under the checkpoint's weights (tools/measure_cache_drift.py "
    "protocol for step_*.pth).",
    "Gradients are w.r.t. the IMAGE columns of the active features (the beta*tau block has no path "
    "to the generator), each encoder weighted by its fixed lambda_e = 1/3; 'all' concatenates them.",
]


def measure(argv) -> None:
    from rdm.sw_lmmd.launch import apply_override
    from rdm.train.launch import load_config
    from rdm.utils.distributed import get_world_size, is_main_process, setup_distributed
    from rdm.utils.logging import setup_logging
    from rdm.utils.seed import fix_random_seeds

    from .train import _git, build_mech_trainer

    ap = argparse.ArgumentParser(prog="python -m experiments.mechanism_ablation.probe")
    ap.add_argument("config")
    ap.add_argument("--ckpt", required=True, help="a main-run step_NNNNNNN.pth (or resume.pth)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--windows", type=int, default=8)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    for o in args.set:
        apply_override(cfg, o)
    rank, world, local_rank = setup_distributed()
    setup_logging(rank=rank)
    fix_random_seeds(getattr(cfg, "seed", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    trainer = build_mech_trainer(cfg, device=device)
    if trainer.mech.variant != "main":
        raise SystemExit("the probe measures the main run: use configs/main_probe.yaml (ref_mode local, both flags off)")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    restored = trainer.load_plain_state(state)
    del state
    if not restored["model"]:
        raise SystemExit(f"{args.ckpt}: no model weights restored (FSDP or no 'model' entry)")
    step0 = trainer.step_idx
    trainer.critic = None                 # L_MMD only (weights-only checkpoint, see the docstring)
    if not trainer.cache.initialized:
        trainer.bootstrap()
    logger.info("[step0] %s: step %d | restored %s | critic OFF | %d warm-up + %d windows",
                args.ckpt, step0, restored, args.warmup, args.windows)

    for _ in range(args.warmup):
        trainer.step()
    windows = []
    for k in range(args.windows):
        logs = trainer.step()
        rec = {"step": logs["step"], "window_step": logs["window_step"]}
        rec.update({key: logs[key] for key in logs if key.startswith("mech/")})
        windows.append(rec)
        logger.info("[step0] window %d/%d (step %d): |g_own|/|g_total| %.3f cos(own,total) %+.3f "
                    "cos(rest,total) %+.3f", k + 1, args.windows, logs["step"],
                    logs["mech/all/share"], logs["mech/all/cos_own_total"],
                    logs["mech/all/cos_rest_total"])

    scopes = list(trainer.encoder_names) + ["all"]
    mean = {s: {q: float(np.mean([w[f"mech/{s}/{q}"] for w in windows])) for q in QUANTITIES}
            for s in scopes}
    std = {s: {q: float(np.std([w[f"mech/{s}/{q}"] for w in windows])) for q in QUANTITIES}
           for s in scopes}
    out = {"checkpoint": os.path.basename(args.ckpt), "checkpoint_path": os.path.abspath(args.ckpt),
           "step": step0, "warmup": args.warmup, "windows": args.windows, "world_size": get_world_size(),
           "restored": restored, "commit": _git("rev-parse", "HEAD"),
           "row_order_hash": trainer.row_order_hash, "deviations": DEVIATIONS,
           "mean_over_windows": mean, "std_over_windows": std, "per_window": windows,
           "seconds": time.time() - t0}
    if is_main_process():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("[step0] step %d: share %.3f (all) -> %s (%.1f min)", step0,
                    mean["all"]["share"], args.out, (time.time() - t0) / 60)


def table(paths) -> str:
    """The Step 0 markdown table: one row per checkpoint and scope (mean +- std over windows)."""
    rows = sorted((json.load(open(p)) for p in paths), key=lambda r: r["step"])
    lines = ["| step | encoder | \\|g_own\\| / \\|g_total\\| | cos(g_own, g_total) | cos(g_rest, g_total) | windows |",
             "|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        for scope in [s for s in r["mean_over_windows"] if s != "all"] + ["all"]:
            m, s = r["mean_over_windows"][scope], r["std_over_windows"][scope]
            name = "**all (concat.)**" if scope == "all" else scope
            lines.append(f"| {r['step']} | {name} | {m['share']:.3f} ± {s['share']:.3f} | "
                         f"{m['cos_own_total']:+.3f} ± {s['cos_own_total']:.3f} | "
                         f"{m['cos_rest_total']:+.3f} ± {s['cos_rest_total']:.3f} | {r['windows']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--table":
        print(table([p for a in sys.argv[2:] for p in sorted(glob.glob(a))]))
    else:
        measure(sys.argv[1:])
