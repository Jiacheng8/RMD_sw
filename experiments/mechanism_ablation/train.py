"""Entry point of the own-reference (mechanism) ablation.

    torchrun --nproc_per_node=2 -m experiments.mechanism_ablation.train \
        experiments/mechanism_ablation/configs/window_nodiag.yaml
    (or: bash experiments/mechanism_ablation/run_arm.sh window-nodiag)

The training loop is ``experiments.ref_ablation.train.train`` -- the loop the Table 2 Global arm
(``global-ablation-nogroup``) was trained with -- called with a :class:`MechanismTrainer` built from
the same ``build_trainer``. Two module-level helpers of that loop are swapped for this ablation:
``run_name`` (the run directory is ``exp_name``, e.g. ``window-nodiag``) and
``format_step_lines`` (adds the own-term line to the console). ``run_info.json`` in the run
directory records the commit, the resolved config, the machine, the GPU count and every
(re)start's wall-clock segment.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import time
from types import SimpleNamespace

import torch

import experiments.ref_ablation.train as ref_train
from experiments.ref_ablation.config import ref_ablation_from_config
from experiments.ref_ablation.nystrom_global import GlobalNystrom
from rdm.sw_lmmd.launch import apply_override, build_trainer

from .config import mechanism_from_config
from .trainer import MechanismTrainer

logger = logging.getLogger("rdm")
_format_ref_lines = ref_train.format_step_lines


def run_name(cfg, ref=None) -> str:
    """``exp_name`` (+ ``-s<seed>`` for seeds other than 0, ``-o<order_seed>`` off 3407)."""
    seed = int(getattr(cfg, "seed", 0))
    name = mechanism_from_config(cfg).arm + (f"-s{seed}" if seed else "")
    order_seed = int(dict(getattr(cfg, "window", None) or {}).get("order_seed", 3407))
    return name if order_seed == 3407 else f"{name}-o{order_seed}"


def format_step_lines(trainer, logs: dict) -> str:
    lines = _format_ref_lines(trainer, logs)
    if "mech/all/share" not in logs:
        return lines
    mech = getattr(trainer, "mech", None)
    tag = {"nodiag": "removed own", "main": "own", "global_self": f"beta*own (beta {mech.self_beta:.4g})"}
    own = "  ".join(f"{n} k_own {logs[f'mech/{n}/own_kernel']:.3e} term {logs[f'mech/{n}/own_term']:+.3e}"
                    for n in trainer.encoder_names)
    return (lines + f"\n  [{tag[mech.variant]}] |g_own|/|g_total| {logs['mech/all/share']:.3f}  "
            f"cos(own,total) {logs['mech/all/cos_own_total']:+.3f}  cos(rest,total) "
            f"{logs['mech/all/cos_rest_total']:+.3f} | {own}")


ref_train.run_name = run_name                  # the loop resolves both at call time
ref_train.format_step_lines = format_step_lines


def build_mech_trainer(cfg, device: str = "cuda") -> MechanismTrainer:
    """``build_trainer`` (unchanged), the reference side, then the mechanism variant."""
    ref = ref_ablation_from_config(cfg)
    mech = mechanism_from_config(cfg)
    have_bundles = ref_train.check_nystrom(cfg, ref)       # fail before the 4B model loads
    base = build_trainer(cfg, device=device)
    nystrom = GlobalNystrom.load(ref.nystrom_dir, base.encoder_names, store=base.store,
                                 joint=base.joint, device=device) if have_bundles else None
    trainer = MechanismTrainer.from_trainer_mech(base, mech, ref_mode=ref.ref_mode,
                                                 ref_seed=ref.ref_seed, nystrom=nystrom,
                                                 grad_norms=ref.grad_norms)
    logger.info("[mechanism] arm %s | variant %s | beta %g | ref_mode %s | row order %s",
                mech.arm, mech.variant, mech.self_beta, ref.ref_mode, trainer.row_order_hash)
    return trainer


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        return None


def _plain(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: _plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj if isinstance(obj, (str, int, float, bool, type(None))) else str(obj)


def write_run_info(cfg, out_dir: str, segment: dict) -> None:
    """Append this (re)start's segment to ``run_info.json``; the first one also stores the setup."""
    path = os.path.join(out_dir, "run_info.json")
    info = json.load(open(path)) if os.path.exists(path) else {}
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] \
        if torch.cuda.is_available() else []
    info.setdefault("arm", mechanism_from_config(cfg).arm)
    info["commit"] = _git("rev-parse", "HEAD")
    info["dirty"] = bool(_git("status", "--porcelain", "--untracked-files=no"))
    info["config"] = _plain(cfg)
    info["machine"] = {"host": socket.gethostname(), "platform": platform.platform(),
                       "gpus_visible": gpus, "torch": torch.__version__}
    info.setdefault("segments", []).append(segment)
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(info, f, indent=2)
    os.replace(tmp, path)


def main() -> None:
    import argparse

    from rdm.train.launch import load_config
    from rdm.utils.distributed import get_world_size, is_main_process, setup_distributed
    from rdm.utils.logging import setup_logging
    from rdm.utils.seed import fix_random_seeds

    ap = argparse.ArgumentParser(description="SW-LMMD own-reference (mechanism) ablation")
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--diagnose-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for override in args.set:
        apply_override(cfg, override)
    if getattr(cfg, "method", None) != "sw_lmmd":
        raise SystemExit(f"{args.config} is not a SW-LMMD config")
    mechanism_from_config(cfg)                           # refuse a bad block before setup

    rank, world, local_rank = setup_distributed()
    setup_logging(rank=rank)
    fix_random_seeds(getattr(cfg, "seed", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    t0, wall0 = time.time(), time.strftime("%Y-%m-%dT%H:%M:%S%z")
    trainer = build_mech_trainer(cfg, device=device)
    out_dir = os.path.join(getattr(cfg, "output_dir", "./work_dirs"), run_name(cfg))
    trainer = ref_train.train(cfg, device=device, trainer=trainer, diagnose_only=args.diagnose_only)
    if is_main_process():
        write_run_info(cfg, out_dir, {
            "started": wall0, "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "wall_seconds": time.time() - t0, "world_size": get_world_size(),
            "resume_from": getattr(cfg, "resume_from", None),
            "last_step": trainer.step_idx, "row_order_hash": trainer.row_order_hash})


if __name__ == "__main__":
    main()
