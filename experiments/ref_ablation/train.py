"""Entry point of the reference-side ablation.

    torchrun --nproc_per_node=2 -m experiments.ref_ablation.train \
        experiments/ref_ablation/configs/ref_random.yaml --set seed=1 --set ref_ablation.ref_seed=1
    (or: bash experiments/ref_ablation/run.sh random 1)

``train`` and ``main`` are COPIED FROM rdm/sw_lmmd/launch.py (commit d5b2e9a) and changed only where
marked ``# [ref-ablation]``:

* the trainer is the one :func:`rdm.sw_lmmd.launch.build_trainer` builds -- unchanged generator,
  battery, store, window, optimizer, critic, lr schedule -- turned into a
  :class:`~experiments.ref_ablation.trainer.RefAblationTrainer`;
* the run directory is ``<ref_mode>-<exp_name>`` (``global-ungrouped``), ``-s<seed>`` added for
  seeds other than 0, so modes and seeds never share one;
* before the first step, :meth:`RefAblationTrainer.attraction_diagnostic` prints the three
  attractions on the same window under the same weights (``pre_diagnostic.json``);
* the console prints one line per encoder; the log carries the peak GPU memory of each step;
* ``save_final: false`` skips the final checkpoint (smoke runs); ``--diagnose-only`` stops after the
  diagnostic.
"""
from __future__ import annotations

import logging
import os

import torch

from rdm.sw_lmmd.launch import (apply_override, build_trainer, cache_from_config,
                                resolve_reference_extension)
from rdm.sw_lmmd.reference_store import ReferenceFeatureStore

from .config import ref_ablation_from_config
from .nystrom_global import GlobalNystrom
from .trainer import RefAblationTrainer

logger = logging.getLogger("rdm")


def run_name(cfg, ref) -> str:
    """``<ref_mode>-<exp_name>`` -- ``global-ungrouped`` for the ablation configs at seed 0 -- plus
    ``-s<seed>`` for any other seed and ``-o<order_seed>`` when the window order is not the configs'
    default 3407."""
    seed = int(getattr(cfg, "seed", 0))
    name = f"{ref.ref_mode}-{getattr(cfg, 'exp_name', 'sw-lmmd')}" + (f"-s{seed}" if seed else "")
    order_seed = int(dict(getattr(cfg, "window", None) or {}).get("order_seed", 3407))
    return name if order_seed == 3407 else f"{name}-o{order_seed}"


def check_nystrom(cfg, ref):
    """Before any model loads: in global mode the bundles must exist and match the store (a cheap
    check -- the store is memory-mapped). Returns whether bundles are available at all."""
    names = list(cfg.encoders)
    if not GlobalNystrom.available(ref.nystrom_dir, names):
        if ref.ref_mode == "global":
            raise FileNotFoundError(
                f"ref_mode=global needs {ref.nystrom_dir}/<encoder>.pt for {names}; build them with "
                f"python -m experiments.ref_ablation.build_landmarks <config>")
        return False
    store = ReferenceFeatureStore(cfg.reference_root, names, require_context=False,
                                  extension=resolve_reference_extension(cfg))
    GlobalNystrom.load(ref.nystrom_dir, names, store=store, joint=getattr(cfg, "joint", True))
    return True


def build_ref_trainer(cfg, device: str = "cuda") -> RefAblationTrainer:
    """``build_trainer`` (unchanged), then the attraction's reference side."""
    ref = ref_ablation_from_config(cfg)
    have_bundles = check_nystrom(cfg, ref)                 # fail before the 4B model loads
    base = build_trainer(cfg, device=device)
    nystrom = GlobalNystrom.load(ref.nystrom_dir, base.encoder_names, store=base.store,
                                 joint=base.joint, device=device) if have_bundles else None
    if nystrom is None:
        logger.info("[ref-ablation] no Nystrom bundles in %s: ref/<enc>/attraction_global is not "
                    "logged", ref.nystrom_dir)
    trainer = RefAblationTrainer.from_trainer(base, ref_mode=ref.ref_mode, ref_seed=ref.ref_seed,
                                              nystrom=nystrom, grad_norms=ref.grad_norms)
    logger.info("[ref-ablation] ref_mode=%s | ref_seed=%d | K=%d attraction rows%s | Nystrom %s",
                ref.ref_mode, ref.ref_seed, trainer.schedule.window_size,
                {"local": " = the window's own rows",
                 "random": f" drawn uniformly from all {trainer.store.num_rows} each step",
                 "global": " -> all N rows through the Nystrom mean embedding"}[ref.ref_mode],
                "off" if nystrom is None else
                f"m={nystrom.num_landmarks(base.encoder_names[0])} ({ref.nystrom_dir})")
    return trainer


def format_diagnostic(diag: dict) -> str:
    """The pre-training table: one row per encoder, the attraction under each reference side."""
    lines = [f"[ref-ablation] attraction bracket on ONE window (window step {diag['window_step']}, "
             f"B={diag['B']} fresh rows vs K={diag['K']} reference rows, N={diag['N']}), "
             f"current weights, this run trains '{diag['ref_mode']}':",
             f"  {'encoder':<14}{'repulsion':>11}{'local':>11}{'(same pr.)':>11}{'(cross)':>11}"
             f"{'random':>11}{'+-std':>10}{'global Nys':>12}{'global ex.':>12}{'Nys err':>9}"
             f"{'grad cos':>9}"]
    for name, r in diag["encoders"].items():
        def v(key, fmt="{:>11.3e}"):
            return fmt.format(r[key]) if key in r else f"{'-':>{len(fmt.format(0.0))}}"
        lines.append(f"  {name:<14}{v('repulsion')}{v('local')}{v('local_same_prompt')}"
                     f"{v('local_cross_prompt')}{v('random_mean')}{v('random_std', '{:>10.1e}')}"
                     f"{v('global_nystrom', '{:>12.3e}')}{v('global_exact', '{:>12.3e}')}"
                     f"{v('global_rel_err_of_mean', '{:>9.2%}')}{v('global_grad_cos', '{:>9.3f}')}")
    for name, r in diag["encoders"].items():
        ratios = [f"random/local {r['random_mean'] / r['local']:.3f}"]
        if "global_nystrom" in r:
            ratios.append(f"global/local {r['global_nystrom'] / r['local']:.3f}")
        if "random_step1" in r:
            ratios.append(f"this run's first draw {r['random_step1']:.3e}")
        lines.append(f"  {name:<14}" + " | ".join(ratios))
    return "\n".join(lines)


def train(cfg, device: str = "cuda", trainer: RefAblationTrainer | None = None,
          diagnose_only: bool = False) -> RefAblationTrainer:
    """Bootstrap, then run ``cfg.steps`` sliding-window updates with drift control.

    COPIED FROM rdm.sw_lmmd.launch.train (commit d5b2e9a); see the module docstring for what changed.
    """
    import json
    import time

    from rdm.sw_lmmd.refresh import probe_drift, refresh_retained, should_refresh
    from rdm.utils.distributed import is_main_process

    ref = ref_ablation_from_config(cfg)                                          # [ref-ablation]
    trainer = trainer if trainer is not None else build_ref_trainer(cfg, device=device)
    save_cache = bool(getattr(cfg, "save_cache", False))
    save_optimizer = bool(getattr(cfg, "save_optimizer", False))
    if save_optimizer and trainer.clip_module is not None:
        raise ValueError("save_optimizer under FSDP would store only rank 0's optimizer shard, "
                         "which cannot resume anything; leave it false")
    save_freq = int(getattr(cfg, "save_freq", 50))
    save_resume = bool(getattr(cfg, "save_resume", False))
    save_final = bool(getattr(cfg, "save_final", True))                          # [ref-ablation]
    resume_every = int(getattr(cfg, "resume_every", 0) or save_freq)
    resume_from = getattr(cfg, "resume_from", None) or None
    cache_cfg = cache_from_config(cfg)
    out_dir = os.path.join(getattr(cfg, "output_dir", "./work_dirs"), run_name(cfg, ref))  # [ref-ablation]
    os.makedirs(out_dir, exist_ok=True)
    jsonl = open(os.path.join(out_dir, "train_log.jsonl"), "a") if is_main_process() else None
    cuda = torch.cuda.is_available() and str(device).startswith("cuda")          # [ref-ablation]

    def write(state, path):
        tmp = path + ".partial"
        torch.save(state, tmp)
        os.replace(tmp, path)

    def checkpoint():
        # EVERY rank builds the state: under FSDP the full state dict is a collective gather,
        # and a rank-0-only call blocks until the NCCL timeout. Only rank 0 writes it.
        state = trainer.state_dict(with_cache=save_cache, with_optimizer=save_optimizer,
                                   with_critic=False)
        if is_main_process():
            path = os.path.join(out_dir, f"step_{trainer.step_idx:07d}.pth")
            write(state, path)
            logger.info("[sw_lmmd] checkpoint -> %s", path)

    def save_resume_state():
        state = trainer.state_dict(with_cache=True, with_optimizer=trainer.clip_module is None)
        if is_main_process():
            t = time.time()
            write(state, os.path.join(out_dir, "resume.pth"))
            logger.info("[sw_lmmd] resume state (step %d) -> %s in %.0f s", trainer.step_idx,
                        os.path.join(out_dir, "resume.pth"), time.time() - t)

    t0 = time.time()
    if resume_from:
        state = torch.load(resume_from, map_location="cpu", weights_only=False, mmap=True)
        restored = trainer.load_state_dict(state)
        del state
        logger.info("[sw_lmmd] resumed %s at step %d (weights %s, optimizer %s, cache %s%s)",
                    resume_from, restored["step"],
                    "loaded" if restored["model"] else "from load_from",
                    "restored" if restored["optimizer"] else "FRESH",
                    "restored" if restored["cache"] else "re-bootstrapped",
                    "" if "critic" not in restored else
                    ", critic " + ("restored" if restored["critic"] else "FRESH"))
    if not trainer.cache.initialized:
        logger.info("[sw_lmmd] bootstrap ...")
        boot = trainer.bootstrap()
        logger.info("[sw_lmmd] bootstrap done: %s rows (%s/rank) in %.1f s",
                    boot["bootstrap_rows"], boot["bootstrap_rows_per_rank"], time.time() - t0)

    if ref.pre_diagnostic or diagnose_only:                                      # [ref-ablation]
        t = time.time()
        diag = trainer.attraction_diagnostic(random_draws=ref.diagnostic_random_draws,
                                             exact_global=ref.diagnostic_exact_global)
        diag["seconds"] = time.time() - t
        if is_main_process():
            logger.info("%s\n  (%.0f s)", format_diagnostic(diag), diag["seconds"])
            with open(os.path.join(out_dir, f"pre_diagnostic_step{trainer.step_idx:07d}.json"),
                      "w") as f:
                json.dump(diag, f, indent=2)
            jsonl.write(json.dumps({"pre_diagnostic": diag}) + "\n")
            jsonl.flush()
    if diagnose_only:                                                            # [ref-ablation]
        if is_main_process():
            jsonl.close()
        return trainer

    probe_every = int(getattr(cfg, "probe_every", 0) or cache_cfg.refresh_every_windows or 0)
    drift = None
    for _ in range(max(0, int(cfg.steps) - trainer.step_idx)):
        if cuda:                                                                 # [ref-ablation]
            torch.cuda.reset_peak_memory_stats(device)
        step_t = time.time()
        logs = trainer.step()
        logs["seconds"] = time.time() - step_t
        if cuda:                                                                 # [ref-ablation]
            logs["mem_peak_gb"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
            logs["mem_reserved_gb"] = torch.cuda.memory_reserved(device) / 2 ** 30

        window = trainer.schedule.peek()          # the window the NEXT step will use
        if probe_every and trainer.step_idx % probe_every == 0:
            stats = probe_drift(trainer, window, cache_cfg.refresh_probe_rows)
            drift = stats.get("drift_mean")
            logs.update(stats)
        if should_refresh(trainer, window, drift, cache_cfg):
            logs.update(refresh_retained(trainer, window))
            drift = None

        if is_main_process():
            if trainer.step_idx % int(getattr(cfg, "print_freq", 1)) == 0:
                gan = "" if "gan/d_acc" not in logs else \
                    " | D acc %.3f real %+.3f fake %+.3f | lambda %s" % (
                        logs["gan/d_acc"], logs["gan/d_real"], logs["gan/d_fake"],
                        "%.4g" % logs["gan/lambda"] if "gan/lambda" in logs else "off")
                logger.info("step %d | force %.6f | raw_mmd2 %.6g | grad_norm %s | lr %.3g | "
                            "cache_age %d | %.1f s%s", logs["step"], logs["force"],
                            logs.get("raw_mmd2", float("nan")), logs["grad_norm"], logs["lr"],
                            logs["cache_max_age"], logs["seconds"], gan)
                logger.info("  %s", format_step_lines(trainer, logs))            # [ref-ablation]
            jsonl.write(json.dumps(logs) + "\n")
            jsonl.flush()
        if trainer.step_idx % save_freq == 0:
            checkpoint()
        if save_resume and trainer.step_idx % resume_every == 0:
            save_resume_state()

    if save_final and trainer.step_idx % save_freq:     # the final step, unless it was just saved
        checkpoint()
    if save_resume and trainer.step_idx % resume_every:
        save_resume_state()
    if is_main_process():
        jsonl.close()
        logger.info("[sw_lmmd] done: %d steps in %.2f h", trainer.step_idx,
                    (time.time() - t0) / 3600)
    return trainer


def format_step_lines(trainer: RefAblationTrainer, logs: dict) -> str:
    """One console line per encoder: the trained attraction, the repulsion, the force, the weighted
    force that is back-propagated, the per-term feature-gradient norms, the window's own attraction
    and the global one (when bundles are loaded). Cross-rank means; see the trainer."""
    out = []
    for name in trainer.encoder_names:
        p = f"ref/{name}/"
        if p + "attraction" not in logs:
            continue
        line = (f"{name:<12} [{trainer.ref_mode}] att {logs[p + 'attraction']:.4e}  "
                f"rep {logs[p + 'repulsion']:.4e}  force {logs[p + 'force']:+.4e}  "
                f"w*force {logs[p + 'force_weighted']:+.4e}")
        if p + "grad_attraction" in logs:
            line += (f"  |g_att| {logs[p + 'grad_attraction']:.3e}  |g_rep| "
                     f"{logs[p + 'grad_repulsion']:.3e}  |g_force| {logs[p + 'grad_force']:.3e}  "
                     f"cos {logs[p + 'grad_cos_att_rep']:+.3f}")
        line += f"  att_local {logs[p + 'attraction_local']:.4e}"
        if p + "attraction_global" in logs:
            line += f"  att_global {logs[p + 'attraction_global']:.4e}"
        out.append(line)
    if "mem_peak_gb" in logs:
        out.append(f"peak mem {logs['mem_peak_gb']:.2f} GB (reserved {logs['mem_reserved_gb']:.2f})")
    return "\n  ".join(out)


def main() -> None:
    """``torchrun --nproc_per_node=2 -m experiments.ref_ablation.train <config> [--set K=V ...]``

    COPIED FROM rdm.sw_lmmd.launch.main (commit d5b2e9a), plus ``--diagnose-only``.
    """
    import argparse

    from rdm.train.launch import load_config
    from rdm.utils.distributed import setup_distributed
    from rdm.utils.logging import setup_logging
    from rdm.utils.seed import fix_random_seeds

    ap = argparse.ArgumentParser(description="SW-LMMD reference-side ablation")
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key; dotted for nested blocks "
                         "(e.g. --set ref_ablation.ref_seed=1 --set steps=5)")
    ap.add_argument("--diagnose-only", action="store_true",                      # [ref-ablation]
                    help="bootstrap, print the three attractions on the first window, exit")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for override in args.set:
        apply_override(cfg, override)
    if getattr(cfg, "method", None) != "sw_lmmd":
        raise SystemExit(f"{args.config} is not a SW-LMMD config (method: {getattr(cfg,'method',None)!r})")
    ref_ablation_from_config(cfg)                    # [ref-ablation] refuse a bad block before setup

    rank, world, local_rank = setup_distributed()
    setup_logging(rank=rank)
    # The window schedule and the per-row noise are seeded identically on every rank by
    # construction; the global seed only affects anything left to torch's default generator.
    fix_random_seeds(getattr(cfg, "seed", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    train(cfg, device=device, diagnose_only=args.diagnose_only)


if __name__ == "__main__":
    main()
