"""Config -> objects for SW-LMMD, keeping the method/machine split honest at build time.

:func:`window_from_config`, :func:`cache_from_config` and :func:`memory_from_config` parse the
three independent blocks of a ``configs/sw_lmmd_*.yaml``. The only place they meet is
:func:`resolve_batching`, which turns the method's ``B`` and the machine's ``micro_batch`` into
a ``grad_accum`` -- and refuses the combination if it does not divide, rather than silently
training on a different effective batch.

:func:`build_trainer` wires the real generator and encoder battery via
:mod:`rdm.train.launch`, so SW-LMMD and iRDM construct their generator identically. It is
therefore only exercisable with the FLUX weights and a built reference store; the parsing
functions above it are pure and are covered by ``tests/test_sw_lmmd_core.py``.
"""
from __future__ import annotations

import logging

import torch

from ..utils.distributed import get_world_size
from .cache import GeneratedWindowCache
from .config import CacheConfig, MemoryPolicy, WindowConfig
from .reference_store import ReferenceFeatureStore
from .trainer import SWLMMDTrainer
from .window_schedule import SlidingWindowSchedule

logger = logging.getLogger("rdm")

_DTYPES = {"fp32": torch.float32, "float32": torch.float32, "bf16": torch.bfloat16,
           "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}


def _block(cfg, name: str) -> dict:
    return dict(getattr(cfg, name, None) or {})


def window_from_config(cfg) -> WindowConfig:
    w = WindowConfig(**{k: v for k, v in _block(cfg, "window").items()
                        if k in WindowConfig.__dataclass_fields__})
    w.validate()
    return w


def cache_from_config(cfg) -> CacheConfig:
    raw = _block(cfg, "cache")
    dtype = _DTYPES.get(str(raw.pop("store_dtype", "float32")).lower(), torch.float32)
    return CacheConfig(store_dtype=dtype,
                       **{k: v for k, v in raw.items() if k in CacheConfig.__dataclass_fields__})


def memory_from_config(cfg) -> MemoryPolicy:
    raw = _block(cfg, "memory")
    dtype = _DTYPES.get(str(raw.pop("param_dtype", "fp32")).lower(), torch.float32)
    policy = MemoryPolicy(param_dtype=dtype,
                          **{k: v for k, v in raw.items()
                             if k in MemoryPolicy.__dataclass_fields__})
    policy.validate()
    return policy


def resolve_batching(cfg, policy: MemoryPolicy, window: WindowConfig,
                     world_size: int | None = None) -> dict:
    """The one place the method's ``B`` meets the machine's micro-batching."""
    world = get_world_size() if world_size is None else int(world_size)
    active = int(getattr(cfg, "active_global_batch", window.stride))
    if active != window.stride:
        raise ValueError(f"active_global_batch {active} must equal the window stride "
                         f"{window.stride}: B is the number of rows the window slides by")
    grad_accum = policy.resolve_grad_accum(active, world)
    return {"active_global": active, "world_size": world, "per_rank": active // world,
            "micro_batch": policy.micro_batch, "grad_accum": grad_accum,
            "max_cache_age": window.max_age_steps}


def wrap_fsdp(model, policy: MemoryPolicy):
    """Shard params / grads / optimizer state across ranks (ZeRO-3). Returns the wrapper.

    Needed only where the 46.5 GB training state does not fit one card (a 24 GB 4090); an
    80 GB card runs faster without it, since every transformer block costs an extra
    all-gather. Three settings here are load-bearing rather than cosmetic:

    * **per-block wrapping.** Sharding the model as one flat unit would all-gather all 3.9 B
      parameters at once, which defeats the point. The auto-wrap policy shards each MM-DiT
      block, so only one block is ever materialized.
    * **``reduce_dtype=float32``.** With bf16 parameters the gradient reduce-scatter would
      otherwise accumulate in bf16's 7 mantissa bits across thousands of values. The parameters
      stay bf16; only the reduction is promoted.
    * **full state dict.** FSDP's default ``state_dict`` is per-rank shards, which no evaluator
      can load. This makes checkpoints ordinary whole-model files.

    The caller must also set ``grad_reduce='none'``: FSDP's reduce-scatter already averages, so
    the trainer's manual all-reduce would average a second time. ``MemoryPolicy.validate``
    refuses that combination.
    """
    import functools

    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    logger.info("[sw_lmmd] FSDP: sharding params/grads/optimizer state (param_dtype=%s, "
                "reduce_dtype=fp32)", policy.param_dtype)
    wrapped = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,               # ZeRO-3
        auto_wrap_policy=functools.partial(size_based_auto_wrap_policy,
                                           min_num_params=int(1e7)),
        mixed_precision=MixedPrecision(param_dtype=policy.param_dtype,
                                       reduce_dtype=torch.float32,
                                       buffer_dtype=policy.param_dtype),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,                 # keeps `model.parameters()` usable by the optimizer
    )
    FSDP.set_state_dict_type(wrapped, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True))
    return wrapped


def build_optimizer(params, cfg, policy: MemoryPolicy):
    """AdamW, optionally with 8-bit moment states.

    The two moments are half of the training state (``2 x 4 bytes x 3.9e9`` = 31 GB for
    klein-4B); holding them in 8 bits drops that to 7.75 GB. Purely a memory/precision trade
    -- the update rule is unchanged, so this does not alter the objective.
    """
    kw = dict(lr=cfg.lr, betas=(getattr(cfg, "beta1", 0.9), getattr(cfg, "beta2", 0.95)),
              weight_decay=getattr(cfg, "weight_decay", 0.0))
    if policy.optimizer == "adamw":
        return torch.optim.AdamW(params, **kw)
    if policy.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError(
                "optimizer='adamw8bit' needs bitsandbytes (pip install bitsandbytes). Use "
                "optimizer='adamw' if you have the headroom -- see configs/sw_lmmd_*.yaml") from e
        return bnb.optim.AdamW8bit(params, **kw)
    raise NotImplementedError(
        f"optimizer={policy.optimizer!r} is not wired (CPU-offload plumbing is not written); "
        f"use 'adamw' or 'adamw8bit'")


def build_trainer(cfg, device: str = "cuda") -> SWLMMDTrainer:
    """Assemble the full SW-LMMD trainer from a parsed config (needs weights + a built store)."""
    from ..representation.battery import Battery
    from ..representation.registry import by_name
    from ..train.launch import build_generator_from_config

    window = window_from_config(cfg)
    policy = memory_from_config(cfg)
    batching = resolve_batching(cfg, policy, window)
    logger.info("[sw_lmmd] K=%d B=%d overlap=%d | world=%d micro_batch=%d grad_accum=%d "
                "| max cache age %d steps", window.size, window.stride, window.overlap,
                batching["world_size"], batching["micro_batch"], batching["grad_accum"],
                batching["max_cache_age"])

    names = list(cfg.encoders)
    store = ReferenceFeatureStore(cfg.reference_root, names,
                                  require_context=getattr(cfg, "joint", True))
    schedule = SlidingWindowSchedule(store.row_order(window.order_seed),
                                     window_size=window.size, stride=window.stride,
                                     cyclic=window.cyclic)
    generator, model = build_generator_from_config(cfg, device, param_dtype=policy.param_dtype)

    battery = Battery([by_name(n) for n in names], device=device)
    if policy.battery_bf16:
        for enc in battery.encoders.values():
            if not getattr(enc, "has_logits", False):
                enc.to(torch.bfloat16)

    clip_module = wrap_fsdp(model, policy) if policy.shard == "fsdp" else None
    noise_shape = (model.in_channels, model.input_size, model.input_size) \
        if getattr(cfg, "mode", "flux") == "flux" else \
        (3, getattr(cfg, "img_size", 256), getattr(cfg, "img_size", 256))

    return SWLMMDTrainer(
        generator, battery, store, schedule,
        build_optimizer([p for p in model.parameters() if p.requires_grad], cfg, policy),
        encoder_names=names, noise_shape=noise_shape,
        cache=GeneratedWindowCache(names, store_dtype=cache_from_config(cfg).store_dtype),
        micro_batch=policy.micro_batch, grad_clip=getattr(cfg, "grad_clip", 4.0),
        kernel_block=int(_block(cfg, "loss").get("kernel_block_size", 256)),
        seed=getattr(cfg, "seed", 0), joint=getattr(cfg, "joint", True),
        grad_reduce=policy.grad_reduce, monitor_every=getattr(cfg, "monitor_every", 1),
        cache_cfg=cache_from_config(cfg), clip_module=clip_module, device=device)


def train(cfg, device: str = "cuda") -> SWLMMDTrainer:
    """Bootstrap, then run ``cfg.steps`` sliding-window updates with drift control.

    The loop owns three things the trainer deliberately does not: the staleness policy (probe,
    then refresh on a schedule or a drift trigger), the machine-readable log, and checkpoints.
    Keeping them out of :meth:`SWLMMDTrainer.step` is what lets the method layer stay testable
    without a filesystem or a clock.
    """
    import json
    import os
    import time

    from ..utils.distributed import is_main_process
    from .refresh import probe_drift, refresh_retained, should_refresh

    trainer = build_trainer(cfg, device=device)
    cache_cfg = cache_from_config(cfg)
    out_dir = os.path.join(getattr(cfg, "output_dir", "./work_dirs"),
                           getattr(cfg, "exp_name", "sw-lmmd"))
    os.makedirs(out_dir, exist_ok=True)
    jsonl = open(os.path.join(out_dir, "train_log.jsonl"), "a") if is_main_process() else None

    t0 = time.time()
    logger.info("[sw_lmmd] bootstrap ...")
    boot = trainer.bootstrap()
    logger.info("[sw_lmmd] bootstrap done: %s rows (%s/rank) in %.1f s",
                boot["bootstrap_rows"], boot["bootstrap_rows_per_rank"], time.time() - t0)

    probe_every = int(getattr(cfg, "probe_every", 0) or cache_cfg.refresh_every_windows or 0)
    drift = None
    for _ in range(int(cfg.steps)):
        step_t = time.time()
        logs = trainer.step()
        logs["seconds"] = time.time() - step_t

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
                logger.info("step %d | force %.6f | raw_mmd2 %.6g | grad_norm %s | "
                            "cache_age %d | %.1f s", logs["step"], logs["force"],
                            logs.get("raw_mmd2", float("nan")), logs["grad_norm"],
                            logs["cache_max_age"], logs["seconds"])
            jsonl.write(json.dumps(logs) + "\n")
            jsonl.flush()
            if trainer.step_idx % int(getattr(cfg, "save_freq", 50)) == 0:
                path = os.path.join(out_dir, f"step_{trainer.step_idx:07d}.pth")
                torch.save(trainer.state_dict(with_cache=bool(getattr(cfg, "save_cache", False))),
                           path)
                logger.info("[sw_lmmd] checkpoint -> %s", path)

    if is_main_process():
        torch.save(trainer.state_dict(with_cache=bool(getattr(cfg, "save_cache", False))),
                   os.path.join(out_dir, f"step_{trainer.step_idx:07d}.pth"))
        jsonl.close()
        logger.info("[sw_lmmd] done: %d steps in %.2f h", trainer.step_idx,
                    (time.time() - t0) / 3600)
    return trainer


def apply_override(cfg, override: str) -> None:
    """Apply one ``key=value`` / ``block.key=value`` override onto a parsed config.

    Values are parsed as YAML so ``true``/``8``/``1e-5`` arrive with the right type; a bare
    string stays a string. Only keys that already exist in a nested block are accepted -- a
    typo in ``--set memroy.shard=fsdp`` would otherwise be silently ignored and the run would
    proceed with the wrong memory policy.
    """
    import yaml

    if "=" not in override:
        raise SystemExit(f"--set expects KEY=VALUE, got {override!r}")
    key, raw = override.split("=", 1)
    value = yaml.safe_load(raw)
    if "." in key:
        block, leaf = key.split(".", 1)
        current = getattr(cfg, block, None)
        if not isinstance(current, dict):
            raise SystemExit(f"--set {key}: config has no block {block!r}")
        if leaf not in current:
            raise SystemExit(f"--set {key}: {block!r} has no key {leaf!r} "
                             f"(has {sorted(current)})")
        current[leaf] = value
    else:
        setattr(cfg, key, value)
    logger.info("[sw_lmmd] override %s = %r", key, value)


def main() -> None:
    """``torchrun --nproc_per_node=2 -m rdm.sw_lmmd.launch configs/sw_lmmd_h100_2gpu.yaml``"""
    import argparse

    from ..train.launch import load_config
    from ..utils.distributed import setup_distributed
    from ..utils.logging import setup_logging
    from ..utils.seed import fix_random_seeds

    ap = argparse.ArgumentParser(description="SW-LMMD training")
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config key; dotted for nested blocks "
                         "(e.g. --set memory.shard=fsdp --set steps=5)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for override in args.set:
        apply_override(cfg, override)
    if getattr(cfg, "method", None) != "sw_lmmd":
        raise SystemExit(f"{args.config} is not a SW-LMMD config (method: {getattr(cfg,'method',None)!r})")

    rank, world, local_rank = setup_distributed()
    setup_logging(rank=rank)
    # The window schedule and the per-row noise are seeded identically on every rank by
    # construction; the global seed only affects anything left to torch's default generator.
    fix_random_seeds(getattr(cfg, "seed", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    train(cfg, device=device)


if __name__ == "__main__":
    main()
