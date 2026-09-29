"""Standalone worker process for the SW-LMMD FSDP test (not collected by pytest).

Runs the real training path the 24 GB-card config depends on -- ``Flux2AdapterModel._run_dit``
over a tiny ``flux2`` MM-DiT, ``FluxGenerator``, :func:`rdm.sw_lmmd.launch.shard_generator`
and the :func:`rdm.sw_lmmd.launch.train` loop with its probe / refresh / checkpointing -- over
gloo on the CPU, launched with the same RANK/WORLD_SIZE environment ``torchrun`` sets. Only the
VAE and the encoder battery are stubbed (``tests/sw_lmmd_fixtures.py``).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sw_lmmd_fixtures import (FLUX_LATENT, NAMES, LatentBattery, LatentTokenizer,  # noqa: E402
                              tiny_flux_adapter)

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fsdp", action="store_true")
    ap.add_argument("--compute-dtype", default="fp32", choices=sorted(_DTYPES))
    ap.add_argument("--amp", action="store_true", help="bf16 autocast in the rollout, as on GPU")
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw8bit"])
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--save-freq", type=int, default=2)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        # Short timeout: a collective only some ranks enter (e.g. a rank-0-only FSDP
        # state_dict) should fail the test in minutes, not after gloo's default 30.
        dist.init_process_group("gloo", rank=rank, world_size=world,
                                timeout=datetime.timedelta(seconds=120))

    from rdm.representation.generators.flux_generator import FluxGenerator
    from rdm.sw_lmmd import (CacheConfig, MemoryPolicy, ReferenceFeatureStore, SWLMMDTrainer,
                             SlidingWindowSchedule)
    from rdm.sw_lmmd.launch import build_optimizer, shard_generator, train

    policy = MemoryPolicy(shard="fsdp" if args.fsdp else "none",
                          grad_reduce="none" if args.fsdp else "mean",
                          optimizer="adamw8bit" if args.optimizer == "adamw8bit" else "adamw",
                          compute_dtype=_DTYPES[args.compute_dtype] if args.fsdp else None,
                          micro_batch=1)
    policy.validate()

    adapter = tiny_flux_adapter()
    generator = FluxGenerator(adapter, {"num_steps": 1}, LatentTokenizer(),
                              args=SimpleNamespace(enable_amp=args.amp))
    clip_module = shard_generator(generator, adapter, policy, "cpu")
    params = [p for p in adapter.parameters() if p.requires_grad]
    # SGD makes the update the gradient itself, so the final weights compare gradients exactly.
    optimizer = torch.optim.SGD(params, lr=args.lr) if args.optimizer == "sgd" else \
        build_optimizer(params, SimpleNamespace(lr=args.lr), policy)

    store = ReferenceFeatureStore(args.store, NAMES)
    schedule = SlidingWindowSchedule(store.row_order(), window_size=args.window,
                                     stride=args.stride)
    trainer = SWLMMDTrainer(generator, LatentBattery(), store, schedule, optimizer,
                            encoder_names=list(NAMES),
                            noise_shape=(128, FLUX_LATENT, FLUX_LATENT),
                            micro_batch=policy.micro_batch, grad_clip=1e6, kernel_block=8,
                            seed=0, grad_reduce=policy.grad_reduce, clip_module=clip_module,
                            cache_cfg=CacheConfig(), device="cpu")

    cfg = SimpleNamespace(steps=args.steps, save_freq=args.save_freq, output_dir=args.out_dir,
                          exp_name="run", print_freq=1, probe_every=1,
                          cache={"refresh_every_windows": 2, "refresh_probe_rows": 4,
                                 "drift_threshold": 0.0})
    train(cfg, device="cpu", trainer=trainer)

    if rank == 0:
        from torch.distributed.fsdp import FlatParameter
        with open(os.path.join(args.out_dir, "run", "worker.json"), "w") as f:
            json.dump({"world": world, "fsdp": args.fsdp,
                       "optimizer_params": len(params),
                       "flat_params_in_optimizer": sum(isinstance(p, FlatParameter)
                                                       for p in params)}, f)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
