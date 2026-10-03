"""Standalone worker process for the SW-LMMD world-size parity test (not collected by pytest).

Launched once per rank with ``RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR`` / ``MASTER_PORT`` in the
environment -- the same contract :func:`rdm.utils.distributed.setup_distributed` reads, so the
test exercises the real distributed path rather than a simulation of it. Rank 0 writes the
post-reduction parameter gradients for the test to compare across world sizes.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sw_lmmd_fixtures import build_trainer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--gan", action="store_true", help="add the adversarial critic")
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("gloo", rank=rank, world_size=world)

    # lr=0 freezes the parameters, so every world size differentiates the SAME function at the
    # SAME point; grad_clip=0 keeps the raw gradient (clipping is norm-dependent and would mask
    # a real discrepancy behind a rescale).
    # The critic keeps its own lr, so with --gan step 2's generator gradient is scored by a
    # critic that has already taken a step -- its weights must agree across world sizes too.
    trainer = build_trainer(args.store, window=args.window, stride=args.stride,
                            micro_batch=args.micro_batch, lr=0.0, grad_clip=0.0,
                            gan={"weight": 0.5} if args.gan else None)
    trainer.bootstrap()
    logs = {}
    for _ in range(args.steps):
        logs = trainer.step()

    if rank == 0:
        torch.save({"grads": [p.grad.detach().clone() for p in trainer._gen_params],
                    "force": logs["force"],
                    "raw_mmd2": logs["raw_mmd2"],
                    "cache": {n: e.features.clone() for n, e in trainer.cache.entries.items()},
                    "cache_rows": {n: e.row_ids for n, e in trainer.cache.entries.items()},
                    "critic": None if trainer.critic is None else
                    {k: v.clone() for k, v in trainer.critic.critic.state_dict().items()},
                    "gan_logs": {k: v for k, v in logs.items() if k.startswith("gan/")},
                    "world": world, "micro_batch": args.micro_batch}, args.out)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
