"""Worker of test E (world-size parity of the random / global modes); not collected by pytest.

Launched once per rank with RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT, as
tests/sw_lmmd_parity_worker.py is. lr = 0 and no clipping, so every world size differentiates the
same function at the same point; rank 0 writes the reduced parameter gradients, the force, the
cross-rank ``ref/`` diagnostics and the attraction rows of the last step.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "tests"), HERE]

from experiments.ref_ablation import GlobalNystrom, RefAblationTrainer  # noqa: E402
from test_ref_ablation import make_trainer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--bundles", required=True)
    ap.add_argument("--mode", required=True, choices=["local", "random", "global"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("gloo", rank=rank, world_size=world)

    base = make_trainer(args.store, grouped=True, unbiased=True, micro_batch=args.micro_batch)
    nystrom = GlobalNystrom.load(args.bundles, base.encoder_names, store=base.store)
    trainer = RefAblationTrainer.from_trainer(base, ref_mode=args.mode, ref_seed=11,
                                              nystrom=nystrom)
    trainer.bootstrap()
    logs = {}
    for _ in range(args.steps):
        logs = trainer.step()

    if rank == 0:
        torch.save({"grads": [p.grad.detach().clone() for p in trainer._gen_params],
                    "force": logs["force"],
                    "ref": {k: v for k, v in logs.items()
                            if k.startswith("ref/") and isinstance(v, float)},
                    "rows": trainer._last_attraction_rows, "world": world}, args.out)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
