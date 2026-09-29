"""The distribution contract: row partition, detached all-gather, per-row noise.

SW-LMMD must produce the **same parameter gradient** on one GPU, on four 4090s, and on
thirty-two H200s. That invariance is not automatic -- it rests on exactly four properties,
three of which live here:

1. **Per-row noise.** ``row_noise`` derives each sample's latent from ``(seed, epoch,
   row_id)``, never from a global RNG consumed in rank order and never from the rank id.
   Two consequences beyond portability:

   * GradCache's second pass must re-generate *the same images* it cached in the first pass;
     a stream-ordered RNG cannot promise that once the chunking changes.
   * The staleness probe re-rolls an old row under new parameters. With per-row noise the
     re-roll reuses that row's original latent, so ``||x_cached - x_refreshed||`` measures
     parameter drift alone rather than drift plus a different noise draw.

2. **Contiguous rank partition.** ``partition_rows`` slices the global active ids in rank
   order, so concatenating the ranks' gathered blocks restores the global row order that the
   reference window is indexed by.

3. **Detached all-gather.** The generated context is a *stop-gradient* quantity (spec sec. 6):
   every row of it, including the rows that happen to be this rank's own active samples, is
   detached. ``all_gather_detached`` therefore has no backward -- unlike
   :func:`rdm.utils.distributed.diff_all_gather`, which the global iRDM loss needs.

The fourth property is the normalization convention, implemented in
:mod:`rdm.sw_lmmd.local_mmd` and :mod:`rdm.sw_lmmd.trainer`: each rank normalizes its force by
``1 / (B_local * K)`` and the parameter gradients are combined with a **mean**, which
telescopes to the global ``2 / (B * K)`` for any world size (and is the identity at world 1).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from ..utils.distributed import get_rank, get_world_size, is_dist

_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """SplitMix64 finalizer -- decorrelates the sequential ``row_id``s we seed from."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


def row_seed(seed: int, epoch: int, row_id: int) -> int:
    """Deterministic per-row seed, independent of rank, chunking and call order."""
    return _splitmix64(_splitmix64(int(seed) * 0x9E3779B1 + int(epoch)) ^ int(row_id)) >> 1


def row_noise(row_ids, shape, seed: int = 0, epoch: int = 0, device="cpu",
              dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Standard-normal latents ``(len(row_ids), *shape)``, one independent draw per row id.

    Drawn on the CPU with a per-row :class:`torch.Generator` so the values do not depend on
    the device, the world size, or how the rows were split into micro-batches -- the
    prerequisite for the world-size parity test and for GradCache's two-pass replay.
    """
    ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    shape = tuple(shape)
    out = torch.empty((len(ids), *shape), dtype=torch.float32)
    gen = torch.Generator(device="cpu")
    for i, rid in enumerate(ids):
        gen.manual_seed(row_seed(seed, epoch, int(rid)))
        out[i] = torch.randn(shape, generator=gen, dtype=torch.float32)
    return out.to(device=device, dtype=dtype)


def partition_rows(row_ids, rank: int | None = None, world: int | None = None) -> np.ndarray:
    """This rank's contiguous share of ``row_ids`` (gather order restores the global order)."""
    ids = np.asarray(row_ids, dtype=np.int64)
    world = get_world_size() if world is None else int(world)
    rank = get_rank() if rank is None else int(rank)
    if world == 1:
        return ids
    if len(ids) % world:
        raise ValueError(f"{len(ids)} rows is not divisible by world_size={world}; all ranks "
                         f"must contribute equal-shaped blocks to the all-gather")
    n = len(ids) // world
    return ids[rank * n:(rank + 1) * n]


@torch.no_grad()
def all_gather_detached(x: torch.Tensor) -> torch.Tensor:
    """Concatenate every rank's ``x`` in rank order, with no autograd edge (no-op at world 1).

    The generated context is stop-gradient by construction, so unlike
    :func:`rdm.utils.distributed.diff_all_gather` this deliberately drops the local graph:
    gradient reaches the generator only through the rank's own *live* active features.
    """
    if not is_dist():
        return x.detach()
    out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    if x.is_cuda:
        # Load-bearing on the iRDM path (see rdm.utils.distributed): sync before the
        # collective so encoder kernels cannot race it under bf16 autocast.
        torch.cuda.current_stream().synchronize()
    dist.all_gather(out, x.detach().contiguous())
    return torch.cat(out, dim=0)


def all_reduce_mean_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place cross-rank average (no-op at world 1)."""
    if is_dist():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


def reduce_scalar(value: float, device=None) -> float:
    """Average a python scalar across ranks (per-rank force values differ by construction).

    The carrier tensor goes where the process group has a transport: NCCL has none for CPU
    tensors ("No backend type associated with device type cpu"), so under NCCL it lives on the
    current CUDA device; gloo keeps it on the CPU.
    """
    if not is_dist():
        return float(value)
    if device is None:
        device = "cuda" if dist.get_backend() == "nccl" else "cpu"
    t = torch.tensor([float(value)], dtype=torch.float64, device=device)
    all_reduce_mean_(t)
    return float(t.item())
