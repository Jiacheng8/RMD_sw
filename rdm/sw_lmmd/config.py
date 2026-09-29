"""Configuration dataclasses for SW-LMMD, split by *layer* rather than by topic.

The package is deliberately layered so the algorithm never learns about the hardware:

* :class:`WindowConfig` / :class:`CacheConfig` / :class:`KernelConfig` describe the
  **method**. They are identical on one 4090 and on thirty-two H200s.
* :class:`MemoryPolicy` describes the **machine**. Everything that differs between a 24 GB
  consumer card and an 80 GB datacenter card lives here and is consumed only by the launch
  layer and the trainer's plumbing -- never by :mod:`rdm.sw_lmmd.local_mmd`,
  :mod:`rdm.sw_lmmd.window_schedule` or :mod:`rdm.sw_lmmd.cache`.

The one number that couples the two is ``B = micro_batch * world_size * grad_accum``, checked
by :meth:`MemoryPolicy.resolve_grad_accum`. The default ``B = 128`` divides 1, 2, 4, 8, 16 and
32 ranks, so a single window config ports across every target without touching the method.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WindowConfig:
    """The sliding window geometry: capacity ``K``, stride / active size ``B``."""

    size: int = 1024                      # K: rows compared by the MMD each step
    stride: int = 128                     # B: rows freshly rolled out and back-propagated
    cyclic: bool = True
    order_seed: int = 3407
    random_start_each_epoch: bool = True
    reverse_probability: float = 0.5

    def validate(self) -> None:
        if self.size <= 0 or self.stride <= 0:
            raise ValueError("window size and stride must be positive")
        if self.stride > self.size:
            raise ValueError(f"stride {self.stride} cannot exceed window size {self.size}")

    @property
    def overlap(self) -> int:
        """``K - B`` -- rows carried over from the previous window (detached)."""
        return self.size - self.stride

    @property
    def full_to_active_gradient_ratio(self) -> float:
        """``K / B``. DIAGNOSTIC ONLY -- it is already inside the ``1/(BK)`` force
        normalization; multiplying the loss by it again double-counts the correction."""
        return self.size / self.stride

    @property
    def max_age_steps(self) -> int:
        """Age (in optimizer steps) of the oldest cached feature in steady state."""
        return -(-self.size // self.stride) - 1          # ceil(K/B) - 1


@dataclass
class KernelConfig:
    """Fixed-bandwidth biased RBF MMD knobs. Distances are always computed in fp32."""

    block_size: int = 256
    biased: bool = True                   # include the k(x_i, x_i) diagonal (the default)


@dataclass
class CacheConfig:
    """Staleness control for the retained generated features."""

    refresh_every_windows: int = 20       # 0 disables the periodic full refresh
    refresh_probe_rows: int = 64          # rows re-rolled to measure drift (never trained on)
    drift_threshold: float = 0.05         # relative drift that triggers an early refresh
    store_dtype: torch.dtype = torch.float32
    # NOTE: the spec suggests fp16 storage. The whole cache is ~55 MB at K=1024 over ten
    # encoders, so fp32 costs nothing measurable and avoids quantizing the context the
    # kernel sees. Set to torch.float16 if you are memory-bound.


@dataclass
class MemoryPolicy:
    """Everything that changes between machines. The method layer never reads this.

    ``grad_reduce`` is the one field with a correctness trap: the parameter gradient must be
    averaged across ranks EXACTLY ONCE. With the repo's manual scheme (no DDP wrapper) the
    trainer does it -> ``"mean"``. Under FSDP/DDP the wrapper's reduce-scatter already
    averages -> ``"none"``; running the manual all-reduce as well would average twice and
    silently shrink every step by a factor of the world size.
    """

    shard: str = "none"                   # none | fsdp  (see launch.wrap_fsdp)
    optimizer: str = "adamw"              # adamw | adamw8bit | adamw_offload
    # Storage dtype of the trainable weights, i.e. what the optimizer updates. Keep fp32: at
    # lr ~3e-6 an AdamW step is smaller than half a bf16 ULP for 96% of klein-4B's weights, so
    # bf16 storage rounds those updates away (compute runs in bf16 under autocast regardless).
    param_dtype: torch.dtype = torch.float32
    # FSDP only: the dtype parameters are all-gathered in for forward/backward, while the fp32
    # shards stay the master copy. None = param_dtype. Without FSDP autocast sets the compute
    # dtype, so this must stay None there.
    compute_dtype: torch.dtype | None = None
    battery_bf16: bool = False            # cast the frozen ViT encoders to bf16
    encoder_offload: bool = False
    micro_batch: int = 1                  # active rows per generator forward, per rank
    grad_reduce: str = "mean"             # mean | none  (see above)

    def resolve_grad_accum(self, active_global: int, world_size: int) -> int:
        """``B = micro_batch * world * grad_accum`` -- returns grad_accum, or explains why not.

        This is the single place the method's ``B`` meets the machine's batching, mirroring
        the ``rollout_size = batch * world * grad_accum`` invariant the iRDM trainer enforces.
        """
        if active_global % world_size:
            raise ValueError(
                f"active batch B={active_global} is not divisible by world_size={world_size}; "
                f"all ranks must gather equal-shaped feature blocks. Pick a B divisible by "
                f"{world_size} (B=128 divides 1/2/4/8/16/32).")
        per_rank = active_global // world_size
        if per_rank % self.micro_batch:
            raise ValueError(
                f"per-rank active rows {per_rank} is not divisible by micro_batch="
                f"{self.micro_batch}")
        return per_rank // self.micro_batch

    def validate(self) -> None:
        if self.shard not in ("none", "fsdp"):
            raise ValueError(f"unknown shard mode {self.shard!r}")
        if self.grad_reduce not in ("mean", "none"):
            raise ValueError(f"unknown grad_reduce {self.grad_reduce!r}")
        if self.shard == "fsdp" and self.grad_reduce == "mean":
            raise ValueError(
                "shard='fsdp' with grad_reduce='mean' would average the parameter gradient "
                "twice (FSDP's reduce-scatter already averages). Set grad_reduce='none'.")
        if self.compute_dtype is not None and self.shard != "fsdp":
            raise ValueError("compute_dtype only applies under shard='fsdp'; without FSDP the "
                             "compute dtype is set by autocast")
