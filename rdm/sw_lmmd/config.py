"""Configuration dataclasses for SW-LMMD, split by *layer* rather than by topic.

The package is deliberately layered so the algorithm never learns about the hardware:

* :class:`WindowConfig` / :class:`CacheConfig` / :class:`KernelConfig` / :class:`GANConfig` /
  :class:`LRSchedule` describe the **method**. They are identical on one 4090 and on thirty-two
  H200s.
* :class:`MemoryPolicy` describes the **machine**. Everything that differs between a 24 GB
  consumer card and an 80 GB datacenter card lives here and is consumed only by the launch
  layer and the trainer's plumbing -- never by :mod:`rdm.sw_lmmd.local_mmd`,
  :mod:`rdm.sw_lmmd.window_schedule` or :mod:`rdm.sw_lmmd.cache`.

The one number that couples the two is ``B = micro_batch * world_size * grad_accum``, checked
by :meth:`MemoryPolicy.resolve_grad_accum`. The default ``B = 128`` divides 1, 2, 4, 8, 16 and
32 ranks, so a single window config ports across every target without touching the method.
"""
from __future__ import annotations

import math
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
class LRSchedule:
    """The generator's learning-rate schedule, a pure function of the step counter.

    It is stateless on purpose. The factor is recomputed from ``step_idx`` before every
    optimizer step, so a resume restores the lr by restoring the step counter, and
    ``resume.pth`` needs nothing extra. A skipped (non-finite) step still advances it, like the
    window. The decay horizon is ``total_steps``, which defaults to the run's ``steps``, so
    extending a run on resume reshapes the rest of the curve.
    """

    name: str = "constant"                # constant | cosine | linear
    warmup_steps: int = 0                 # linear ramp lr * (s + 1) / warmup_steps
    min_lr_ratio: float = 0.0             # decay floor as a fraction of lr (cosine / linear)
    total_steps: int | None = None        # decay horizon; None = the run's `steps`

    def validate(self) -> None:
        if self.name not in ("constant", "cosine", "linear"):
            raise ValueError(f"lr_schedule.name must be constant / cosine / linear, "
                             f"got {self.name!r}")
        if self.warmup_steps < 0:
            raise ValueError("lr_schedule.warmup_steps must be >= 0")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("lr_schedule.min_lr_ratio must be in [0, 1]")
        if self.total_steps is not None and self.total_steps <= 0:
            raise ValueError("lr_schedule.total_steps must be positive")

    def factor(self, step: int) -> float:
        """Multiplier on the base lr for the optimizer step taken at 0-based ``step``.

        Warm-up never yields 0, since a zero-lr step would waste a full rollout.
        """
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if self.name == "constant":
            return 1.0
        if self.total_steps is None:
            raise ValueError("a decaying lr schedule needs total_steps (the launcher fills it "
                             "from `steps`)")
        progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps -
                                                              self.warmup_steps))
        shape = 0.5 * (1.0 + math.cos(math.pi * progress)) if self.name == "cosine" else \
            1.0 - progress
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * shape


@dataclass
class GANConfig:
    """The optional adversarial term (:mod:`rdm.sw_lmmd.adversarial`). Off by default.

    ``weight`` is the knob that matters. With ``adaptive`` it is a *ratio*: the adversarial
    gradient on the cached image features is scaled to ``weight x`` the MMD force's gradient
    there, so it neither vanishes nor swamps the force as the critic sharpens. Without
    ``adaptive`` it multiplies the raw adversarial loss, whose gradient scale is unrelated to
    the force's ``1/(BK)`` and drifts with the critic -- a value that works early stops working
    later.
    """

    enabled: bool = False
    encoders: list | None = None          # critic heads; None = every training encoder
    weight: float = 0.25                  # lambda (a gradient-norm ratio under ``adaptive``)
    adaptive: bool = True
    min_norm_ratio: float = 1e-3          # adaptive: floor on |g_adv| / |g_mmd| (caps lambda)
    loss: str = "hinge"                   # hinge | ns (non-saturating logistic)
    lr: float = 2e-4
    betas: tuple = (0.0, 0.99)
    hidden: int = 1024
    depth: int = 2                        # hidden layers per head
    spectral_norm: bool = True
    r1_gamma: float = 0.0                 # R1 penalty on real features (0 = off)
    g_start_step: int = 50                # critic-only warm-up: the generator ignores it before
    conditional: bool = True              # projection on tau(c); off when ``joint`` is off
    stats_rows: int = 8192                # reference rows that fix the input standardization
    seed: int = 0

    def validate(self) -> None:
        if self.loss not in ("hinge", "ns"):
            raise ValueError(f"gan.loss must be 'hinge' or 'ns', got {self.loss!r}")
        if self.weight < 0 or self.lr < 0 or self.r1_gamma < 0:
            raise ValueError("gan.weight, gan.lr and gan.r1_gamma must be non-negative")
        if self.hidden <= 0 or self.depth <= 0 or self.stats_rows <= 1:
            raise ValueError("gan.hidden, gan.depth must be positive and gan.stats_rows > 1")
        if len(tuple(self.betas)) != 2:
            raise ValueError(f"gan.betas must be two numbers, got {self.betas!r}")
        if not 0.0 < self.min_norm_ratio <= 1.0:
            raise ValueError("gan.min_norm_ratio must be in (0, 1]")


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
