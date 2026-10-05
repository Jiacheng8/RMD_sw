"""Exact biased two-sample MMD on one window, and the active-only force that trains it.

For window ``W`` with generated joint features ``X`` and reference joint features ``Y`` (both
``K`` rows, row-aligned to the same prompts) the monitored quantity is the biased empirical

    MMD^2 = K_xx - 2 K_xy + K_yy,    K_ab = (1/K^2) sum_{i,j} k(a_i, b_j)

"exact" meaning the kernel sums are computed in full -- no Nystrom landmarks, no random
features. Biased (the ``i == j`` diagonal is kept) because ``k(x_i, x_i) = 1`` contributes no
gradient under an RBF, the estimator stays non-negative, and it maps directly onto dense
blocks.

Only the ``B`` newest rows carry gradient. The full local MMD's force on a generated point is

    dL/dx_i = (2/K^2) sum_{j in W} [ grad_xi k(x_i, x_j) - grad_xi k(x_i, y_j) ]

so back-propagating through the active rows alone is reproduced exactly by

    L_force = (2/(B*K)) sum_{i in A} sum_{j in W} [ k(x_i, sg(x_j)) - k(x_i, y_j) ]

Three things this form encodes, each a documented way to get it wrong:

1. **The second kernel argument is always detached** -- including where it happens to be one
   of this step's own active rows. Letting both arguments carry gradient and then adding a
   symmetry factor of 2 double-counts.
2. **The ``K/B`` correction is already inside the ``1/(BK)`` denominator.** Relative to the
   full MMD's ``1/K^2``, dividing by ``B*K`` *is* the factor ``K/B``; multiplying the loss by
   it again (the config's ``full_to_active_gradient_ratio``, which is diagnostic only) applies
   it twice.
3. **The forward scalar is not an MMD.** It is a surrogate whose gradient is right; it can be
   negative and its magnitude is not comparable across ``K``. Plot it separately from the
   monitored ``mmd2``.

Distances and kernel sums run in **at least** fp32 regardless of the storage dtype -- with a
small bandwidth a bf16 exponent underflows the whole Gram matrix to zero and the force
silently becomes zero. fp64 inputs are left alone rather than downcast (see
:func:`at_least_fp32`). The blockwise reductions reuse :mod:`rdm.compare.kernels`, so this
module shares its squared-distance and RBF arithmetic with the global iRDM loss rather than
restating it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..compare.kernels import cross_kernel_mean, gamma_from_sigma, gaussian_gram


@dataclass
class MMDMonitor:
    """The three biased kernel means and the assembled ``MMD^2`` for one window."""

    mmd2: torch.Tensor
    k_xx: torch.Tensor
    k_xy: torch.Tensor
    k_yy: torch.Tensor

    def as_log(self, prefix: str = "") -> dict:
        return {f"{prefix}mmd2": float(self.mmd2), f"{prefix}k_xx": float(self.k_xx),
                f"{prefix}k_xy": float(self.k_xy), f"{prefix}k_yy": float(self.k_yy)}


def at_least_fp32(t: torch.Tensor) -> torch.Tensor:
    """Promote bf16/fp16 to fp32, leave fp32/fp64 alone.

    The rule is a *floor* on precision, not a target: with a small bandwidth a half-precision
    exponent underflows the whole Gram matrix to zero and the force silently vanishes. Casting
    unconditionally with ``.float()`` would instead *downcast* fp64, which shows up as a
    world-size parity failure long before it shows up as a training bug.
    """
    return t if t.dtype in (torch.float32, torch.float64) else t.float()


@torch.no_grad()
def blockwise_mean_nograd(a: torch.Tensor, b: torch.Tensor, gamma: float,
                          block: int = 256) -> torch.Tensor:
    """``(1/(n_a n_b)) sum_{i,j} k(a_i, b_j)``, blocked over rows of ``a``.

    The monitoring counterpart of :func:`rdm.compare.kernels.cross_kernel_mean`: no autograd
    graph and no gradient checkpointing, so it neither allocates nor warns on detached input.
    """
    a32, b32 = at_least_fp32(a), at_least_fp32(b)
    acc = torch.zeros((), dtype=a32.dtype, device=a32.device)
    for lo in range(0, a32.shape[0], block):
        acc += gaussian_gram(a32[lo:lo + block], b32, gamma).sum()
    return acc / float(a32.shape[0] * b32.shape[0])


@torch.no_grad()
def within_group_kernel_mean(x: torch.Tensor, group: int, sigma: float) -> torch.Tensor:
    """Mean ``k(x_a, x_b)`` over distinct pairs inside consecutive blocks of ``group`` rows.

    With a prompt-grouped window each block is one prompt's seeds, so this is the same-prompt
    similarity: computed on generated rows and on their references, a generated value well
    above the reference one means the seeds of a prompt have collapsed toward one image.
    """
    if group < 2 or x.shape[0] % group:
        raise ValueError(f"need >= 2 rows per group and {x.shape[0]} rows divisible by {group}")
    blocks = at_least_fp32(x).reshape(-1, group, x.shape[-1])
    d2 = torch.cdist(blocks, blocks).square()
    k = torch.exp(-gamma_from_sigma(sigma) * d2)
    off = ~torch.eye(group, dtype=torch.bool, device=x.device)
    return k[:, off].mean()


class ExactLocalMMD:
    """Window-local biased MMD: the training force, and the monitored value."""

    def __init__(self, block_size: int = 256):
        self.block_size = int(block_size)

    def active_force(self, active: torch.Tensor, generated_context: torch.Tensor,
                     reference: torch.Tensor, sigma: float) -> tuple[torch.Tensor, dict]:
        """``2 * (mean_k(active, context) - mean_k(active, reference))``.

        Args:
            active: ``(B_local, d)`` live features of THIS rank's fresh rows (carry grad).
            generated_context: ``(K, d)`` full generated window -- retained cache plus every
                rank's active rows, all detached.
            reference: ``(K, d)`` reference joint features for the same rows, detached.
            sigma: the encoder's fixed RBF bandwidth.

        Both means divide by ``B_local * K``, which is the ``1/(BK)`` normalization; combined
        with a cross-rank gradient **mean** it telescopes to the global ``2/(BK)`` for any
        world size.
        """
        if not active.requires_grad:
            raise ValueError(
                "active features do not require grad -- the generator would receive no "
                "gradient. Only the context and the reference may be detached; do not wrap "
                "the encoder forward in torch.no_grad().")
        if generated_context.shape[0] != reference.shape[0]:
            raise ValueError(f"context has {generated_context.shape[0]} rows but reference has "
                             f"{reference.shape[0]}; both must be the same window")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        gamma = gamma_from_sigma(sigma)
        a = at_least_fp32(active)
        repel = cross_kernel_mean(a, at_least_fp32(generated_context).detach().to(a.dtype),
                                  gamma, self.block_size)
        attract = cross_kernel_mean(a, at_least_fp32(reference).detach().to(a.dtype),
                                    gamma, self.block_size)
        force = 2.0 * (repel - attract)
        stats = {"active_repulsion": repel.detach(), "active_attraction": attract.detach(),
                 "force": force.detach()}
        return force, stats

    @torch.no_grad()
    def monitor(self, generated_context: torch.Tensor, reference: torch.Tensor, sigma: float,
                precomputed_k_yy: torch.Tensor | float | None = None) -> MMDMonitor:
        """The full-window biased ``MMD^2`` -- reported, never back-propagated.

        ``K_yy`` depends only on the frozen reference rows, so it may be precomputed per
        window; it is included because the *value* of MMD^2 is meaningless without it.
        """
        gamma = gamma_from_sigma(sigma)
        x, y = generated_context.detach(), reference.detach()
        k_xx = blockwise_mean_nograd(x, x, gamma, self.block_size)
        k_xy = blockwise_mean_nograd(x, y, gamma, self.block_size)
        if precomputed_k_yy is None:
            k_yy = blockwise_mean_nograd(y, y, gamma, self.block_size)
        else:
            k_yy = torch.as_tensor(precomputed_k_yy, dtype=k_xx.dtype, device=x.device)
        return MMDMonitor(mmd2=(k_xx - 2.0 * k_xy + k_yy).clamp_min(0.0),
                          k_xx=k_xx, k_xy=k_xy, k_yy=k_yy)
