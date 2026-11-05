"""The SW-LMMD active force with a switchable attraction reference side.

COPIED FROM rdm/sw_lmmd/local_mmd.py, ``ExactLocalMMD.active_force`` (commit d5b2e9a), and changed in
one place only -- where the attraction's reference rows come from:

    force   = 2 * (repel - attract)
    repel   = (1/(B K)) sum_{i in A} sum_{j in W} k(x_i, sg(x_j))  [+ sibling weighting]   unchanged
    attract = (1/B) sum_{i in A} [ (1/K) sum_{j in W}   k(x_i, y_j) ]     local:  the window's K rows
            = (1/B) sum_{i in A} [ (1/K) sum_{j in R_t} k(x_i, y_j) ]     random: K rows drawn this step
            = (1/B) sum_{i in A} psi(x_i)^T mu_bar                        global: Nystrom, all N rows

(``B`` is the rank's local active count; the cross-rank gradient mean telescopes it to the global
``2/B`` exactly as in the original.) The repulsion, the sibling weighting, the fp32 floor and the
detach rules are the original lines, in the original order -- repulsion, attraction, then the
sibling term -- so ``ref_mode="local"`` reproduces ``ExactLocalMMD.active_force`` bit for bit,
values and gradients (``tests/test_ref_ablation.py``, test A). :meth:`active_force_parts`
additionally returns the live ``repel`` and ``attract`` tensors, for the per-term gradient
diagnostics and for test D.
"""
from __future__ import annotations

import torch

from rdm.compare.kernels import cross_kernel_mean, gamma_from_sigma
from rdm.sw_lmmd.local_mmd import at_least_fp32, sibling_kernel_sum

from .config import REF_MODES


class RefAblationLoss:
    """``ExactLocalMMD.active_force`` with the attraction taken against ``ref_mode``'s reference."""

    def __init__(self, mode: str, block_size: int = 256, nystrom=None):
        if mode not in REF_MODES:
            raise ValueError(f"ref_mode must be one of {REF_MODES}, got {mode!r}")
        if mode == "global" and nystrom is None:
            raise ValueError("ref_mode=global needs the Nystrom bundles (GlobalNystrom)")
        self.mode = mode
        self.block_size = int(block_size)
        self.nystrom = nystrom

    def attraction(self, name: str, a: torch.Tensor, reference: torch.Tensor | None,
                   gamma: float) -> torch.Tensor:
        """The bracket averaged over the rows of ``a``; differentiable in ``a`` only."""
        if self.mode == "global":
            return self.nystrom.attraction(name, a)
        # local / random: the original line, against whichever K rows the caller passed
        return cross_kernel_mean(a, at_least_fp32(reference).detach().to(a.dtype),
                                 gamma, self.block_size)

    def active_force_parts(self, name: str, active: torch.Tensor, generated_context: torch.Tensor,
                           reference: torch.Tensor | None, sigma: float, group: int = 1,
                           sibling_weight: float = 1.0):
        """``(force, stats, repel, attract)``; ``repel`` / ``attract`` keep their autograd graph.

        Args:
            name: encoder name (picks the Nystrom bundle in global mode).
            active: ``(B_local, d)`` live features of THIS rank's fresh rows (carry grad).
            generated_context: ``(K, d)`` full generated window, detached (unchanged).
            reference: ``(K, d)`` attraction reference rows -- the window's (local) or this step's
                random draw (random); ``None`` in global mode.
            sigma, group, sibling_weight: as in ``ExactLocalMMD.active_force``.
        """
        if not active.requires_grad:
            raise ValueError(
                "active features do not require grad -- the generator would receive no "
                "gradient. Only the context and the reference may be detached; do not wrap "
                "the encoder forward in torch.no_grad().")
        if self.mode != "global":
            if reference is None:
                raise ValueError(f"ref_mode={self.mode} needs the K reference rows")
            if generated_context.shape[0] != reference.shape[0]:
                raise ValueError(f"context has {generated_context.shape[0]} rows but reference has "
                                 f"{reference.shape[0]}; the attraction must average over K rows")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        gamma = gamma_from_sigma(sigma)
        a = at_least_fp32(active)
        repel = cross_kernel_mean(a, at_least_fp32(generated_context).detach().to(a.dtype),
                                  gamma, self.block_size)
        attract = self.attraction(name, a, reference, gamma)          # [ref-ablation] the one change
        stats = {}
        if sibling_weight != 1.0:
            sibling = sibling_kernel_sum(a, int(group), gamma) / float(a.shape[0] *
                                                                      generated_context.shape[0])
            repel = repel + (float(sibling_weight) - 1.0) * sibling
            stats["sibling_repulsion"] = sibling.detach()       # unweighted, in force units
        force = 2.0 * (repel - attract)
        stats.update({"active_repulsion": repel.detach(), "active_attraction": attract.detach(),
                      "force": force.detach()})
        return force, stats, repel, attract

    def active_force(self, name: str, active: torch.Tensor, generated_context: torch.Tensor,
                     reference: torch.Tensor | None, sigma: float, group: int = 1,
                     sibling_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
        """``(force, stats)``, the original's return signature."""
        force, stats, _, _ = self.active_force_parts(name, active, generated_context, reference,
                                                     sigma, group=group,
                                                     sibling_weight=sibling_weight)
        return force, stats
