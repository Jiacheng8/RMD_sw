"""The active force of the three mechanism variants, built on the reference-ablation loss.

Per encoder, ``B`` = this rank's active rows, ``K`` = window rows, ``A`` = active rows, ``y_i`` = the
stored reference row of active row ``i`` (the same row id; its text block is ``i``'s own caption, so
the joint kernel ``k(x_i, y_i)`` equals the image kernel -- "text kernel = 1 for the own row"):

    main         force = 2 (repel - att),  att = (1/(B K)) sum_{i in A} sum_{j in W} k(x_i, y_j)
                 -- RefAblationLoss(local) unchanged, i.e. ExactLocalMMD.active_force bit for bit.
                 Its own (diagonal) part ``own = -(2/(B K)) sum_i k(x_i, y_i)`` is only measured.
    nodiag       force = 2 (repel - att_nd), att_nd = (K/(K-1)) (1/(B K)) sum_i sum_{j != i} k(x_i, y_j)
                                                    = (1/(B (K-1))) sum_i sum_{j != i} k(x_i, y_j)
                 the repulsion as in main (its diagonal k(x_i, sg(x_i)) = 1 carries no gradient).
    global_self  force = 2 (repel - psi(x)^T mu_bar)  +  own,   own = beta (2/B) sum_i [1 - k(x_i, y_i)]
                 -- RefAblationLoss(global) plus the own-reference term. With the trainer's fixed
                 encoder weights 1/|E| the summed loss gains beta (2/(|E| B)) sum_e sum_i [1 - kappa_e].

``own`` is returned with its autograd graph so the trainer can split the feature gradient into
``g_own`` and ``g_rest`` (for ``nodiag`` it is the REMOVED term, reported but not trained).
"""
from __future__ import annotations

import torch

from rdm.compare.kernels import gamma_from_sigma, gaussian_gram
from rdm.sw_lmmd.local_mmd import at_least_fp32

from experiments.ref_ablation.ref_loss import RefAblationLoss

from .config import REF_MODE_OF, VARIANTS


class MechanismLoss(RefAblationLoss):
    """``RefAblationLoss`` (local or global) with the own-row term removed, measured or added."""

    def __init__(self, variant: str, beta: float = 0.0, block_size: int = 256, nystrom=None):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        super().__init__(REF_MODE_OF[variant], block_size=block_size, nystrom=nystrom)
        self.variant = variant
        self.beta = float(beta)
        self.own_reference = None     # callable(name) -> (B_local, d) reference rows of the active rows
        self.own_columns = None       # (B_local,) LongTensor: their positions in the window
        self.last_own = {}            # name -> own term tensor (with graph), per step
        self.last_own_kernel = {}     # name -> mean k(x_i, y_i), detached

    def active_force_parts(self, name, active, generated_context, reference, sigma, group: int = 1,
                           sibling_weight: float = 1.0):
        if self.own_reference is None:
            raise RuntimeError("MechanismLoss.own_reference is not set: the trainer sets it every "
                               "step before the loss runs")
        if group != 1 or sibling_weight != 1.0:
            raise ValueError("the mechanism ablation runs on ungrouped windows only (K128 · GAN)")
        force, stats, repel, attract = super().active_force_parts(
            name, active, generated_context, reference, sigma, group=group,
            sibling_weight=sibling_weight)
        gamma = gamma_from_sigma(sigma)
        a = at_least_fp32(active)
        K = int(generated_context.shape[0])
        if self.variant in ("main", "nodiag"):
            # the window's (K, d) reference, as RefAblationLoss(local) used it
            ref = at_least_fp32(reference).detach().to(a.dtype)
            cols = self.own_columns.to(a.device)
            gram = gaussian_gram(a, ref, gamma)                               # (B, K), with graph
            rows = torch.arange(a.shape[0], device=a.device)
            own_k = gram[rows, cols]                                          # k(x_i, y_i)
            own = -2.0 * own_k.sum() / float(a.shape[0] * K)                  # the diagonal of -2 att
            if self.variant == "nodiag":
                keep = torch.ones_like(gram, dtype=torch.bool)
                keep[rows, cols] = False
                attract = (gram * keep).sum() / float(a.shape[0] * (K - 1))   # K/(K-1) rescaled
                force = 2.0 * (repel - attract)
                stats.update({"active_attraction": attract.detach(), "force": force.detach()})
        else:
            y = at_least_fp32(self.own_reference(name)).detach().to(a.dtype)
            if y.shape != a.shape:
                raise ValueError(f"own reference rows {tuple(y.shape)} do not match the active rows "
                                 f"{tuple(a.shape)}")
            own_k = torch.exp(-gamma * (a - y).pow(2).sum(-1))                # k(x_i, y_i)
            own = self.beta * 2.0 * (1.0 - own_k).mean()                      # beta (2/B) sum [1 - k]
            force = force + own
            stats["force"] = force.detach()
        self.last_own[name] = own
        self.last_own_kernel[name] = own_k.detach().mean()
        stats.update({"own_kernel": own_k.detach().mean(), "own_term": own.detach()})
        return force, stats, repel, attract
