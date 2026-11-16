"""``RefAblationTrainer`` with the mechanism variants of :mod:`.loss` and the own/rest gradient split.

Nothing of ``RefAblationTrainer.step`` is copied: the step calls ``self._attraction_rows(window)``
once before its loss closure and ``self.ref_loss.active_force_parts`` /
``self._collect_ref_diagnostics`` / ``self._reduce_ref_diagnostics`` inside it, and those four are
the hooks used here:

* ``_attraction_rows`` also records this rank's active row ids and their columns in the window, so
  the loss can find each active row's own reference ``y_i``;
* ``ref_loss`` is a :class:`MechanismLoss` (``main``: the method unchanged; ``nodiag``;
  ``global_self``) -- what it returns is what is trained, and the critic's adaptive weight is
  balanced against it (``global_self``: the own term is inside, as the own pull is in the main run);
* the diagnostics add, per step, the split of ``dL_MMD/du`` over the active image features (every
  encoder weighted by its fixed ``lambda_e``) into ``g_own`` (the own-row term) and
  ``g_rest = g_total - g_own``: per encoder and concatenated over encoders (``mech/all/...``).

Logged per step (cross-rank, world-size invariant):
    mech/<enc>/own_kernel   mean k(x_i, y_i) over the B active rows
    mech/<enc>/own_term     the own term's value in force units (nodiag: the removed one)
    mech/<s>/grad_own, grad_total, share = |g_own| / |g_total|, cos_own_total, cos_rest_total
                            for s in each encoder and ``all``
    mech/all/sq_own, sq_total, dot_own_total   the raw sums the beta calibration solves with
For ``nodiag`` ``g_own`` is the gradient the removed diagonal WOULD have had (not trained) and
``g_total`` the trained one.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from rdm.sw_lmmd.sharding import partition_rows
from rdm.sw_lmmd.trainer import SWLMMDTrainer
from rdm.sw_lmmd.window_schedule import order_hash
from rdm.utils.distributed import get_world_size

from experiments.ref_ablation.trainer import RefAblationTrainer

from .config import MechanismConfig
from .loss import MechanismLoss

_SUM = "mechsum/"


class MechanismTrainer(RefAblationTrainer):
    """See the module docstring."""

    @classmethod
    def from_trainer_mech(cls, trainer: SWLMMDTrainer, mech: MechanismConfig, *, ref_mode: str,
                          ref_seed: int = 0, nystrom=None,
                          grad_norms: bool = True) -> "MechanismTrainer":
        obj = cls.from_trainer(trainer, ref_mode=ref_mode, ref_seed=ref_seed, nystrom=nystrom,
                               grad_norms=grad_norms)
        obj._setup_mechanism(mech)
        return obj

    def _setup_mechanism(self, mech: MechanismConfig) -> None:
        mech.validate(self.ref_mode)
        if getattr(self.schedule, "group_size", 1) != 1:
            raise ValueError("the mechanism ablation needs ungrouped windows (K128 · GAN)")
        self.mech = mech
        self.ref_loss = MechanismLoss(mech.variant, beta=mech.self_beta,
                                      block_size=self.mmd.block_size, nystrom=self.nystrom)
        self.ref_loss.own_reference = lambda name: self._reference_window(name, self._own_ids)
        self._own_ids = None
        self.row_order_hash = order_hash(self.schedule._base_order)
        if mech.expected_order_hash and self.row_order_hash != mech.expected_order_hash:
            raise RuntimeError(f"window row order hash {self.row_order_hash} != the main run's "
                               f"{mech.expected_order_hash}: another store or order_seed")

    # ------------------------------------------------------------------ hooks
    def _attraction_rows(self, window):
        ids = partition_rows(window.active_ids)                 # the step's own partition
        all_ids = np.asarray(window.all_ids, dtype=np.int64)
        pos = {int(r): p for p, r in enumerate(all_ids)}
        if len(pos) != all_ids.size:
            raise RuntimeError("a window holds the same row twice: the own column is ambiguous")
        self._own_ids = ids
        self.ref_loss.own_columns = torch.as_tensor([pos[int(r)] for r in ids], dtype=torch.long)
        self.ref_loss.last_own, self.ref_loss.last_own_kernel = {}, {}
        return super()._attraction_rows(window)

    def _collect_ref_diagnostics(self, name, active, force, repel, attract, reference,
                                 means: dict, sums: dict) -> None:
        super()._collect_ref_diagnostics(name, active, force, repel, attract, reference, means, sums)
        own = self.ref_loss.last_own[name]
        w = float(self.encoder_weights[name])
        d = self.image_dims[name]
        g_tot = torch.autograd.grad(w * force, active, retain_graph=True)[0][:, :d].double()
        g_own = torch.autograd.grad(w * own, active, retain_graph=True)[0][:, :d].double()
        p = f"mech/{name}/"
        means[p + "own_term"] = own.detach()
        means[p + "own_kernel"] = self.ref_loss.last_own_kernel[name]
        if self.mech.variant == "global_self" and active.shape[1] > d:
            with torch.no_grad():     # own row and active row share one caption: the text block is equal
                y = self._reference_window(name, self._own_ids).to(active.dtype)
                means[p + "own_text_gap"] = (active.detach()[:, d:] - y[:, d:]).abs().max()
        # squared norms and the dot product: summed over ranks (all-reduced as "means" x world)
        means[_SUM + name + "/own2"] = g_own.pow(2).sum()
        means[_SUM + name + "/tot2"] = g_tot.pow(2).sum()
        means[_SUM + name + "/dot"] = (g_own * g_tot).sum()

    def _reduce_ref_diagnostics(self, means: dict, sums: dict) -> dict:
        out = super()._reduce_ref_diagnostics(means, sums)
        world = get_world_size()
        raw = {k[len(_SUM):]: out.pop(k) * world for k in list(out) if k.startswith(_SUM)}
        if not raw:
            return out
        acc = {"own2": 0.0, "tot2": 0.0, "dot": 0.0}
        for name in self.encoder_names:
            vals = {q: raw[f"{name}/{q}"] for q in acc}
            for q in acc:
                acc[q] += vals[q]
            out.update(_split(f"mech/{name}/", vals, world))
        out.update(_split("mech/all/", acc, world))
        # world-size invariant raw sums (each rank's force is normalized by its own B/R rows)
        out["mech/all/sq_own"] = acc["own2"] / world ** 2
        out["mech/all/sq_total"] = acc["tot2"] / world ** 2
        out["mech/all/dot_own_total"] = acc["dot"] / world ** 2
        return out

    # ------------------------------------------------------------------ checkpointing
    def state_dict(self, with_cache: bool = True, with_optimizer: bool = True,
                   with_critic: bool = True) -> dict:
        state = super().state_dict(with_cache=with_cache, with_optimizer=with_optimizer,
                                   with_critic=with_critic)
        state["mechanism"] = {"arm": self.mech.arm, "variant": self.mech.variant,
                              "self_beta": self.mech.self_beta,
                              "row_order_hash": self.row_order_hash}
        return state

    def load_state_dict(self, state: dict) -> dict:
        saved = state.get("mechanism")
        if saved is None:
            raise RuntimeError("not a mechanism-ablation checkpoint (no 'mechanism' entry); a main-run "
                               "checkpoint is loaded with load_plain_state (probe only)")
        mine = {"arm": self.mech.arm, "variant": self.mech.variant, "self_beta": self.mech.self_beta}
        if any(saved.get(k) != v for k, v in mine.items()):
            raise RuntimeError(f"mechanism mismatch on resume: checkpoint {saved}, config {mine}")
        return super().load_state_dict(state)

    def load_plain_state(self, state: dict) -> dict:
        """A checkpoint of the plain SW-LMMD trainer (the main run): weights, step, window position."""
        return SWLMMDTrainer.load_state_dict(self, state)


def _split(prefix: str, v: dict, world: int) -> dict:
    """|g_own|, |g_total|, the share and the two cosines from summed squares and the dot product."""
    own2, tot2, dot = v["own2"], v["tot2"], v["dot"]
    rest2 = max(tot2 - 2.0 * dot + own2, 0.0)
    n_own, n_tot, n_rest = math.sqrt(max(own2, 0.0)), math.sqrt(max(tot2, 0.0)), math.sqrt(rest2)
    tiny = 1e-300
    return {prefix + "grad_own": n_own / world, prefix + "grad_total": n_tot / world,
            prefix + "grad_rest": n_rest / world,
            prefix + "share": n_own / max(n_tot, tiny),
            prefix + "cos_own_total": dot / max(n_own * n_tot, tiny),
            prefix + "cos_rest_total": (tot2 - dot) / max(n_rest * n_tot, tiny)}
