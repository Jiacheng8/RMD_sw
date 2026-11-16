"""The two config flags of the own-reference (mechanism) ablation, both OFF by default.

    attraction:
      skip_diagonal: false   # true: drop k(X_i, Y_i) from the window attraction, rescale the rest
                             #       by K/(K-1)                          (run b, ``window-nodiag``)
    own_term:
      beta: 0.0              # > 0: add beta * (2/(|E| B)) sum_e sum_i [1 - kappa_e(u_i, v_i)] to the
                             #      Global arm's loss                     (run a, ``global-plus-self-*``)

They are read by ``experiments.mechanism_ablation.train`` -- the entry point of the Table 2 Global
arm (``experiments.ref_ablation``) plus these flags -- not by ``rdm.sw_lmmd.launch``. With both off
the loss is the reference ablation's unchanged: ``ref_mode: local`` is the main run's
``ExactLocalMMD`` force bit for bit, ``ref_mode: global`` the Global arm's
(tests/test_mechanism_ablation.py). The resulting variant:

    ref_mode local,  flags off          -> ``main``         (the method; own term only measured)
    ref_mode local,  skip_diagonal      -> ``nodiag``
    ref_mode global, own_term.beta >= 0 -> ``global_self``  (beta 0 = the Global arm itself)

The run directory is ``exp_name``. ``expected_row_order_hash`` (top level) is checked at start-up.
"""
from __future__ import annotations

from dataclasses import dataclass

VARIANTS = ("main", "nodiag", "global_self")
REF_MODE_OF = {"main": "local", "nodiag": "local", "global_self": "global"}
#: sha1[:16] of the window row order with order_seed 3407 on the FLUX.2 [klein] COCO store -- the
#: main run's (analysis/out/step2000_w_gan/gen_meta.json). Empty string disables the check (tests).
MAIN_ORDER_HASH = "6559fefb9a872952"


@dataclass
class MechanismConfig:
    arm: str = ""                       # run directory name (exp_name)
    skip_diagonal: bool = False
    self_beta: float = 0.0
    ref_mode: str = "local"
    expected_order_hash: str = MAIN_ORDER_HASH

    @property
    def variant(self) -> str:
        if self.ref_mode == "global":
            return "global_self"
        return "nodiag" if self.skip_diagonal else "main"

    def validate(self, ref_mode: str | None = None) -> None:
        ref_mode = ref_mode or self.ref_mode
        if ref_mode != self.ref_mode:
            raise ValueError(f"ref_mode mismatch: {ref_mode} vs {self.ref_mode}")
        if ref_mode not in ("local", "global"):
            raise ValueError(f"the mechanism flags need ref_ablation.ref_mode local or global, got {ref_mode}")
        if self.skip_diagonal and ref_mode != "local":
            raise ValueError("attraction.skip_diagonal needs ref_ablation.ref_mode: local (the window "
                             "attraction has a diagonal; the global one has none)")
        if self.self_beta < 0:
            raise ValueError(f"own_term.beta must be >= 0, got {self.self_beta}")
        if self.self_beta > 0 and ref_mode != "global":
            raise ValueError("own_term.beta > 0 needs ref_ablation.ref_mode: global (run a is the Global "
                             "arm plus the own term; the window already contains it)")
        if not self.arm:
            raise ValueError("exp_name (the run directory) must be set")


def _block(cfg, name: str, known: set) -> dict:
    raw = dict(getattr(cfg, name, None) or {})
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{name}: unknown key(s) {unknown} (known: {sorted(known)})")
    return raw


def mechanism_from_config(cfg) -> MechanismConfig:
    """Parse ``attraction`` / ``own_term``; unknown keys are refused (a typo must not silently train
    the plain method)."""
    from experiments.ref_ablation.config import ref_ablation_from_config

    att = _block(cfg, "attraction", {"skip_diagonal"})
    own = _block(cfg, "own_term", {"beta"})
    skip = att.get("skip_diagonal", False)
    if not isinstance(skip, bool):
        raise ValueError(f"attraction.skip_diagonal must be true or false, got {skip!r}")
    mech = MechanismConfig(arm=str(getattr(cfg, "exp_name", "") or ""), skip_diagonal=skip,
                           self_beta=float(own.get("beta", 0.0) or 0.0),
                           ref_mode=ref_ablation_from_config(cfg).ref_mode,
                           expected_order_hash=str(getattr(cfg, "expected_row_order_hash", MAIN_ORDER_HASH)))
    mech.validate()
    return mech
