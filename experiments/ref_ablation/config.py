"""The ``ref_ablation:`` config block -- the only setting the three ablation configs disagree on.

Everything else (window, cache, encoders, bandwidths, lr, steps, GAN, seeds) comes from the SW-LMMD
config the ablation configs extend, and is consumed by :func:`rdm.sw_lmmd.launch.build_trainer`
exactly as for a normal run.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The attraction's reference side: the window's own rows (the method), K rows drawn uniformly from
#: the whole reference each step, or the whole reference through a frozen Nystrom mean embedding.
REF_MODES = ("local", "random", "global")


@dataclass
class RefAblationConfig:
    ref_mode: str = "local"               # local | random | global
    ref_seed: int = 0                     # random mode: seeds the row sampler (its own stream)
    nystrom_dir: str | None = None        # per-encoder landmark bundles (build_landmarks.py);
                                          # required by global, optional elsewhere (extra logs)
    pre_diagnostic: bool = True           # before the first step: all three attractions on one window
    diagnostic_random_draws: int = 8      # random draws averaged in that diagnostic
    diagnostic_exact_global: bool = True  # ... and the exact mean over all N rows (reads the store once)
    grad_norms: bool = True               # per-encoder |d attraction / dx|, |d repulsion / dx| every step

    def validate(self) -> None:
        if self.ref_mode not in REF_MODES:
            raise ValueError(f"ref_ablation.ref_mode must be one of {REF_MODES}, got {self.ref_mode!r}")
        if self.ref_mode == "global" and not self.nystrom_dir:
            raise ValueError("ref_ablation.ref_mode: global needs ref_ablation.nystrom_dir (the bundles "
                             "python -m experiments.ref_ablation.build_landmarks writes)")
        if int(self.diagnostic_random_draws) < 1:
            raise ValueError("ref_ablation.diagnostic_random_draws must be >= 1")


def ref_ablation_from_config(cfg) -> RefAblationConfig:
    """Parse ``cfg.ref_ablation``. Unknown keys are refused: a typo would silently fall back to a
    default -- and a default ``ref_mode`` would silently train the method instead of the ablation."""
    raw = dict(getattr(cfg, "ref_ablation", None) or {})
    unknown = sorted(set(raw) - set(RefAblationConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"ref_ablation: unknown key(s) {unknown} (known: "
                         f"{sorted(RefAblationConfig.__dataclass_fields__)})")
    ref = RefAblationConfig(**raw)
    ref.validate()
    return ref
