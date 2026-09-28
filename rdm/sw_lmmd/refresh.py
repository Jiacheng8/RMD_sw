"""Cache staleness: measure the drift of retained features, and optionally refresh them.

At ``K=1024, B=128`` the oldest context row was generated seven optimizer steps ago, so the
generated context is not a sample from the current student. That bias is the method's central
research risk, which makes drift a first-class metric rather than a debug aid:

    delta_e = || x_cached - x_refreshed || / (|| x_refreshed || + eps)

bucketed by cache age. Because :func:`rdm.sw_lmmd.sharding.row_noise` derives each latent from
its row id, the re-roll reuses the row's original noise -- so ``delta`` isolates the change in
the *parameters* instead of mixing it with a different noise draw. Without that property the
metric would be dominated by sampling variance and would say nothing about staleness.

Both entry points are gradient-free and side-effect-free with respect to the method's state:
they never move the window, never change visit counts, and never contribute to a training
gradient. A refresh changes only how accurately the cached context represents the current
student; it never touches the reference, which is frozen by construction.
"""
from __future__ import annotations

import numpy as np
import torch

from .sharding import all_gather_detached, partition_rows


def _rollout_rows(trainer, row_ids: np.ndarray) -> dict:
    """No-grad re-encode of ``row_ids``, gathered so every rank holds identical features."""
    local = partition_rows(row_ids)
    feats = trainer._encode_no_grad(trainer._chunks(local))
    return {name: all_gather_detached(feats[name]) for name in trainer.encoder_names}


@torch.no_grad()
def probe_drift(trainer, window, n_rows: int = 64, seed: int | None = None) -> dict:
    """Re-roll a sample of retained rows and report drift by age bucket (cache untouched).

    ``n_rows`` is rounded down to a multiple of ``world_size * micro_batch`` so every rank
    contributes an equal-shaped block to the all-gather.
    """
    from ..utils.distributed import get_world_size
    retained = np.asarray(window.retained_ids, dtype=np.int64)
    granule = max(1, get_world_size() * trainer.micro_batch)
    n = min(int(n_rows), retained.size) // granule * granule
    if n == 0:
        return {}
    rng = np.random.default_rng(trainer.seed if seed is None else seed)
    # Sample positions, then sort: the cache stores rows in window order, and a sorted
    # selection keeps the gathered block aligned with the ids we ask the cache for.
    sel = np.sort(rng.choice(retained.size, size=n, replace=False))
    rows = retained[sel]

    fresh = _rollout_rows(trainer, rows)
    entry = trainer.cache.entries[trainer.encoder_names[0]]
    pos = trainer.cache._positions_of(entry.row_ids, rows)
    ages = (window.step - entry.generated_at_steps[pos]).tolist()

    out, per_bucket = {}, {}
    for name in trainer.encoder_names:
        cached = trainer.cache.entries[name].features[pos].float()
        new = fresh[name].float().to(cached.device)
        drift = ((cached - new).norm(dim=1) / (new.norm(dim=1) + 1e-8)).tolist()
        out[f"{name}/drift_mean"] = float(np.mean(drift))
        out[f"{name}/drift_max"] = float(np.max(drift))
        for age, d in zip(ages, drift):
            per_bucket.setdefault(int(age), []).append(float(d))
    for age, vals in sorted(per_bucket.items()):
        out[f"drift_age_{age}"] = float(np.mean(vals))
    out["drift_probe_rows"] = int(n)
    out["drift_mean"] = float(np.mean([v for vals in per_bucket.values() for v in vals]))
    return out


@torch.no_grad()
def refresh_retained(trainer, window) -> dict:
    """Recompute every retained row under the current parameters and overwrite the cache."""
    rows = np.asarray(window.retained_ids, dtype=np.int64)
    fresh = _rollout_rows(trainer, rows)
    drifts = []
    for name in trainer.encoder_names:
        d = trainer.cache.refresh_retained(name, rows, fresh[name], step=window.step)
        drifts.append(float(d.mean()))
    return {"refreshed_rows": int(rows.size), "refresh_drift_mean": float(np.mean(drifts))}


def should_refresh(trainer, window, measured_drift: float | None, cfg) -> bool:
    """Periodic schedule OR a drift trigger (spec sec. 16)."""
    every = int(getattr(cfg, "refresh_every_windows", 0) or 0)
    if every and window.step > 0 and window.step % every == 0:
        return True
    threshold = float(getattr(cfg, "drift_threshold", 0.0) or 0.0)
    return bool(threshold and measured_drift is not None and measured_drift > threshold)
