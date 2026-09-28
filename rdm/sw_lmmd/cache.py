"""The cross-step cache of generated joint features -- the one stale quantity in SW-LMMD.

The cache holds exactly the current window: ``K`` rows per encoder, replicated identically on
every rank (it is filled from a detached all-gather, so no rank holds a private view). Each
step it yields the ``K - B`` retained rows as the detached half of the generated context, then
absorbs the ``B`` freshly computed active rows and slides.

Two properties are worth stating because they are easy to get wrong:

* **Rows age individually.** A row enters as active and survives up to ``ceil(K/B) - 1``
  further slides, so at ``K=1024, B=128`` the oldest context row was produced seven optimizer
  steps ago. ``generated_at_steps`` records per-row provenance; "the cache is one step stale"
  is false and would understate the bias.
* **Row identity is checked, not assumed.** :meth:`GeneratedWindowCache.retained` refuses to
  serve rows whose ids do not match what the schedule asked for. Serving misaligned rows
  would pair each generated sample with another prompt's reference features -- a bias that
  does not show up as a NaN, a crash, or even an obviously wrong loss.

This is distinct from the offline *reference* feature cache (fixed target, never stale) and
from GradCache's within-step feature cache (same parameters, discarded at the end of the step).
Only this one deliberately reuses features computed under older parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EncoderCacheEntry:
    """One encoder's window: row ids, ``(K, d_joint)`` features, per-row birth step."""

    row_ids: np.ndarray
    features: torch.Tensor
    generated_at_steps: torch.Tensor


class GeneratedWindowCache:
    """Per-encoder store of the current window's detached generated joint features."""

    def __init__(self, encoder_names, store_dtype: torch.dtype = torch.float32):
        self.encoder_names = list(encoder_names)
        self.store_dtype = store_dtype
        self.entries: dict[str, EncoderCacheEntry] = {}

    # ---------------------------------------------------------------- state
    @property
    def initialized(self) -> bool:
        return bool(self.entries)

    def clear(self) -> None:
        """Drop everything -- required at an epoch boundary, where the overlap identity
        (and therefore the meaning of the retained rows) no longer holds."""
        self.entries = {}

    def initialize(self, row_ids, features: dict, step: int = -1) -> None:
        """Seed the cache with the bootstrap rows (usually the first window's retained half)."""
        ids = np.asarray(row_ids, dtype=np.int64).copy()
        for name in self.encoder_names:
            feat = features[name].detach().to(self.store_dtype).contiguous()
            if feat.shape[0] != ids.size:
                raise ValueError(f"{name}: {feat.shape[0]} features for {ids.size} row ids")
            self.entries[name] = EncoderCacheEntry(
                row_ids=ids,
                features=feat,
                generated_at_steps=torch.full((feat.shape[0],), int(step), dtype=torch.int64))

    # ---------------------------------------------------------------- read
    def retained(self, encoder_name: str, expected_row_ids) -> torch.Tensor:
        """The detached features for ``expected_row_ids`` (the window's retained prefix)."""
        entry = self.entries[encoder_name]
        expected = np.asarray(expected_row_ids, dtype=np.int64)
        actual = entry.row_ids[-expected.size:] if expected.size else entry.row_ids[:0]
        if not np.array_equal(actual, expected):
            raise RuntimeError(
                f"cache row mismatch for {encoder_name!r}: schedule asked for "
                f"{expected[:3]}... but the cache tail holds {actual[:3]}.... The generated "
                f"context would be paired with the wrong prompts.")
        return entry.features[-expected.size:].detach() if expected.size else entry.features[:0]

    def max_age(self, current_step: int, encoder_name: str | None = None) -> int:
        """Age in optimizer steps of the oldest row currently in the window."""
        name = encoder_name or self.encoder_names[0]
        return int((current_step - self.entries[name].generated_at_steps).max().item())

    def age_buckets(self, current_step: int, encoder_name: str | None = None) -> dict:
        """``{age_in_steps: row_count}`` -- the staleness histogram worth logging every step."""
        name = encoder_name or self.encoder_names[0]
        ages = (current_step - self.entries[name].generated_at_steps).tolist()
        out: dict[int, int] = {}
        for a in ages:
            out[int(a)] = out.get(int(a), 0) + 1
        return out

    # ---------------------------------------------------------------- write
    def advance(self, full_row_ids, retained_count: int, active_features: dict,
                step: int) -> None:
        """Slide: keep the last ``retained_count`` rows, append the new active features."""
        ids = np.asarray(full_row_ids, dtype=np.int64).copy()
        for name in self.encoder_names:
            old = self.entries[name]
            retained = old.features[-retained_count:].detach() if retained_count else \
                old.features[:0].detach()
            active = active_features[name].detach().to(self.store_dtype)
            feat = torch.cat([retained, active], dim=0).contiguous()
            if feat.shape[0] != ids.size:
                raise ValueError(f"{name}: advanced to {feat.shape[0]} rows for {ids.size} ids")
            old_steps = old.generated_at_steps[-retained_count:] if retained_count else \
                old.generated_at_steps[:0]
            new_steps = torch.full((active.shape[0],), int(step), dtype=torch.int64)
            self.entries[name] = EncoderCacheEntry(
                row_ids=ids, features=feat,
                generated_at_steps=torch.cat([old_steps, new_steps], dim=0))

    def refresh_retained(self, encoder_name: str, row_ids, features: torch.Tensor,
                         step: int) -> torch.Tensor:
        """Overwrite cached rows with features recomputed under the current parameters.

        Returns the per-row relative drift ``||x_cached - x_fresh|| / (||x_fresh|| + eps)``
        *before* the overwrite. Refreshing changes only how accurate the context is; it never
        moves the window, changes visit counts, or produces a training gradient.
        """
        entry = self.entries[encoder_name]
        ids = np.asarray(row_ids, dtype=np.int64)
        pos = self._positions_of(entry.row_ids, ids)
        fresh = features.detach().to(entry.features.dtype)
        cached = entry.features[pos]
        drift = (cached.float() - fresh.float()).norm(dim=1) / (fresh.float().norm(dim=1) + 1e-8)
        entry.features[pos] = fresh
        entry.generated_at_steps[pos] = int(step)
        return drift

    @staticmethod
    def _positions_of(haystack: np.ndarray, needles: np.ndarray) -> torch.Tensor:
        """Index of each needle in ``haystack`` (raises if any row is absent)."""
        lookup = {int(r): i for i, r in enumerate(haystack)}
        try:
            return torch.tensor([lookup[int(r)] for r in needles], dtype=torch.long)
        except KeyError as e:
            raise RuntimeError(f"row {e.args[0]} is not in the current cache window") from e

    # ---------------------------------------------------------------- checkpointing
    def state_dict(self) -> dict:
        return {name: {"row_ids": e.row_ids, "features": e.features.cpu(),
                       "generated_at_steps": e.generated_at_steps.cpu()}
                for name, e in self.entries.items()}

    def load_state_dict(self, state: dict, device="cpu") -> None:
        self.entries = {
            name: EncoderCacheEntry(row_ids=np.asarray(v["row_ids"], dtype=np.int64),
                                    features=v["features"].to(device),
                                    generated_at_steps=v["generated_at_steps"])
            for name, v in state.items()}
