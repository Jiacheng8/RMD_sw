"""The ``random`` attraction's rows: K reference rows drawn uniformly from the whole store, every step.

The draw is independent of the window: a fresh uniform K-subset of all N rows (without
replacement inside a step, independent across steps). Each drawn row is read through
:meth:`rdm.sw_lmmd.reference_store.ReferenceFeatureStore.reference_joint_features`, so it carries
its OWN caption's text block ``beta * tau(c_row)`` -- a complete reference pair -- never the
generator-side prompt.

The sampler owns a private :class:`numpy.random.Generator`. It never touches torch's or numpy's
global RNG, the per-row latent noise (:func:`rdm.sw_lmmd.sharding.row_noise` derives it from row
ids) or the window permutation, so switching ``ref_mode`` leaves the generator side bit-identical.
Every rank builds it from the same seed and draws once per step, so all ranks read the same rows.
"""
from __future__ import annotations

import copy

import numpy as np


class RandomReferenceSampler:
    """Uniform K-subsets of ``range(num_rows)``, one per :meth:`draw`, from a private stream."""

    #: Domain tags mixed into the seed, so the training stream and the diagnostic stream (and any
    #: other generator seeded with the same integer) never coincide.
    STREAMS = {"train": 0x52414E44, "diagnostic": 0x44494147}

    def __init__(self, num_rows: int, k: int, seed: int = 0, stream: str = "train"):
        if stream not in self.STREAMS:
            raise ValueError(f"unknown stream {stream!r} (known: {sorted(self.STREAMS)})")
        if not 0 < int(k) <= int(num_rows):
            raise ValueError(f"cannot draw {k} distinct rows from {num_rows}")
        self.num_rows, self.k, self.seed, self.stream = int(num_rows), int(k), int(seed), stream
        self.rng = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([self.seed, self.STREAMS[stream]])))
        self.draws = 0

    def draw(self) -> np.ndarray:
        """``k`` distinct row ids, uniformly at random (order random too)."""
        rows = self.rng.choice(self.num_rows, size=self.k, replace=False).astype(np.int64)
        self.draws += 1
        return rows

    def state_dict(self) -> dict:
        return {"num_rows": self.num_rows, "k": self.k, "seed": self.seed, "stream": self.stream,
                "draws": self.draws, "bit_generator": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state: dict) -> None:
        saved = (int(state["num_rows"]), int(state["k"]), int(state["seed"]), state["stream"])
        mine = (self.num_rows, self.k, self.seed, self.stream)
        if saved != mine:
            raise RuntimeError(f"random reference sampler mismatch on resume: the checkpoint drew "
                               f"(num_rows, k, seed, stream) = {saved}, this run {mine}")
        self.rng.bit_generator.state = copy.deepcopy(state["bit_generator"])
        self.draws = int(state["draws"])
