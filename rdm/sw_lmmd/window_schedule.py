"""The overlapping window schedule over a fixed random permutation of the reference rows.

Window ``t`` is ``K`` consecutive positions of ``row_order`` starting at ``t * B``; its last
``B`` rows are *active* (freshly rolled out, back-propagated) and its first ``K - B`` are
*retained* (served from the generated-feature cache, detached). Sliding by exactly ``B`` makes

    previous.all_ids[B:] == current.retained_ids

an identity, which the cache relies on to hand back the right rows. It is asserted on every
transition, cyclic wrap included, rather than trusted -- a silent mismatch here would train
the generator against the wrong prompts' reference features and still look healthy in the loss.

Order is a **seeded random permutation** by default, not a semantic ordering. A window is
therefore a random size-``K`` subset ("local" means *this finite subset*, not *semantically
close*); semantic orderings are a controlled ablation because they shrink within-window
diversity and invite sequential forgetting.

**Prompt-grouped orders** (``window.group_by_prompt``, off by default). The reference holds
several rows per prompt (one per kept teacher seed), and a row-level permutation scatters them:
at K=1024 over 331k rows a row meets one of its siblings in its window with probability ~0.01.
The window then holds one generated sample and one reference per prompt, the same-prompt
attraction is never matched by a same-prompt repulsion, and the local MMD degenerates into a
per-prompt regression that drives every seed to one image. :func:`build_grouped_row_order`
permutes *prompts* instead and keeps each prompt's ``G`` rows adjacent; with ``G | K``, ``G | B``
and every offset a multiple of ``G``, each prompt enters the window as one active block of ``G``
fresh samples against its ``G`` references, and leaves it together.

Crossing an epoch boundary changes the permutation (or the cyclic offset), which destroys the
overlap identity above: the old cache is meaningless under the new order and must be dropped
and re-bootstrapped. :meth:`SlidingWindowSchedule.new_epoch` returns the new epoch index and
the trainer treats that as a cache-invalidation signal.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowBatch:
    """One window: the full row set, the retained prefix, the active suffix."""

    all_ids: np.ndarray
    retained_ids: np.ndarray
    active_ids: np.ndarray
    step: int
    epoch: int


def build_row_order(num_rows: int, seed: int = 3407) -> np.ndarray:
    """The canonical reproducible permutation ``row_order[position] -> row_id``."""
    return np.random.default_rng(seed).permutation(int(num_rows)).astype(np.int64)


def build_grouped_row_order(prompt_ids, seed: int = 3407) -> tuple[np.ndarray, int]:
    """A permutation that shuffles prompts but keeps each prompt's rows adjacent.

    Returns ``(order, G)``: ``order[G*j : G*(j+1)]`` are all the rows of one prompt, and the
    prompts appear in a seeded random order. Every prompt must own exactly ``G`` rows -- a
    ragged store cannot be tiled into windows that never split a prompt.
    """
    pid = np.asarray(prompt_ids, dtype=np.int64)
    if pid.ndim != 1 or pid.size == 0:
        raise ValueError("prompt_ids must be a non-empty 1-D array")
    counts = np.bincount(pid)
    counts = counts[counts > 0]
    group = int(counts[0])
    if not np.all(counts == group):
        raise ValueError(f"prompt-grouped windows need the same number of rows per prompt; "
                         f"this store has between {counts.min()} and {counts.max()}")
    rows_by_prompt = np.argsort(pid, kind="stable").reshape(-1, group)
    perm = np.random.default_rng(seed).permutation(rows_by_prompt.shape[0])
    return rows_by_prompt[perm].reshape(-1).astype(np.int64), group


def order_hash(order: np.ndarray) -> str:
    """Short digest of a row order, recorded in checkpoints so a resume cannot silently
    continue under a different permutation."""
    return hashlib.sha1(np.ascontiguousarray(order, dtype=np.int64).tobytes()).hexdigest()[:16]


class SlidingWindowSchedule:
    """Yields overlapping :class:`WindowBatch`es and guards the overlap identity."""

    def __init__(self, row_order, window_size: int = 1024, stride: int = 128,
                 cyclic: bool = True, start_offset: int = 0, reverse: bool = False,
                 epoch: int = 0, group_size: int = 1):
        order = np.asarray(row_order, dtype=np.int64)
        if order.ndim != 1 or order.size == 0:
            raise ValueError("row_order must be a non-empty 1-D array of row ids")
        if stride <= 0 or window_size <= 0:
            raise ValueError("window size and stride must be positive")
        if stride > window_size:
            raise ValueError(f"stride {stride} must be <= window_size {window_size}")
        if window_size > order.size:
            raise ValueError(f"window_size {window_size} exceeds the {order.size} available rows")
        group_size = int(group_size)
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        if group_size > 1:
            bad = {name: v for name, v in (("window_size", window_size), ("stride", stride),
                                           ("num_rows", order.size),
                                           ("start_offset", start_offset)) if v % group_size}
            if bad:
                raise ValueError(f"prompt-grouped windows need every size and offset to be a "
                                 f"multiple of the group size {group_size}; not: {bad}")
        self.group_size = group_size
        self._base_order = order
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.cyclic = bool(cyclic)
        self.epoch = int(epoch)
        self._configure(start_offset, reverse)

    # ---------------------------------------------------------------- internals
    def _configure(self, start_offset: int, reverse: bool) -> None:
        self.reverse = bool(reverse)
        self.order = self._base_order[::-1].copy() if self.reverse else self._base_order
        self.start_offset = int(start_offset) % self.order.size
        self.step = 0
        self._prev_all_ids: np.ndarray | None = None

    @property
    def overlap(self) -> int:
        return self.window_size - self.stride

    @property
    def num_rows(self) -> int:
        return int(self.order.size)

    def _positions(self, start: int, length: int) -> np.ndarray:
        pos = np.arange(start, start + length, dtype=np.int64)
        if self.cyclic:
            return pos % self.order.size
        if pos[-1] >= self.order.size:
            raise StopIteration("non-cyclic schedule exhausted; call new_epoch()")
        return pos

    def _window_at(self, step: int) -> WindowBatch:
        start = self.start_offset + step * self.stride
        ids = self.order[self._positions(start, self.window_size)]
        return WindowBatch(all_ids=ids, retained_ids=ids[:self.overlap],
                           active_ids=ids[self.overlap:], step=step, epoch=self.epoch)

    # ---------------------------------------------------------------- public API
    def peek(self) -> WindowBatch:
        """The window :meth:`next` would return, without advancing."""
        return self._window_at(self.step)

    def next(self) -> WindowBatch:
        """Advance one stride and return the new window (asserting the overlap identity)."""
        batch = self._window_at(self.step)
        if self._prev_all_ids is not None:
            expected = self._prev_all_ids[self.stride:]
            if not np.array_equal(expected, batch.retained_ids):
                raise RuntimeError(
                    "window overlap broken: previous.all_ids[stride:] != current.retained_ids "
                    f"at step {self.step} (expected {expected[:3]}..., got "
                    f"{batch.retained_ids[:3]}...). The generated cache would be misaligned.")
        self._prev_all_ids = batch.all_ids
        self.step += 1
        return batch

    def new_epoch(self, seed: int | None = None, random_start: bool = True,
                  reverse_probability: float = 0.5) -> int:
        """Reshuffle / re-offset for the next epoch. **Invalidates the generated cache.**

        Returns the new epoch index. Full coverage is preserved: the permutation is
        re-drawn or the traversal re-anchored, never truncated or re-weighted.
        """
        self.epoch += 1
        rng = np.random.default_rng(self.epoch if seed is None else seed)
        g = self.group_size
        if seed is not None:              # permute whole prompt groups, never split one
            groups = self._base_order.reshape(-1, g)
            self._base_order = groups[rng.permutation(groups.shape[0])].reshape(-1)
        offset = g * int(rng.integers(self.order.size // g)) if random_start else 0
        reverse = bool(rng.random() < reverse_probability)
        self._configure(offset, reverse)
        return self.epoch

    # ---------------------------------------------------------------- checkpointing
    def state_dict(self) -> dict:
        # Hash the BASE order: it is independent of `reverse` / `start_offset`, which are
        # recorded separately, so the digest identifies the permutation alone.
        return {"step": self.step, "start_offset": self.start_offset, "reverse": self.reverse,
                "epoch": self.epoch, "order_hash": order_hash(self._base_order),
                "window_size": self.window_size, "stride": self.stride,
                "group_size": self.group_size}

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        if strict and state.get("order_hash") not in (None, order_hash(self._base_order)):
            # Resuming under a different permutation would keep the step counter but change
            # which rows it points at -- silently training on a different schedule.
            raise RuntimeError("row order hash mismatch on resume: this checkpoint was written "
                               "under a different permutation. Rebuild row_order with the "
                               "recorded seed, or resume with strict=False and re-bootstrap.")
        saved_group = int(state.get("group_size", 1))
        if strict and saved_group != self.group_size:
            raise RuntimeError(f"window grouping mismatch on resume: the checkpoint was written "
                               f"with group_size={saved_group}, this config has "
                               f"{self.group_size}. Resume with the run's own config.")
        saved = (int(state.get("window_size", self.window_size)),
                 int(state.get("stride", self.stride)))
        if strict and saved != (self.window_size, self.stride):
            # The step counter means "slid `step` strides": under another K/B it names other
            # windows, and a resume without the cache would train on them without complaint.
            raise RuntimeError(f"window mismatch on resume: the checkpoint was written with "
                               f"K={saved[0]} B={saved[1]}, this config has K={self.window_size} "
                               f"B={self.stride}. Resume with the run's own config.")
        self.epoch = int(state.get("epoch", 0))
        self._configure(int(state.get("start_offset", 0)), bool(state.get("reverse", False)))
        self.step = int(state["step"])
