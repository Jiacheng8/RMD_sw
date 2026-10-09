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

**Smaller groups** (``window.group_size``, e.g. 2 for a store with 4 rows per prompt).
:func:`split_groups` cuts each prompt's rows into random sub-groups of that size, once per run
(seeded by ``order_seed``); each sub-group is then scheduled as a group of its own, at its own
position in the order. Every reference row is still used once per lap, and the same ``B``
holds ``B/G`` prompts instead of ``B/4`` -- more prompts per step, fewer seeds per prompt. With
two seeds per prompt the biased force's same-prompt repulsion is half the attraction; a
``window.sibling_weight`` above 1 (1.5 restores the 4-seed balance of 3/4, ``unbiased_siblings``
the full 1) sets it back (:mod:`rdm.sw_lmmd.local_mmd`).

**Mixed schedules** (``window.group_pattern``, e.g. ``"GU"``). Grouped and ungrouped steps pull in
opposite directions: grouped windows match each prompt's seed *distribution* (diverse, but no
better than the reference), ungrouped ones regress each sample onto one reference (sharper, more
often correct, collapsed). :class:`MixedGroupSchedule` alternates them step by step following a
repeating pattern -- ``"GU"`` one grouped step then one ungrouped, ``"GUU"`` one in three
grouped. Averaged over steps this weakens the same-prompt repulsion relative to the attraction
(roughly 0.8x for ``"GU"``, ~0.55x for ``"GUU"``, in units of the grouped force), i.e. it targets a
sharpened version of the reference: a knob between faithful diversity and collapse. One network
sees no flag telling the two step types apart, so it converges to one compromise distribution,
not to two behaviours.

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
    # True when the active block is whole prompt groups (G seeds of each prompt together)
    active_grouped: bool = False


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


def _equal_groups(grouping_ids) -> np.ndarray:
    """``(n_groups, G)`` row ids, one group per row of the result (equal-sized groups only)."""
    gid = np.asarray(grouping_ids, dtype=np.int64)
    if gid.ndim != 1 or gid.size == 0:
        raise ValueError("grouping ids must be a non-empty 1-D array")
    counts = np.bincount(gid)
    counts = counts[counts > 0]
    group = int(counts[0])
    if not np.all(counts == group):
        raise ValueError(f"grouped windows need the same number of rows per group; this store "
                         f"has between {counts.min()} and {counts.max()}")
    return np.argsort(gid, kind="stable").reshape(-1, group)


def split_groups(grouping_ids, size: int, seed: int = 3407) -> np.ndarray:
    """New grouping ids that cut every group into random sub-groups of ``size`` rows.

    Which rows of a group end up together is drawn per group (seeded), so a sub-group is not
    always "the two best-ranked references". ``size`` must divide the (equal) group size; equal
    to it, the result is the same partition renumbered.
    """
    size = int(size)
    groups = _equal_groups(grouping_ids)
    g = groups.shape[1]
    if size < 2 or g % size:
        raise ValueError(f"group_size {size} must be >= 2 and divide the store's {g} rows per group")
    rng = np.random.default_rng(seed)
    rows = groups[np.arange(groups.shape[0])[:, None], np.argsort(rng.random(groups.shape), axis=1)]
    out = np.empty(groups.size, dtype=np.int64)
    out[rows.reshape(-1)] = np.arange(groups.size) // size
    return out


def _scatter_rows(groups: np.ndarray, separation: int, rng) -> np.ndarray:
    """Rows of ``groups`` ``(n, G)`` in a sequence where two rows of one group are always at least
    ``separation`` positions apart (cyclically).

    Column ``k`` of the sequence holds one row of EVERY group, groups in a fresh random order, so
    a group's rows sit about ``n`` apart. Which of a group's rows goes to which column is drawn
    per group -- otherwise the first column (the part a short run sees) would hold only rank-0
    references. Near a column boundary a group could still end one column and start the next;
    those heads are swapped with rows from the column's middle.
    """
    n, g = groups.shape
    sep = int(separation)
    if g > 1 and n < sep:
        raise ValueError(f"{n} scattered groups are too few to keep a group's rows {sep} apart")
    rows = groups[np.arange(n)[:, None], np.argsort(rng.random((n, g)), axis=1)]
    if n >= 3 * sep:                                   # independent columns, boundaries repaired
        cols = [rng.permutation(n) for _ in range(g)]
        for k in list(range(1, g)) + [0]:              # column 0 last: it follows column g-1
            tail = set(cols[k - 1][n - sep:].tolist())
            cur = cols[k]
            free = [j for j in range(sep, n - sep) if cur[j] not in tail]
            for i in range(sep):
                if cur[i] in tail:
                    j = free.pop()
                    cur[i], cur[j] = cur[j], cur[i]
    else:                                              # small pool: one order for every column,
        perm = rng.permutation(n)                      # so a group's rows sit exactly n apart
        cols = [perm] * g
    return np.concatenate([rows[cols[k], k] for k in range(g)])


def build_mixed_row_order(grouping_ids, pattern: str, segment: int, seed: int = 3407,
                          separation: int | None = None,
                          phase: int = 0) -> tuple[np.ndarray, int, np.ndarray]:
    """A row order made of ``segment``-row blocks, grouped or scattered as ``pattern`` repeats.

    Returns ``(order, G, segment_grouped)``. Block ``i`` covers ``order[i*segment:(i+1)*segment]``;
    ``segment_grouped[i]`` is ``pattern[(i + phase) % len(pattern)] == "G"``. A grouped block is
    ``segment // G`` whole groups (G rows of one prompt each, adjacent); a scattered block is
    ``segment`` rows of OTHER groups, and two rows of one scattered group are at least
    ``separation`` (default ``segment``; the schedule passes its window size) positions apart in
    the scattered sequence -- hence never in one block, nor in one window. (A prompt that owns
    several groups -- a GenEval-block prompt has up to 24 -- can still meet itself through two
    different groups; that adds a little same-prompt repulsion, nothing else.) The groups are split between the two kinds
    in proportion to the pattern; each row is used at most once. The block count is a multiple of
    ``len(pattern)`` so the pattern also holds across the cyclic wrap; the few rows that do not
    fill whole blocks (fewer than ``len(pattern) * segment``) sit out this epoch.
    """
    pattern = str(pattern).upper()
    if not pattern or set(pattern) - {"G", "U"}:
        raise ValueError(f"group_pattern must be a non-empty string of G/U, got {pattern!r}")
    groups = _equal_groups(grouping_ids)
    g = groups.shape[1]
    if segment % g:
        raise ValueError(f"segment (the stride B={segment}) must be a multiple of the group size {g}")
    n_seg = groups.size // segment // len(pattern) * len(pattern)
    if n_seg == 0:
        raise ValueError(f"{groups.size} rows do not fill one {len(pattern)}x{segment}-row cycle")
    kinds = np.array([pattern[(i + phase) % len(pattern)] == "G" for i in range(n_seg)])
    n_grouped = int(kinds.sum()) * segment // g             # groups spent on grouped blocks
    n_scattered = int((~kinds).sum()) * segment // g        # groups whose rows are scattered

    rng = np.random.default_rng(seed)
    groups = groups[rng.permutation(groups.shape[0])]
    grouped = groups[:n_grouped].reshape(-1)                # already group-contiguous
    scattered = _scatter_rows(groups[n_grouped:n_grouped + n_scattered],
                              segment if separation is None else separation, rng)

    order = np.empty(n_seg * segment, dtype=np.int64)
    gi = si = 0
    for i, is_grouped in enumerate(kinds):
        block = slice(i * segment, (i + 1) * segment)
        if is_grouped:
            order[block] = grouped[gi:gi + segment]
            gi += segment
        else:
            order[block] = scattered[si:si + segment]
            si += segment
    return order, g, kinds


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
                           active_ids=ids[self.overlap:], step=step, epoch=self.epoch,
                           active_grouped=self.group_size > 1)

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
        # (MixedGroupSchedule adds "group_pattern"; a plain schedule is pattern "G" or none)

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        if strict and type(self) is SlidingWindowSchedule and state.get("group_pattern", "G") != "G":
            raise RuntimeError(f"group_pattern mismatch on resume: the checkpoint was written with "
                               f"{state['group_pattern']!r}, this config has no mixed pattern. "
                               f"Resume with the run's own config.")
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


class MixedGroupSchedule(SlidingWindowSchedule):
    """Grouped and ungrouped steps alternating as ``group_pattern`` repeats ("GU", "GUU", ...).

    The window still slides over ONE fixed order (:func:`build_mixed_row_order`), so the cache,
    the overlap identity and resume are untouched; only the order's layout changes. With
    ``K`` and every offset multiples of ``B``, each step's active block is exactly one segment,
    and :attr:`WindowBatch.active_grouped` says which kind it is.
    """

    def __init__(self, grouping_ids, group_pattern: str, window_size: int = 1024,
                 stride: int = 128, cyclic: bool = True, start_offset: int = 0,
                 reverse: bool = False, epoch: int = 0, seed: int = 3407):
        if window_size % stride:
            raise ValueError(f"a mixed schedule needs the window K={window_size} to be a multiple of "
                             f"the stride B={stride}, so each step's active block is one segment")
        if start_offset % stride:
            raise ValueError(f"start_offset {start_offset} must be a multiple of the stride {stride}")
        self.group_pattern = str(group_pattern).upper()
        self._grouping_ids = np.asarray(grouping_ids, dtype=np.int64)
        # phase: window 0's active block is segment (K-B)/B; make it pattern[0], so "GU" starts
        # with a grouped step
        self._phase = -((window_size - stride) // stride)
        order, group, kinds = build_mixed_row_order(self._grouping_ids, self.group_pattern,
                                                    stride, seed, separation=window_size,
                                                    phase=self._phase)
        self._base_kinds = kinds
        super().__init__(order, window_size=window_size, stride=stride, cyclic=cyclic,
                         start_offset=start_offset, reverse=reverse, epoch=epoch, group_size=group)

    def _configure(self, start_offset: int, reverse: bool) -> None:
        super()._configure(start_offset, reverse)
        # reversing the order reverses the segment sequence; each segment stays whole
        self.kinds = self._base_kinds[::-1].copy() if self.reverse else self._base_kinds

    def _window_at(self, step: int) -> WindowBatch:
        batch = super()._window_at(step)
        active_start = self.start_offset + step * self.stride + self.overlap
        segment = (active_start // self.stride) % self.kinds.size
        return WindowBatch(all_ids=batch.all_ids, retained_ids=batch.retained_ids,
                           active_ids=batch.active_ids, step=batch.step, epoch=batch.epoch,
                           active_grouped=bool(self.kinds[segment]))

    def new_epoch(self, seed: int | None = None, random_start: bool = True,
                  reverse_probability: float = 0.5) -> int:
        """Re-split the groups between the two kinds (with ``seed``) and/or re-anchor on a
        segment boundary. **Invalidates the generated cache.**"""
        self.epoch += 1
        rng = np.random.default_rng(self.epoch if seed is None else seed)
        if seed is not None:
            order, _, kinds = build_mixed_row_order(self._grouping_ids, self.group_pattern,
                                                    self.stride, int(rng.integers(2**31)),
                                                    separation=self.window_size, phase=self._phase)
            self._base_order, self._base_kinds = order, kinds
        offset = self.stride * int(rng.integers(self._base_kinds.size)) if random_start else 0
        reverse = bool(rng.random() < reverse_probability)
        self._configure(offset, reverse)
        return self.epoch

    def state_dict(self) -> dict:
        return {**super().state_dict(), "group_pattern": self.group_pattern}

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        saved = state.get("group_pattern", "G")
        if strict and saved != self.group_pattern:
            raise RuntimeError(f"group_pattern mismatch on resume: the checkpoint was written with "
                               f"{saved!r}, this config has {self.group_pattern!r}. Resume with the "
                               f"run's own config.")
        super().load_state_dict(state, strict=strict)
