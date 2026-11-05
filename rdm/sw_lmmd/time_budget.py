"""Train for a fixed amount of GPU time instead of a fixed number of steps.

The first ``calib_steps`` steps of a run are timed, wall clock, whatever the loop does between them
(probes, refreshes, checkpoints). The total is then ``steps so far + time left / s per step``: the lr
schedule decays over it and checkpoints are spread over it. A step that would overrun the budget is
not started, so the run ends within one step of it. The clock includes the bootstrap, not model loading.

``<run dir>/time_budget.json`` keeps the time spent and the plan, so a resumed run continues them.
Rank 0's clock decides and every rank follows it.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque

import torch
import torch.distributed as dist

from ..utils.distributed import get_world_size, is_dist, is_main_process

logger = logging.getLogger("rdm")

CHECKPOINTS = 4


class TimeBudget:
    def __init__(self, gpu_hours: float, out_dir: str, calib_steps: int = 60, device="cpu"):
        if gpu_hours <= 0 or calib_steps <= 0:
            raise ValueError("the time budget and its calibration steps must be positive")
        self.seconds = float(gpu_hours) * 3600 / get_world_size()
        self.gpu_hours = float(gpu_hours)
        self.calib_steps = int(calib_steps)
        self.path = os.path.join(out_dir, "time_budget.json")
        self.device = device
        self.total: int | None = None
        self.spent = 0.0
        if os.path.exists(self.path):
            with open(self.path) as f:
                saved = json.load(f)
            self.spent, self.total = float(saved["spent_seconds"]), saved["total_steps"]
        self.started = self.last = time.time()
        self.intervals: deque = deque(maxlen=self.calib_steps)

    def elapsed(self) -> float:
        return self.spent + time.time() - self.started

    def checkpoint_every(self) -> int | None:
        """Steps between checkpoints once the plan is known."""
        return None if self.total is None else max(1, self.total // CHECKPOINTS)

    def _agree(self, total: int, stop: bool) -> tuple[int, bool]:
        decision = torch.tensor([total, int(stop)], dtype=torch.int64, device=self.device)
        if is_dist():
            dist.broadcast(decision, 0)
        return int(decision[0]), bool(decision[1])

    def _save(self, step: int, finished: bool) -> None:
        if not is_main_process():
            return
        state = {"gpu_hours": self.gpu_hours, "seconds": self.seconds, "spent_seconds": self.elapsed(),
                 "total_steps": self.total, "step": step, "finished": finished}
        tmp = self.path + ".partial"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.path)

    def after_step(self, step: int, lr_schedule=None) -> bool:
        """Call after every step, on every rank. True: stop here."""
        now = time.time()
        self.intervals.append(now - self.last)
        self.last = now
        pace = sum(self.intervals) / len(self.intervals)
        left = self.seconds - self.elapsed()
        total = self.total or 0
        if not total and len(self.intervals) >= self.calib_steps:
            total = step + max(0, int(left / pace))
        total, stop = self._agree(total, left < pace or (total > 0 and step >= total))
        if total and self.total is None:
            self.total = total
            logger.info("[time budget] %.1f s/step -> %d steps in %.2f GPU-h (%.2f h wall)", pace, total,
                        self.gpu_hours, self.seconds / 3600)
        if self.total and lr_schedule is not None:
            lr_schedule.total_steps = self.total
        if stop:
            self.total = step                              # what the run came to
        self._save(step, stop)
        return stop

    def apply_saved_plan(self, lr_schedule=None) -> None:
        """A resumed run: the lr keeps decaying over the planned total."""
        if self.total and lr_schedule is not None:
            lr_schedule.total_steps = self.total
