"""rdm.sw_lmmd.time_budget: the plan, the stop, the resume."""
import json

from rdm.sw_lmmd.config import LRSchedule
from rdm.sw_lmmd.time_budget import TimeBudget


def clocked(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("rdm.sw_lmmd.time_budget.time.time", lambda: now[0])
    return now


def run(budget, now, seconds_per_step, schedule=None, first=1):
    step = first
    while True:
        now[0] += seconds_per_step
        if budget.after_step(step, schedule):
            return step
        step += 1


def test_plans_decays_and_stops_on_budget(tmp_path, monkeypatch):
    now = clocked(monkeypatch)
    schedule = LRSchedule(name="cosine", warmup_steps=2, min_lr_ratio=0.1, total_steps=10 ** 6)
    budget = TimeBudget(1.0, str(tmp_path), calib_steps=5)        # 1 GPU-h on one rank
    assert run(budget, now, 10.0, schedule) == 360
    assert schedule.total_steps == 360 and budget.checkpoint_every() == 90
    saved = json.loads((tmp_path / "time_budget.json").read_text())
    assert saved["finished"] and saved["total_steps"] == 360 and abs(saved["spent_seconds"] - 3600) < 1e-6


def test_a_slowdown_cannot_overrun(tmp_path, monkeypatch):
    now = clocked(monkeypatch)
    budget = TimeBudget(1.0, str(tmp_path), calib_steps=5)
    for step in range(1, 6):
        now[0] += 10.0
        assert not budget.after_step(step)
    assert budget.total == 360
    step = run(budget, now, 100.0, first=6)
    assert 3500 < budget.elapsed() <= 3600 + 100 and step < 360


def test_one_slow_step_is_enough_to_plan(tmp_path, monkeypatch):
    now = clocked(monkeypatch)
    budget = TimeBudget(4.0, str(tmp_path), calib_steps=1)
    assert run(budget, now, 1200.0) == 12


def test_resume_keeps_the_time_and_the_plan(tmp_path, monkeypatch):
    now = clocked(monkeypatch)
    budget = TimeBudget(1.0, str(tmp_path), calib_steps=5)
    for step in range(1, 181):
        now[0] += 10.0
        assert not budget.after_step(step)
    schedule = LRSchedule(name="cosine", total_steps=10 ** 6)
    resumed = TimeBudget(1.0, str(tmp_path), calib_steps=5)
    resumed.apply_saved_plan(schedule)
    assert schedule.total_steps == 360 and abs(resumed.spent - 1800) < 1e-6
    assert run(resumed, now, 10.0, schedule, first=181) == 360
