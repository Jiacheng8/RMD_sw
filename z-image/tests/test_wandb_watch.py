"""The W&B watcher's parsing helpers (no wandb run)."""
import json

from zimage_distill.wandb_watch import flat_numbers, image_score, step_of, world_of


def test_rows_become_metrics():
    row = {"step": 3, "loss": 0.5, "skipped": False, "grad_norm": None, "s1/force": float("nan"),
           "note": "x", "active_rows": 64, "active_rows_per_rank": 16}
    m = flat_numbers(row, "train/")
    assert m == {"train/step": 3, "train/loss": 0.5, "train/skipped": 0, "train/active_rows": 64,
                 "train/active_rows_per_rank": 16}
    assert step_of(row) == 3 and step_of({"iteration": 7}) == 7 and step_of({}) is None
    assert world_of(row, None) == 4 and world_of({"rows": 128, "rows_per_rank": 16}, None) == 8
    assert world_of(row, 2) == 2


def test_image_score_is_the_mean_facet_score():
    out = {"Quality": {"Realism": {"A": {"score": 0}, "B": {"score": 2}}},
           "Alignment": {"Attributes": {"C": {"score": 1}, "D": {"score": "N/A"}}}}
    assert image_score({"judge_model_output": json.dumps(out)}) == (0 + 100 + 60) / 3
    assert image_score({"judge_model_output": ""}) is None
