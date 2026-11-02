"""The W&B watcher's parsing helpers (no wandb run)."""
import json

from zimage_distill.wandb_watch import OURS_LOSSES, flat_numbers, image_score, section, step_of, world_of


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


def test_probe_and_refresh_keys_get_their_own_section():
    for k in ("s2/siglip2/drift_mean", "s1/dinov3_l/drift_max", "drift_age_3", "drift_probe_rows",
              "drift_mean", "refreshed_rows", "refresh_drift_mean"):
        assert section(k) == "cache/", k
    for k in ("loss", "force", "raw_mmd2", "s2/siglip2/mmd2", "s2/siglip2/repulsion", "gan/d_acc"):
        assert section(k) == "train/", k


def test_ours_sends_only_the_chosen_losses():
    for k in ("grad_norm", "s1/gan/dinov3_l/g_loss", "s2/gan/siglip2/d_loss", "s1/siglip2/mmd2", "s2/dinov3_l/mmd2"):
        assert OURS_LOSSES.fullmatch(k), k
    for k in ("loss_total", "raw_mmd2", "lr", "s1/force", "s2/gan/d_acc", "s2/gan/g_loss", "s2/gan/aimv2_huge/d_loss",
              "s2/aimv2_huge/mmd2", "s2/siglip2/force", "s2/gan/siglip2/d_acc", "s2/siglip2/drift_mean"):
        assert not OURS_LOSSES.fullmatch(k), k
