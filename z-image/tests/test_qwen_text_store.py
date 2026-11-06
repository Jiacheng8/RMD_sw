"""zimage_distill.qwen_text_store: which rows it fills, and the pooling."""
import numpy as np
import torch

from rdm.sw_lmmd.window_schedule import SlidingWindowSchedule
from zimage_distill.qwen_text_store import pool_context, rows_read, stats_rows


def test_rows_read_are_exactly_the_schedule_s():
    order = np.random.default_rng(0).permutation(1000)
    schedule = SlidingWindowSchedule(order, window_size=16, stride=4)
    seen = np.unique(np.concatenate([schedule.next().all_ids for _ in range(20)]))
    assert set(seen) == set(rows_read(order, 20, 16, 4))


def test_stats_rows_cover_the_critics_sample():
    for seed in (0, 1):
        want = np.random.default_rng(seed).choice(500, size=64, replace=False)
        assert set(want) <= set(stats_rows(500, 64, [0, 1]))


def test_pool_is_a_unit_mean():
    ctx = torch.tensor([[3.0, 0.0], [3.0, 8.0]], dtype=torch.bfloat16)
    assert np.allclose(pool_context(ctx), [0.6, 0.8])


def test_from_hub_assembles_next_to_the_source(tmp_path, monkeypatch):
    import json

    import huggingface_hub
    import pytest

    from zimage_distill.qwen_text_store import BUILT, LINKED, from_hub

    source, out = tmp_path / "reference_store", tmp_path / "reference_store_qwen"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps({"prompt_hash": "p", "num_rows": 3, "num_prompts": 3}))
    (source / "z_context.json").write_text("{}")

    def fake_download(repo, name, repo_type, local_dir):
        assert repo_type == "dataset"
        path = tmp_path / local_dir / name
        if name == "text_features.npy":
            np.save(path, np.zeros((3, 8), dtype=np.float32))
        else:
            path.write_text(json.dumps({"prompt_hash": hash_on_hub, "num_rows": 3, "text_dim": 8}))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    hash_on_hub = "other"
    with pytest.raises(SystemExit):
        from_hub("r", source, out)
    assert not out.exists()
    hash_on_hub = "p"
    from_hub("r", source, out)
    assert all((out / n).exists() for n in BUILT) and (out / "z_context.json").exists()
    assert all((out / n).is_symlink() and str((out / n).readlink()).startswith("..") for n in LINKED)
