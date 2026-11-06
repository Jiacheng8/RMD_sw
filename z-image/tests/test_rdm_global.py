"""The RDM (local-to-global) baseline on the toy few-step student (CPU, float64)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from rdm.compare.grad_balance import self_normalize
from rdm.compare.mmd_nystrom import mmd_nystrom_loss
from sw_lmmd_fixtures import NAMES
from test_zimage_multistep import DT, LATENT_CHANNELS, SIDE, StubBattery, student, store_root  # noqa: F401
from zimage_distill.rdm_global import (GlobalRDMTrainer, build_global_bundles, checkpoint_steps,
                                       gpu_seconds_of_run, load_bundles, plan_ready, plan_steps)
from zimage_distill.store import ZImageReferenceStore

ROWS = 8


@pytest.fixture(scope="module")
def packs(store_root, tmp_path_factory):  # noqa: F811
    store = ZImageReferenceStore(store_root, NAMES)
    paths = build_global_bundles(store, NAMES, tmp_path_factory.mktemp("nys"), n_landmarks=8)
    return load_bundles(paths, store, "cpu")        # fp32, as mmd_nystrom_loss computes


def build(store_root, packs, *, micro_batch=1, lr=0.0, rows=ROWS):  # noqa: F811
    store = ZImageReferenceStore(store_root, NAMES)
    model, gen = student()
    return GlobalRDMTrainer(gen, StubBattery(), store, packs,
                            torch.optim.AdamW(model.parameters(), lr=lr), encoders=list(NAMES),
                            num_steps=2, rows_per_step=rows, micro_batch=micro_batch,
                            noise_shape=(2, LATENT_CHANNELS, SIDE, SIDE), grad_clip=0.0,
                            use_store_context=False, device="cpu", feature_dtype=DT)


def test_bundles_cover_the_whole_store_and_are_cached(store_root, packs, tmp_path):  # noqa: F811
    store = ZImageReferenceStore(store_root, NAMES)
    for e in NAMES:
        assert packs[e]["Z"].shape == (8, store.dim(e) + store.text.shape[1])   # joint landmarks
        assert packs[e]["sigma"] == pytest.approx(store.sigma(e))
    paths = build_global_bundles(store, NAMES, tmp_path, n_landmarks=8)
    mtime = {e: (tmp_path / f"{e}_M8.pt").stat().st_mtime for e in NAMES}
    build_global_bundles(store, NAMES, tmp_path, n_landmarks=8)                # cached: no rebuild
    assert all((tmp_path / f"{e}_M8.pt").stat().st_mtime == mtime[e] for e in NAMES)
    assert set(paths) == set(NAMES)


def test_fresh_rows_every_step_and_the_rdm_loss(store_root, packs):  # noqa: F811
    t = build(store_root, packs)
    r0, r1 = t.step_rows(0), t.step_rows(1)
    assert r0.size == ROWS and np.unique(r0).size == ROWS and not np.array_equal(r0, r1)
    np.testing.assert_array_equal(r0, t.step_rows(0))                            # reproducible
    feats = t._encode(r0)
    loss, logs = t._loss(feats)
    expect = 0.0
    for k in (1, 2):
        for e in NAMES:
            p = packs[e]
            raw = mmd_nystrom_loss(feats[f"s{k}/{e}"], p["Z"], p["Z2"], p["alpha"], p["sigma"],
                                   p["k_rr"])
            expect = expect + 0.5 * self_normalize(raw)
            assert logs[f"s{k}/{e}/mmd2"] == pytest.approx(float(raw.detach()))
    torch.testing.assert_close(loss, expect)


def test_gradient_does_not_depend_on_the_micro_batch(store_root, packs):  # noqa: F811
    def grad(micro):
        t = build(store_root, packs, micro_batch=micro)
        t.optimizer.step = lambda: None
        t.step()
        return [p.grad.detach().clone() for p in t.generator.model.parameters()]
    for a, b in zip(grad(1), grad(ROWS)):
        torch.testing.assert_close(a, b)


def test_trains_and_writes_ours_checkpoint_format(store_root, packs):  # noqa: F811
    t = build(store_root, packs, lr=1e-2)
    before = [p.detach().clone() for p in t.generator.model.parameters()]
    for _ in range(3):
        logs = t.step()
        assert np.isfinite(logs["loss"]) and not logs["skipped"]
        assert logs["rows"] == ROWS and "s1/mmd2" in logs and "s2/mmd2" in logs
    assert any(not torch.equal(a, b) for a, b in zip(before, t.generator.model.parameters()))
    state = t.state_dict(with_optimizer=False)
    assert {"model", "step", "num_steps", "sampling_args"} <= set(state) and state["step"] == 3


def test_oom_probe_rolls_out_only_the_first_chunks(store_root, packs):  # noqa: F811
    t = build(store_root, packs, lr=1e-2)
    rolled = []
    real = t._encode
    t._encode = lambda ids: rolled.append(len(ids)) or real(ids)
    r = t.oom_probe(real_chunks=2)
    assert rolled == [1, 1, 1, 1]                       # 2 chunks of 8, pass 1 and pass 2
    assert r["rows"] == ROWS and r["real_chunks"] == 2 and np.isfinite(r["loss"])
    assert all(p.grad is None for p in t.generator.model.parameters())
    assert t.step_idx == 0


def test_gpu_time_plan(tmp_path):
    run = tmp_path / "ours"
    run.mkdir()
    with open(run / "train_log.jsonl", "w") as f:
        for s in range(1, 11):
            f.write(json.dumps({"step": s, "seconds": 90.0, "active_rows": 32,
                                "active_rows_per_rank": 16}) + "\n")
        f.write(json.dumps({"step": 10, "seconds": 90.0, "active_rows": 32,     # resumed: re-logged
                            "active_rows_per_rank": 16}) + "\n")
    gpu_s, world, steps = gpu_seconds_of_run(run)
    assert (world, steps) == (2, 10) and gpu_s == pytest.approx(10 * 90 * 2)
    # 8 GPUs at 30 s/step (the slow first step ignored): 1800 GPU-s -> 7.5 steps
    assert plan_steps(gpu_s, 8, [200] + [30] * 9, done=10) == 10      # never below what is done
    assert plan_steps(gpu_s, 8, [200, 30, 30], done=3) == 8          # 7.5 -> 8 (rounded)
    # 10240 rows: one step spends a large share of the budget -> plan after it, not after 10
    assert plan_ready(100 * 3600, 8, [2 * 3600], calib=10)            # 16 of 100 GPU-h
    # 100 GPU-h at ~16 GPU-h per 10240-row step: the nearest step count, not the truncated one
    assert plan_steps(100 * 3600, 8, [2 * 3600], done=1) == 6          # 6.25 -> 6
    assert plan_steps(100 * 3600, 4, [3.4 * 3600], done=1) == 7        # 7.35 -> 7
    assert plan_steps(100 * 3600, 2, [7.4 * 3600], done=1) == 7        # 6.76 -> 7 (int: 6)
    assert not plan_ready(100 * 3600, 8, [25.0] * 9, calib=10)
    assert checkpoint_steps(2000, 4) == {500, 1000, 1500, 2000}
    assert checkpoint_steps(7, 4) == {2, 4, 6, 7} and checkpoint_steps(2, 4) == {1, 2}
