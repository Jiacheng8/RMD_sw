"""End-to-end SW-LMMD trainer tests on a toy generator + mock battery (CPU, no weights).

Exercises the full control flow -- bootstrap, GradCache two-pass, detached gather, per-encoder
force, cache slide, drift probe, refresh -- and the invariants that would otherwise only fail
silently at scale.
"""
import numpy as np
import pytest
import torch

from rdm.sw_lmmd import ReferenceFeatureStore, WindowConfig, probe_drift, refresh_retained
from rdm.sw_lmmd.reference_store import write_reference_store
from sw_lmmd_fixtures import DIMS, D_TXT, NAMES, build_toy_store, build_trainer


@pytest.fixture(scope="module")
def store_root(tmp_path_factory):
    return build_toy_store(str(tmp_path_factory.mktemp("sw_store")), num_rows=64)


# --------------------------------------------------------------------------------------
# reference store
# --------------------------------------------------------------------------------------
def test_store_reads_rows_in_requested_order(store_root):
    store = ReferenceFeatureStore(store_root, NAMES)
    assert store.num_rows == 64
    rows = np.array([5, 1, 63, 1])
    img = store.reference_image_features("enc_a", rows)
    assert img.shape == (4, DIMS[0])
    torch.testing.assert_close(img[1], img[3])                 # same row id -> same features
    joint = store.reference_joint_features("enc_a", rows)
    assert joint.shape == (4, DIMS[0] + D_TXT)
    torch.testing.assert_close(joint[:, :DIMS[0]], img)
    torch.testing.assert_close(joint[:, DIMS[0]:],
                               store.beta("enc_a") * store.text_features(rows))


def test_store_marginal_mode_drops_the_text_block(store_root):
    store = ReferenceFeatureStore(store_root, NAMES)
    marginal = store.reference_joint_features("enc_b", np.arange(3), joint=False)
    assert marginal.shape == (3, DIMS[1])


def test_store_rejects_row_count_mismatch(tmp_path):
    root = tmp_path / "bad"
    write_reference_store(str(root), encoder_features={"enc_a": np.zeros((4, 6), np.float32)},
                          text_features=np.zeros((4, 3), np.float32),
                          bandwidths={"enc_a": {"sigma": 1.0, "beta": 1.0}})
    (root / "metadata.json").write_text('{"num_rows": 5, "encoder_feature_dims": {"enc_a": 6}}')
    with pytest.raises(ValueError, match="rows"):
        ReferenceFeatureStore(str(root), ["enc_a"], require_context=False)


def test_store_requires_a_fixed_bandwidth(tmp_path):
    """SW-LMMD never re-estimates sigma per window; a missing one must fail loudly."""
    root = tmp_path / "nobw"
    write_reference_store(str(root), encoder_features={"enc_a": np.zeros((4, 6), np.float32)},
                          text_features=np.zeros((4, 3), np.float32), bandwidths={})
    with pytest.raises(ValueError, match="bandwidth"):
        ReferenceFeatureStore(str(root), ["enc_a"], require_context=False)


def test_store_supports_sharded_context(tmp_path):
    """The real context pool is 61 GB, so it may be a shard directory rather than one .npy."""
    rows, ctx_dim = 12, 5
    ctx = np.arange(rows * ctx_dim, dtype=np.float32).reshape(rows, ctx_dim)
    root = tmp_path / "sharded"
    write_reference_store(str(root), encoder_features={"e": np.zeros((rows, 2), np.float32)},
                          text_features=np.zeros((rows, 3), np.float32), context=ctx,
                          bandwidths={"e": {"sigma": 1.0, "beta": 1.0}}, context_shards=4)
    store = ReferenceFeatureStore(str(root), ["e"])
    picked = np.array([11, 0, 7, 4])
    torch.testing.assert_close(store.generator_context(picked),
                               torch.from_numpy(ctx[picked]).float())


# --------------------------------------------------------------------------------------
# trainer control flow
# --------------------------------------------------------------------------------------
def test_bootstrap_fills_only_the_retained_half(store_root):
    trainer = build_trainer(store_root, window=16, stride=4)
    logs = trainer.bootstrap()
    assert logs["bootstrap_rows"] == 12
    for name in NAMES:
        entry = trainer.cache.entries[name]
        assert entry.features.shape[0] == 12
        assert int(entry.generated_at_steps.max()) == -1


def test_step_requires_bootstrap(store_root):
    with pytest.raises(RuntimeError, match="bootstrap"):
        build_trainer(store_root).step()


def test_step_updates_parameters_and_slides_the_window(store_root):
    trainer = build_trainer(store_root, window=16, stride=4, lr=1e-2)
    trainer.bootstrap()
    before = [p.detach().clone() for p in trainer._gen_params]

    logs = trainer.step()

    assert logs["step"] == 1 and logs["active_rows"] == 4
    assert not logs["skipped"] and np.isfinite(logs["force"])
    assert any(not torch.equal(b, p) for b, p in zip(before, trainer._gen_params))
    for name in NAMES:
        entry = trainer.cache.entries[name]
        assert entry.features.shape[0] == 16                   # cache now holds a full window
        assert entry.features.shape[1] == DIMS[NAMES.index(name)] + D_TXT


def test_cache_rows_track_the_schedule(store_root):
    """Every step the cache must hold exactly the window that was just trained on."""
    trainer = build_trainer(store_root, window=16, stride=4, lr=1e-3)
    trainer.bootstrap()
    max_age_cfg = WindowConfig(size=16, stride=4).max_age_steps
    for step in range(8):
        logs = trainer.step()
        expected = trainer.schedule._window_at(step).all_ids
        assert np.array_equal(trainer.cache.entries["enc_a"].row_ids, expected)
        assert logs["cache_max_age"] <= max_age_cfg
    assert logs["cache_max_age"] == max_age_cfg                # steady state reached


def test_micro_batching_does_not_change_the_gradient(store_root):
    """grad_accum is a memory knob; splitting the rank's rows must not move the gradient."""
    grads = []
    for micro_batch in (1, 2, 4):
        trainer = build_trainer(store_root, window=16, stride=4, micro_batch=micro_batch)
        trainer.bootstrap()
        trainer.step()                                          # lr=0 -> parameters unchanged
        grads.append([p.grad.detach().clone() for p in trainer._gen_params])
    for other in grads[1:]:
        for a, b in zip(grads[0], other):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)


def test_monitor_reports_the_three_kernel_terms(store_root):
    trainer = build_trainer(store_root, window=16, stride=4, monitor_every=1)
    trainer.bootstrap()
    logs = trainer.step()
    for name in NAMES:
        assert logs[f"{name}/mmd2"] >= 0.0
        assert {f"{name}/k_xx", f"{name}/k_xy", f"{name}/k_yy"} <= set(logs)
        assert np.isfinite(logs[f"{name}/force"])
    assert np.isfinite(logs["raw_mmd2"])


def test_force_and_mmd2_are_reported_separately(store_root):
    """The trained surrogate is not the MMD; conflating them misreads every training curve."""
    trainer = build_trainer(store_root, window=16, stride=4)
    trainer.bootstrap()
    logs = trainer.step()
    assert logs["force"] != pytest.approx(logs["raw_mmd2"])


def test_marginal_mode_drops_the_text_block(store_root):
    trainer = build_trainer(store_root, window=16, stride=4, joint=False, lr=1e-3)
    trainer.bootstrap()
    trainer.step()
    assert trainer.cache.entries["enc_a"].features.shape[1] == DIMS[0]


def test_non_finite_force_skips_the_optimizer_step(store_root, monkeypatch):
    trainer = build_trainer(store_root, window=16, stride=4, lr=1e-2)
    trainer.bootstrap()
    before = [p.detach().clone() for p in trainer._gen_params]
    monkeypatch.setattr(trainer.mmd, "active_force",
                        lambda active, *a, **k: ((active.sum() * float("nan")), {
                            "force": torch.tensor(float("nan")),
                            "active_repulsion": torch.tensor(0.0),
                            "active_attraction": torch.tensor(0.0)}))
    logs = trainer.step()
    assert logs["skipped"]
    for b, p in zip(before, trainer._gen_params):
        torch.testing.assert_close(b, p)


def test_epoch_boundary_rebootstraps_the_cache(store_root):
    """A new permutation destroys the overlap identity, so the old cache must not survive."""
    trainer = build_trainer(store_root, window=16, stride=4, lr=1e-3)
    trainer.bootstrap()
    trainer.step()
    old_rows = trainer.cache.entries["enc_a"].row_ids.copy()
    logs = trainer.start_epoch()
    assert logs["epoch"] == 1
    assert trainer.cache.entries["enc_a"].features.shape[0] == 12      # freshly bootstrapped
    assert not np.array_equal(trainer.cache.entries["enc_a"].row_ids, old_rows)


def test_train_loop_bootstraps_automatically(store_root):
    trainer = build_trainer(store_root, window=16, stride=4, lr=1e-3)
    seen = []
    assert trainer.train(3, log_fn=seen.append) == 3
    assert [s["step"] for s in seen] == [1, 2, 3]


def test_state_dict_carries_schedule_and_cache(store_root):
    trainer = build_trainer(store_root, window=16, stride=4, lr=1e-3)
    trainer.bootstrap()
    trainer.step()
    state = trainer.state_dict()
    assert state["step"] == 1
    assert state["schedule"]["step"] == 1
    assert set(state["cache"]) == set(NAMES)
    assert "order_hash" in state["schedule"]


# --------------------------------------------------------------------------------------
# staleness
# --------------------------------------------------------------------------------------
def test_drift_probe_is_zero_when_parameters_have_not_moved(store_root):
    """Per-row noise makes drift measure PARAMETER change only -- at lr=0 it must vanish."""
    trainer = build_trainer(store_root, window=16, stride=4, lr=0.0)
    trainer.bootstrap()
    trainer.step()
    window = trainer.schedule.peek()
    stats = probe_drift(trainer, window, n_rows=8)
    assert stats["drift_probe_rows"] == 8
    assert stats["drift_mean"] == pytest.approx(0.0, abs=1e-9)


def test_drift_probe_detects_moved_parameters(store_root):
    trainer = build_trainer(store_root, window=16, stride=4, lr=0.5)
    trainer.bootstrap()
    trainer.step()
    stats = probe_drift(trainer, trainer.schedule.peek(), n_rows=8)
    assert stats["drift_mean"] > 1e-6
    assert any(k.startswith("drift_age_") for k in stats)


def test_probe_does_not_mutate_cache_or_schedule(store_root):
    trainer = build_trainer(store_root, window=16, stride=4, lr=0.5)
    trainer.bootstrap()
    trainer.step()
    window = trainer.schedule.peek()
    before = trainer.cache.entries["enc_a"].features.clone()
    ages = trainer.cache.entries["enc_a"].generated_at_steps.clone()
    step_before = trainer.schedule.step
    probe_drift(trainer, window, n_rows=8)
    torch.testing.assert_close(trainer.cache.entries["enc_a"].features, before)
    torch.testing.assert_close(trainer.cache.entries["enc_a"].generated_at_steps, ages)
    assert trainer.schedule.step == step_before


def test_refresh_resets_ages_and_zeroes_drift(store_root):
    trainer = build_trainer(store_root, window=16, stride=4, lr=0.5)
    trainer.bootstrap()
    trainer.step()
    window = trainer.schedule.peek()
    stats = refresh_retained(trainer, window)
    assert stats["refreshed_rows"] == 12
    assert probe_drift(trainer, window, n_rows=8)["drift_mean"] == pytest.approx(0.0, abs=1e-9)


def test_store_prompt_id_indirection(tmp_path):
    """Several reference rows per prompt: image features are per (prompt, seed), while the
    text table and generator context stay per prompt (replicating the 61 GB ctx would not fit)."""
    n_prompts, seeds = 5, 4
    n_rows = n_prompts * seeds
    root = tmp_path / "multi_seed"
    rng = np.random.default_rng(0)
    write_reference_store(
        str(root),
        encoder_features={"e": rng.standard_normal((n_rows, 3)).astype(np.float32)},
        text_features=np.arange(n_prompts * 2, dtype=np.float32).reshape(n_prompts, 2),
        context=np.arange(n_prompts * 3, dtype=np.float32).reshape(n_prompts, 3),
        bandwidths={"e": {"sigma": 1.0, "beta": 1.0}},
        prompt_ids=np.repeat(np.arange(n_prompts), seeds))

    store = ReferenceFeatureStore(str(root), ["e"])
    assert store.num_rows == n_rows and store.num_prompts == n_prompts
    # the four rows of prompt 2 share its text/context but have distinct image features
    rows = np.array([8, 9, 10, 11])
    assert np.array_equal(store.prompt_rows(rows), [2, 2, 2, 2])
    txt = store.text_features(rows)
    assert txt.shape == (4, 2) and torch.equal(txt[0], txt[3])
    assert torch.equal(store.generator_context(rows)[0], store.generator_context(rows)[2])
    img = store.reference_image_features("e", rows)
    assert not torch.equal(img[0], img[1])


def test_store_rejects_out_of_range_prompt_ids(tmp_path):
    root = tmp_path / "bad_ids"
    write_reference_store(str(root), encoder_features={"e": np.zeros((4, 3), np.float32)},
                          text_features=np.zeros((2, 2), np.float32),
                          bandwidths={"e": {"sigma": 1.0, "beta": 1.0}},
                          prompt_ids=np.array([0, 1, 2, 3]))
    with pytest.raises(ValueError, match="prompt_ids indexes row"):
        ReferenceFeatureStore(str(root), ["e"], require_context=False)
