"""Prompt-grouped windows (``window.group_by_prompt``): ordering, schedule invariants, monitor.

The property everything else rests on: every window -- across slides, the cyclic wrap, epoch
reshuffles and resume -- is made of WHOLE prompts, and each step's active block holds every
seed of the prompts it introduces.
"""
import numpy as np
import pytest
import torch

from rdm.compare.kernels import gamma_from_sigma
from rdm.sw_lmmd import ReferenceFeatureStore, SlidingWindowSchedule, build_grouped_row_order
from rdm.sw_lmmd.local_mmd import within_group_kernel_mean
from rdm.sw_lmmd.reference_store import write_reference_store
from sw_lmmd_fixtures import CTX_DIM, D_TXT, DIMS, NAMES, build_trainer

G = 4


def _prompt_ids(num_prompts, group=G, seed=0):
    """Rows of one prompt deliberately scattered, as a row-level store may lay them out."""
    pid = np.repeat(np.arange(num_prompts), group)
    return pid[np.random.default_rng(seed).permutation(pid.size)]


def _assert_whole_groups(ids, pid, group=G):
    blocks = pid[np.asarray(ids)].reshape(-1, group)
    assert (blocks == blocks[:, :1]).all(), "a window split a prompt's rows"


# --------------------------------------------------------------------------------------
# ordering
# --------------------------------------------------------------------------------------
def test_grouped_order_is_a_permutation_of_whole_prompts():
    pid = _prompt_ids(50)
    order, group = build_grouped_row_order(pid, seed=3407)
    assert group == G
    assert np.array_equal(np.sort(order), np.arange(pid.size))
    _assert_whole_groups(order, pid)
    again, _ = build_grouped_row_order(pid, seed=3407)
    other, _ = build_grouped_row_order(pid, seed=1)
    assert np.array_equal(order, again) and not np.array_equal(order, other)


def test_grouped_order_rejects_ragged_prompts():
    pid = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    with pytest.raises(ValueError, match="same number of rows per prompt"):
        build_grouped_row_order(pid)


@pytest.mark.parametrize("kw", [dict(window_size=10, stride=4), dict(window_size=12, stride=6),
                                dict(window_size=12, stride=4, start_offset=2)])
def test_grouped_schedule_rejects_unaligned_geometry(kw):
    order, _ = build_grouped_row_order(_prompt_ids(20))
    with pytest.raises(ValueError, match="multiple of the group size"):
        SlidingWindowSchedule(order, group_size=G, **kw)


# --------------------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------------------
def test_every_window_holds_whole_prompts_across_slides_wrap_and_epochs():
    pid = _prompt_ids(10)                                   # 40 rows: windows wrap quickly
    order, _ = build_grouped_row_order(pid)
    sched = SlidingWindowSchedule(order, window_size=16, stride=8, group_size=G)
    for epoch in range(4):
        for _ in range(12):                                 # > 2 laps of 40 rows
            w = sched.next()
            _assert_whole_groups(w.all_ids, pid)
            _assert_whole_groups(w.active_ids, pid)
            _assert_whole_groups(w.retained_ids, pid)
        sched.new_epoch(seed=epoch, random_start=True, reverse_probability=0.5)
        assert sched.start_offset % G == 0


def test_grouped_schedule_covers_every_prompt_once_per_lap():
    pid = _prompt_ids(16)
    order, _ = build_grouped_row_order(pid)
    sched = SlidingWindowSchedule(order, window_size=16, stride=8, group_size=G)
    active = np.concatenate([sched.next().active_ids for _ in range(pid.size // 8)])
    assert np.array_equal(np.sort(active), np.arange(pid.size))


def test_grouping_is_recorded_and_guarded_on_resume():
    pid = _prompt_ids(20)
    order, _ = build_grouped_row_order(pid)
    sched = SlidingWindowSchedule(order, window_size=16, stride=8, group_size=G)
    for _ in range(3):
        sched.next()
    state = sched.state_dict()
    assert state["group_size"] == G

    twin = SlidingWindowSchedule(order, window_size=16, stride=8, group_size=G)
    twin.load_state_dict(state)
    assert np.array_equal(twin.next().all_ids, sched.next().all_ids)

    ungrouped = SlidingWindowSchedule(order, window_size=16, stride=8)
    with pytest.raises(RuntimeError, match="grouping mismatch"):
        ungrouped.load_state_dict(state)


def test_default_schedule_is_unchanged():
    """group_size defaults to 1: old checkpoints (no group_size key) still resume."""
    order = np.random.default_rng(0).permutation(40)
    sched = SlidingWindowSchedule(order, window_size=16, stride=4)
    assert sched.group_size == 1
    state = sched.state_dict()
    state.pop("group_size")
    SlidingWindowSchedule(order, window_size=16, stride=4).load_state_dict(state)


# --------------------------------------------------------------------------------------
# monitor
# --------------------------------------------------------------------------------------
def test_within_group_kernel_mean_matches_brute_force():
    x = torch.randn(12, 5, dtype=torch.float64)
    sigma = 1.7
    gamma = gamma_from_sigma(sigma)
    vals = [torch.exp(-gamma * (x[b + i] - x[b + j]).square().sum())
            for b in range(0, 12, G) for i in range(G) for j in range(G) if i != j]
    torch.testing.assert_close(within_group_kernel_mean(x, G, sigma), torch.stack(vals).mean())
    collapsed = x[::G].repeat_interleave(G, 0)              # every seed identical
    assert float(within_group_kernel_mean(collapsed, G, sigma)) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# trainer, end to end on a grouped toy store
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def grouped_store(tmp_path_factory):
    num_prompts = 24
    rng = np.random.default_rng(5)
    pid = _prompt_ids(num_prompts, seed=5)
    text = rng.standard_normal((num_prompts, D_TXT)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    root = str(tmp_path_factory.mktemp("grouped_store"))
    return write_reference_store(
        root, prompt_ids=pid, text_features=text, row_order=np.arange(pid.size),
        context=rng.standard_normal((num_prompts, CTX_DIM)).astype(np.float32),
        encoder_features={n: rng.standard_normal((pid.size, d)).astype(np.float32)
                          for n, d in zip(NAMES, DIMS)},
        bandwidths={n: {"sigma": 2.0, "beta": 1.0} for n in NAMES})


def test_grouped_trainer_steps_and_logs_same_prompt_similarity(grouped_store):
    store = ReferenceFeatureStore(grouped_store, NAMES)
    order, group = build_grouped_row_order(store.prompt_ids)
    trainer = build_trainer(grouped_store, window=16, stride=8, micro_batch=4, lr=1e-2)
    trainer.schedule = SlidingWindowSchedule(order, window_size=16, stride=8, group_size=group)
    trainer.bootstrap()
    before = [p.detach().clone() for p in trainer.generator.model.parameters()]
    for _ in range(3):
        logs = trainer.step()
        _assert_whole_groups(trainer.cache.entries[NAMES[0]].row_ids, store.prompt_ids)
    for name in NAMES:
        assert 0.0 < logs[f"{name}/group_k_gen"] <= 1.0
        assert 0.0 < logs[f"{name}/group_k_ref"] <= 1.0
    assert any(not torch.equal(a, b) for a, b in
               zip(before, trainer.generator.model.parameters()))


def test_ungrouped_trainer_logs_no_group_keys(grouped_store):
    trainer = build_trainer(grouped_store, window=16, stride=8)
    trainer.bootstrap()
    logs = trainer.step()
    assert not any(k.endswith("group_k_gen") for k in logs)


def test_grouped_config_differs_from_the_gan_run_only_in_grouping_and_micro_batch():
    import dataclasses

    from rdm.sw_lmmd.launch import (gan_from_config, memory_from_config, resolve_batching,
                                    window_from_config)
    from rdm.train.launch import load_config

    base = load_config("configs/sw_lmmd_train_h100_2gpu_gan.yaml")
    cfg = load_config("configs/sw_lmmd_train_h100_2gpu_gan_grouped.yaml")
    w = window_from_config(cfg)
    assert w.group_by_prompt and not window_from_config(base).group_by_prompt
    assert dataclasses.replace(w, group_by_prompt=False) == window_from_config(base)
    assert w.size % G == 0 and w.stride % G == 0
    policy = memory_from_config(cfg)
    assert dataclasses.replace(policy, micro_batch=8) == memory_from_config(base)
    assert gan_from_config(cfg) == gan_from_config(base)
    for key in ("encoders", "lr", "grad_clip", "steps", "joint", "loss", "cache", "lr_schedule"):
        assert getattr(cfg, key) == getattr(base, key), key
    assert cfg.exp_name != base.exp_name
    assert resolve_batching(cfg, policy, w, world_size=2)["grad_accum"] == 4


def test_k1024_config_differs_from_the_k128_geneval_run_only_in_the_window():
    import dataclasses

    from rdm.sw_lmmd.launch import (gan_from_config, memory_from_config, resolve_batching,
                                    window_from_config)
    from rdm.train.launch import load_config

    base = load_config("configs/sw_lmmd_train_h100_2gpu_gan_grouped_geneval.yaml")
    cfg = load_config("configs/sw_lmmd_train_h100_gan_grouped_geneval_k1024b128.yaml")
    w = window_from_config(cfg)
    assert (w.size, w.stride, w.group_by_prompt, w.group_pattern) == (1024, 128, True, "G")
    assert dataclasses.replace(w, size=128, stride=32) == window_from_config(base)
    assert cfg.active_global_batch == 128
    assert memory_from_config(cfg) == memory_from_config(base)
    assert gan_from_config(cfg) == gan_from_config(base)
    for key in ("encoders", "lr", "lr_schedule", "steps", "save_freq", "joint", "loss", "cache",
                "reference_extension", "grad_clip"):
        assert getattr(cfg, key) == getattr(base, key), key
    assert cfg.exp_name != base.exp_name
    policy = memory_from_config(cfg)
    assert {n: resolve_batching(cfg, policy, w, world_size=n)["grad_accum"]
            for n in (2, 4, 8)} == {2: 16, 4: 8, 8: 4}
