"""Unbiased sibling repulsion (``window.unbiased_siblings``) and smaller groups (``window.group_size``).

What has to hold:

* the weighted force is exactly the full-window force with each same-group repulsion term
  multiplied by ``G/(G-1)`` -- nothing else changes, and weight 1 is the old force bit for bit;
* it removes the biased force's pull toward narrower seed distributions (Monte Carlo below);
* split groups are random sub-groups of one prompt, every row used once per lap, and windows
  never split a sub-group;
* the trainer applies the weight on grouped steps only, logs it, and refuses to resume across
  a change of the setting.
"""
import dataclasses

import numpy as np
import pytest
import torch

from rdm.compare.kernels import gamma_from_sigma
from rdm.sw_lmmd import (MixedGroupSchedule, ReferenceFeatureStore, SlidingWindowSchedule,
                         WindowConfig, build_grouped_row_order, split_groups)
from rdm.sw_lmmd.local_mmd import ExactLocalMMD, sibling_kernel_sum
from rdm.sw_lmmd.reference_store import write_reference_store
from sw_lmmd_fixtures import CTX_DIM, D_TXT, DIMS, NAMES, build_trainer


def _prompt_ids(num_prompts, group=4, seed=0):
    pid = np.repeat(np.arange(num_prompts), group)
    return pid[np.random.default_rng(seed).permutation(pid.size)]


# --------------------------------------------------------------------------------------
# the force
# --------------------------------------------------------------------------------------
def _explicit_force(a, context, reference, sigma, group, weight, offset):
    """``(2/(B K)) sum_i sum_j [W_ij k(a_i, sg(c_j)) - k(a_i, y_j)]`` with an explicit weight
    matrix: ``weight`` on the same-group columns of ``a_i`` (self excluded), 1 elsewhere."""
    gamma = gamma_from_sigma(sigma)
    k_xc = torch.exp(-gamma * ((a[:, None] - context.detach()[None]) ** 2).sum(-1))
    k_xy = torch.exp(-gamma * ((a[:, None] - reference[None]) ** 2).sum(-1))
    w = torch.ones_like(k_xc)
    for i in range(a.shape[0]):
        p = offset + i                                  # position of a_i in the context
        lo = offset + (i // group) * group
        for j in range(lo, lo + group):
            if j != p:
                w[i, j] = weight
    return 2.0 * ((w * k_xc).sum() - k_xy.sum()) / (a.shape[0] * context.shape[0])


@pytest.mark.parametrize("group", [2, 4])
def test_weighted_force_matches_the_explicit_weighted_sum(group):
    g = torch.Generator().manual_seed(group)
    B, K, d = 4 * group, 6 * group, 5
    a = torch.randn(B, d, generator=g, dtype=torch.float64, requires_grad=True)
    retained = torch.randn(K - B, d, generator=g, dtype=torch.float64)
    context = torch.cat([retained, a.detach()])
    reference = torch.randn(K, d, generator=g, dtype=torch.float64)
    weight = group / (group - 1)

    force, stats = ExactLocalMMD(block_size=5).active_force(a, context, reference, 1.3,
                                                            group=group, sibling_weight=weight)
    (grad,) = torch.autograd.grad(force, a)
    a2 = a.detach().clone().requires_grad_(True)
    expected = _explicit_force(a2, context, reference, 1.3, group, weight, offset=K - B)
    (grad_expected,) = torch.autograd.grad(expected, a2)
    torch.testing.assert_close(force, expected)
    torch.testing.assert_close(grad, grad_expected)
    assert float(stats["sibling_repulsion"]) > 0


def test_weight_one_is_the_old_force_exactly():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(8, 3, generator=g, dtype=torch.float64, requires_grad=True)
    context = torch.cat([torch.randn(8, 3, generator=g, dtype=torch.float64), a.detach()])
    reference = torch.randn(16, 3, generator=g, dtype=torch.float64)
    mmd = ExactLocalMMD()
    old, old_stats = mmd.active_force(a, context, reference, 1.0)
    new, new_stats = mmd.active_force(a, context, reference, 1.0, group=4, sibling_weight=1.0)
    assert torch.equal(old, new) and "sibling_repulsion" not in new_stats
    assert torch.equal(torch.autograd.grad(old, a)[0], torch.autograd.grad(new, a)[0])


def test_sibling_kernel_sum_needs_whole_groups():
    with pytest.raises(ValueError, match="divisible"):
        sibling_kernel_sum(torch.zeros(6, 2), 4, 1.0)
    with pytest.raises(ValueError, match=">= 2 rows"):
        sibling_kernel_sum(torch.zeros(6, 2), 1, 1.0)


@pytest.mark.parametrize("group", [2, 4])
def test_biased_force_narrows_the_seeds_and_the_unbiased_one_does_not(group):
    """Per-prompt toy: generated seeds ``s * z`` vs references ``w``, z and w both N(0, I), each
    prompt far from the others in the text block. At ``s = 1`` the two distributions are equal,
    so a consistent objective has zero expected gradient in ``s``. The biased force's is
    positive (it lowers the loss by shrinking ``s``, i.e. collapsing the seeds); the unbiased
    one's is zero up to Monte Carlo noise."""
    n_prompts, d = 600, 2
    mmd = ExactLocalMMD(block_size=4096)
    slopes = {}
    for weight in (1.0, group / (group - 1)):
        vals = []
        for rep in range(2):
            g = torch.Generator().manual_seed(rep)
            z = torch.randn(n_prompts * group, d, generator=g, dtype=torch.float64)
            w = torch.randn(n_prompts * group, d, generator=g, dtype=torch.float64)
            text = 50.0 * torch.repeat_interleave(torch.arange(n_prompts, dtype=torch.float64),
                                                  group)[:, None]
            s = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
            x = torch.cat([s * z, text], 1)
            force, _ = mmd.active_force(x, x.detach(), torch.cat([w, text], 1), 1.0,
                                        group=group, sibling_weight=weight)
            (ds,) = torch.autograd.grad(force, s)
            vals.append(float(ds) * x.shape[0])          # per-row slope
        slopes[weight] = float(np.mean(vals))
    assert slopes[1.0] > 0.3                             # measured ~0.45 for G = 2 and 4
    assert abs(slopes[group / (group - 1)]) < 0.1


# --------------------------------------------------------------------------------------
# split groups
# --------------------------------------------------------------------------------------
def test_split_groups_cuts_each_prompt_into_random_sub_groups():
    pid = _prompt_ids(200)
    sub = split_groups(pid, 2, seed=3407)
    counts = np.bincount(sub)
    assert counts.size == 400 and (counts == 2).all()
    for s in range(400):                                  # a sub-group never mixes two prompts
        assert np.unique(pid[sub == s]).size == 1
    assert np.array_equal(sub, split_groups(pid, 2, seed=3407))
    assert not np.array_equal(sub, split_groups(pid, 2, seed=1))
    # the pairing is drawn per prompt: the rows of a prompt are not always split the same way
    rank = np.empty_like(pid)
    for p in range(200):
        rows = np.flatnonzero(pid == p)
        rank[rows] = np.arange(4)
    partners = {tuple(sorted(rank[sub == s])) for s in range(400)}
    assert len(partners) == 6                             # all C(4,2) pairings occur


def test_split_groups_full_size_is_the_same_partition():
    pid = _prompt_ids(30)
    sub = split_groups(pid, 4)
    for p in range(30):
        assert np.unique(sub[pid == p]).size == 1


@pytest.mark.parametrize("size", [1, 3, 8])
def test_split_groups_rejects_sizes_that_do_not_divide(size):
    with pytest.raises(ValueError, match="divide"):
        split_groups(_prompt_ids(10), size)


def test_split_schedule_keeps_pairs_whole_and_covers_every_row_once_per_lap():
    pid = _prompt_ids(16)                                 # 64 rows -> 32 pairs
    sub = split_groups(pid, 2)
    order, group = build_grouped_row_order(sub)
    assert group == 2
    sched = SlidingWindowSchedule(order, window_size=16, stride=8, group_size=2)
    active = []
    for _ in range(pid.size // 8):
        w = sched.next()
        for ids in (w.all_ids, w.active_ids, w.retained_ids):
            blocks = sub[ids].reshape(-1, 2)
            assert (blocks == blocks[:, :1]).all()
        active.append(w.active_ids)
    assert np.array_equal(np.sort(np.concatenate(active)), np.arange(pid.size))


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw, match", [
    (dict(unbiased_siblings=True), "group_by_prompt"),
    (dict(group_size=2), "group_by_prompt"),
    (dict(group_by_prompt=True, group_size=1), "group_size"),
    (dict(group_by_prompt=True, group_size=-2), "group_size"),
    (dict(sibling_weight=1.5), "group_by_prompt"),
    (dict(group_by_prompt=True, sibling_weight=0.0), "positive"),
    (dict(group_by_prompt=True, sibling_weight=1.5, unbiased_siblings=True), "not both"),
])
def test_window_config_rejects_inconsistent_settings(kw, match):
    with pytest.raises(ValueError, match=match):
        WindowConfig(**kw).validate()


def test_defaults_are_off():
    w = WindowConfig()
    assert w.group_size == 0 and w.unbiased_siblings is False and w.sibling_weight == 1.0


# --------------------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def grouped_store(tmp_path_factory):
    num_prompts = 48                                      # enough for a "GU" schedule at K=16
    rng = np.random.default_rng(5)
    pid = _prompt_ids(num_prompts, seed=5)
    text = rng.standard_normal((num_prompts, D_TXT)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    return write_reference_store(
        str(tmp_path_factory.mktemp("unbiased_store")), prompt_ids=pid, text_features=text,
        row_order=np.arange(pid.size),
        context=rng.standard_normal((num_prompts, CTX_DIM)).astype(np.float32),
        encoder_features={n: rng.standard_normal((pid.size, d)).astype(np.float32)
                          for n, d in zip(NAMES, DIMS)},
        bandwidths={n: {"sigma": 2.0, "beta": 1.0} for n in NAMES})


def _grouped_trainer(store_root, *, size=0, unbiased=True, lr=0.0):
    store = ReferenceFeatureStore(store_root, NAMES)
    gids = store.grouping_ids() if not size else split_groups(store.grouping_ids(), size)
    order, group = build_grouped_row_order(gids)
    trainer = build_trainer(store_root, window=16, stride=8, micro_batch=4, lr=lr)
    trainer.schedule = SlidingWindowSchedule(order, window_size=16, stride=8, group_size=group)
    trainer.unbiased_siblings = unbiased
    return trainer


@pytest.mark.parametrize("size, weight", [(0, 4 / 3), (2, 2.0)])
def test_trainer_applies_and_logs_the_sibling_weight(grouped_store, size, weight):
    trainer = _grouped_trainer(grouped_store, size=size, lr=1e-2)
    trainer.bootstrap()
    before = [p.detach().clone() for p in trainer.generator.model.parameters()]
    for _ in range(3):
        logs = trainer.step()
        assert logs["sibling_weight"] == pytest.approx(weight)
        assert np.isfinite(logs["force"]) and not logs["skipped"]
        for name in NAMES:
            assert logs[f"{name}/sibling_repulsion"] > 0
            assert f"{name}/group_k_gen" in logs
    assert any(not torch.equal(a, b) for a, b in
               zip(before, trainer.generator.model.parameters()))


def test_unbiased_trainer_adds_exactly_the_weighted_sibling_force(grouped_store):
    """Same parameters, same window: the two trainers' forces differ by
    ``(w - 1) * 2 * sibling_repulsion`` per encoder, weighted like the force."""
    logs = {}
    for unbiased in (False, True):
        trainer = _grouped_trainer(grouped_store, unbiased=unbiased)
        trainer.bootstrap()
        logs[unbiased] = trainer.step()
    for name in NAMES:
        delta = logs[True][f"{name}/repulsion"] - logs[False][f"{name}/repulsion"]
        assert delta == pytest.approx((4 / 3 - 1) * logs[True][f"{name}/sibling_repulsion"])
        assert logs[True][f"{name}/attraction"] == pytest.approx(logs[False][f"{name}/attraction"])
    assert "sibling_weight" not in logs[False]


def test_mixed_schedule_weighs_grouped_steps_only(grouped_store):
    store = ReferenceFeatureStore(grouped_store, NAMES)
    trainer = build_trainer(grouped_store, window=16, stride=8, micro_batch=4)
    trainer.schedule = MixedGroupSchedule(store.grouping_ids(), "GU", window_size=16, stride=8)
    trainer.unbiased_siblings = True
    trainer.bootstrap()
    for _ in range(4):
        logs = trainer.step()
        grouped = bool(logs["grouped_step"])
        assert logs["sibling_weight"] == (4 / 3 if grouped else 1.0)
        assert (f"{NAMES[0]}/sibling_repulsion" in logs) == grouped


def test_trainer_refuses_unbiased_without_groups_and_a_changed_setting_on_resume(grouped_store):
    from rdm.sw_lmmd import SWLMMDTrainer

    plain = build_trainer(grouped_store, window=16, stride=8)
    with pytest.raises(ValueError, match="prompt-grouped"):
        SWLMMDTrainer(plain.generator, plain.battery, plain.store, plain.schedule,
                      plain.optimizer, encoder_names=list(NAMES), noise_shape=plain.noise_shape,
                      device="cpu", unbiased_siblings=True)

    biased = _grouped_trainer(grouped_store, unbiased=False)
    biased.bootstrap()
    biased.step()
    state = biased.state_dict()
    assert "unbiased_siblings" not in state                 # default files are unchanged
    unbiased = _grouped_trainer(grouped_store, unbiased=True)
    with pytest.raises(RuntimeError, match="unbiased_siblings mismatch"):
        unbiased.load_state_dict(state)
    biased_again = _grouped_trainer(grouped_store, unbiased=False)
    assert biased_again.load_state_dict(state)["step"] == 1
    unbiased.bootstrap()
    unbiased.step()
    state = unbiased.state_dict()
    assert state["unbiased_siblings"] is True
    with pytest.raises(RuntimeError, match="unbiased_siblings mismatch"):
        _grouped_trainer(grouped_store, unbiased=False).load_state_dict(state)
    assert _grouped_trainer(grouped_store, unbiased=True).load_state_dict(state)["step"] == 1


def test_grouped_trainer_validates_geometry_at_construction(grouped_store):
    from rdm.sw_lmmd import SWLMMDTrainer

    t = _grouped_trainer(grouped_store, unbiased=False)
    built = SWLMMDTrainer(t.generator, t.battery, t.store, t.schedule, t.optimizer,
                          encoder_names=list(NAMES), noise_shape=t.noise_shape, device="cpu",
                          unbiased_siblings=True)
    assert built.unbiased_siblings


# --------------------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------------------
def _load(path):
    from rdm.train.launch import load_config
    return load_config(path)


def test_experiment_configs_change_one_thing_each_and_hold_no_geneval_block():
    from rdm.sw_lmmd.launch import gan_from_config, memory_from_config, window_from_config

    exp1 = _load("configs/sw_lmmd_train_h100_gan_grouped_geneval_k1024b128.yaml")
    exp5 = _load("configs/sw_lmmd_train_h100_gan_grouped_k1024b128.yaml")
    exp4 = _load("configs/sw_lmmd_train_h100_gan_grouped_unbiased_k1024b128.yaml")
    exp3 = _load("configs/sw_lmmd_train_h100_gan_grouped_g2_unbiased_k1024b128.yaml")
    w1, w5, w4, w3 = (window_from_config(c) for c in (exp1, exp5, exp4, exp3))

    assert w5 == w1                                               # 5 = 1 minus the GenEval block
    assert dataclasses.replace(w4, unbiased_siblings=False) == w5
    assert dataclasses.replace(w3, group_size=0) == w4
    assert (w3.group_size, w3.unbiased_siblings, w3.size, w3.stride) == (2, True, 1024, 128)
    for cfg in (exp5, exp4, exp3):
        assert getattr(cfg, "reference_extension", None) in (None, "")
        assert cfg.active_global_batch == 128 and cfg.resume_every == 100
        assert memory_from_config(cfg) == memory_from_config(exp1)
        assert gan_from_config(cfg) == gan_from_config(exp1)
        for key in ("encoders", "lr", "lr_schedule", "steps", "save_freq", "joint", "loss",
                    "cache", "grad_clip", "reference_root"):
            assert getattr(cfg, key) == getattr(exp1, key), key
    assert len({c.exp_name for c in (exp1, exp5, exp4, exp3)}) == 4


def test_g2_biased_config_differs_from_the_k128_grouped_run_only_in_the_group_size():
    from rdm.sw_lmmd.launch import (gan_from_config, memory_from_config, resolve_batching,
                                    window_from_config)

    base = _load("configs/sw_lmmd_train_h100_2gpu_gan_grouped.yaml")
    cfg = _load("configs/sw_lmmd_train_h100_2gpu_gan_grouped_g2.yaml")
    w = window_from_config(cfg)
    assert (w.size, w.stride, w.group_size, w.unbiased_siblings) == (128, 32, 2, False)
    assert dataclasses.replace(w, group_size=0) == window_from_config(base)
    assert getattr(cfg, "reference_extension", None) in (None, "")
    assert memory_from_config(cfg) == memory_from_config(base)
    assert gan_from_config(cfg) == gan_from_config(base)
    for key in ("encoders", "lr", "lr_schedule", "steps", "save_freq", "joint", "loss", "cache",
                "grad_clip", "reference_root"):
        assert getattr(cfg, key) == getattr(base, key), key
    assert cfg.exp_name != base.exp_name
    assert resolve_batching(cfg, memory_from_config(cfg), w, world_size=2)["grad_accum"] == 4


def test_g2_biased_trainer_trains_pairs_without_a_sibling_weight(grouped_store):
    trainer = _grouped_trainer(grouped_store, size=2, unbiased=False, lr=1e-2)
    assert trainer.schedule.group_size == 2
    trainer.bootstrap()
    for _ in range(3):
        logs = trainer.step()
        assert np.isfinite(logs["force"]) and not logs["skipped"]
        assert "sibling_weight" not in logs and f"{NAMES[0]}/sibling_repulsion" not in logs
        assert 0.0 < logs[f"{NAMES[0]}/group_k_gen"] <= 1.0      # pairs monitored as groups


# --------------------------------------------------------------------------------------
# sibling_weight: a free weight on the same-prompt repulsion
# --------------------------------------------------------------------------------------
def test_sibling_weight_scales_only_the_sibling_part(grouped_store):
    """Same parameters, same window: weight w adds (w - 1) x the unweighted sibling repulsion."""
    logs = {}
    for w in (1.0, 1.5, 0.5):
        trainer = _grouped_trainer(grouped_store, size=2, unbiased=False)
        trainer.sibling_weight = w
        trainer.bootstrap()
        logs[w] = trainer.step()
    for w in (1.5, 0.5):
        assert logs[w]["sibling_weight"] == w
        for name in NAMES:
            delta = logs[w][f"{name}/repulsion"] - logs[1.0][f"{name}/repulsion"]
            assert delta == pytest.approx((w - 1.0) * logs[w][f"{name}/sibling_repulsion"])
            assert logs[w][f"{name}/attraction"] == pytest.approx(logs[1.0][f"{name}/attraction"])
    assert "sibling_weight" not in logs[1.0]


def test_g2_weight_1p5_has_the_g4_balance_in_the_toy():
    """Weight 1.5 at G = 2 restores the G = 4 biased balance (repulsion 3/4 of the attraction):
    in the per-prompt toy the shrink slope per unit of attraction matches G = 4 at weight 1."""
    n_rows, d = 2400, 2
    mmd = ExactLocalMMD(block_size=4096)

    def slope(group, weight):
        vals = []
        for rep in range(2):
            g = torch.Generator().manual_seed(rep)
            z = torch.randn(n_rows, d, generator=g, dtype=torch.float64)
            w = torch.randn(n_rows, d, generator=g, dtype=torch.float64)
            text = 50.0 * torch.repeat_interleave(torch.arange(n_rows // group,
                                                               dtype=torch.float64), group)[:, None]
            s = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
            x = torch.cat([s * z, text], 1)
            force, _ = mmd.active_force(x, x.detach(), torch.cat([w, text], 1), 1.0,
                                        group=group, sibling_weight=weight)
            vals.append(float(torch.autograd.grad(force, s)[0]) * n_rows / group)  # per attraction term
        return float(np.mean(vals))

    g4, g2_w15, g2_w1 = slope(4, 1.0), slope(2, 1.5), slope(2, 1.0)
    assert g2_w15 == pytest.approx(g4, rel=0.35)
    assert g2_w1 > 1.5 * g4                       # plain G = 2 shrinks twice as hard


def test_trainer_refuses_a_changed_sibling_weight_on_resume(grouped_store):
    trainer = _grouped_trainer(grouped_store, size=2, unbiased=False)
    trainer.sibling_weight = 1.5
    trainer.bootstrap()
    trainer.step()
    state = trainer.state_dict()
    assert state["sibling_weight"] == 1.5
    plain = _grouped_trainer(grouped_store, size=2, unbiased=False)
    with pytest.raises(RuntimeError, match="sibling_weight mismatch"):
        plain.load_state_dict(state)
    again = _grouped_trainer(grouped_store, size=2, unbiased=False)
    again.sibling_weight = 1.5
    assert again.load_state_dict(state)["step"] == 1
    assert "sibling_weight" not in plain.state_dict()                 # default files unchanged


def test_trainer_rejects_both_sibling_settings(grouped_store):
    from rdm.sw_lmmd import SWLMMDTrainer

    t = _grouped_trainer(grouped_store, unbiased=False)
    with pytest.raises(ValueError, match="not both"):
        SWLMMDTrainer(t.generator, t.battery, t.store, t.schedule, t.optimizer,
                      encoder_names=list(NAMES), noise_shape=t.noise_shape, device="cpu",
                      unbiased_siblings=True, sibling_weight=1.5)


def test_g2_w15_config_differs_from_experiment_6_only_in_the_weight():
    from rdm.sw_lmmd.launch import gan_from_config, memory_from_config, window_from_config

    exp6 = _load("configs/sw_lmmd_train_h100_2gpu_gan_grouped_g2.yaml")
    exp7 = _load("configs/sw_lmmd_train_h100_2gpu_gan_grouped_g2_w15.yaml")
    w6, w7 = window_from_config(exp6), window_from_config(exp7)
    assert (w7.group_size, w7.sibling_weight, w7.unbiased_siblings) == (2, 1.5, False)
    assert dataclasses.replace(w7, sibling_weight=1.0) == w6
    assert getattr(exp7, "reference_extension", None) in (None, "")
    assert memory_from_config(exp7) == memory_from_config(exp6)
    assert gan_from_config(exp7) == gan_from_config(exp6)
    for key in ("encoders", "lr", "lr_schedule", "steps", "save_freq", "joint", "loss", "cache",
                "grad_clip", "reference_root"):
        assert getattr(exp7, key) == getattr(exp6, key), key
    assert exp7.exp_name != exp6.exp_name


def test_s3000_config_is_experiment_5_with_3000_steps():
    from rdm.sw_lmmd.launch import (gan_from_config, lr_schedule_from_config, memory_from_config,
                                    resolve_batching, window_from_config)

    exp5 = _load("configs/sw_lmmd_train_h100_gan_grouped_k1024b128.yaml")
    exp8 = _load("configs/sw_lmmd_train_h100_gan_grouped_k1024b128_s3000.yaml")
    assert exp8.steps == 3000 and exp5.steps == 2000
    assert window_from_config(exp8) == window_from_config(exp5)
    w = window_from_config(exp8)
    assert (w.size, w.stride, w.group_by_prompt, w.group_size, w.unbiased_siblings,
            w.sibling_weight) == (1024, 128, True, 0, False, 1.0)
    assert getattr(exp8, "reference_extension", None) in (None, "")
    assert memory_from_config(exp8) == memory_from_config(exp5)
    assert gan_from_config(exp8) == gan_from_config(exp5) and gan_from_config(exp8).enabled
    for key in ("encoders", "lr", "save_freq", "joint", "loss", "cache", "grad_clip",
                "reference_root", "active_global_batch", "resume_every"):
        assert getattr(exp8, key) == getattr(exp5, key), key
    assert exp8.exp_name != exp5.exp_name
    sched = lr_schedule_from_config(exp8)                 # cosine over all 3000 steps
    assert sched.name == "cosine" and sched.total_steps == 3000
    assert sched.factor(2000) > 0.3 and sched.factor(2999) == pytest.approx(0.1, abs=1e-3)
    assert resolve_batching(exp8, memory_from_config(exp8), w, world_size=2)["grad_accum"] == 16
