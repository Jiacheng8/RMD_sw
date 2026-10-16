"""Mixed grouped/ungrouped schedules (``window.group_pattern``, e.g. "GU").

Contracts: every step's active block is one whole segment of the pattern's kind -- grouped
blocks are whole prompt groups, ungrouped blocks never hold two rows of one prompt -- and the
pattern holds across slides, the cyclic wrap, reverse, epoch re-splits and resume. Pattern "G"
is exactly the plain grouped schedule.
"""
import numpy as np
import pytest
import torch

from rdm.sw_lmmd import (MixedGroupSchedule, ReferenceFeatureStore, SlidingWindowSchedule,
                         WindowConfig, build_grouped_row_order, build_mixed_row_order)
from rdm.sw_lmmd.reference_store import write_reference_store
from sw_lmmd_fixtures import CTX_DIM, D_TXT, DIMS, NAMES, build_trainer

G = 4


def _gids(num_groups, seed=0):
    """Group id per row, rows of one group scattered (as a store may lay them out)."""
    gid = np.repeat(np.arange(num_groups), G)
    return gid[np.random.default_rng(seed).permutation(gid.size)]


def _whole_groups(ids, gid):
    blocks = gid[np.asarray(ids)].reshape(-1, G)
    return bool((blocks == blocks[:, :1]).all())


def _grouped_blocks(sched, w):
    """The window's grouped segments (sets of row ids)."""
    first = (sched.start_offset + w.step * sched.stride) // sched.stride
    n = sched.window_size // sched.stride
    out = []
    for i in range(n):
        seg = (first + i) % sched.kinds.size
        if sched.kinds[seg]:
            out.append(set(w.all_ids[i * sched.stride:(i + 1) * sched.stride].tolist()))
    return out


def _all_distinct_groups(ids, gid):
    g = gid[np.asarray(ids)]
    return np.unique(g).size == g.size


# --------------------------------------------------------------------------------------
# the order
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("pattern", ["GU", "GUU", "UG", "GGU", "U"])
def test_mixed_order_segments_follow_the_pattern(pattern):
    gid = _gids(200)
    order, g, kinds = build_mixed_row_order(gid, pattern, segment=16, seed=1)
    assert g == G and order.size == kinds.size * 16
    assert kinds.size % len(pattern) == 0
    assert np.unique(order).size == order.size                      # no row twice
    assert gid.size - order.size < len(pattern) * 16                # only a sliver sits out
    for i, grouped in enumerate(kinds):
        assert grouped == (pattern[i % len(pattern)] == "G")
        seg = order[i * 16:(i + 1) * 16]
        if grouped:
            assert _whole_groups(seg, gid)
    # scattered rows come only from groups never used as grouped blocks
    grouped_rows = np.concatenate([order[i * 16:(i + 1) * 16] for i in np.flatnonzero(kinds)] or [[]])
    scattered_rows = np.concatenate([order[i * 16:(i + 1) * 16] for i in np.flatnonzero(~kinds)] or [[]])
    assert not set(gid[grouped_rows.astype(int)]) & set(gid[scattered_rows.astype(int)])


def test_mixed_order_is_seeded_and_validated():
    gid = _gids(100)
    a = build_mixed_row_order(gid, "GU", 16, seed=3)[0]
    assert np.array_equal(a, build_mixed_row_order(gid, "GU", 16, seed=3)[0])
    assert not np.array_equal(a, build_mixed_row_order(gid, "GU", 16, seed=4)[0])
    with pytest.raises(ValueError, match="G/U"):
        build_mixed_row_order(gid, "GX", 16)
    with pytest.raises(ValueError, match="multiple of the group size"):
        build_mixed_row_order(gid, "GU", 14)


# --------------------------------------------------------------------------------------
# the schedule
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("num_groups", [120, 400])   # small-pool path / independent columns
def test_steps_alternate_and_blocks_are_whole_across_wrap_reverse_and_epochs(num_groups):
    gid = _gids(num_groups)
    sched = MixedGroupSchedule(gid, "GU", window_size=32, stride=16, seed=0)
    laps = sched.kinds.size
    for epoch in range(4):
        kinds = []
        for _ in range(2 * laps + 5):                 # > 2 laps: the cyclic wrap twice
            w = sched.next()
            kinds.append(w.active_grouped)
            if w.active_grouped:
                assert _whole_groups(w.active_ids, gid)
            else:
                assert _all_distinct_groups(w.active_ids, gid)
            # a scattered prompt never has two rows in one window
            scattered = [r for r in w.all_ids if not any(r in b for b in _grouped_blocks(sched, w))]
            assert _all_distinct_groups(scattered, gid)
        assert all(a != b for a, b in zip(kinds, kinds[1:])), "G and U must alternate"
        sched.new_epoch(seed=epoch, random_start=True, reverse_probability=0.5)
        assert sched.start_offset % 16 == 0


def test_pattern_g_is_exactly_the_plain_grouped_schedule():
    """The default pattern never builds a mixed order: grouped runs and their resumes are unchanged."""
    w = WindowConfig(size=32, stride=16, group_by_prompt=True)
    assert w.group_pattern == "G"
    gid = _gids(40)
    order, g = build_grouped_row_order(gid, seed=7)
    plain = SlidingWindowSchedule(order, window_size=32, stride=16, group_size=g)
    assert plain.next().active_grouped and not hasattr(plain, "group_pattern")


def test_mixed_state_round_trip_and_mismatch_guards():
    gid = _gids(80)
    sched = MixedGroupSchedule(gid, "GU", window_size=32, stride=16, seed=0)
    for _ in range(5):
        sched.next()
    state = sched.state_dict()
    assert state["group_pattern"] == "GU"

    twin = MixedGroupSchedule(gid, "GU", window_size=32, stride=16, seed=0)
    twin.load_state_dict(state)
    a, b = twin.next(), sched.next()
    assert np.array_equal(a.all_ids, b.all_ids) and a.active_grouped == b.active_grouped

    with pytest.raises(RuntimeError, match="group_pattern mismatch"):
        MixedGroupSchedule(gid, "GUU", window_size=32, stride=16, seed=0).load_state_dict(state)
    order, g = build_grouped_row_order(gid, seed=0)
    with pytest.raises(RuntimeError, match="group_pattern mismatch"):
        SlidingWindowSchedule(order, window_size=32, stride=16, group_size=g).load_state_dict(state)


def test_mixed_schedule_rejects_unaligned_windows():
    gid = _gids(80)
    with pytest.raises(ValueError, match="multiple of the stride"):
        MixedGroupSchedule(gid, "GU", window_size=40, stride=16)
    with pytest.raises(ValueError, match="needs window.group_by_prompt"):
        WindowConfig(size=32, stride=16, group_pattern="GU").validate()
    with pytest.raises(ValueError, match="G/U"):
        WindowConfig(size=32, stride=16, group_by_prompt=True, group_pattern="GZ").validate()


# --------------------------------------------------------------------------------------
# the trainer, on a grouped toy store
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def grouped_store(tmp_path_factory):
    num_prompts = 48
    rng = np.random.default_rng(5)
    pid = _gids(num_prompts, seed=5)
    text = rng.standard_normal((num_prompts, D_TXT)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    return write_reference_store(
        str(tmp_path_factory.mktemp("mixed_store")), prompt_ids=pid, row_order=np.arange(pid.size),
        text_features=text,
        context=rng.standard_normal((num_prompts, CTX_DIM)).astype(np.float32),
        encoder_features={n: rng.standard_normal((pid.size, d)).astype(np.float32)
                          for n, d in zip(NAMES, DIMS)},
        bandwidths={n: {"sigma": 2.0, "beta": 1.0} for n in NAMES})


def test_trainer_alternates_and_monitors_only_grouped_steps(grouped_store):
    store = ReferenceFeatureStore(grouped_store, NAMES)
    trainer = build_trainer(grouped_store, window=16, stride=8, micro_batch=4, lr=1e-2)
    trainer.schedule = MixedGroupSchedule(store.grouping_ids(), "GU", window_size=16, stride=8)
    trainer.bootstrap()
    before = [p.detach().clone() for p in trainer.generator.model.parameters()]
    seen = []
    for _ in range(6):
        logs = trainer.step()
        seen.append(logs["grouped_step"])
        has_k = f"{NAMES[0]}/group_k_gen" in logs
        assert has_k == bool(logs["grouped_step"])     # same-prompt similarity on G steps only
        assert np.isfinite(logs["force"]) and not logs["skipped"]
    assert seen == [1, 0, 1, 0, 1, 0]                   # "GU": the first trained step is grouped
    assert any(not torch.equal(a, b) for a, b in zip(before, trainer.generator.model.parameters()))


def test_mixed_config_differs_from_its_base_only_in_the_pattern():
    import dataclasses

    from rdm.sw_lmmd.launch import gan_from_config, memory_from_config, window_from_config
    from rdm.train.launch import load_config

    base = load_config("configs/sw_lmmd_train_h100_2gpu_gan_grouped_geneval.yaml")
    cfg = load_config("configs/sw_lmmd_train_h100_2gpu_gan_grouped_geneval_mixGU.yaml")
    w = window_from_config(cfg)
    assert w.group_pattern == "GU"
    assert dataclasses.replace(w, group_pattern="G") == window_from_config(base)
    assert memory_from_config(cfg) == memory_from_config(base)
    assert gan_from_config(cfg) == gan_from_config(base)
    for key in ("encoders", "lr", "steps", "joint", "loss", "cache", "lr_schedule",
                "reference_extension"):
        assert getattr(cfg, key) == getattr(base, key), key
    assert cfg.exp_name != base.exp_name


# --------------------------------------------------------------------------------------
# block GAN critics on a mixed schedule: one per step kind (rdm.sw_lmmd.adversarial.PerKindCritic)
# --------------------------------------------------------------------------------------
_BLOCK = {"critic": "block", "block_width": 16, "block_heads": 2, "block_layers": 1}


@pytest.mark.parametrize("num_groups", [120, 400])
def test_block_kinds_are_the_window_segments_and_end_with_the_active_kind(num_groups):
    gid = _gids(num_groups)
    sched = MixedGroupSchedule(gid, "GUU", window_size=64, stride=16, seed=2)
    prev = None
    for step in range(3 * sched.kinds.size + 5):            # through the cyclic wrap
        if step and step % 7 == 0:
            sched.new_epoch(seed=step)                       # re-split, re-anchor, maybe reverse
            prev = None
        w = sched.next()
        kinds = sched.block_kinds(w)
        assert kinds.size == 4 and kinds[-1] == w.active_grouped
        grouped = [set(w.all_ids[i * 16:(i + 1) * 16].tolist()) for i in np.flatnonzero(kinds)]
        assert grouped == _grouped_blocks(sched, w)
        for i in np.flatnonzero(kinds):
            assert _whole_groups(w.all_ids[i * 16:(i + 1) * 16], gid)
        if prev is not None:
            assert np.array_equal(kinds[:-1], prev[1:])      # a block keeps its kind as it slides
        prev = kinds
    plain = SlidingWindowSchedule(np.arange(64), window_size=16, stride=4)
    assert not plain.block_kinds(plain.next()).any()


def test_gu_trains_one_block_critic_per_kind(grouped_store):
    trainer = build_trainer(grouped_store, window=16, stride=4, lr=1e-2, pattern="GU",
                            gan={**_BLOCK, "weight": 0.5})
    critics = trainer.critic.critics
    start = {k: [p.detach().clone() for p in c.critic.parameters()] for k, c in critics.items()}
    trainer.bootstrap()
    for i in range(4):
        logs = trainer.step()
        kind = "grouped" if logs["grouped_step"] else "ungrouped"
        for label in ("grouped", "ungrouped"):          # 2 + 2 blocks of each kind per window
            assert f"gan/{label}/d_acc" in logs and not logs[f"gan/{label}/d_skipped"]
        assert logs["gan/d_acc"] == logs[f"gan/{kind}/d_acc"]     # the scoring critic's numbers
        assert "gan/lambda" in logs and not logs["skipped"]
        assert critics[True].updates == critics[False].updates == i + 1
    for k, c in critics.items():
        assert any(not torch.equal(a, b) for a, b in zip(start[k], c.critic.parameters()))
    state = trainer.state_dict()["critic"]
    assert set(state) == {"grouped", "ungrouped"}


@pytest.mark.parametrize("steps", [1, 2])                 # step 1 is grouped, step 2 ungrouped
def test_gu_generator_term_is_scored_by_the_critic_of_the_step_kind(grouped_store, steps):
    """GradCache with per-kind critics == one graph through the force and the critic of the
    current step's kind (critic lr 0 and no spectral norm, so the critic step changes nothing)."""
    trainer = build_trainer(grouped_store, window=16, stride=4, micro_batch=1, pattern="GU",
                            gan={**_BLOCK, "weight": 0.5, "lr": 0.0, "spectral_norm": False})
    trainer.bootstrap()
    for _ in range(steps - 1):
        trainer.step()
    window = trainer.schedule.peek() if trainer._pending_window is None else trainer._pending_window

    feats = trainer._encode(window.active_ids)
    force = None
    for name in NAMES:
        context = torch.cat([trainer.cache.retained(name, window.retained_ids),
                             feats[name].detach()])
        reference = trainer._reference_window(name, window.all_ids)
        f, _ = trainer.mmd.active_force(feats[name], context, reference, trainer.sigmas[name])
        term = trainer.encoder_weights[name] * f
        force = term if force is None else force + term
    scorer = trainer.critic.select(window.active_grouped)
    tau = trainer.store.text_features(window.active_ids, dtype=torch.float64)
    adv, _ = scorer.generator_loss(feats, tau)
    lam, _ = scorer.balance(force, adv, feats, trainer.image_dims)
    trainer.optimizer.zero_grad(set_to_none=True)          # drop step 1's gradient first
    (force + lam * adv).backward()
    want = [p.grad.clone() for p in trainer._gen_params]
    trainer.optimizer.zero_grad(set_to_none=True)

    logs = trainer.step()
    assert logs["grouped_step"] == int(steps == 1)
    assert logs["gan/lambda"] == pytest.approx(lam, rel=1e-10)
    for got, ref in zip(trainer._gen_params, want):
        torch.testing.assert_close(got.grad, ref, rtol=1e-9, atol=1e-12)


def test_block_critics_match_the_schedule_kinds(grouped_store):
    with pytest.raises(ValueError, match="one block critic per step kind"):
        build_trainer(grouped_store, window=16, stride=4, pattern="GU", gan=_BLOCK, per_kind=False)
    with pytest.raises(ValueError, match="need a mixed G/U schedule"):
        build_trainer(grouped_store, window=16, stride=4, gan=_BLOCK, per_kind=True)
    with pytest.raises(ValueError, match="block critic"):
        build_trainer(grouped_store, window=16, stride=4, pattern="GU", gan={}, per_kind=True)
    # a row critic on a mixed schedule stays the one shared critic it always was
    assert not getattr(build_trainer(grouped_store, window=16, stride=4, pattern="GU",
                                     gan={}).critic, "per_kind", False)


def test_per_kind_critic_state_round_trip(grouped_store):
    a = build_trainer(grouped_store, window=16, stride=4, pattern="GU", gan={**_BLOCK, "lr": 1e-2})
    a.bootstrap()
    a.step()
    b = build_trainer(grouped_store, window=16, stride=4, pattern="GU", gan={**_BLOCK, "seed": 9})
    b.critic.load_state_dict(a.critic.state_dict())
    for x, y in zip(a.critic.critic.state_dict().values(), b.critic.critic.state_dict().values()):
        assert torch.equal(x, y)
    assert b.critic.updates == a.critic.updates == 2


def test_gu_block_critic_config_differs_from_experiment_9_only_in_the_pattern():
    import dataclasses

    from rdm.sw_lmmd.launch import gan_from_config, memory_from_config, window_from_config
    from rdm.train.launch import load_config

    base = load_config("configs/sw_lmmd_train_h100_2gpu_gan_grouped_blockcritic.yaml")
    cfg = load_config("configs/sw_lmmd_train_h100_2gpu_gan_mixGU_blockcritic.yaml")
    w = window_from_config(cfg)
    assert (w.size, w.stride, w.group_pattern) == (128, 32, "GU")
    assert dataclasses.replace(w, group_pattern="G") == window_from_config(base)
    assert gan_from_config(cfg) == gan_from_config(base) and gan_from_config(cfg).critic == "block"
    assert memory_from_config(cfg) == memory_from_config(base)
    for key in ("encoders", "lr", "steps", "joint", "loss", "cache", "lr_schedule"):
        assert getattr(cfg, key) == getattr(base, key), key
    assert not getattr(cfg, "reference_extension", None)          # COCO only
    assert cfg.exp_name != base.exp_name



def test_gu_active_rows_step_only_the_critic_of_the_step_kind(grouped_store):
    trainer = build_trainer(grouped_store, window=16, stride=4, lr=1e-2, pattern="GU",
                            gan={**_BLOCK, "weight": 0.5, "critic_rows": "active"})
    critics = trainer.critic.critics
    trainer.bootstrap()
    for i in range(4):
        before = {k: [p.detach().clone() for p in c.critic.parameters()] for k, c in critics.items()}
        logs = trainer.step()
        kind, other = (True, False) if logs["grouped_step"] else (False, True)
        label = "grouped" if kind else "ungrouped"
        assert f"gan/{label}/d_acc" in logs and logs["gan/d_acc"] == logs[f"gan/{label}/d_acc"]
        assert not any(k.startswith(f"gan/{'ungrouped' if kind else 'grouped'}/") for k in logs)
        assert any(not torch.equal(a, b) for a, b in zip(before[kind], critics[kind].critic.parameters()))
        assert all(torch.equal(a, b) for a, b in zip(before[other], critics[other].critic.parameters()))
    assert critics[True].updates == critics[False].updates == 2
