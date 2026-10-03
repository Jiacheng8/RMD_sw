"""Method-layer tests for SW-LMMD: window transitions, cache, kernel, force gradient, noise.

Everything here is pure CPU and needs no weights -- the layering exists so the algorithm can
be proved correct before any GPU or checkpoint is involved.
"""
import numpy as np
import pytest
import torch

from rdm.compare.kernels import gamma_from_sigma, gaussian_gram
from rdm.sw_lmmd import (ExactLocalMMD, GeneratedWindowCache, SlidingWindowSchedule, WindowConfig,
                         build_row_order, partition_rows, row_noise, row_seed)
from rdm.sw_lmmd.config import MemoryPolicy


# --------------------------------------------------------------------------------------
# window schedule
# --------------------------------------------------------------------------------------
def test_overlap_is_exact():
    """previous.all_ids[stride:] == current.retained_ids -- the identity the cache relies on."""
    sched = SlidingWindowSchedule(np.arange(4096), window_size=1024, stride=128)
    w0 = sched.next()
    w1 = sched.next()
    assert np.array_equal(w0.all_ids[128:], w1.retained_ids)
    assert np.array_equal(w1.all_ids[:896], w1.retained_ids)
    assert np.array_equal(w1.all_ids[896:], w1.active_ids)
    assert w1.active_ids.size == 128


def test_overlap_holds_across_cyclic_wrap():
    """The wrap is where an off-by-one would hide; walk a full cycle and then some."""
    n, k, b = 40, 16, 4
    sched = SlidingWindowSchedule(build_row_order(n, seed=1), window_size=k, stride=b, cyclic=True)
    prev = sched.next()
    for _ in range(3 * n // b):
        cur = sched.next()                      # next() itself asserts the identity
        assert np.array_equal(prev.all_ids[b:], cur.retained_ids)
        prev = cur


def test_schedule_covers_every_row_uniformly():
    """Cyclic windows must not over- or under-weight any row (spec sec. 22.7)."""
    n, k, b = 64, 16, 4
    sched = SlidingWindowSchedule(build_row_order(n, seed=2), window_size=k, stride=b)
    counts = np.zeros(n, dtype=int)
    for _ in range(n // b):                     # exactly one full cycle of active windows
        counts[sched.next().active_ids] += 1
    assert counts.min() == counts.max() == 1


def test_non_cyclic_schedule_raises_when_exhausted():
    sched = SlidingWindowSchedule(np.arange(20), window_size=8, stride=4, cyclic=False)
    with pytest.raises(StopIteration):
        for _ in range(10):
            sched.next()


def test_schedule_state_round_trip_and_order_guard():
    order = build_row_order(64, seed=3)
    sched = SlidingWindowSchedule(order, window_size=16, stride=4)
    for _ in range(3):
        sched.next()
    state = sched.state_dict()
    expected = sched.peek()

    resumed = SlidingWindowSchedule(order, window_size=16, stride=4)
    resumed.load_state_dict(state)
    assert np.array_equal(resumed.peek().all_ids, expected.all_ids)

    other = SlidingWindowSchedule(build_row_order(64, seed=999), window_size=16, stride=4)
    with pytest.raises(RuntimeError, match="row order hash"):
        other.load_state_dict(state)

    # Same permutation, other window: the step counter would name different rows.
    for k, b in ((8, 4), (16, 2)):
        with pytest.raises(RuntimeError, match="window mismatch"):
            SlidingWindowSchedule(order, window_size=k, stride=b).load_state_dict(state)
    legacy = {key: v for key, v in state.items() if key not in ("window_size", "stride")}
    SlidingWindowSchedule(order, window_size=8, stride=2).load_state_dict(legacy)  # predates them


def test_new_epoch_changes_traversal():
    sched = SlidingWindowSchedule(build_row_order(64, seed=4), window_size=16, stride=4)
    first = sched.next().all_ids.copy()
    assert sched.new_epoch(seed=5) == 1
    assert sched.step == 0
    assert not np.array_equal(sched.next().all_ids, first)


def test_window_config_reports_max_age():
    cfg = WindowConfig(size=1024, stride=128)
    assert cfg.overlap == 896
    assert cfg.full_to_active_gradient_ratio == 8.0
    assert cfg.max_age_steps == 7                      # ceil(K/B) - 1
    assert WindowConfig(size=1024, stride=120).max_age_steps == 8


# --------------------------------------------------------------------------------------
# generated feature cache
# --------------------------------------------------------------------------------------
def _cache_with_window(names=("e",), k=16, b=4, d=3):
    sched = SlidingWindowSchedule(build_row_order(64, seed=7), window_size=k, stride=b)
    cache = GeneratedWindowCache(names)
    w0 = sched.next()
    feats = {n: torch.randn(k - b, d) for n in names}
    cache.initialize(w0.retained_ids, feats, step=-1)
    return sched, cache, w0


def test_cache_transition_keeps_rows_and_ages():
    k, b, d = 16, 4, 3
    sched, cache, w0 = _cache_with_window(k=k, b=b, d=d)
    active0 = torch.randn(b, d)
    cache.advance(w0.all_ids, k - b, {"e": active0}, step=0)
    entry = cache.entries["e"]
    assert np.array_equal(entry.row_ids, w0.all_ids)
    torch.testing.assert_close(entry.features[-b:], active0)
    assert cache.max_age(0) == 1                       # bootstrap rows were written at step -1

    w1 = sched.next()
    retained = cache.retained("e", w1.retained_ids)
    assert retained.shape[0] == k - b
    torch.testing.assert_close(retained, entry.features[b:])
    cache.advance(w1.all_ids, k - b, {"e": torch.randn(b, d)}, step=1)
    assert cache.max_age(1) == 2
    assert sum(cache.age_buckets(1).values()) == k


def test_cache_rejects_misaligned_rows():
    """A silent row mismatch would pair samples with other prompts' references."""
    _, cache, w0 = _cache_with_window()
    with pytest.raises(RuntimeError, match="cache row mismatch"):
        cache.retained("e", w0.retained_ids[::-1].copy())


def test_cache_max_age_matches_window_config():
    k, b, d = 16, 4, 2
    sched, cache, window = _cache_with_window(k=k, b=b, d=d)
    for step in range(12):
        cache.advance(window.all_ids, k - b, {"e": torch.randn(b, d)}, step=step)
        assert cache.max_age(step) <= WindowConfig(size=k, stride=b).max_age_steps
        window = sched.next()
        cache.retained("e", window.retained_ids)


def test_cache_refresh_overwrites_and_reports_drift():
    k, b, d = 16, 4, 3
    _, cache, w0 = _cache_with_window(k=k, b=b, d=d)
    cache.advance(w0.all_ids, k - b, {"e": torch.randn(b, d)}, step=0)
    rows = w0.all_ids[:4]
    fresh = torch.zeros(4, d)
    drift = cache.refresh_retained("e", rows, fresh, step=5)
    assert drift.shape == (4,)
    torch.testing.assert_close(cache.entries["e"].features[:4], fresh)
    # refreshed rows become age 0; the untouched bootstrap rows (written at step -1) are 5-(-1)
    assert cache.age_buckets(5)[0] == 4
    assert cache.max_age(5) == 6


def test_cache_state_round_trip():
    _, cache, w0 = _cache_with_window()
    cache.advance(w0.all_ids, 12, {"e": torch.randn(4, 3)}, step=0)
    restored = GeneratedWindowCache(["e"])
    restored.load_state_dict(cache.state_dict())
    torch.testing.assert_close(restored.entries["e"].features, cache.entries["e"].features)
    assert np.array_equal(restored.entries["e"].row_ids, cache.entries["e"].row_ids)


# --------------------------------------------------------------------------------------
# the force surrogate -- the headline correctness property
# --------------------------------------------------------------------------------------
def test_active_force_matches_scaled_partial_gradient():
    """grad(force, active) == (K/B) * grad(full biased MMD, x)[-B:]   (spec sec. 6 / 19.3).

    This is what licenses back-propagating through B rows instead of K: the surrogate
    reproduces the full local MMD's per-point force on the active rows, scaled by K/B, with
    that factor already inside the 1/(BK) normalization.
    """
    torch.manual_seed(0)
    K, B, D, sigma = 8, 4, 6, 1.3
    gamma = gamma_from_sigma(sigma)
    x = torch.randn(K, D, dtype=torch.float64, requires_grad=True)
    y = torch.randn(K, D, dtype=torch.float64)

    full = (gaussian_gram(x, x, gamma).mean()
            - 2.0 * gaussian_gram(x, y, gamma).mean()
            + gaussian_gram(y, y, gamma).mean())
    grad_full = torch.autograd.grad(full, x)[0]

    active = x[-B:].detach().clone().requires_grad_(True)
    context = torch.cat([x[:-B].detach(), active.detach()], dim=0)
    force, stats = ExactLocalMMD(block_size=3).active_force(active, context, y, sigma)
    grad_force = torch.autograd.grad(force, active)[0]

    torch.testing.assert_close(grad_force.double(), (K / B) * grad_full[-B:],
                               rtol=1e-5, atol=1e-6)
    assert float(stats["force"]) == pytest.approx(float(force.detach()))


@pytest.mark.parametrize("K,B", [(8, 4), (12, 3), (16, 16)])
def test_force_identity_across_shapes(K, B):
    torch.manual_seed(K * 100 + B)
    D, sigma = 5, 0.9
    gamma = gamma_from_sigma(sigma)
    x = torch.randn(K, D, dtype=torch.float64, requires_grad=True)
    y = torch.randn(K, D, dtype=torch.float64)
    full = (gaussian_gram(x, x, gamma).mean() - 2.0 * gaussian_gram(x, y, gamma).mean()
            + gaussian_gram(y, y, gamma).mean())
    grad_full = torch.autograd.grad(full, x)[0]
    active = x[-B:].detach().clone().requires_grad_(True)
    context = torch.cat([x[:-B].detach(), active.detach()], dim=0)
    force, _ = ExactLocalMMD(block_size=4).active_force(active, context, y, sigma)
    grad_force = torch.autograd.grad(force, active)[0]
    torch.testing.assert_close(grad_force.double(), (K / B) * grad_full[-B:],
                               rtol=1e-5, atol=1e-6)


def test_blockwise_kernel_parity_forward_and_gradient():
    """Block size must not change the value or the gradient -- only the peak memory."""
    torch.manual_seed(1)
    active = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    context = torch.randn(10, 4, dtype=torch.float64)
    reference = torch.randn(10, 4, dtype=torch.float64)

    f_small, _ = ExactLocalMMD(block_size=2).active_force(active, context, reference, 1.1)
    g_small = torch.autograd.grad(f_small, active)[0]
    f_big, _ = ExactLocalMMD(block_size=1000).active_force(active, context, reference, 1.1)
    g_big = torch.autograd.grad(f_big, active)[0]

    torch.testing.assert_close(f_small, f_big)
    torch.testing.assert_close(g_small, g_big)


def test_monitor_matches_dense_biased_mmd():
    torch.manual_seed(2)
    x, y, sigma = torch.randn(9, 5), torch.randn(9, 5), 1.4
    gamma = gamma_from_sigma(sigma)
    expected = (gaussian_gram(x, x, gamma).mean() - 2.0 * gaussian_gram(x, y, gamma).mean()
                + gaussian_gram(y, y, gamma).mean())
    m = ExactLocalMMD(block_size=2).monitor(x, y, sigma)
    torch.testing.assert_close(m.mmd2, expected.clamp_min(0).float(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(m.k_xx, gaussian_gram(x, x, gamma).mean().float(),
                               rtol=1e-5, atol=1e-6)
    assert set(m.as_log("e/")) == {"e/mmd2", "e/k_xx", "e/k_xy", "e/k_yy"}


def test_monitor_is_nonnegative_and_zero_on_identical_sets():
    x = torch.randn(7, 3)
    m = ExactLocalMMD().monitor(x, x.clone(), 1.0)
    assert float(m.mmd2) == pytest.approx(0.0, abs=1e-6)


def test_active_force_rejects_detached_active():
    """The classic failure: wrapping the encoder forward in no_grad kills all training."""
    mmd = ExactLocalMMD()
    with pytest.raises(ValueError, match="do not require grad"):
        mmd.active_force(torch.randn(2, 3), torch.randn(4, 3), torch.randn(4, 3), 1.0)


def test_active_force_rejects_mismatched_window():
    mmd = ExactLocalMMD()
    with pytest.raises(ValueError, match="same window"):
        mmd.active_force(torch.randn(2, 3, requires_grad=True), torch.randn(4, 3),
                         torch.randn(5, 3), 1.0)


def test_reference_term_carries_no_gradient():
    active = torch.randn(3, 4, requires_grad=True)
    reference = torch.randn(6, 4, requires_grad=True)
    context = torch.randn(6, 4)
    force, _ = ExactLocalMMD().active_force(active, context, reference, 1.0)
    force.backward()
    assert reference.grad is None                  # the target is frozen by construction


# --------------------------------------------------------------------------------------
# the distribution contract
# --------------------------------------------------------------------------------------
def test_row_noise_is_deterministic_per_row():
    rows = np.array([7, 3, 99, 3])
    a = row_noise(rows, (2, 2), seed=11)
    b = row_noise(rows, (2, 2), seed=11)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a[1], a[3])               # same row id -> same latent
    assert not torch.allclose(a[0], a[1])


def test_row_noise_is_independent_of_chunking_and_rank():
    """The property that makes the world-size parity test possible at all."""
    rows = np.arange(16)
    whole = row_noise(rows, (3,), seed=5)
    chunked = torch.cat([row_noise(rows[i:i + 4], (3,), seed=5) for i in range(0, 16, 4)])
    torch.testing.assert_close(whole, chunked)
    shard = torch.cat([row_noise(partition_rows(rows, r, 4), (3,), seed=5) for r in range(4)])
    torch.testing.assert_close(whole, shard)


def test_row_noise_varies_with_seed_and_epoch():
    rows = np.arange(4)
    base = row_noise(rows, (3,), seed=1, epoch=0)
    assert not torch.allclose(base, row_noise(rows, (3,), seed=2, epoch=0))
    assert not torch.allclose(base, row_noise(rows, (3,), seed=1, epoch=1))


def test_row_seed_is_distinct_for_neighbouring_rows():
    seeds = {row_seed(0, 0, i) for i in range(1000)}
    assert len(seeds) == 1000


def test_partition_rows_is_contiguous_and_exhaustive():
    ids = np.arange(12) * 3
    parts = [partition_rows(ids, r, 4) for r in range(4)]
    assert all(p.size == 3 for p in parts)
    assert np.array_equal(np.concatenate(parts), ids)
    with pytest.raises(ValueError, match="divisible"):
        partition_rows(np.arange(10), 0, 4)


# --------------------------------------------------------------------------------------
# memory policy: the method/machine seam
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("world", [1, 2, 4, 8, 16, 32])
def test_b128_divides_every_target_world_size(world):
    """One window config has to port from a single 4090 to a 32-rank H200 job."""
    assert MemoryPolicy(micro_batch=1).resolve_grad_accum(128, world) == 128 // world


def test_grad_accum_invariant():
    policy = MemoryPolicy(micro_batch=4)
    assert policy.resolve_grad_accum(128, 8) == 4          # 4 * 8 * 4 == 128
    with pytest.raises(ValueError, match="divisible by world_size"):
        policy.resolve_grad_accum(128, 6)
    with pytest.raises(ValueError, match="micro_batch"):
        MemoryPolicy(micro_batch=3).resolve_grad_accum(128, 8)


def test_fsdp_forbids_double_gradient_reduction():
    with pytest.raises(ValueError, match="twice"):
        MemoryPolicy(shard="fsdp", grad_reduce="mean").validate()
    MemoryPolicy(shard="fsdp", grad_reduce="none").validate()


# --------------------------------------------------------------------------------------
# config -> objects (the shipped YAMLs must resolve to the intended geometry)
# --------------------------------------------------------------------------------------
def test_shipped_configs_share_one_method_and_differ_only_in_memory():
    """The portability claim, asserted against the actual files."""
    from rdm.sw_lmmd.launch import memory_from_config, resolve_batching, window_from_config
    from rdm.train.launch import load_config

    debug = load_config("configs/sw_lmmd_debug_4x4090.yaml")
    h100 = load_config("configs/sw_lmmd_h100_8gpu.yaml")

    w_debug, w_h100 = window_from_config(debug), window_from_config(h100)
    assert (w_debug.size, w_debug.stride) == (w_h100.size, w_h100.stride) == (1024, 128)
    assert w_debug.overlap == 896 and w_debug.max_age_steps == 7

    b_debug = resolve_batching(debug, memory_from_config(debug), w_debug, world_size=4)
    assert (b_debug["per_rank"], b_debug["micro_batch"], b_debug["grad_accum"]) == (32, 1, 32)

    b_h100 = resolve_batching(h100, memory_from_config(h100), w_h100, world_size=8)
    assert (b_h100["per_rank"], b_h100["micro_batch"], b_h100["grad_accum"]) == (16, 8, 2)


def test_debug_config_avoids_double_gradient_reduction():
    from rdm.sw_lmmd.launch import memory_from_config
    from rdm.train.launch import load_config

    debug = memory_from_config(load_config("configs/sw_lmmd_debug_4x4090.yaml"))
    assert debug.shard == "fsdp" and debug.grad_reduce == "none"
    assert debug.param_dtype == torch.bfloat16 and debug.battery_bf16

    h100 = memory_from_config(load_config("configs/sw_lmmd_h100_8gpu.yaml"))
    assert h100.shard == "none" and h100.grad_reduce == "mean"


def test_cache_config_parses_refresh_policy():
    from rdm.sw_lmmd.launch import cache_from_config
    from rdm.train.launch import load_config

    cfg = cache_from_config(load_config("configs/sw_lmmd_flux.yaml"))
    assert cfg.refresh_every_windows == 20 and cfg.drift_threshold == 0.05


def test_h100_2gpu_config_fits_80gb():
    """The shipped 2xH100 config must fit an 80 GB card with room for fragmentation.

    `shard: none` is DATA parallel -- every rank holds a full copy of the training state, so
    the per-card budget is independent of world size and adding GPUs never buys headroom.
    """
    from rdm.sw_lmmd.launch import memory_from_config, resolve_batching, window_from_config
    from rdm.train.launch import load_config

    cfg = load_config("configs/sw_lmmd_h100_2gpu.yaml")
    policy = memory_from_config(cfg)
    assert policy.shard == "none" and policy.grad_reduce == "mean"
    assert policy.param_dtype == torch.bfloat16, "fp32 params put this at 76 GB on an 80 GB card"
    assert policy.battery_bf16, "an fp32 battery adds 9 GB this config has no room for"

    params_b = 3.875                                  # klein-4B, from the released checkpoints
    bytes_per = 2 if policy.param_dtype == torch.bfloat16 else 4
    moments = params_b * 2 if policy.optimizer == "adamw8bit" else params_b * 8
    train_state = params_b * bytes_per * 2 + moments   # params + grads + AdamW's TWO moments
    encoders = 7.8 if len(cfg.encoders) >= 10 else 0.95 * len(cfg.encoders)
    activations = 2.0 * policy.micro_batch          # rough, and the only estimated term
    total = train_state + encoders + 0.2 + activations
    assert total < 80 * 0.85, f"per-card {total:.1f} GB leaves no fragmentation margin on 80 GB"

    b = resolve_batching(cfg, policy, window_from_config(cfg), world_size=2)
    assert b["per_rank"] == 64 and b["grad_accum"] == 8
    assert b["per_rank"] == policy.micro_batch * b["grad_accum"]


def test_optimizer_builder_rejects_unwired_modes():
    from rdm.sw_lmmd.config import MemoryPolicy
    from rdm.sw_lmmd.launch import build_optimizer

    p = [torch.nn.Parameter(torch.zeros(2))]
    cfg = type("C", (), {"lr": 1e-5})()
    assert isinstance(build_optimizer(p, cfg, MemoryPolicy(optimizer="adamw")), torch.optim.AdamW)
    with pytest.raises(NotImplementedError, match="offload"):
        build_optimizer(p, cfg, MemoryPolicy(optimizer="adamw_offload"))


# --------------------------------------------------------------------------------------
# generator lr schedule
# --------------------------------------------------------------------------------------
def test_lr_schedule_shapes():
    from rdm.sw_lmmd.config import LRSchedule

    cos = LRSchedule("cosine", warmup_steps=10, min_lr_ratio=0.1, total_steps=110)
    f = [cos.factor(s) for s in range(120)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0)     # (s+1)/10, never 0
    assert f[10] == pytest.approx(1.0) and f[60] == pytest.approx(0.55)  # cosine midpoint
    assert all(a >= b for a, b in zip(f[10:], f[11:]))                   # non-increasing
    assert f[110] == f[119] == pytest.approx(0.1)                        # floor past the horizon
    lin = LRSchedule("linear", total_steps=4)
    assert [lin.factor(s) for s in range(5)] == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])
    assert LRSchedule().factor(0) == LRSchedule().factor(10 ** 9) == 1.0
    assert LRSchedule(warmup_steps=4).factor(1) == 0.5                   # warm-up, then flat
    with pytest.raises(ValueError, match="total_steps"):
        LRSchedule("cosine").factor(5)
    with pytest.raises(ValueError, match="constant / cosine / linear"):
        LRSchedule("step").validate()


def test_lr_schedule_configs():
    from types import SimpleNamespace

    from rdm.sw_lmmd.launch import lr_schedule_from_config
    from rdm.train.launch import load_config

    flux = lr_schedule_from_config(load_config("configs/sw_lmmd_flux.yaml"))
    assert (flux.name, flux.warmup_steps) == ("constant", 0)           # default: the plain lr
    assert lr_schedule_from_config(load_config("configs/sw_lmmd_train_4x4090.yaml")) == flux
    for name in ("sw_lmmd_train_h100_2gpu.yaml", "sw_lmmd_train_h100_2gpu_gan.yaml"):
        h100 = lr_schedule_from_config(load_config(f"configs/{name}"))
        assert (h100.name, h100.warmup_steps, h100.min_lr_ratio, h100.total_steps) == \
            ("cosine", 100, 0.1, 2000)
    cfg = load_config("configs/sw_lmmd_train_h100_2gpu.yaml")
    cfg.steps = 20                                                        # --set steps=20 (gate)
    assert lr_schedule_from_config(cfg).total_steps == 20
    with pytest.raises(ValueError, match="unknown key"):
        lr_schedule_from_config(SimpleNamespace(lr_schedule={"warmup": 10}, steps=5))
