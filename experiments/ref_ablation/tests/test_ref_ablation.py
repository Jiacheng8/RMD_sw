"""Tests of the reference-side ablation (small tensors and toy stores; no model is loaded).

A. ``ref_mode=local`` IS the existing method: the new force equals ``ExactLocalMMD.active_force``
   bit for bit (value and gradient), and a local-mode trainer reproduces the existing
   ``SWLMMDTrainer`` step for step -- logs, parameters, cache, critic -- with every diagnostic on.
B. ``random``: the drawn rows' text block is their OWN caption's ``beta * tau`` (read straight from
   the .npy files), consecutive steps draw different rows, the sampler is isolated from every
   other RNG, and a resume continues its stream.
C. ``global``: ``psi(x)^T psi(y)`` vs ``k(x, y)`` and ``psi(x)^T mu_bar`` vs the exact mean similarity
   -- exact when the landmarks are the whole reference, converging as m grows, and the
   ``k(x, L) . v`` shortcut equal to the explicit feature map; bundles are refused for another store.
D. The repulsion -- value and gradient -- is identical in all three modes.
E. (extra) random and global keep the world-size parity of the method: 1 vs 2 ranks.
"""
from __future__ import annotations

import copy
import os
import socket
import subprocess
import sys

import numpy as np
import pytest
import torch

from rdm.compare.kernels import cross_kernel_mean, gamma_from_sigma, squared_distance
from rdm.compare.median_heuristic import median_bandwidth
from rdm.sw_lmmd import (CacheConfig, GANConfig, ReferenceFeatureStore, SlidingWindowSchedule,
                         SWLMMDTrainer, build_critic, build_grouped_row_order, split_groups)
from rdm.sw_lmmd.local_mmd import ExactLocalMMD
from rdm.sw_lmmd.reference_store import write_reference_store
from sw_lmmd_fixtures import CTX_DIM, D_TXT, DIMS, NAMES, NOISE_DIM, MockBattery, ToyGenerator

from experiments.ref_ablation import (GlobalNystrom, RandomReferenceSampler, RefAblationLoss,
                                      RefAblationTrainer, ref_ablation_from_config)
from experiments.ref_ablation.build_landmarks import (build_bundle, load_joint_pool,
                                                      nystrom_from_pool, quality_report)
from experiments.ref_ablation.nystrom_global import bundle_path, check_against_store

F64 = torch.float64


# =============================================================================================
# toy data
# =============================================================================================
def write_grouped_store(root: str, num_prompts: int = 64, group: int = 4, scale: float = 0.25,
                        seed: int = 0) -> str:
    """A store laid out like the real one: ``group`` reference rows per prompt (``prompt_ids``),
    prompt-indexed text / context, image features clustered by prompt, ``sigma = scale * median``
    and ``beta = sigma`` (s_txt = 1), as bandwidths.json holds for the COCO store."""
    rng = np.random.default_rng(seed)
    rows = num_prompts * group
    feats = {}
    for name, d in zip(NAMES, DIMS):
        centers = rng.standard_normal((8, d)) * 2.0
        proto = centers[rng.integers(8, size=num_prompts)] + 0.6 * rng.standard_normal((num_prompts, d))
        feats[name] = (np.repeat(proto, group, 0) +
                       0.35 * rng.standard_normal((rows, d))).astype(np.float32)
    text = rng.standard_normal((num_prompts, D_TXT)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    context = rng.standard_normal((num_prompts, CTX_DIM)).astype(np.float32)
    bandwidths = {}
    for name, f in feats.items():
        sigma = scale * float(median_bandwidth(torch.from_numpy(f).float()))
        bandwidths[name] = {"sigma": sigma, "beta": sigma, "sigma_scale": scale, "s_txt": 1.0}
    prompt_ids = np.repeat(np.arange(num_prompts), group)[rng.permutation(rows)]
    return write_reference_store(root, encoder_features=feats, text_features=text, context=context,
                                 bandwidths=bandwidths, prompt_ids=prompt_ids,
                                 extra_metadata={"reference_type": "toy-grouped",
                                                 "row_order_hash": "toy"})


def make_trainer(store_root: str, *, window: int = 16, stride: int = 8, grouped: bool = False,
                 group_size: int = 0, unbiased: bool = False, sibling_weight: float = 1.0,
                 gan: dict | None = None, lr: float = 0.0, micro_batch: int = 1,
                 grad_clip: float = 0.0) -> SWLMMDTrainer:
    """The existing trainer over a toy store, as tests/sw_lmmd_fixtures.build_trainer builds it,
    plus prompt grouping / sibling weighting."""
    store = ReferenceFeatureStore(store_root, NAMES)
    if grouped:
        grouping = store.grouping_ids()
        if group_size:
            grouping = split_groups(grouping, group_size, 3407)
        order, group = build_grouped_row_order(grouping, 3407)
    else:
        order, group = np.random.default_rng(3407).permutation(store.num_rows), 1
    schedule = SlidingWindowSchedule(order, window_size=window, stride=stride, group_size=group)
    generator = ToyGenerator(seed=0, dtype=F64)
    optimizer = torch.optim.AdamW(generator.model.parameters(), lr=lr)
    critic = None
    if gan is not None:
        cfg = GANConfig(**{"enabled": True, "hidden": 16, "g_start_step": 0, "block_width": 16,
                           "block_heads": 2, "block_layers": 1, **gan})
        critic = build_critic(cfg, store, list(NAMES), joint=True, device="cpu", dtype=F64,
                              block=stride if cfg.critic == "block" else None)
    return SWLMMDTrainer(generator, MockBattery(dtype=F64), store, schedule, optimizer,
                         encoder_names=list(NAMES), noise_shape=(NOISE_DIM,),
                         micro_batch=micro_batch, grad_clip=grad_clip, kernel_block=8, seed=0,
                         joint=True, monitor_every=1, cache_cfg=CacheConfig(store_dtype=F64),
                         device="cpu", feature_dtype=F64, context_dtype=F64, critic=critic,
                         unbiased_siblings=unbiased, sibling_weight=sibling_weight)


def build_toy_bundles(store_root: str, out_dir: str, m: int = 64) -> str:
    store = ReferenceFeatureStore(store_root, NAMES, require_context=False)
    os.makedirs(out_dir, exist_ok=True)
    for name in NAMES:
        pool = load_joint_pool(store, name, device="cpu")
        torch.save(build_bundle(store, name, pool, m=m, iters=10, seed=0),
                   str(bundle_path(out_dir, name)))
    return out_dir


@pytest.fixture(scope="module")
def grouped_store(tmp_path_factory):
    return write_grouped_store(str(tmp_path_factory.mktemp("refabl_store")))


@pytest.fixture(scope="module")
def toy_nystrom(grouped_store, tmp_path_factory):
    out = build_toy_bundles(grouped_store, str(tmp_path_factory.mktemp("refabl_bundles")))
    store = ReferenceFeatureStore(grouped_store, NAMES, require_context=False)
    return GlobalNystrom.load(out, NAMES, store=store)


def _force_inputs(group: int, seed: int, dtype=F64, d: int = 7):
    """Active rows, a window context that contains them, a window reference, and an independent
    second K-row reference -- the shapes active_force sees."""
    g = torch.Generator().manual_seed(seed)
    B, K = 4 * group, 12 * group
    active = torch.randn(B, d, generator=g, dtype=F64).to(dtype).requires_grad_(True)
    context = torch.cat([torch.randn(K - B, d, generator=g, dtype=F64).to(dtype), active.detach()])
    reference = (torch.randn(K, d, generator=g, dtype=F64) * 0.9 + 0.2).to(dtype)
    other = (torch.randn(K, d, generator=g, dtype=F64) * 1.1 - 0.1).to(dtype)
    sigma = float(median_bandwidth(torch.cat([context, reference]).double()))
    return active, context, reference, other, sigma


SIBLING_CASES = [(1, 1.0), (4, 4.0 / 3.0), (2, 1.5)]   # plain; G=4 unbiased; G=2 weight 1.5


# =============================================================================================
# A. local mode == the existing method
# =============================================================================================
@pytest.mark.parametrize("dtype", [torch.float32, F64])
@pytest.mark.parametrize("group,weight", SIBLING_CASES)
def test_A_local_force_is_the_existing_force(dtype, group, weight):
    active, context, reference, _, sigma = _force_inputs(group, seed=group, dtype=dtype)
    f0, s0 = ExactLocalMMD(block_size=8).active_force(active, context, reference, sigma,
                                                      group=group, sibling_weight=weight)
    g0, = torch.autograd.grad(f0, active)
    f1, s1 = RefAblationLoss("local", block_size=8).active_force(
        "enc", active, context, reference, sigma, group=group, sibling_weight=weight)
    g1, = torch.autograd.grad(f1, active)
    assert torch.equal(f0, f1)
    assert torch.equal(g0, g1)
    assert s0.keys() == s1.keys()
    for key in s0:
        assert torch.equal(s0[key], s1[key]), key


TRAINER_CASES = {
    "ungrouped": dict(),
    "grouped_unbiased_mb2": dict(grouped=True, unbiased=True, micro_batch=2),
    "grouped_g2_w15_rowgan": dict(grouped=True, group_size=2, sibling_weight=1.5,
                                  gan={"weight": 0.5}),
    "grouped_blockgan_active": dict(grouped=True, gan={"weight": 1.5, "critic": "block",
                                                       "critic_rows": "active"}),
}


@pytest.mark.parametrize("case", sorted(TRAINER_CASES))
def test_A_local_trainer_reproduces_the_existing_trainer(grouped_store, toy_nystrom, case):
    """Same toy run through the existing trainer and through the ablation trainer in local mode
    (Nystrom bundles loaded and every diagnostic on): identical logs, weights, cache, critic."""
    kw = dict(TRAINER_CASES[case], lr=1e-2, grad_clip=1.0)
    old = make_trainer(grouped_store, **kw)
    new = RefAblationTrainer.from_trainer(make_trainer(grouped_store, **kw), ref_mode="local",
                                          nystrom=toy_nystrom, grad_norms=True)
    old.bootstrap()
    new.bootstrap()
    for _ in range(4):
        a, b = old.step(), new.step()
        for key, value in a.items():
            assert b[key] == value, (case, key, value, b[key])
        assert {k for k in b if k not in a} >= {"ref_mode", "ref/enc_a/grad_attraction",
                                                "ref/enc_b/attraction_global"}
        for p, q in zip(old._gen_params, new._gen_params):
            assert torch.equal(p, q)
        for name in NAMES:
            assert torch.equal(old.cache.entries[name].features, new.cache.entries[name].features)
    if old.critic is not None:
        sa, sb = old.critic.critic.state_dict(), new.critic.critic.state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_A_local_diagnostics_are_consistent(grouped_store):
    """In local mode the trained attraction IS the window's attraction, and |g_force| is the norm
    of 2 (grad repel - grad attract)."""
    tr = RefAblationTrainer.from_trainer(make_trainer(grouped_store, grouped=True), ref_mode="local")
    tr.bootstrap()
    logs = tr.step()
    for name in NAMES:
        p = f"ref/{name}/"
        assert logs[p + "attraction"] == pytest.approx(logs[p + "attraction_local"], rel=1e-12)
        assert logs[p + "attraction"] == pytest.approx(logs[f"{name}/attraction"], rel=1e-12)
        assert logs[p + "force_weighted"] == pytest.approx(0.5 * logs[p + "force"], rel=1e-12)
        a, r, c = logs[p + "grad_attraction"], logs[p + "grad_repulsion"], logs[p + "grad_cos_att_rep"]
        assert logs[p + "grad_force"] ** 2 == pytest.approx(a * a + r * r - 2 * a * r * c, rel=1e-9)


# =============================================================================================
# B. random mode
# =============================================================================================
def _spy_references(trainer):
    """Record the reference tensor every active_force call receives."""
    seen = []
    original = trainer.ref_loss.active_force_parts

    def spy(name, active, context, reference, *args, **kwargs):
        seen.append((name, None if reference is None else reference.detach().clone()))
        return original(name, active, context, reference, *args, **kwargs)

    trainer.ref_loss.active_force_parts = spy
    return seen


def test_B_random_rows_carry_their_own_captions(grouped_store):
    tr = RefAblationTrainer.from_trainer(make_trainer(grouped_store, grouped=True),
                                         ref_mode="random", ref_seed=7)
    seen = _spy_references(tr)
    text = np.load(os.path.join(grouped_store, "text_features.npy"))
    pid = np.load(os.path.join(grouped_store, "prompt_ids.npy"))
    feats = {n: np.load(os.path.join(grouped_store, "encoder_features", f"{n}.npy")) for n in NAMES}
    K, N = tr.schedule.window_size, tr.store.num_rows
    tr.bootstrap()
    drawn = []
    for _ in range(6):
        window = tr._pending_window or tr.schedule.peek()
        tr.step()
        rows = tr._last_attraction_rows
        drawn.append(rows)
        assert rows.shape == (K,) and np.unique(rows).size == K and rows.min() >= 0 and rows.max() < N
        for (name, ref), d in zip(seen[-len(NAMES):], DIMS):
            beta = tr.store.beta(name)
            np.testing.assert_array_equal(ref[:, :d].numpy(), feats[name][rows].astype(np.float64))
            # the text block is the drawn rows' OWN captions ...
            np.testing.assert_allclose(ref[:, d:].numpy(), beta * text[pid[rows]].astype(np.float64),
                                       rtol=0, atol=1e-6)
            # ... not the prompts the generator rolled out in this window
            assert not np.allclose(ref[:, d:].numpy(), beta * text[pid[window.all_ids]])
    for r0, r1 in zip(drawn, drawn[1:]):
        assert set(r0.tolist()) != set(r1.tolist())


def test_B_sampler_is_isolated_and_reproducible():
    torch_state, np_state = torch.get_rng_state(), np.random.get_state()
    s = RandomReferenceSampler(1000, 16, seed=3)
    rows = [s.draw() for _ in range(5)]
    assert torch.equal(torch.get_rng_state(), torch_state)
    after = np.random.get_state()
    assert np_state[0] == after[0] and np.array_equal(np_state[1], after[1]) and np_state[2:] == after[2:]
    again = RandomReferenceSampler(1000, 16, seed=3)
    assert all(np.array_equal(r, again.draw()) for r in rows)
    other_seed = RandomReferenceSampler(1000, 16, seed=4).draw()
    diag_stream = RandomReferenceSampler(1000, 16, seed=3, stream="diagnostic").draw()
    assert not np.array_equal(other_seed, rows[0]) and not np.array_equal(diag_stream, rows[0])
    resumed = RandomReferenceSampler(1000, 16, seed=3)
    one = RandomReferenceSampler(1000, 16, seed=3)
    one.draw()
    resumed.load_state_dict(one.state_dict())
    assert all(np.array_equal(r, resumed.draw()) for r in rows[1:])
    with pytest.raises(RuntimeError, match="mismatch"):
        RandomReferenceSampler(1000, 16, seed=9).load_state_dict(one.state_dict())


def test_B_random_mode_leaves_the_generator_side_untouched(grouped_store):
    """lr = 0: the local and the random trainer roll out the same samples on the same windows --
    the sampler touches neither the noise nor the window order."""
    loc = RefAblationTrainer.from_trainer(make_trainer(grouped_store, grouped=True), ref_mode="local")
    rnd = RefAblationTrainer.from_trainer(make_trainer(grouped_store, grouped=True),
                                          ref_mode="random", ref_seed=1)
    loc.bootstrap()
    rnd.bootstrap()
    for _ in range(4):
        a, b = loc.step(), rnd.step()
        assert a["window_step"] == b["window_step"]
        for name in NAMES:
            assert torch.equal(loc.cache.entries[name].features, rnd.cache.entries[name].features)
            assert a[f"{name}/repulsion"] == b[f"{name}/repulsion"]
            assert a[f"{name}/mmd2"] == b[f"{name}/mmd2"]           # monitor: the window's refs
            assert a[f"ref/{name}/attraction_local"] == b[f"ref/{name}/attraction_local"]
            assert a[f"{name}/attraction"] != b[f"{name}/attraction"]


def test_B_resume_continues_the_random_stream(grouped_store):
    ran = RefAblationTrainer.from_trainer(make_trainer(grouped_store, grouped=True, lr=1e-2),
                                          ref_mode="random", ref_seed=2)
    ran.bootstrap()
    ran.step()
    ran.step()
    state = copy.deepcopy(ran.state_dict())
    resumed = RefAblationTrainer.from_trainer(make_trainer(grouped_store, grouped=True, lr=1e-2),
                                              ref_mode="random", ref_seed=2)
    restored = resumed.load_state_dict(state)
    assert restored["ref_sampler"] and restored["cache"]
    a, b = ran.step(), resumed.step()
    assert np.array_equal(ran._last_attraction_rows, resumed._last_attraction_rows)
    assert a["force"] == b["force"]
    with pytest.raises(RuntimeError, match="ref_ablation mismatch"):
        RefAblationTrainer.from_trainer(make_trainer(grouped_store, grouped=True),
                                        ref_mode="local").load_state_dict(state)
    plain = make_trainer(grouped_store, grouped=True)
    plain.bootstrap()
    with pytest.raises(RuntimeError, match="not written by the reference-ablation"):
        RefAblationTrainer.from_trainer(make_trainer(grouped_store, grouped=True),
                                        ref_mode="random").load_state_dict(plain.state_dict())


# =============================================================================================
# C. global mode
# =============================================================================================
def _cloud(num_prompts: int = 256, group: int = 4, scale: float = 0.25, seed: int = 0):
    """Joint features [img | beta * txt] with prompt structure, sigma = scale * median, beta = sigma."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((12, 6)) * 2.0
    proto = centers[rng.integers(12, size=num_prompts)] + 0.6 * rng.standard_normal((num_prompts, 6))
    img = np.repeat(proto, group, 0) + 0.35 * rng.standard_normal((num_prompts * group, 6))
    txt = rng.standard_normal((num_prompts, 3))
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    sigma = scale * float(median_bandwidth(torch.from_numpy(img).float()))
    pool = np.concatenate([img, sigma * np.repeat(txt, group, 0)], 1)
    return torch.from_numpy(pool).double(), sigma


def _as_nystrom(bundle: dict, name: str = "enc") -> GlobalNystrom:
    return GlobalNystrom({name: dict(bundle, encoder=name)})


def _errors(pool: torch.Tensor, sigma: float, bundle: dict, subset: int = 300) -> dict:
    """psi.psi vs k on a subset; psi.mu_bar vs the exact mean similarity over the whole pool."""
    nys = _as_nystrom(bundle)
    x = pool[torch.randperm(pool.shape[0], generator=torch.Generator().manual_seed(1))[:subset]]
    k = torch.exp(-gamma_from_sigma(sigma) * squared_distance(x, x))
    psi = nys.psi("enc", x)
    exact = torch.exp(-gamma_from_sigma(sigma) * squared_distance(x, pool)).mean(1)
    approx = nys.attraction_rows("enc", x)
    mu = bundle["mu_bar"]
    return {"kernel_abs_mean": float((psi @ psi.T - k).abs().mean()),
            "kernel_abs_max": float((psi @ psi.T - k).abs().max()),
            "mean_rel_of_mean": abs(float(approx.mean() - exact.mean())) / float(exact.mean()),
            "mean_rel_rows_median": float(((approx - exact).abs() / exact).median()),
            # ||mu - P_L mu||^2 in the RKHS: >= 0, and 0 when the landmarks span the data
            "rkhs_residual": float(torch.exp(-gamma_from_sigma(sigma) *
                                             squared_distance(pool, pool)).mean() - mu @ mu),
            "v_form_vs_psi": float((approx - psi @ mu).abs().max())}


def test_C_exact_when_the_landmarks_are_the_whole_reference():
    pool, sigma = _cloud(num_prompts=64)                                  # N = 256 = m
    bundle = nystrom_from_pool(pool, sigma, m=pool.shape[0], iters=5, seed=0, jitter=1e-12)
    err = _errors(pool, sigma, bundle, subset=256)
    assert err["kernel_abs_max"] < 1e-9
    assert err["mean_rel_of_mean"] < 1e-10 and err["mean_rel_rows_median"] < 1e-10
    assert abs(err["rkhs_residual"]) < 1e-10
    # for ANY query (not only reference rows): psi(x)^T mu_bar = (1/N) sum_n k(x, y_n)
    q = pool[:50] + 0.3 * torch.randn(50, pool.shape[1], generator=torch.Generator().manual_seed(2),
                                      dtype=F64)
    exact = torch.exp(-gamma_from_sigma(sigma) * squared_distance(q, pool)).mean(1)
    torch.testing.assert_close(_as_nystrom(bundle).attraction_rows("enc", q), exact,
                               rtol=1e-9, atol=1e-14)


def test_C_converges_as_landmarks_grow():
    """N = 1024 rows at the cold bandwidth sigma = 0.25 x median (as the store): every error falls
    as m grows; with m = N/2 the mean similarity is within 1%."""
    pool, sigma = _cloud(num_prompts=256)
    errs = {m: _errors(pool, sigma, nystrom_from_pool(pool, sigma, m=m, iters=20, seed=0))
            for m in (32, 128, 512)}
    for small, big in ((32, 128), (128, 512)):
        for key in ("kernel_abs_mean", "mean_rel_of_mean", "mean_rel_rows_median", "rkhs_residual"):
            assert errs[big][key] < errs[small][key], (key, small, big, errs)
    assert all(e["rkhs_residual"] > -1e-12 for e in errs.values())
    assert errs[512]["mean_rel_of_mean"] < 0.01 and errs[512]["mean_rel_rows_median"] < 0.01
    assert all(e["v_form_vs_psi"] < 1e-12 for e in errs.values())


def test_C_real_pipeline_on_a_store(grouped_store, toy_nystrom, tmp_path):
    """build_bundle + quality_report + GlobalNystrom on a store: the report agrees with direct
    computation, the bundle carries the store's sigma / beta, and another store is refused."""
    store = ReferenceFeatureStore(grouped_store, NAMES, require_context=False)
    for name in NAMES:
        b = toy_nystrom.bundles[name]
        assert b["sigma"] == store.sigma(name) and b["beta"] == store.beta(name)
        assert b["num_rows"] == store.num_rows and b["m"] == 64
        pool = load_joint_pool(store, name).double()
        report = quality_report(store, name, b, pool, prompts=16, seed=0)
        assert report["rows"] == 64 and report["kernel_pairs"]["same_prompt"]["pairs"] == 16 * 12
        me = report["mean_embedding"]
        assert me["v_form_max_abs_diff"] < 1e-12
        x = pool[:20]
        exact = torch.exp(-gamma_from_sigma(store.sigma(name)) * squared_distance(x, pool)).mean(1)
        approx = toy_nystrom.attraction_rows(name, x)
        assert float(((approx - exact).abs() / exact).median()) < 0.5   # 64 landmarks for 256 rows
    other = write_grouped_store(str(tmp_path / "other"), seed=5)
    with pytest.raises(ValueError, match="does not match"):
        check_against_store(toy_nystrom.bundles["enc_a"],
                            ReferenceFeatureStore(other, NAMES, require_context=False), "enc_a")
    with pytest.raises(ValueError, match="used as"):
        check_against_store(toy_nystrom.bundles["enc_a"], store, "enc_b")


def test_C_global_trainer_trains_on_the_nystrom_attraction(grouped_store, toy_nystrom):
    """The global-mode trainer's attraction is mean_i psi(x_i)^T mu_bar of its live rows, and the
    pre-training diagnostic reports all three attractions on the same samples."""
    tr = RefAblationTrainer.from_trainer(make_trainer(grouped_store, grouped=True),
                                         ref_mode="global", nystrom=toy_nystrom)
    seen = _spy_references(tr)
    tr.bootstrap()
    diag = tr.attraction_diagnostic(random_draws=4, exact_global=True)
    window = tr._pending_window
    logs = tr.step()
    assert all(ref is None for _, ref in seen)
    feats = tr.cache.entries                     # the step's live rows are the cache's newest B
    B = window.active_ids.size
    for name in NAMES:
        x = feats[name].features[-B:]
        expected = toy_nystrom.attraction(name, x)
        assert logs[f"{name}/attraction"] == pytest.approx(float(expected), rel=1e-12)
        d = diag["encoders"][name]
        assert d["global_nystrom"] == pytest.approx(float(expected), rel=1e-12)
        assert d["local"] == pytest.approx(logs[f"ref/{name}/attraction_local"], rel=1e-12)
        assert d["local_same_prompt"] + d["local_cross_prompt"] == pytest.approx(d["local"], rel=1e-12)
        exact = torch.exp(-gamma_from_sigma(tr.sigmas[name]) * squared_distance(
            x, tr.store.reference_joint_features(name, np.arange(tr.store.num_rows), dtype=F64))).mean()
        assert d["global_exact"] == pytest.approx(float(exact), rel=1e-10)
        assert -1.0 <= d["global_grad_cos"] <= 1.0


# =============================================================================================
# D. the repulsion is the same in every mode
# =============================================================================================
@pytest.mark.parametrize("group,weight", SIBLING_CASES)
def test_D_repulsion_identical_in_every_mode(toy_nystrom, group, weight):
    d = DIMS[0] + D_TXT
    active, context, reference, other, sigma = _force_inputs(group, seed=10 + group, d=d)
    sigma = toy_nystrom.bundles["enc_a"]["sigma"]
    refs = {"local": reference, "random": other, "global": None}
    out = {}
    for mode, ref in refs.items():
        loss = RefAblationLoss(mode, block_size=8, nystrom=toy_nystrom)
        force, stats, repel, attract = loss.active_force_parts("enc_a", active, context, ref, sigma,
                                                               group=group, sibling_weight=weight)
        g_rep, = torch.autograd.grad(repel, active, retain_graph=True)
        g_att, = torch.autograd.grad(attract, active, retain_graph=True)
        g_force, = torch.autograd.grad(force, active)
        torch.testing.assert_close(g_force, 2.0 * (g_rep - g_att), rtol=1e-12, atol=1e-15)
        out[mode] = (stats["active_repulsion"], g_rep, stats["active_attraction"])
    base, _ = ExactLocalMMD(block_size=8).active_force(active, context, reference, sigma,
                                                       group=group, sibling_weight=weight)
    for mode in ("random", "global"):
        assert torch.equal(out[mode][0], out["local"][0])                 # value, bit for bit
        assert torch.equal(out[mode][1], out["local"][1])                 # gradient, bit for bit
        assert not torch.equal(out[mode][2], out["local"][2])             # the attraction differs
    expected_repel = (base / 2.0) + out["local"][2]                       # force = 2 (rep - att)
    torch.testing.assert_close(out["local"][0], expected_repel, rtol=1e-12, atol=1e-15)


def test_D_config_refuses_unknown_keys_and_modes():
    from types import SimpleNamespace
    assert ref_ablation_from_config(SimpleNamespace(ref_ablation={"ref_mode": "random"})).ref_mode == "random"
    with pytest.raises(ValueError, match="unknown key"):
        ref_ablation_from_config(SimpleNamespace(ref_ablation={"ref_mod": "random"}))
    with pytest.raises(ValueError, match="ref_mode must be"):
        ref_ablation_from_config(SimpleNamespace(ref_ablation={"ref_mode": "glob"}))
    with pytest.raises(ValueError, match="nystrom_dir"):
        ref_ablation_from_config(SimpleNamespace(ref_ablation={"ref_mode": "global"}))


# =============================================================================================
# F. the copied training loop, end to end on a toy trainer
# =============================================================================================
def _loop_cfg(out_dir: str, steps: int, **kw):
    from types import SimpleNamespace
    cfg = dict(exp_name="toy", seed=0, steps=steps, output_dir=out_dir, save_freq=2,
               save_resume=True, resume_every=2, probe_every=2, window={"order_seed": 3407},
               cache={"refresh_every_windows": 3, "refresh_probe_rows": 4, "drift_threshold": 0.05},
               ref_ablation={"ref_mode": "random", "ref_seed": 3, "pre_diagnostic": True,
                             "diagnostic_random_draws": 2})
    cfg.update(kw)
    return SimpleNamespace(**cfg)


def test_F_train_loop_runs_and_resumes_exactly(grouped_store, toy_nystrom, tmp_path):
    import json

    from experiments.ref_ablation.train import train

    def trainer():
        return RefAblationTrainer.from_trainer(make_trainer(grouped_store, grouped=True, lr=1e-2,
                                                            grad_clip=1.0),
                                               ref_mode="random", ref_seed=3, nystrom=toy_nystrom)

    straight = train(_loop_cfg(str(tmp_path / "a"), 5), device="cpu", trainer=trainer())
    run_dir = tmp_path / "a" / "random-toy"
    records = [json.loads(line) for line in open(run_dir / "train_log.jsonl")]
    assert "pre_diagnostic" in records[0]
    diag = records[0]["pre_diagnostic"]["encoders"]["enc_a"]
    assert {"local", "random_mean", "random_step1", "global_nystrom", "global_exact"} <= set(diag)
    assert [r["step"] for r in records[1:]] == [1, 2, 3, 4, 5]
    assert all("ref/enc_b/grad_repulsion" in r for r in records[1:])
    assert (run_dir / "pre_diagnostic_step0000000.json").exists()
    assert sorted(p.name for p in run_dir.glob("step_*.pth")) == \
        ["step_0000002.pth", "step_0000004.pth", "step_0000005.pth"]

    first = train(_loop_cfg(str(tmp_path / "b"), 3), device="cpu", trainer=trainer())
    resumed = train(_loop_cfg(str(tmp_path / "b"), 5, resume_from=str(
        tmp_path / "b" / "random-toy" / "resume.pth")), device="cpu", trainer=trainer())
    assert first.step_idx == 3 and resumed.step_idx == 5
    for p, q in zip(straight._gen_params, resumed._gen_params):
        assert torch.equal(p, q)
    assert np.array_equal(straight._last_attraction_rows, resumed._last_attraction_rows)


# =============================================================================================
# E. world-size parity of the new modes (real gloo processes, as tests/test_sw_lmmd_parity.py)
# =============================================================================================
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_parity_worker.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_world(world: int, mode: str, store: str, bundles: str, out_dir: str, micro_batch: int = 1):
    out = os.path.join(out_dir, f"{mode}_w{world}_mb{micro_batch}.pt")
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port()),
               WORLD_SIZE=str(world), OMP_NUM_THREADS="1")
    cmd = [sys.executable, WORKER, "--store", store, "--bundles", bundles, "--mode", mode,
           "--out", out, "--micro-batch", str(micro_batch)]
    procs = [subprocess.Popen(cmd, env=dict(env, RANK=str(r)), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for r in range(world)]
    for rank, proc in enumerate(procs):
        output, _ = proc.communicate(timeout=600)
        if proc.returncode != 0:
            pytest.fail(f"{mode} world={world} rank={rank} rc={proc.returncode}\n{output}")
    return torch.load(out, weights_only=False)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
@pytest.mark.parametrize("mode", ["random", "global"])
def test_E_world_size_parity(grouped_store, tmp_path_factory, mode):
    bundles = build_toy_bundles(grouped_store, str(tmp_path_factory.mktemp(f"par_b_{mode}")), m=32)
    out_dir = str(tmp_path_factory.mktemp(f"par_o_{mode}"))
    one = _run_world(1, mode, grouped_store, bundles, out_dir)
    for world, mb in ((2, 1), (2, 2)):
        other = _run_world(world, mode, grouped_store, bundles, out_dir, micro_batch=mb)
        for g1, g2 in zip(one["grads"], other["grads"]):
            torch.testing.assert_close(g2, g1, rtol=1e-8, atol=1e-12)
        assert other["force"] == pytest.approx(one["force"], rel=1e-9, abs=1e-15)
        np.testing.assert_array_equal(other["rows"], one["rows"]) if mode == "random" else None
        for key, value in one["ref"].items():
            assert other["ref"][key] == pytest.approx(value, rel=1e-8, abs=1e-15), key
