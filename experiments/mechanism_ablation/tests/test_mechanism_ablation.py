"""Tests of the own-reference (mechanism) ablation: small tensors and toy stores, no model.

1. ``main`` is the method: the force equals RefAblationLoss(local) / ExactLocalMMD bit for bit, and
   a main-variant trainer reproduces the plain local trainer step for step.
2. ``nodiag`` equals a brute-force (1/(B(K-1))) sum_{j != i} attraction (value and gradient), and
   its repulsion is unchanged.
3. ``global_self`` equals the global force plus beta (2/B) sum_i [1 - k(x_i, y_i)].
4. In a trainer the own row is the active row's own reference row (same id, same caption, text
   block identical), and the logged split is consistent: |g_own| / |g_total|, the cosines, and
   the raw sums the calibration uses (sq_own scales as beta^2; solve_beta recovers a known beta).
5. Checkpoints refuse another arm / beta; the config refuses inconsistent blocks.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rdm.compare.kernels import gamma_from_sigma, gaussian_gram
from rdm.compare.median_heuristic import median_bandwidth
from sw_lmmd_fixtures import NAMES
from test_ref_ablation import F64, build_toy_bundles, make_trainer, write_grouped_store

from experiments.mechanism_ablation import MechanismConfig, MechanismLoss, MechanismTrainer
from experiments.mechanism_ablation.calibrate import solve_beta
from experiments.mechanism_ablation.config import mechanism_from_config
from experiments.ref_ablation import GlobalNystrom, RefAblationLoss, RefAblationTrainer
from rdm.sw_lmmd.reference_store import ReferenceFeatureStore


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return write_grouped_store(str(tmp_path_factory.mktemp("mech_store")))


@pytest.fixture(scope="module")
def nystrom(store, tmp_path_factory):
    out = build_toy_bundles(store, str(tmp_path_factory.mktemp("mech_bundles")))
    return GlobalNystrom.load(out, NAMES, store=ReferenceFeatureStore(store, NAMES, require_context=False))


def _inputs(seed=0, B=4, K=12, d=7):
    g = torch.Generator().manual_seed(seed)
    active = torch.randn(B, d, generator=g, dtype=F64).requires_grad_(True)
    context = torch.cat([torch.randn(K - B, d, generator=g, dtype=F64), active.detach()])
    reference = torch.randn(K, d, generator=g, dtype=F64) * 0.9 + 0.2
    sigma = float(median_bandwidth(torch.cat([context, reference])))
    cols = torch.arange(K - B, K)           # active rows are the window's last B
    return active, context, reference, sigma, cols


def _loss(variant, cols, own_rows=None, beta=0.0, nystrom=None):
    loss = MechanismLoss(variant, beta=beta, block_size=4, nystrom=nystrom)
    loss.own_columns = cols
    loss.own_reference = lambda name: own_rows
    return loss


def mech(variant, arm="t", beta=0.0):
    return MechanismConfig(arm=arm, skip_diagonal=variant == "nodiag", self_beta=beta,
                           ref_mode="global" if variant == "global_self" else "local",
                           expected_order_hash="")


def mech_trainer(store, variant, beta=0.0, nystrom=None, **kw):
    ref_mode = "global" if variant == "global_self" else "local"
    return MechanismTrainer.from_trainer_mech(make_trainer(store, **kw), mech(variant, beta=beta),
                                              ref_mode=ref_mode, nystrom=nystrom)


# ------------------------------------------------------------------------------------------ 1
def test_main_force_is_the_method():
    active, context, reference, sigma, cols = _inputs()
    f0, s0 = RefAblationLoss("local", block_size=4).active_force("e", active, context, reference, sigma)
    g0, = torch.autograd.grad(f0, active)
    loss = _loss("main", cols)
    f1, s1 = loss.active_force("e", active, context, reference, sigma)
    g1, = torch.autograd.grad(f1, active, retain_graph=True)
    assert torch.equal(f0, f1) and torch.equal(g0, g1)
    gam = gamma_from_sigma(sigma)
    diag = torch.exp(-gam * (active - reference[cols]).pow(2).sum(-1))
    own = loss.last_own["e"]
    assert float(own.detach()) == pytest.approx(float(-2.0 * diag.sum() / (4 * 12)), rel=1e-12)


def test_main_trainer_reproduces_the_local_trainer(store):
    kw = dict(lr=1e-2, grad_clip=1.0, gan={"weight": 1.5})
    old = RefAblationTrainer.from_trainer(make_trainer(store, **kw), ref_mode="local")
    new = mech_trainer(store, "main", **kw)
    old.bootstrap()
    new.bootstrap()
    for _ in range(3):
        a, b = old.step(), new.step()
        for key, value in a.items():
            assert b[key] == value, key
        assert 0.0 < b["mech/all/share"] <= 1.0 + 1e-12
        for p, q in zip(old._gen_params, new._gen_params):
            assert torch.equal(p, q)


# ------------------------------------------------------------------------------------------ 2
def test_nodiag_is_the_offdiagonal_attraction():
    active, context, reference, sigma, cols = _inputs(seed=1)
    B, K = active.shape[0], context.shape[0]
    gam = gamma_from_sigma(sigma)
    loss = _loss("nodiag", cols)
    f, stats, repel, attract = loss.active_force_parts("e", active, context, reference, sigma)
    g, = torch.autograd.grad(f, active)

    x = active.detach().clone().requires_grad_(True)
    gram = gaussian_gram(x, reference, gam)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[torch.arange(B), cols] = False
    att = gram[mask].sum() / (B * (K - 1))
    assert float(att) == pytest.approx(float(K / (K - 1) * (gram.sum() - gram[~mask].sum()) / (B * K)),
                                       rel=1e-12)
    rep = gaussian_gram(x, context, gam).mean()
    f_ref = 2.0 * (rep - att)
    g_ref, = torch.autograd.grad(f_ref, x)
    assert float(f) == pytest.approx(float(f_ref), rel=1e-12)
    assert torch.allclose(g, g_ref, rtol=1e-10, atol=1e-14)
    f_loc, _, rep_loc, _ = RefAblationLoss("local", block_size=4).active_force_parts(
        "e", active, context, reference, sigma)
    assert torch.equal(repel, rep_loc)                       # repulsion untouched
    assert float(stats["force"]) == float(f)


# ------------------------------------------------------------------------------------------ 3
def test_global_self_adds_the_own_term(nystrom):
    torch.manual_seed(0)
    d = nystrom.bundles["enc_a"]["landmarks"].shape[1]
    active = (torch.randn(4, d, dtype=F64) * 0.3).requires_grad_(True)
    own_rows = (active.detach() + 0.05 * torch.randn(4, d, dtype=F64))
    context = torch.cat([torch.randn(8, d, dtype=F64) * 0.3, active.detach()])
    sigma, beta = 0.7, 0.37
    f_g, _, _, _ = RefAblationLoss("global", block_size=4, nystrom=nystrom).active_force_parts(
        "enc_a", active, context, None, sigma)
    loss = _loss("global_self", None, own_rows=own_rows, beta=beta, nystrom=nystrom)
    f, stats, _, _ = loss.active_force_parts("enc_a", active, context, None, sigma)
    k = torch.exp(-gamma_from_sigma(sigma) * (active - own_rows).pow(2).sum(-1))
    expected = f_g + beta * 2.0 * (1.0 - k).mean()
    assert float(f) == pytest.approx(float(expected), rel=1e-12)
    g, = torch.autograd.grad(f, active)
    g_ref, = torch.autograd.grad(expected, active)
    assert torch.allclose(g, g_ref, rtol=1e-10, atol=1e-14)


# ------------------------------------------------------------------------------------------ 4
@pytest.mark.parametrize("variant", ["main", "nodiag", "global_self"])
def test_trainer_own_row_and_split(store, nystrom, variant):
    tr = mech_trainer(store, variant, beta=0.05 if variant == "global_self" else 0.0,
                      nystrom=nystrom)
    tr.bootstrap()
    window = tr._pending_window or tr.schedule.peek()
    logs = tr.step()
    assert np.array_equal(tr._own_ids, window.active_ids)              # one rank: all active rows
    cols = tr.ref_loss.own_columns.numpy()
    assert np.array_equal(np.asarray(window.all_ids)[cols], window.active_ids)
    B = window.active_ids.size
    for name in NAMES:
        x = tr.cache.entries[name].features[-B:]
        y = tr.store.reference_joint_features(name, window.active_ids, dtype=F64)
        d = tr.image_dims[name]
        assert torch.equal(x[:, d:], y[:, d:])                          # same caption: text block equal
        k = torch.exp(-gamma_from_sigma(tr.sigmas[name]) * (x - y).pow(2).sum(-1))
        assert logs[f"mech/{name}/own_kernel"] == pytest.approx(float(k.mean()), rel=1e-10)
        if variant == "global_self":
            assert logs[f"mech/{name}/own_text_gap"] == 0.0
    for s in list(NAMES) + ["all"]:
        p = f"mech/{s}/"
        o, t, r = logs[p + "grad_own"], logs[p + "grad_total"], logs[p + "grad_rest"]
        assert logs[p + "share"] == pytest.approx(o / t, rel=1e-9)
        c1, c2 = logs[p + "cos_own_total"], logs[p + "cos_rest_total"]
        assert -1 - 1e-9 <= c1 <= 1 + 1e-9 and -1 - 1e-9 <= c2 <= 1 + 1e-9
        assert r ** 2 == pytest.approx(o * o + t * t - 2 * o * t * c1, rel=1e-6, abs=1e-30)
        assert o * c1 + r * c2 == pytest.approx(t, rel=1e-6)            # g_own + g_rest = g_total
    if variant == "nodiag":     # the removed term is not trained: the total is the off-diagonal force
        f = sum(0.5 * logs[f"{n}/force"] for n in NAMES)
        assert logs["force"] == pytest.approx(f, rel=1e-12)


def test_global_self_raw_sums_and_solver(store, nystrom):
    """Same state, two betas: sq_own scales as beta^2, and solve_beta recovers the second beta
    from the first run's sums and the second run's share."""
    logs = {}
    for beta in (0.02, 0.11):
        tr = mech_trainer(store, "global_self", beta=beta, nystrom=nystrom)
        tr.bootstrap()
        logs[beta] = tr.step()
    a, b = logs[0.02], logs[0.11]
    assert b["mech/all/sq_own"] / a["mech/all/sq_own"] == pytest.approx((0.11 / 0.02) ** 2, rel=1e-9)
    assert solve_beta([a], 0.02, b["mech/all/share"]) == pytest.approx(0.11, rel=1e-6)


# ------------------------------------------------------------------------------------------ 5
def test_checkpoint_refuses_another_arm(store, nystrom):
    tr = mech_trainer(store, "global_self", beta=0.05, nystrom=nystrom)
    tr.bootstrap()
    tr.step()
    state = tr.state_dict()
    other = mech_trainer(store, "global_self", beta=0.2, nystrom=nystrom)
    with pytest.raises(RuntimeError, match="mechanism mismatch"):
        other.load_state_dict(state)
    same = mech_trainer(store, "global_self", beta=0.05, nystrom=nystrom)
    assert same.load_state_dict(state)["step"] == 1
    plain = make_trainer(store).state_dict()
    with pytest.raises(RuntimeError, match="not a mechanism-ablation checkpoint"):
        same.load_state_dict(plain)
    assert mech_trainer(store, "main").load_plain_state(plain)["step"] == 0


def test_config_validation(store):
    def cfg(att=None, own=None, ref_mode="local", name="a"):
        c = SimpleNamespace(exp_name=name, ref_ablation={"ref_mode": ref_mode, "nystrom_dir": "/x"})
        if att is not None:
            c.attraction = att
        if own is not None:
            c.own_term = own
        return mechanism_from_config(c)
    assert cfg().variant == "main"                                   # no blocks: the method
    assert cfg({"skip_diagonal": False}, {"beta": 0.0}).variant == "main"
    assert cfg({"skip_diagonal": True}).variant == "nodiag"
    assert cfg(own={"beta": 0.0}, ref_mode="global").variant == "global_self"
    assert cfg(own={"beta": 0.01}, ref_mode="global").self_beta == 0.01
    with pytest.raises(ValueError, match="unknown"):
        cfg({"skip_diag": True})
    with pytest.raises(ValueError, match="unknown"):
        cfg(own={"weight": 1.0})
    with pytest.raises(ValueError, match="true or false"):
        cfg({"skip_diagonal": "yes"})
    with pytest.raises(ValueError, match="needs ref_ablation.ref_mode: local"):
        cfg({"skip_diagonal": True}, ref_mode="global")
    with pytest.raises(ValueError, match="needs ref_ablation.ref_mode: global"):
        cfg(own={"beta": 0.1})
    with pytest.raises(ValueError, match=">= 0"):
        cfg(own={"beta": -1.0}, ref_mode="global")
    with pytest.raises(ValueError, match="exp_name"):
        cfg(name="")
    with pytest.raises(ValueError, match="ungrouped"):
        MechanismTrainer.from_trainer_mech(make_trainer(store, grouped=True), mech("main"),
                                           ref_mode="local")


# ------------------------------------------------------------------------------------------ 6
def test_train_loop_names_the_run_and_resumes_exactly(store, nystrom, tmp_path):
    """The ref-ablation loop with a MechanismTrainer: run dir = mechanism.arm, the own-term logs in
    every step, and a resume from resume.pth continues bit for bit."""
    import json

    import experiments.mechanism_ablation.train  # noqa: F401  (swaps run_name in the loop)
    from experiments.ref_ablation.train import train

    def cfg(out, steps, **kw):
        c = dict(exp_name="global-plus-self-b1", seed=0, steps=steps, output_dir=str(out), save_freq=2,
                 save_resume=True, resume_every=2, probe_every=2, window={"order_seed": 3407},
                 cache={"refresh_every_windows": 3, "refresh_probe_rows": 4, "drift_threshold": 0.05},
                 ref_ablation={"ref_mode": "global", "pre_diagnostic": False, "nystrom_dir": "/x"})
        c.update(kw)
        return SimpleNamespace(**c)

    def trainer():
        return mech_trainer(store, "global_self", beta=0.05, nystrom=nystrom, lr=1e-2, grad_clip=1.0,
                            gan={"weight": 1.5})

    straight = train(cfg(tmp_path / "a", 5), device="cpu", trainer=trainer())
    run_dir = tmp_path / "a" / "global-plus-self-b1"
    records = [json.loads(line) for line in open(run_dir / "train_log.jsonl")]
    assert [r["step"] for r in records] == [1, 2, 3, 4, 5]
    assert all("mech/all/share" in r and "mech/enc_a/own_term" in r for r in records)
    first = train(cfg(tmp_path / "b", 3), device="cpu", trainer=trainer())
    resumed = train(cfg(tmp_path / "b", 5, resume_from=str(tmp_path / "b" / "global-plus-self-b1" /
                                                           "resume.pth")), device="cpu", trainer=trainer())
    assert first.step_idx == 3 and resumed.step_idx == 5
    for p, q in zip(straight._gen_params, resumed._gen_params):
        assert torch.equal(p, q)


# ------------------------------------------------------------------------------------------ 7
def test_flags_off_reproduce_the_global_arm(store, nystrom):
    """own_term.beta = 0 on ref_mode global is the Table 2 Global arm, step for step."""
    kw = dict(lr=1e-2, grad_clip=1.0, gan={"weight": 1.5})
    old = RefAblationTrainer.from_trainer(make_trainer(store, **kw), ref_mode="global", nystrom=nystrom)
    new = mech_trainer(store, "global_self", beta=0.0, nystrom=nystrom, **kw)
    old.bootstrap()
    new.bootstrap()
    for _ in range(3):
        a, b = old.step(), new.step()
        for key, value in a.items():
            assert b[key] == value, key
        assert b["mech/all/grad_own"] == 0.0
        for p, q in zip(old._gen_params, new._gen_params):
            assert torch.equal(p, q)


@pytest.mark.parametrize("flags", [{"attraction": {"skip_diagonal": True}},
                                   {"own_term": {"beta": 0.05}, "ref": "global"}])
def test_each_flag_trains_20_steps_on_a_small_window(store, nystrom, tmp_path, flags):
    """Config flags -> variant -> the ref-ablation training loop, 20 steps on a K=16 / B=8 window with the
    critic on: finite, every step logs the own term, and the weights move."""
    import json

    import experiments.mechanism_ablation.train  # noqa: F401
    from experiments.ref_ablation.train import train

    ref_mode = flags.pop("ref", "local")
    cfg = SimpleNamespace(exp_name="flag-test", seed=0, steps=20, output_dir=str(tmp_path), save_freq=10,
                          save_resume=True, resume_every=10, probe_every=5, window={"order_seed": 3407},
                          cache={"refresh_every_windows": 7, "refresh_probe_rows": 4, "drift_threshold": 0.05},
                          ref_ablation={"ref_mode": ref_mode, "pre_diagnostic": False, "nystrom_dir": "/x"},
                          expected_row_order_hash="", **flags)
    m = mechanism_from_config(cfg)
    tr = MechanismTrainer.from_trainer_mech(make_trainer(store, lr=1e-2, grad_clip=1.0, gan={"weight": 1.5}),
                                            m, ref_mode=ref_mode, nystrom=nystrom)
    before = [p.detach().clone() for p in tr._gen_params]
    train(cfg, device="cpu", trainer=tr)
    rows = [json.loads(l) for l in open(tmp_path / "flag-test" / "train_log.jsonl")]
    assert [r["step"] for r in rows] == list(range(1, 21))
    assert not any(r["skipped"] for r in rows)
    assert all(np.isfinite(r["force"]) and np.isfinite(r["mech/all/share"]) for r in rows)
    assert all("gan/lambda" in r for r in rows)
    if m.variant == "global_self":
        assert all(r["mech/all/grad_own"] > 0 for r in rows)
    assert any(not torch.equal(p, q) for p, q in zip(before, tr._gen_params))
    assert (tmp_path / "flag-test" / "step_0000020.pth").exists()
