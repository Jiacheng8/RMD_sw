"""The optional adversarial term: critic math, its place inside GradCache, balance, configs.

CPU, float64 toy fixtures. World-size parity and resume with the critic live with their
siblings in ``test_sw_lmmd_parity.py`` and ``test_sw_lmmd_resume.py``.
"""
import dataclasses
import io
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rdm.sw_lmmd import GANConfig, MemoryPolicy, ReferenceFeatureStore, build_critic
from rdm.sw_lmmd.adversarial import adversarial_loss, critic_loss
from sw_lmmd_fixtures import DIMS, NAMES, build_toy_store, build_trainer

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
F64 = torch.float64


@pytest.fixture(scope="module")
def store_root(tmp_path_factory):
    return build_toy_store(str(tmp_path_factory.mktemp("sw_gan_store")), num_rows=64)


@pytest.fixture(scope="module")
def store(store_root):
    return ReferenceFeatureStore(store_root, NAMES)


def _critic(store, **kw):
    return build_critic(GANConfig(**{"enabled": True, "hidden": 32, **kw}), store, list(NAMES),
                        dtype=F64)


# --------------------------------------------------------------------------------------
# losses and the critic on its own
# --------------------------------------------------------------------------------------
def test_losses_match_their_definitions():
    real, fake = torch.tensor([2.0, 0.5]), torch.tensor([-2.0, 0.5])
    assert float(critic_loss(real, fake, "hinge")) == pytest.approx(0.25 + 0.75)
    assert float(adversarial_loss(fake, "hinge")) == pytest.approx(0.75)
    sp = torch.nn.functional.softplus
    assert float(critic_loss(real, fake, "ns")) == pytest.approx(
        float(sp(-real).mean() + sp(fake).mean()))
    assert float(adversarial_loss(fake, "ns")) == pytest.approx(float(sp(-fake).mean()))
    with pytest.raises(ValueError, match="unknown"):
        critic_loss(real, fake, "wgan")


def test_critic_input_is_standardized_by_the_reference(store):
    """stats_rows >= num_rows uses every row, so the reference standardizes to mean 0, RMS 1."""
    critic = _critic(store, stats_rows=10_000).critic
    rows = np.arange(store.num_rows)
    phi = {n: store.reference_image_features(n, rows, dtype=F64) for n in NAMES}
    x, t = critic.standardize(phi, store.text_features(rows, dtype=F64))
    for v in (*x.values(), t):
        torch.testing.assert_close(v.mean(0), torch.zeros(v.shape[1], dtype=F64),
                                   atol=1e-12, rtol=0)
        assert float(v.pow(2).mean().sqrt()) == pytest.approx(1.0, rel=1e-9)


def test_critic_step_learns_to_separate(store):
    gan = _critic(store, lr=1e-3)
    rows = np.arange(32)
    real = {n: store.reference_image_features(n, rows, dtype=F64) for n in NAMES}
    # unit-variance toy features: a shift of 3 per dim is separable (Bayes accuracy > 0.999)
    fake = {n: v + 3.0 for n, v in real.items()}
    tau = store.text_features(rows, dtype=F64)
    first = gan.update(fake, real, tau)
    for _ in range(199):
        logs = gan.update(fake, real, tau)
    assert first["gan/d_acc"] < 0.9
    assert logs["gan/d_acc"] > 0.95 and logs["gan/d_real"] > logs["gan/d_fake"]
    assert not logs["gan/d_skipped"] and gan.updates == 200
    assert all(p.grad is None for p in gan.critic.parameters())   # nothing left for the generator


def test_r1_penalty_is_reported_and_finite(store):
    gan = _critic(store, r1_gamma=1.0)
    rows = np.arange(16)
    real = {n: store.reference_image_features(n, rows, dtype=F64) for n in NAMES}
    logs = gan.update({n: v * 1.5 for n, v in real.items()}, real,
                      store.text_features(rows, dtype=F64))
    assert logs["gan/r1"] > 0 and np.isfinite(logs["gan/d_loss"])


def test_generator_loss_reaches_image_features_not_the_critic(store):
    gan = _critic(store)
    rows = np.arange(8)
    feats = {n: store.reference_joint_features(n, rows, dtype=F64).requires_grad_(True)
             for n in NAMES}
    loss, _ = gan.generator_loss(feats, store.text_features(rows, dtype=F64))
    loss.backward()
    assert all(p.grad is None for p in gan.critic.parameters())
    assert all(p.requires_grad for p in gan.critic.parameters())  # unfrozen again afterwards
    for name, d in zip(NAMES, DIMS):
        g = feats[name].grad
        assert float(g[:, :d].abs().sum()) > 0                     # phi(x) is pushed...
        assert torch.all(g[:, d:] == 0)                            # ...the beta*tau block is not


def test_critic_state_round_trip(store):
    rows = np.arange(16)
    real = {n: store.reference_image_features(n, rows, dtype=F64) for n in NAMES}
    fake = {n: v * 0.5 for n, v in real.items()}
    tau = store.text_features(rows, dtype=F64)

    a = _critic(store)
    for _ in range(3):
        a.update(fake, real, tau)
    b = _critic(store, seed=1)                   # different init AND standardization sample
    # Through bytes, as a resume does: Optimizer.load_state_dict keeps same-dtype tensors by
    # reference, so an in-memory hand-over would share a's Adam moments with b.
    buf = io.BytesIO()
    torch.save(a.state_dict(), buf)
    buf.seek(0)
    b.load_state_dict(torch.load(buf, weights_only=False))
    for (ka, va), (kb, vb) in zip(a.critic.state_dict().items(), b.critic.state_dict().items()):
        assert ka == kb and torch.equal(va, vb), ka          # incl. spectral-norm u/v, stats
    a.update(fake, real, tau)
    b.update(fake, real, tau)
    for pa, pb in zip(a.critic.parameters(), b.critic.parameters()):
        assert torch.equal(pa, pb)


def test_critic_rejects_a_non_training_encoder(store):
    with pytest.raises(ValueError, match="not training encoders"):
        _critic(store, encoders=["enc_a", "dinov3_l"])


def test_marginal_objective_gets_an_unconditional_critic(store):
    gan = build_critic(GANConfig(enabled=True, hidden=8), store, list(NAMES), joint=False,
                       dtype=F64)
    assert not gan.conditional
    rows = np.arange(4)
    feats = {n: store.reference_image_features(n, rows, dtype=F64) for n in NAMES}
    loss, _ = gan.generator_loss(feats)
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------------------
# inside the trainer
# --------------------------------------------------------------------------------------
def test_warmup_leaves_the_generator_gradient_untouched(store_root):
    """Before g_start_step the critic trains, and the generator sees exactly the force."""
    plain = build_trainer(store_root, window=16, stride=4)
    plain.bootstrap()
    want = plain.step()
    warm = build_trainer(store_root, window=16, stride=4, gan={"g_start_step": 10 ** 6})
    warm.bootstrap()
    got = warm.step()
    assert "gan/d_acc" in got and "gan/lambda" not in got and warm.critic.updates == 1
    assert got["force"] == want["force"] and got["loss_total"] == want["force"]
    for a, b in zip(plain._gen_params, warm._gen_params):
        assert torch.equal(a.grad, b.grad)


def test_gradcache_matches_the_full_graph_with_the_adversarial_term(store_root):
    """GradCache (4 micro-batches, critic step in the middle) == one graph through everything.

    The critic's lr is 0 and spectral norm is off, so its step leaves it unchanged and the
    oracle can score the same critic.
    """
    trainer = build_trainer(store_root, window=16, stride=4, micro_batch=1,
                            gan={"weight": 0.5, "lr": 0.0, "spectral_norm": False})
    trainer.bootstrap()
    window = trainer._pending_window

    feats = trainer._encode(window.active_ids)                    # one live graph, B rows
    force = None
    for name in NAMES:
        context = torch.cat([trainer.cache.retained(name, window.retained_ids),
                             feats[name].detach()])
        reference = trainer._reference_window(name, window.all_ids)
        f, _ = trainer.mmd.active_force(feats[name], context, reference, trainer.sigmas[name])
        term = trainer.encoder_weights[name] * f
        force = term if force is None else force + term
    tau = trainer.store.text_features(window.active_ids, dtype=F64)
    adv, _ = trainer.critic.generator_loss(feats, tau)
    lam, _ = trainer.critic.balance(force, adv, feats, trainer.image_dims)
    (force + lam * adv).backward()
    want = [p.grad.clone() for p in trainer._gen_params]
    trainer.optimizer.zero_grad(set_to_none=True)

    logs = trainer.step()
    assert logs["gan/lambda"] == pytest.approx(lam, rel=1e-10)
    assert logs["force"] == pytest.approx(float(force.detach()), rel=1e-10)
    assert logs["loss_total"] == pytest.approx(float((force + lam * adv).detach()),
                                                rel=1e-10)
    for got, ref in zip(trainer._gen_params, want):
        torch.testing.assert_close(got.grad, ref, rtol=1e-9, atol=1e-12)


def test_adversarial_gradient_is_micro_batch_invariant(store_root):
    grads = []
    for micro_batch in (1, 2, 4):
        trainer = build_trainer(store_root, window=16, stride=4, micro_batch=micro_batch,
                                gan={"weight": 0.5})
        trainer.bootstrap()
        trainer.step()
        grads.append([p.grad.detach().clone() for p in trainer._gen_params])
    for other in grads[1:]:
        for a, b in zip(grads[0], other):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("weight", [0.1, 1.0])
def test_adaptive_weight_sets_the_feature_gradient_ratio(store_root, weight):
    trainer = build_trainer(store_root, window=16, stride=4, gan={"weight": weight})
    trainer.bootstrap()
    logs = trainer.step()
    assert logs["gan/grad_ratio"] == pytest.approx(weight, rel=1e-9)
    assert logs["gan/lambda"] == pytest.approx(weight / logs["gan/grad_ratio_raw"], rel=1e-9)


def test_fixed_weight_is_used_as_given(store_root):
    trainer = build_trainer(store_root, window=16, stride=4,
                            gan={"weight": 0.3, "adaptive": False})
    trainer.bootstrap()
    logs = trainer.step()
    assert logs["gan/lambda"] == 0.3 and "gan/grad_ratio" not in logs
    assert logs["loss_total"] == pytest.approx(logs["force"] + 0.3 * logs["gan/g_loss"])


def test_training_with_the_critic_moves_both_players(store_root):
    trainer = build_trainer(store_root, window=16, stride=4, lr=1e-2, gan={"weight": 0.5})
    gen0 = [p.detach().clone() for p in trainer._gen_params]
    critic0 = [p.detach().clone() for p in trainer.critic.critic.parameters()]
    seen = []
    trainer.train(4, log_fn=seen.append)
    assert all(not r["skipped"] and not r["gan/d_skipped"] for r in seen)
    assert any(not torch.equal(a, b) for a, b in zip(gen0, trainer._gen_params))
    assert any(not torch.equal(a, b) for a, b in zip(critic0, trainer.critic.critic.parameters()))
    assert trainer.critic.updates == 4


def test_critic_needs_a_window_divisible_by_world_size(store_root, monkeypatch):
    import rdm.sw_lmmd.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "get_world_size", lambda: 3)
    with pytest.raises(ValueError, match="window size 16"):
        build_trainer(store_root, window=16, stride=4, gan={})


def test_state_dict_carries_the_critic_only_when_asked(store_root):
    trainer = build_trainer(store_root, window=16, stride=4, gan={})
    trainer.bootstrap()
    trainer.step()
    assert "critic" in trainer.state_dict()
    assert "critic" not in trainer.state_dict(with_critic=False)
    assert "critic" not in build_trainer(store_root).state_dict()


# --------------------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------------------
def test_gan_is_off_in_the_method_config_and_every_training_config():
    from rdm.sw_lmmd.launch import gan_from_config
    from rdm.train.launch import load_config

    gan = gan_from_config(load_config(os.path.join(CONFIGS, "sw_lmmd_flux.yaml")))
    assert not gan.enabled and gan.adaptive and gan.loss == "hinge"
    assert gan.lr == pytest.approx(2e-4) and gan.betas == (0.0, 0.99)
    for name in ("sw_lmmd_train_4x4090.yaml", "sw_lmmd_train_h100_2gpu.yaml"):
        assert gan_from_config(load_config(os.path.join(CONFIGS, name))) == gan


def test_gan_config_differs_from_its_base_only_in_gan_and_the_battery():
    from rdm.sw_lmmd.launch import (gan_from_config, memory_from_config, resolve_batching,
                                    window_from_config)
    from rdm.train.launch import load_config

    base = load_config(os.path.join(CONFIGS, "sw_lmmd_train_h100_2gpu.yaml"))
    cfg = load_config(os.path.join(CONFIGS, "sw_lmmd_train_h100_2gpu_gan.yaml"))
    gan = gan_from_config(cfg)
    assert gan.enabled
    assert dataclasses.replace(gan, enabled=False) == gan_from_config(base)
    policy = memory_from_config(cfg)
    assert not policy.battery_bf16
    assert dataclasses.replace(policy, battery_bf16=True) == memory_from_config(base)
    assert window_from_config(cfg) == window_from_config(base)
    for key in ("method", "encoders", "lr", "grad_clip", "steps", "joint", "loss", "cache",
                "reference_root", "output_dir", "save_resume", "resume_every"):
        assert getattr(cfg, key) == getattr(base, key), key
    assert cfg.exp_name != base.exp_name                       # never resumes into the base run
    assert resolve_batching(cfg, policy, window_from_config(cfg), world_size=2)["grad_accum"] == 2


def test_gan_block_rejects_unknown_keys_and_bad_values():
    from rdm.sw_lmmd.launch import gan_from_config

    with pytest.raises(ValueError, match="unknown key"):
        gan_from_config(SimpleNamespace(gan={"enabled": True, "wieght": 1.0}))
    with pytest.raises(ValueError, match="hinge"):
        gan_from_config(SimpleNamespace(gan={"loss": "wgan"}))
    with pytest.raises(ValueError, match="non-negative"):
        GANConfig(weight=-1.0).validate()
    assert not gan_from_config(SimpleNamespace()).enabled        # no block at all -> off


def test_set_override_turns_the_critic_on():
    from rdm.sw_lmmd.launch import apply_override, gan_from_config
    from rdm.train.launch import load_config

    cfg = load_config(os.path.join(CONFIGS, "sw_lmmd_train_4x4090.yaml"))
    apply_override(cfg, "gan.enabled=true")
    apply_override(cfg, "gan.weight=0.5")
    gan = gan_from_config(cfg)
    assert gan.enabled and gan.weight == 0.5
    with pytest.raises(SystemExit, match="no key"):
        apply_override(cfg, "gan.wieght=1.0")


def test_gan_with_a_bf16_battery_warns(caplog):
    from rdm.sw_lmmd.launch import warn_gan_pipeline

    with caplog.at_level("WARNING", logger="rdm"):
        warn_gan_pipeline(GANConfig(enabled=True), MemoryPolicy(battery_bf16=True))
    assert "battery_bf16" in caplog.text and "53-56% held out" in caplog.text   # no stray %%
    caplog.clear()
    with caplog.at_level("WARNING", logger="rdm"):
        warn_gan_pipeline(GANConfig(enabled=True), MemoryPolicy(battery_bf16=False))
        warn_gan_pipeline(GANConfig(enabled=False), MemoryPolicy(battery_bf16=True))
    assert not caplog.text
