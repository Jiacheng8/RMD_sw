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


# --------------------------------------------------------------------------------------
# block critic (gan.critic: block): one sample = one step's B fresh rows
# --------------------------------------------------------------------------------------
_BLOCK = {"critic": "block", "block_width": 16, "block_heads": 2, "block_layers": 1}


def _block_critic(store, block=4, joint=True, **kw):
    return build_critic(GANConfig(**{"enabled": True, "hidden": 16, **_BLOCK, **kw}), store,
                        list(NAMES), joint=joint, dtype=F64, block=block)


def test_block_critic_scores_one_logit_per_block_and_ignores_row_order(store):
    gan = _block_critic(store, block=4)
    rows = np.arange(16)
    phi = {n: store.reference_image_features(n, rows, dtype=F64) for n in NAMES}
    tau = store.text_features(rows, dtype=F64)
    blocks = {n: gan._samples(v) for n, v in phi.items()}
    logits = gan.critic(blocks, gan._samples(tau))
    assert all(v.shape == (4,) for v in logits.values())
    perm = torch.randperm(4, generator=torch.Generator().manual_seed(0))   # same within every block
    shuffled = gan.critic({n: v[:, perm] for n, v in blocks.items()}, gan._samples(tau)[:, perm])
    for n in NAMES:
        torch.testing.assert_close(shuffled[n], logits[n], rtol=1e-10, atol=1e-12)


def test_block_critic_sees_collapsed_seeds_a_row_critic_cannot(store):
    """Real blocks: 4 independent rows. Fake blocks: one row repeated 4 times. Every single row
    has the same distribution either way, so a row critic stays at chance; the block critic
    sees the rows of a block agree and separates them."""
    g = torch.Generator().manual_seed(0)

    def draw():
        real = {n: torch.randn(64, d, generator=g, dtype=F64) for n, d in zip(NAMES, DIMS)}
        fake = {n: torch.randn(16, 1, d, generator=g, dtype=F64).expand(16, 4, d).reshape(64, d)
                for n, d in zip(NAMES, DIMS)}
        return fake, real

    def train(gan):
        accs = []
        for _ in range(400):
            fake, real = draw()
            accs.append(gan.update(fake, real)["gan/d_acc"])
        return float(np.mean(accs[-100:]))

    block = train(_block_critic(store, block=4, joint=False, lr=1e-3, block_width=32))
    row = train(build_critic(GANConfig(enabled=True, hidden=32, lr=1e-3), store, list(NAMES),
                             joint=False, dtype=F64))
    assert block > 0.85, block
    assert row < 0.6, row


def test_block_critic_needs_its_block_size(store):
    with pytest.raises(ValueError, match="block = the window stride"):
        build_critic(GANConfig(enabled=True, **_BLOCK), store, list(NAMES), dtype=F64)
    gan = _block_critic(store, block=4)
    with pytest.raises(ValueError, match="whole blocks of 4"):
        gan.update({n: torch.zeros(6, d, dtype=F64) for n, d in zip(NAMES, DIMS)},
                   {n: torch.zeros(6, d, dtype=F64) for n, d in zip(NAMES, DIMS)},
                   store.text_features(np.arange(6), dtype=F64))


def test_block_critic_state_round_trip(store):
    a, b = _block_critic(store, seed=1), _block_critic(store, seed=2)
    rows = np.arange(8)
    phi = {n: store.reference_image_features(n, rows, dtype=F64) for n in NAMES}
    tau = store.text_features(rows, dtype=F64)
    a.update({n: v + 1.0 for n, v in phi.items()}, phi, tau)
    buf = io.BytesIO()
    torch.save(a.state_dict(), buf)
    buf.seek(0)
    b.load_state_dict(torch.load(buf, weights_only=False))
    want, got = a.generator_loss(phi, tau)[0], b.generator_loss(phi, tau)[0]
    assert float(got) == float(want) and b.updates == a.updates == 1


def test_gradcache_matches_the_full_graph_with_the_block_critic(store_root):
    """GradCache (4 micro-batches, critic step in the middle) == one graph through everything,
    with the generator term scoring the step's whole block of B rows."""
    trainer = build_trainer(store_root, window=16, stride=4, micro_batch=1,
                            gan={**_BLOCK, "weight": 0.5, "lr": 0.0, "spectral_norm": False})
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
    assert logs["loss_total"] == pytest.approx(float((force + lam * adv).detach()), rel=1e-10)
    for got, ref in zip(trainer._gen_params, want):
        torch.testing.assert_close(got.grad, ref, rtol=1e-9, atol=1e-12)


def test_block_critic_gradient_is_micro_batch_invariant(store_root):
    grads = []
    for micro_batch in (1, 2, 4):
        trainer = build_trainer(store_root, window=16, stride=4, micro_batch=micro_batch,
                                gan={**_BLOCK, "weight": 0.5})
        trainer.bootstrap()
        logs = trainer.step()
        assert "gan/lambda" in logs and not logs["gan/d_skipped"]
        grads.append([p.grad.detach().clone() for p in trainer._gen_params])
    for other in grads[1:]:
        for a, b in zip(grads[0], other):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)


def test_block_critic_needs_whole_blocks_per_rank(store_root, monkeypatch):
    with pytest.raises(ValueError, match="whole blocks of B=4"):
        build_trainer(store_root, window=18, stride=4, gan=_BLOCK)          # K not a multiple of B
    import rdm.sw_lmmd.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "get_world_size", lambda: 8)
    with pytest.raises(ValueError, match="split evenly over world_size=8"):
        build_trainer(store_root, window=16, stride=4, gan=_BLOCK)          # 4 blocks, 8 ranks


def test_gan_critic_kind_is_validated():
    with pytest.raises(ValueError, match="'row' or 'block'"):
        GANConfig(critic="set").validate()
    with pytest.raises(ValueError, match="divide block_width"):
        GANConfig(critic="block", block_width=30, block_heads=4).validate()
    assert GANConfig().critic == "row"                       # the default stays the row critic


def test_block_critic_config_differs_from_the_grouped_run_only_in_the_critic():
    """Experiment 9 = K128 · GAN · grouped with gan.critic: block -- a clean A/B on the critic,
    on the default K=128 / B=32 window and the COCO-only reference."""
    from rdm.sw_lmmd.launch import gan_from_config, memory_from_config, window_from_config
    from rdm.train.launch import load_config

    base = load_config(os.path.join(CONFIGS, "sw_lmmd_train_h100_2gpu_gan_grouped.yaml"))
    cfg = load_config(os.path.join(CONFIGS, "sw_lmmd_train_h100_2gpu_gan_grouped_blockcritic.yaml"))
    gan = gan_from_config(cfg)
    assert gan.enabled and gan.critic == "block" and gan_from_config(base).critic == "row"
    assert gan.critic_rows == "active"                    # the step's fresh block trains the critic
    assert dataclasses.replace(gan, critic="row", critic_rows="window") == gan_from_config(base)
    window = window_from_config(cfg)
    assert window == window_from_config(base)
    assert (window.size, window.stride, window.group_by_prompt) == (128, 32, True)
    assert (window.size // window.stride) % 2 == 0                # 4 blocks: 1, 2 or 4 ranks
    assert memory_from_config(cfg) == memory_from_config(base)
    for key in ("method", "encoders", "lr", "steps", "loss", "reference_root", "output_dir"):
        assert getattr(cfg, key) == getattr(base, key), key
    assert not getattr(cfg, "reference_extension", None)          # COCO only, no GenEval block
    assert cfg.exp_name != base.exp_name


def test_ungrouped_block_critic_config_differs_from_the_row_critic_run_only_in_the_critic():
    """Experiment 11 = K128 · GAN (ungrouped) with gan.critic: block; micro_batch 4 is memory
    only (the gradient is micro-batch invariant)."""
    from rdm.sw_lmmd.launch import gan_from_config, memory_from_config, window_from_config
    from rdm.train.launch import load_config

    base = load_config(os.path.join(CONFIGS, "sw_lmmd_train_h100_2gpu_gan.yaml"))
    cfg = load_config(os.path.join(CONFIGS, "sw_lmmd_train_h100_2gpu_gan_blockcritic.yaml"))
    gan = gan_from_config(cfg)
    assert gan.critic == "block" and gan.critic_rows == "active"
    assert dataclasses.replace(gan, critic="row", critic_rows="window") == gan_from_config(base)
    window = window_from_config(cfg)
    assert window == window_from_config(base) and not window.group_by_prompt
    assert (window.size, window.stride) == (128, 32)
    assert dataclasses.replace(memory_from_config(cfg), micro_batch=8) == memory_from_config(base)
    for key in ("method", "encoders", "lr", "steps", "loss", "reference_root", "output_dir"):
        assert getattr(cfg, key) == getattr(base, key), key
    assert not getattr(cfg, "reference_extension", None)          # COCO only
    assert cfg.exp_name != base.exp_name



# --------------------------------------------------------------------------------------
# gan.critic_rows: active -- the critic trains on the step's B fresh rows only
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["row", "block"])
def test_active_rows_train_the_critic_on_the_fresh_rows_only(store_root, kind):
    """After one step the critic equals a fresh copy stepped once on (the B active generated
    rows, their B reference rows) -- no cached row of the window took part."""
    extra = {"critic": kind, "critic_rows": "active", "lr": 1e-2}
    if kind == "block":
        extra = {**_BLOCK, **extra}
    trainer = build_trainer(store_root, window=16, stride=4, gan=extra)
    trainer.bootstrap()
    window = trainer._pending_window
    with torch.no_grad():
        fake = trainer._encode(window.active_ids)
    real = {n: trainer._reference_window(n, window.active_ids) for n in NAMES}
    tau = trainer.store.text_features(window.active_ids, dtype=F64)

    twin = build_critic(dataclasses.replace(trainer.critic.cfg), trainer.store, list(NAMES),
                        dtype=F64, block=4 if kind == "block" else None)
    twin.update(fake, real, tau)
    logs = trainer.step()
    assert trainer.critic.updates == 1 and not logs["gan/d_skipped"]
    # atol 1e-9: an attention key bias has an exactly zero gradient (softmax ignores a per-query
    # shift), so its computed gradient is rounding noise that AdamW turns into a step of up to
    # lr * |noise| / eps ~ 1e-10. It cannot change any score; every other entry agrees to ~1e-16.
    for (key, got), want in zip(trainer.critic.critic.state_dict().items(),
                                twin.critic.state_dict().values()):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-9, msg=key)


def test_active_rows_lift_the_whole_blocks_per_rank_rule(store_root, monkeypatch):
    import rdm.sw_lmmd.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "get_world_size", lambda: 8)
    build_trainer(store_root, window=16, stride=4, gan={**_BLOCK, "critic_rows": "active"})
    with pytest.raises(ValueError, match="split evenly over world_size=8"):
        build_trainer(store_root, window=16, stride=4, gan=_BLOCK)


def test_critic_rows_is_validated():
    with pytest.raises(ValueError, match="'window' or 'active'"):
        GANConfig(critic_rows="cache").validate()
    assert GANConfig().critic_rows == "window"               # the old runs' behaviour stays default
