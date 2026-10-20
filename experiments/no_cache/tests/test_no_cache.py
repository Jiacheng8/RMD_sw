"""No-cache control (K = B): it trains, keeps no cache, its gradient is the full-window MMD's, the
stock training loop runs it with checkpoints and resume, and the config changes only what it must."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.no_cache.train import NoCacheTrainer, check_config
from rdm.compare.kernels import gamma_from_sigma
from sw_lmmd_fixtures import NAMES, build_toy_store, build_trainer

K = 16


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_toy_store(str(tmp_path_factory.mktemp("nocache_store")))


def _nocache(store, **kw):
    return NoCacheTrainer.from_trainer(build_trainer(store, window=K, stride=K, **kw))


def test_refuses_a_window_with_retained_rows(store):
    with pytest.raises(ValueError, match="K == B"):
        NoCacheTrainer.from_trainer(build_trainer(store, window=16, stride=4))


def test_trains_with_an_always_empty_cache(store):
    t = _nocache(store, micro_batch=4, lr=1e-2)
    boot = t.bootstrap()
    assert boot["bootstrap_rows"] == 0
    before = [p.detach().clone() for p in t.generator.model.parameters()]
    seen = []
    for _ in range(4):
        logs = t.step()
        assert np.isfinite(logs["force"]) and not logs["skipped"]
        assert logs["active_rows"] == K and logs["cache_max_age"] == 0
        for name in NAMES:                      # the cache holds the last window, retains nothing
            assert t.cache.entries[name].features.shape[0] == K
            assert t.cache.retained(name, np.empty(0, dtype=np.int64)).shape[0] == 0
        seen.append(t._last_logs["window_step"])
    assert seen == [0, 1, 2, 3]
    assert any(not torch.equal(a, b) for a, b in zip(before, t.generator.model.parameters()))


def test_gradient_is_the_full_window_mmd_gradient(store):
    """With K = B the force's gradient on the generated features equals the gradient of the
    biased full-window MMD^2 (both arguments live), checked on the encoded features."""
    t = _nocache(store, lr=0.0)
    t.bootstrap()
    window = t._pending_window
    feats = t._encode(window.active_ids)
    for name in NAMES:
        x = feats[name].detach().clone().requires_grad_(True)
        y = t._reference_window(name, window.all_ids).detach()
        sigma = t.sigmas[name]
        force, _ = t.mmd.active_force(x, x.detach(), y, sigma)
        (g_force,) = torch.autograd.grad(force, x)
        x2 = x.detach().clone().requires_grad_(True)
        gm = gamma_from_sigma(sigma)
        k = lambda a, b: torch.exp(-gm * torch.cdist(a, b).pow(2))
        mmd2 = k(x2, x2).mean() - 2 * k(x2, y).mean() + k(y, y).mean()
        (g_full,) = torch.autograd.grad(mmd2, x2)
        torch.testing.assert_close(g_force, g_full, rtol=1e-6, atol=1e-9)


def test_stock_training_loop_runs_and_resumes(store, tmp_path):
    from rdm.sw_lmmd.launch import train

    def cfg(steps, **kw):
        return SimpleNamespace(steps=steps, save_freq=2, output_dir=str(tmp_path), exp_name="nc",
                               print_freq=1, probe_every=0,
                               cache={"refresh_every_windows": 0, "refresh_probe_rows": 4,
                                      "drift_threshold": 0.0},
                               save_resume=True, resume_every=2, **kw)
    full = train(cfg(6), device="cpu", trainer=_nocache(store, lr=1e-2))
    part_dir = tmp_path / "part"
    t1 = _nocache(store, lr=1e-2)
    c = cfg(3); c.output_dir = str(part_dir)
    train(c, device="cpu", trainer=t1)
    c = cfg(6); c.output_dir = str(part_dir); c.resume_from = str(part_dir / "nc" / "resume.pth")
    resumed = train(c, device="cpu", trainer=_nocache(store, lr=1e-2))
    # resume.pth was written at step 2: steps 3-6 re-run from there, identically
    assert resumed.step_idx == full.step_idx == 6
    for a, b in zip(full.generator.model.parameters(), resumed.generator.model.parameters()):
        assert torch.equal(a, b)


def test_config_is_k128_gan_with_k_equal_b():
    from rdm.sw_lmmd.launch import (cache_from_config, gan_from_config, lr_schedule_from_config,
                                    memory_from_config, resolve_batching, window_from_config)
    from rdm.train.launch import load_config

    base = load_config("configs/sw_lmmd_train_h100_2gpu_gan.yaml")
    cfg = load_config("experiments/no_cache/configs/nocache_k128_b128.yaml")
    check_config(cfg)
    w, wb = window_from_config(cfg), window_from_config(base)
    assert (w.size, w.stride) == (128, 128) and (wb.size, wb.stride) == (128, 32)
    assert dataclasses.replace(w, stride=32) == wb
    assert not w.group_by_prompt and cfg.active_global_batch == 128
    assert getattr(cfg, "reference_extension", None) in (None, "")
    assert gan_from_config(cfg) == gan_from_config(base) and gan_from_config(cfg).enabled
    assert lr_schedule_from_config(cfg) == lr_schedule_from_config(base)
    assert dataclasses.replace(memory_from_config(cfg), micro_batch=memory_from_config(base).micro_batch) \
        == memory_from_config(base)
    c = cache_from_config(cfg)
    assert (c.refresh_every_windows, c.drift_threshold, cfg.probe_every) == (0, 0.0, 0)
    for key in ("encoders", "lr", "steps", "save_freq", "joint", "loss", "grad_clip", "seed",
                "reference_root"):
        assert getattr(cfg, key) == getattr(base, key), key
    assert cfg.exp_name != base.exp_name
    assert resolve_batching(cfg, memory_from_config(cfg), w, world_size=2)["grad_accum"] == 16
