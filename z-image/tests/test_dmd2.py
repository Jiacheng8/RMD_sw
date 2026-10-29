"""The DMD2 baseline on toy DiTs (CPU, float32)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from test_zimage_multistep import CAP_DIM, FakeTransformer, StubText, process_root  # noqa: F401
from zimage_distill.dmd2 import DMD2Trainer, GANHead, ScoreModel, dmd_loss, plan_iterations
from rdm.sw_lmmd.sharding import row_noise
from zimage_distill.dmd2_latents import FORMAT, Latents, chosen_shards
from zimage_distill.model import LATENT_CHANNELS, ZImageStudent
from zimage_distill.store import ZImageReferenceStore
from sw_lmmd_fixtures import NAMES

LAT = 32                      # latent side -> 16 x 16 image tokens, enough for 4 stride-2 blocks
DIM = 32


class ToyDiT(nn.Module):
    """``ZImageTransformer2DModel``'s call signature with a ``layers`` list (the hooked blocks):
    2x2 patches -> tokens [image | caption] -> blocks -> image tokens -> patches."""

    def __init__(self, seed: int = 0, n_layers: int = 4):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Linear(LATENT_CHANNELS * 4, DIM)
        self.cap = nn.Linear(CAP_DIM, DIM)
        self.time = nn.Linear(1, DIM)
        self.layers = nn.ModuleList(nn.Linear(DIM, DIM) for _ in range(n_layers))
        self.final = nn.Linear(DIM, LATENT_CHANNELS * 4)
        self.config = type("C", (), {"dim": DIM})()

    def forward(self, x, t, cap_feats, return_dict=True):
        b, (c, _, h, w) = len(x), x[0].shape
        img = torch.stack([xi[:, 0] for xi in x])                                   # (b, c, h, w)
        tok = img.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c * 4)
        n_img = tok.shape[1]
        cap = torch.stack([self.cap(ci.float()).mean(0) for ci in cap_feats])[:, None]
        u = torch.cat([self.embed(tok) + self.time(t[:, None].float())[:, None], cap], 1)
        for layer in self.layers:
            u = torch.tanh(layer(u)) + u
        out = self.final(u[:, :n_img]).reshape(b, h // 2, w // 2, c, 2, 2)
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return ([o[:, None] for o in out],)


@pytest.fixture()
def toy_latents(tmp_path):
    d = tmp_path / "latents"
    d.mkdir()
    available = np.array([0, 3, 5, 9, 12], dtype=np.int64)
    rng = np.random.default_rng(0)
    data = torch.from_numpy(rng.standard_normal((available.size, LATENT_CHANNELS, LAT, LAT))
                            .astype(np.float32)).to(torch.bfloat16)
    data.view(torch.int16).numpy().tofile(d / "shard_00.bin")
    index = np.full((16, 2), -1, dtype=np.int64)
    index[available] = np.stack([np.zeros(available.size, dtype=np.int64), np.arange(available.size)], 1)
    np.save(d / "index.npy", index)
    np.save(d / "available.npy", available)
    (d / "shard_01.bin").write_bytes(b"")           # a rank that got no tar shard
    (d / "manifest.json").write_text(json.dumps({"format": FORMAT, "shape": [LATENT_CHANNELS, LAT, LAT],
                                                 "dtype": "bfloat16",
                                                 "shards": ["shard_00.bin", "shard_01.bin"]}))
    return Latents(d), data


def build(process_root, latents, *, ratio=2):  # noqa: F811
    store = ZImageReferenceStore(process_root, NAMES)
    student = ZImageStudent(image_resolution=LAT * 8, param_dtype=torch.float32,
                            gradient_checkpointing=False, transformer=FakeTransformer(0).float())
    real = ScoreModel(ToyDiT(1)).requires_grad_(False)
    fake = ScoreModel(ToyDiT(1), feature_layer=2)
    head = GANHead(DIM, width=32)
    return DMD2Trainer(student, real, fake, head,
                       torch.optim.AdamW(student.parameters(), lr=1e-2),
                       torch.optim.AdamW(list(fake.parameters()) + list(head.parameters()), lr=1e-2),
                       store, latents, StubText().table[0].float(), sigmas=[1.0, 0.75], batch=4,
                       ratio=ratio, micro_batch=2, device="cpu", amp=False)


def test_dmd_loss_gradient_is_the_normalized_score_difference():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 5, 5, requires_grad=True)
    x0_real, x0_fake = torch.randn(3, 4, 5, 5), torch.randn(3, 4, 5, 5)
    dmd_loss(x, x0_real, x0_fake).backward()
    expect = (x0_fake - x0_real) / (x.detach() - x0_real).abs().mean(dim=(1, 2, 3), keepdim=True)
    torch.testing.assert_close(x.grad, expect / x.numel())


def test_score_model_is_the_flow_parametrisation_and_keeps_features():
    m = ScoreModel(ToyDiT(0), feature_layer=1)
    x = torch.randn(2, LATENT_CHANNELS, LAT, LAT)
    out = m(x, torch.tensor([0.3, 0.9]), [torch.randn(3, CAP_DIM), torch.randn(5, CAP_DIM)])
    assert out.shape == x.shape
    assert m.features.shape == (2, (LAT // 2) ** 2 + 1, DIM)          # image tokens, then caption
    assert GANHead(DIM, width=32)(m.features, side=LAT // 2).shape == (2,)


def test_one_cycle_trains_the_student_on_its_turn_only(process_root, toy_latents):  # noqa: F811
    lat, data = toy_latents
    torch.testing.assert_close(lat.get([3, 12]), data[[1, 4]])
    with pytest.raises(KeyError):
        lat.get([1])
    t = build(process_root, lat)
    snap = lambda m: [p.detach().clone() for p in m.parameters()]  # noqa: E731
    s0, r0, f0, h0 = snap(t.student), snap(t.real), snap(t.fake), snap(t.head)
    logs = t.step()                                          # iteration 0: student + fake
    assert logs["student_step"] and all(np.isfinite(logs[k]) for k in
                                        ("loss_dmd", "loss_gan_g", "loss_fake", "loss_gan_d"))
    s1, f1 = snap(t.student), snap(t.fake)
    assert any(not torch.equal(a, b) for a, b in zip(s0, s1))
    assert any(not torch.equal(a, b) for a, b in zip(f0, f1))
    assert any(not torch.equal(a, b) for a, b in zip(h0, snap(t.head)))
    logs = t.step()                                          # iteration 1: fake only
    assert not logs["student_step"] and "loss_dmd" not in logs
    assert all(torch.equal(a, b) for a, b in zip(s1, snap(t.student)))
    assert any(not torch.equal(a, b) for a, b in zip(f1, snap(t.fake)))
    assert all(torch.equal(a, b) for a, b in zip(r0, snap(t.real)))   # the real score is frozen


def test_plan_counts_whole_cycles():
    # ratio 5: student iterations 100 s, fake-only 20 s -> 36 s on average; 2 GPUs, 3600 GPU-s
    secs = ([100, 20, 20, 20, 20] * 3)
    assert plan_iterations(3600, 2, secs, ratio=5, done=15) == 50
    assert plan_iterations(3600, 2, secs, ratio=5, done=60) == 60
    assert chosen_shards(24)[:3] == [0, 4, 7] and len(chosen_shards(24)) == 24
    assert chosen_shards(24)[-1] == 86 and chosen_shards(1) == [0]


def test_backward_simulation_step_is_the_same_on_every_rank(process_root, toy_latents, monkeypatch):  # noqa: F811
    # under FSDP each student step is a collective: ranks that drew different k would deadlock
    import zimage_distill.dmd2 as dmd2_mod
    t = build(process_root, toy_latents[0])
    ks = []
    for rank in (0, 1, 3):
        monkeypatch.setattr(dmd2_mod, "get_rank", lambda r=rank: r)
        ks.append([t._steps_for(c) for c in range(16)])
    assert ks[0] == ks[1] == ks[2] and set(ks[0]) == {1, 2}


def test_real_prompt_ids_map_to_rows_of_that_prompt(process_root, toy_latents):  # noqa: F811
    t = build(process_root, toy_latents[0])
    pids = t.latents.available
    rows = t._row_of_prompt[pids]
    assert (rows >= 0).all()
    np.testing.assert_array_equal(t.store.prompt_rows(rows), pids)


def test_score_model_returns_the_features_it_keeps():
    # the GAN loss must take the features from the forward's return value (FSDP's root unit
    # enters its backward through its outputs; a side door left it idle on 4 x A100)
    m = ScoreModel(ToyDiT(0), feature_layer=1)
    x = torch.randn(2, LATENT_CHANNELS, LAT, LAT)
    caps = [torch.randn(3, CAP_DIM), torch.randn(5, CAP_DIM)]
    out, feats = m(x, torch.tensor([0.3, 0.9]), caps, return_features=True)
    assert feats is m.features and out.shape == x.shape
    torch.testing.assert_close(out, m(x, torch.tensor([0.3, 0.9]), caps))


def test_rollout_is_the_students_own_k_step_loop(process_root, toy_latents):  # noqa: F811
    # step k from its re-noised input (earlier steps without autograd) == the student's k steps
    t = build(process_root, toy_latents[0])
    rows = np.array([1, 4])
    for k in (1, 2):
        t._steps_for = lambda c, k=k: k
        got = t._rollout(rows, 0, with_grad=True)
        noise = row_noise(rows, (2, *t.latents.shape), seed=t.seed, epoch=t.iteration,
                          device="cpu", dtype=torch.float32)
        want = t.student(noise, t._ctx(rows), {"sigmas": t.sigmas[:k], "renoise": True,
                                               "detach_between_steps": True})[-1]
        torch.testing.assert_close(got, want)
        assert got.requires_grad
