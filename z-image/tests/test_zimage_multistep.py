"""The few-step student and its trainer, on a tiny fake transformer (CPU, float64).

Checks the sampler math against the pipeline's Euler loop, the cut between steps, and that the
trainer gives every step its own stream, cache and critic, trains, and resumes through the stock
training loop.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from rdm.compare.median_heuristic import median_bandwidth
from rdm.sw_lmmd import CacheConfig, GANConfig, SlidingWindowSchedule, build_critic, build_row_order
from rdm.sw_lmmd.cache import GeneratedWindowCache
from rdm.sw_lmmd.reference_store import write_reference_store
from sw_lmmd_fixtures import D_TXT, DIMS, NAMES
from zimage_distill.model import LATENT_CHANNELS, ZImageGenerator, ZImageStudent, turbo_sigmas
from zimage_distill.store import StepStreamStore, ZImageReferenceStore, prompt_dir, stream_names
from zimage_distill.trainer import MultiStepTrainer

DT = torch.float64
RES = 32                      # -> 4 x 4 latents
SIDE = RES // 8
CAP_DIM = 6
PROMPTS = 16
KEEP = 4
K, B = 16, 4


class FakeTransformer(nn.Module):
    """The ``ZImageTransformer2DModel`` call signature on a tiny differentiable map."""

    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.mix = nn.Parameter(torch.randn(LATENT_CHANNELS, LATENT_CHANNELS, generator=g) * 0.2)
        self.cap = nn.Parameter(torch.randn(CAP_DIM, LATENT_CHANNELS, generator=g) * 0.2)
        self.time = nn.Parameter(torch.randn(LATENT_CHANNELS, generator=g) * 0.2)

    def forward(self, x, t, cap_feats, return_dict=True):
        outs = []
        for xi, ti, ci in zip(x, t, cap_feats):          # xi: (C, 1, h, w), ci: (L_i, CAP_DIM)
            h = torch.einsum("cd,dfhw->cfhw", self.mix, xi)
            bias = (ci.mean(0) @ self.cap + ti * self.time)[:, None, None, None]
            outs.append(torch.tanh(h + bias))
        return (outs,)


class StubText:
    """``p{i}`` -> a fixed ``(L_i, CAP_DIM)`` context, a different length per prompt."""

    def __init__(self):
        g = torch.Generator().manual_seed(7)
        self.table = [torch.randn(2 + i % 3, CAP_DIM, generator=g, dtype=DT) for i in range(PROMPTS)]

    def __call__(self, prompts):
        return [self.table[int(p[1:])] for p in prompts]


class StubVAE:
    def decode(self, z):
        return torch.tanh(z)                             # (n, C, h, w) "pixels"


class StubBattery:
    """Fixed linear read-outs of the flattened images; records the image shapes it saw."""

    def __init__(self, seed: int = 1, side: int = SIDE):
        g = torch.Generator().manual_seed(seed)
        n_in = LATENT_CHANNELS * side * side
        self.W = {n: torch.randn(n_in, d, generator=g, dtype=DT) / n_in ** 0.5
                  for n, d in zip(NAMES, DIMS)}
        self.shapes = []

    def __call__(self, images, only=None):
        self.shapes.append(tuple(images.shape))
        x = images.flatten(1)
        return {n: x @ w for n, w in self.W.items() if only is None or n in only}


def student(sigmas=(1.0, 0.75), renoise=True, detach=True, seed=0):
    model = ZImageStudent(image_resolution=RES, param_dtype=DT, gradient_checkpointing=False,
                          transformer=FakeTransformer(seed))
    gen = ZImageGenerator(model, StubVAE(), StubText(),
                          {"sigmas": list(sigmas), "renoise": renoise,
                           "detach_between_steps": detach}, enable_amp=False)
    return model, gen


@pytest.fixture(scope="module")
def store_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("zimage_store")
    rng = np.random.default_rng(3)
    rows = PROMPTS * KEEP
    feats = {n: rng.standard_normal((rows, d)).astype(np.float32) for n, d in zip(NAMES, DIMS)}
    text = rng.standard_normal((PROMPTS, D_TXT)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    bands = {n: {"sigma": float(median_bandwidth(torch.from_numpy(f))),
                 "beta": float(median_bandwidth(torch.from_numpy(f)))} for n, f in feats.items()}
    write_reference_store(str(root), encoder_features=feats, text_features=text, bandwidths=bands,
                          row_order=build_row_order(rows, seed=3),
                          prompt_ids=np.arange(rows) // KEEP,
                          extra_metadata={"reference_type": "toy-zimage"})
    with open(root / "prompts.jsonl", "w") as f:
        for i in range(PROMPTS):
            f.write(json.dumps({"prompt": f"p{i}"}) + "\n")
    return str(root)


@pytest.fixture(scope="module")
def process_root(tmp_path_factory):
    """The /data/ironman/.../z-image/process layout: prompts and contexts in the render root,
    z_context.json and the extraction contract (ADM 512 crop) in reference_store/."""
    root = tmp_path_factory.mktemp("zimage_process")
    store = root / "reference_store"
    rng = np.random.default_rng(4)
    rows = PROMPTS * KEEP
    feats = {n: rng.standard_normal((rows, d)).astype(np.float32) for n, d in zip(NAMES, DIMS)}
    text = rng.standard_normal((PROMPTS, D_TXT)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    bands = {n: {"sigma": float(median_bandwidth(torch.from_numpy(f))),
                 "beta": float(median_bandwidth(torch.from_numpy(f)))} for n, f in feats.items()}
    write_reference_store(str(store), encoder_features=feats, text_features=text, bandwidths=bands,
                          row_order=build_row_order(rows, seed=3),
                          prompt_ids=np.arange(rows) // KEEP,
                          extra_metadata={"reference_type": "z_image_turbo_teacher",
                                          "prompt_hash": "abc"})
    (store / "z_context.json").write_text(json.dumps({"root": "..", "prompt_hash": "abc"}))
    for n in NAMES:
        (store / "encoder_features" / f"{n}.done.json").write_text(
            json.dumps({"contract": {"img_size": 512, "encoder": n}}))
    with open(root / "prompts.jsonl", "w") as f:
        for i in range(PROMPTS):
            f.write(json.dumps({"prompt_id": i, "image_id": f"img{i}", "prompt": f"p{i}"}) + "\n")
    ctx = StubText()
    for i in range(PROMPTS):
        folder = prompt_dir(root, i)
        folder.mkdir(parents=True)
        torch.save(ctx.table[i].to(torch.bfloat16), folder / "context.pt")
    (root / "render_config.json").write_text(json.dumps({"height": 32, "width": 32, "keep": KEEP}))
    return str(store)


def build(store_root, *, gan=True, lr=0.0, step_weights=None, detach=True, seed=0,
          feature_pool=1, text=True):
    store = ZImageReferenceStore(store_root, NAMES)
    schedule = SlidingWindowSchedule(store.row_order(), window_size=K, stride=B)
    model, gen = student(detach=detach, seed=seed)
    if not text:
        gen.text = None                                  # contexts must come from the store
    critics = None
    if gan:
        critics = [build_critic(GANConfig(enabled=True, hidden=16, g_start_step=0, seed=k), store,
                                list(NAMES), joint=True, device="cpu", dtype=DT) for k in range(2)]
    return MultiStepTrainer(
        gen, StubBattery(side=SIDE // feature_pool), store, schedule,
        torch.optim.AdamW(model.parameters(), lr=lr),
        encoders=list(NAMES), num_steps=2, step_weights=step_weights, critics=critics,
        feature_pool=feature_pool,
        noise_shape=(2, LATENT_CHANNELS, SIDE, SIDE),
        cache=GeneratedWindowCache(stream_names(2, NAMES), store_dtype=DT), micro_batch=2,
        grad_clip=0.0, kernel_block=8, seed=0, joint=True, cache_cfg=CacheConfig(store_dtype=DT),
        device="cpu", feature_dtype=DT, context_dtype=DT)


# ------------------------------------------------------------------------------ sampler
def test_turbo_sigmas_match_the_pipeline_schedule():
    assert turbo_sigmas(2) == pytest.approx([1.0, 0.75])
    assert turbo_sigmas(9) == pytest.approx([1.0, 0.96, 0.913043, 0.857143, 0.789474, 0.705882,
                                             0.6, 0.461538, 0.272727], abs=1e-5)


def test_ode_mode_is_the_pipeline_euler_loop():
    """Teacher schedule, no re-noise: the last x0_hat is the pipeline's final latent."""
    sigmas = turbo_sigmas(9)
    model, gen = student(sigmas=sigmas, renoise=False, detach=False)
    noise = torch.randn(3, 9, LATENT_CHANNELS, SIDE, SIDE, dtype=DT)
    cap = StubText()(["p0", "p5", "p9"])
    with torch.no_grad():
        outs = model(noise, cap, gen.sampling_args)
        x = noise[:, 0].clone()                          # ZImagePipeline's loop, verbatim math
        for s, s_next in zip(sigmas, sigmas[1:] + [0.0]):
            noise_pred = -model.velocity_out(x, s, cap)
            x = x + (s_next - s) * noise_pred
    assert len(outs) == 9
    torch.testing.assert_close(outs[-1], x)


def test_renoise_starts_step_two_from_step_one_and_cuts_the_graph():
    model, gen = student(detach=True)
    noise = torch.randn(2, 2, LATENT_CHANNELS, SIDE, SIDE, dtype=DT)
    cap = StubText()(["p1", "p2"])
    x0_1, x0_2 = model(noise, cap, gen.sampling_args)
    # step 2 by hand from sg(x0_1)
    x = 0.25 * x0_1.detach() + 0.75 * noise[:, 1]
    torch.testing.assert_close(x0_2, x + 0.75 * model.velocity_out(x, 0.75, cap))

    params = list(model.parameters())
    g_cut = torch.autograd.grad(x0_2.sum(), params)
    g_hand = torch.autograd.grad((x + 0.75 * model.velocity_out(x, 0.75, cap)).sum(), params)
    for a, b in zip(g_cut, g_hand):
        torch.testing.assert_close(a, b)
    model_full, gen_full = student(detach=False)
    _, x0_2_full = model_full(noise, cap, gen_full.sampling_args)
    g_full = torch.autograd.grad(x0_2_full.sum(), list(model_full.parameters()))
    assert any(not torch.allclose(a, b) for a, b in zip(g_cut, g_full))


# ------------------------------------------------------------------------------ store
def test_store_reads_prompts_and_streams_map_to_their_encoder(store_root):
    store = ZImageReferenceStore(store_root, NAMES)
    assert store.prompts([0, 3, 4, 63]) == ["p0", "p0", "p1", "p15"]
    view = StepStreamStore(store)
    for k in (1, 2):
        name = f"s{k}/{NAMES[0]}"
        assert view.sigma(name) == store.sigma(NAMES[0])
        torch.testing.assert_close(view.reference_joint_features(name, [5, 9]),
                                   store.reference_joint_features(NAMES[0], [5, 9]))
    assert not store.has_context and store.feature_image_size() is None
    with pytest.raises(RuntimeError, match="no precomputed context"):
        store.generator_context([0])


def test_process_store_contexts_and_feature_size(process_root):
    from zimage_distill.launch import check_render_config, resolve_feature_pool
    store = ZImageReferenceStore(process_root, NAMES)
    assert store.has_context and store.prompts([0, 5]) == ["p0", "p1"]
    ctx = store.generator_context([0, 5, 63], dtype=DT)
    table = StubText().table
    for c, p in zip(ctx, (0, 1, 15)):
        torch.testing.assert_close(c, table[p].to(torch.bfloat16).to(DT))
    assert store.feature_image_size() == 512
    assert resolve_feature_pool(store, 1024) == 2 and resolve_feature_pool(store, 512) == 1
    assert resolve_feature_pool(store, 1024, setting=4) == 4
    with pytest.raises(ValueError, match="power-of-two"):
        resolve_feature_pool(store, 768)
    with pytest.raises(ValueError, match="teacher renders are 32x32"):
        check_render_config(store, 1024)
    check_render_config(store, 32)


def test_trainer_reads_store_contexts_and_pools_before_the_encoders(process_root):
    t = build(process_root, text=False, feature_pool=2, lr=1e-2)
    assert t.use_store_context
    t.bootstrap()
    logs = t.step()
    assert np.isfinite(logs["force"]) and not logs["skipped"]
    assert {s[1:] for s in t.battery.shapes} == {(LATENT_CHANNELS, SIDE // 2, SIDE // 2)}


# ------------------------------------------------------------------------------ trainer
def test_every_step_has_its_own_stream_cache_and_critic(store_root):
    t = build(store_root, lr=1e-2)
    assert t.encoder_names == stream_names(2, NAMES)
    assert t.encoder_weights[f"s1/{NAMES[0]}"] == pytest.approx(0.5)
    t.bootstrap()
    assert set(t.cache.entries) == set(stream_names(2, NAMES))

    seen = {0: [], 1: []}
    for k, critic in enumerate(t.step_critics):
        def spy(fake, real, tau=None, _orig=critic.update, _k=k):
            seen[_k].append({n: v.detach().clone() for n, v in fake.items()})
            return _orig(fake, real, tau)
        critic.update = spy

    before = [p.detach().clone() for p in t.generator.model.parameters()]
    for _ in range(3):
        logs = t.step()
        assert np.isfinite(logs["force"]) and not logs["skipped"]
        for k in (1, 2):
            assert np.isfinite(logs[f"s{k}/force"])
            assert f"s{k}/gan/d_acc" in logs and f"s{k}/gan/lambda" in logs
        assert logs["gan/d_acc"] == logs["s2/gan/d_acc"]            # the console line: last step
        # critic k trained on step k's window, whose last B rows are this step's fresh rows
        for k in (0, 1):
            fake = seen[k][-1]
            for n in NAMES:
                cached = t.cache.entries[f"s{k + 1}/{n}"].features
                torch.testing.assert_close(fake[n][-B:], cached[-B:])
        assert not torch.allclose(seen[0][-1][NAMES[0]], seen[1][-1][NAMES[0]])
    after = list(t.generator.model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_step_gradients_add_up(store_root):
    """Without critics, the S-step gradient is the sum of the single-step-weighted ones."""
    def grad(weights):
        t = build(store_root, gan=False, step_weights=weights)
        t.optimizer.step = lambda: None                  # keep the gradient, skip the update
        t.bootstrap()
        t.step()
        return [p.grad.detach().clone() for p in t.generator.model.parameters()]
    both, first, second = grad([1.0, 1.0]), grad([1.0, 0.0]), grad([0.0, 1.0])
    for a, b, c in zip(both, first, second):
        torch.testing.assert_close(a, b + c)
    assert any(b.abs().sum() > 0 for b in first) and any(c.abs().sum() > 0 for c in second)


def test_resume_restores_both_critics_and_refuses_another_step_count(store_root):
    t = build(store_root, lr=1e-2)
    t.bootstrap()
    t.step()
    state = t.state_dict()
    assert len(state["critics"]) == 2 and state["num_steps"] == 2
    fresh = build(store_root, lr=1e-2)
    restored = fresh.load_state_dict(state)
    assert restored["critic"] and restored["cache"]
    for a, b in zip(t.step_critics, fresh.step_critics):
        for p, q in zip(a.critic.parameters(), b.critic.parameters()):
            assert torch.equal(p, q)
    with pytest.raises(RuntimeError, match="3-step student"):
        fresh.load_state_dict({**state, "num_steps": 3})


def test_stock_training_loop_runs_and_resumes(store_root, tmp_path):
    from rdm.sw_lmmd.launch import train

    def cfg(steps, out):
        return SimpleNamespace(steps=steps, save_freq=2, output_dir=str(out), exp_name="z",
                               print_freq=1, probe_every=2,
                               cache={"refresh_every_windows": 3, "refresh_probe_rows": 4,
                                      "drift_threshold": 0.0},
                               save_resume=True, resume_every=2)
    full = train(cfg(5, tmp_path / "full"), device="cpu", trainer=build(store_root, lr=1e-2))
    train(cfg(3, tmp_path / "part"), device="cpu", trainer=build(store_root, lr=1e-2))
    c = cfg(5, tmp_path / "part")
    c.resume_from = str(tmp_path / "part" / "z" / "resume.pth")
    resumed = train(c, device="cpu", trainer=build(store_root, lr=1e-2))
    assert resumed.step_idx == full.step_idx == 5
    for a, b in zip(full.generator.model.parameters(), resumed.generator.model.parameters()):
        torch.testing.assert_close(a, b)
    logs = [json.loads(line) for line in open(tmp_path / "full" / "z" / "train_log.jsonl")]
    assert any("s1/drift_mean" not in r and f"s1/{NAMES[0]}/drift_mean" in r for r in logs)


def test_student_block_parsing():
    from zimage_distill.launch import student_from_config
    s = student_from_config(SimpleNamespace(student={"num_steps": 2}))
    assert s["sampling_args"]["sigmas"] == pytest.approx([1.0, 0.75])
    assert s["sampling_args"]["renoise"] and s["sampling_args"]["detach_between_steps"]
    assert s["step_weights"] == [1.0, 1.0]
    for bad in ({"num_steps": 2, "sigmas": [1.0]}, {"num_steps": 2, "sigmas": [0.9, 0.5]},
                {"num_steps": 2, "sigmas": [1.0, 0.0]}, {"num_steps": 2, "step_weight": [1, 1]}):
        with pytest.raises(ValueError):
            student_from_config(SimpleNamespace(student=bad))


def test_reference_pixels_is_the_png_then_adm_crop_path():
    """Forward: exactly the pipeline's PNG through ``center_crop_arr``; backward: the average pool."""
    from PIL import Image

    from rdm.data.imagenet import center_crop_arr
    from zimage_distill.model import reference_pixels

    g = torch.Generator().manual_seed(0)
    raw = (torch.rand(2, 3, 64, 64, generator=g) * 2.2 - 1.1).to(torch.bfloat16)   # VAE output
    x = (raw.float() * 0.5 + 0.5).clamp(0, 1)                     # ZImageVAE.decode
    png = torch.round(255 * (raw / 2 + 0.5).clamp(0, 1).float())  # ZImagePipeline postprocess
    ref = torch.stack([torch.from_numpy(np.array(center_crop_arr(
        Image.fromarray(img.permute(1, 2, 0).numpy().astype(np.uint8)), 32))).permute(2, 0, 1)
        for img in png]).float() / 255
    x = x.requires_grad_(True)
    out = reference_pixels(x, 2)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    w = torch.randn(out.shape, generator=g)
    (g_ste,) = torch.autograd.grad((out * w).sum(), x)
    (g_pool,) = torch.autograd.grad((torch.nn.functional.avg_pool2d(x, 2) * w).sum(), x)
    torch.testing.assert_close(g_ste, g_pool)


# ------------------------------------------------------------------------------ context rebuild
class _CharTokenizer:
    """One token per character of the chat-wrapped prompt; pad id 0."""
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "<" + messages[0]["content"] + ">"

    def __call__(self, text, truncation, max_length):
        return {"input_ids": [[ord(c) for c in t][:max_length] for t in text]}


class _CausalToy(nn.Module):
    """Causal by construction (running mean of embeddings): a token never sees what follows it,
    pads included -- the property the bucketed rebuild relies on."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = nn.Parameter(torch.randn(256, 2560, generator=g).to(torch.bfloat16))

    def forward(self, input_ids, attention_mask, output_hidden_states):
        h = self.emb[input_ids].float()
        run = h.cumsum(1) / torch.arange(1, h.shape[1] + 1)[None, :, None]
        return SimpleNamespace(hidden_states=(h, run.to(torch.bfloat16), h))


def test_bucketed_contexts_equal_one_by_one_encoding(tmp_path):
    from zimage_distill.contexts import FlatContexts, build, encode_batch, finalize, plan_batches
    prompts = [("x" * n) for n in (5, 40, 7, 12, 3, 40, 22, 9, 1, 30)]
    with open(tmp_path / "prompts.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")
    (tmp_path / "prepare.json").write_text(json.dumps({"prompt_hash": "h"}))
    tok, model = _CharTokenizer(), _CausalToy()
    for r in range(2):
        build(tmp_path, r, 2, None, max_tokens=64, device="cpu", tokenizer=tok, model=model)
    finalize(tmp_path, 2)
    ctx = FlatContexts(tmp_path / "contexts")
    assert len(ctx) == len(prompts) and ctx.manifest["prompt_hash"] == "h"
    for i, p in enumerate(prompts):
        ids = [ord(c) for c in "<" + p + ">"]
        alone = encode_batch(model, [ids], 0, "cpu")[0]          # no padding at all
        assert ctx.get(i).dtype == torch.bfloat16
        torch.testing.assert_close(ctx.get(i), alone, rtol=0, atol=0)
    batches = plan_batches([len(p) + 2 for p in prompts], max_tokens=64)
    assert sorted(i for b in batches for i in b) == list(range(len(prompts)))
    assert all(len(b) * max(len(prompts[i]) + 2 for i in b) <= 64 or len(b) == 1 for b in batches)


def test_pack_store_reads_the_rebuilt_contexts(tmp_path):
    """A training pack: reference_store/ + prompts.jsonl + contexts/, no renders/."""
    from zimage_distill.contexts import FlatContexts, build, finalize
    store_dir = tmp_path / "reference_store"
    rng = np.random.default_rng(5)
    rows = PROMPTS * KEEP
    feats = {n: rng.standard_normal((rows, d)).astype(np.float32) for n, d in zip(NAMES, DIMS)}
    text = rng.standard_normal((PROMPTS, D_TXT)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    write_reference_store(str(store_dir), encoder_features=feats, text_features=text,
                          bandwidths={n: {"sigma": 1.0, "beta": 1.0} for n in NAMES},
                          row_order=build_row_order(rows, seed=3), prompt_ids=np.arange(rows) // KEEP,
                          extra_metadata={"prompt_hash": "abc"})
    (store_dir / "z_context.json").write_text(json.dumps({"root": "..", "prompt_hash": "abc"}))
    (tmp_path / "prepare.json").write_text(json.dumps({"prompt_hash": "abc"}))
    with open(tmp_path / "prompts.jsonl", "w") as f:
        for i in range(PROMPTS):
            f.write(json.dumps({"prompt": f"prompt number {i}"}) + "\n")
    assert not ZImageReferenceStore(str(store_dir), NAMES).has_context    # before the rebuild
    build(tmp_path, 0, 1, None, max_tokens=64, device="cpu", tokenizer=_CharTokenizer(),
          model=_CausalToy())
    finalize(tmp_path, 1)
    store = ZImageReferenceStore(str(store_dir), NAMES)
    assert store.has_context
    flat = FlatContexts(tmp_path / "contexts")
    for row, c in zip([0, 9, 63], store.generator_context([0, 9, 63])):
        torch.testing.assert_close(c, flat.get(row // KEEP), rtol=0, atol=0)
