"""Pixel GradCache vs full end-to-end gradients; no weights, GPUs, or network."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

# macOS's system torch is on Python3.9; existing rdm.train/__init__ eagerly imports
# unrelated modules with Python3.10-only runtime annotations. Exercise this standalone
# torch-only module directly there, and the normal package import on the training Python.
if sys.version_info < (3, 10):
    _spec = importlib.util.spec_from_file_location(
        "pixel_grad_caching_under_test", Path(__file__).parents[1] / "rdm/train/pixel_grad_caching.py")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    pixel_gradcache_backward = _module.pixel_gradcache_backward
else:
    from rdm.train.pixel_grad_caching import pixel_gradcache_backward


def _problem():
    torch.manual_seed(17)
    generator = nn.Linear(3, 4)
    encoder = nn.Linear(4, 6).requires_grad_(False)
    critic = nn.Linear(6, 1).requires_grad_(False)
    inputs = torch.randn(7, 3)
    return generator, encoder, critic, inputs


def _callbacks(generator, encoder, critic):
    counters = {"generate": 0, "encode": 0, "encoder_leaf_checks": 0}

    def generate(x):
        counters["generate"] += 1
        first = generator(x).tanh()
        # The second generated image depends on the first; phase3 must preserve this.
        return [first.reshape(-1, 1, 2, 2), (first.square() + 0.2 * first).reshape(-1, 1, 2, 2)]

    def encode(images, x):
        counters["encode"] += 1
        if torch.is_grad_enabled():
            assert all(image.is_leaf and image.requires_grad for image in images)
            counters["encoder_leaf_checks"] += 1
        output = {}
        for k, image in enumerate(images, 1):
            features = encoder(image.flatten(1)).sin()
            # Global feature includes a constant caption coordinate, spatial feature is 4D.
            output[f"s{k}/global"] = torch.cat([features, torch.full_like(features[:, :1], 0.3)], 1)
            output[f"s{k}/spatial"] = features.reshape(-1, 2, 1, 3)
        return output

    def terms(features, warmup=False):
        output = {}
        for k, weight in ((1, 0.25), (2, 1.0)):
            global_features = features[f"s{k}/global"]
            spatial = features[f"s{k}/spatial"].flatten(1)
            # Coupled across the full batch: plain per-chunk loss accumulation is wrong.
            force = weight * (global_features.mean(0).square().sum() +
                              0.2 * (global_features[:, None] - global_features[None, :]).square().mean())
            adv = None if warmup else -weight * critic(spatial).tanh().mean()
            output[k] = (force, adv)
        return output, {"example": 7}

    return generate, encode, terms, counters


def _balance(k, mmd, adv):
    return 1.5 * mmd / max(adv, 1e-12), {"balance_target": 1.5}


def _naive(generator, encoder, critic, inputs, warmup=False):
    # This oracle keeps both the complete student and encoder graphs at once.
    first = generator(inputs).tanh()
    images = [first.reshape(-1, 1, 2, 2), (first.square() + 0.2 * first).reshape(-1, 1, 2, 2)]
    features = {}
    for k, image in enumerate(images, 1):
        value = encoder(image.flatten(1)).sin()
        features[f"s{k}/global"] = torch.cat([value, torch.full_like(value[:, :1], 0.3)], 1)
        features[f"s{k}/spatial"] = value.reshape(-1, 2, 1, 3)
    _, _, terms_fn, _ = _callbacks(generator, encoder, critic)
    terms, _ = terms_fn(features, warmup=warmup)
    total = 0
    norms = {}
    for k, (force, adv) in terms.items():
        mmd = torch.autograd.grad(force, images[k - 1], retain_graph=True)[0]
        mmd_norm = float(mmd.double().norm())
        adv_norm = 0.0
        if adv is None:
            weight = 0
        else:
            gradient = torch.autograd.grad(adv, images[k - 1], retain_graph=True)[0]
            adv_norm = float(gradient.double().norm())
            weight, _ = _balance(k, mmd_norm, adv_norm)
        norms[k] = (mmd_norm, adv_norm, weight)
        total = total + force + (0 if adv is None else weight * adv)
    total.backward()
    return float(total.detach()), norms


@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_matches_naive_coupled_loss_two_step_weights_and_chunk_sizes(chunk_size):
    generator, encoder, critic, inputs = _problem()
    expected_generator = copy.deepcopy(generator)
    expected_loss, norms = _naive(expected_generator, encoder, critic, inputs)
    generate, encode, terms, counters = _callbacks(generator, encoder, critic)
    chunks = list(inputs.split(chunk_size))
    loss, logs = pixel_gradcache_backward(chunks, generate, encode, terms, _balance)
    assert loss == pytest.approx(expected_loss, abs=2e-6)
    for actual, expected in zip(generator.parameters(), expected_generator.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
    for k, (mmd, adv, weight) in norms.items():
        assert logs[f"s{k}/pixel_mmd_grad_norm"] == pytest.approx(mmd, rel=2e-5)
        assert logs[f"s{k}/pixel_adv_grad_norm"] == pytest.approx(adv, rel=2e-5)
        assert logs[f"s{k}/pixel_adv_lambda"] == pytest.approx(weight, rel=2e-5)
    assert counters == {"generate": 2 * len(chunks), "encode": 2 * len(chunks),
                        "encoder_leaf_checks": len(chunks)}
    assert logs["pixel_gc/generator_replay_max_abs"] == 0
    assert all(p.grad is None for p in [*encoder.parameters(), *critic.parameters()])


def test_warmup_none_adv_is_zero_without_balance_or_none_grad_failure():
    generator, encoder, critic, inputs = _problem()
    expected_generator = copy.deepcopy(generator)
    expected_loss, _ = _naive(expected_generator, encoder, critic, inputs, warmup=True)
    generate, encode, terms, _ = _callbacks(generator, encoder, critic)

    def not_called(*args):
        raise AssertionError("Warmup must not balance an absent adversarial objective")
    loss, logs = pixel_gradcache_backward(list(inputs.split(2)), generate, encode,
                                          lambda f: terms(f, warmup=True), not_called)
    assert loss == pytest.approx(expected_loss, abs=2e-6)
    for actual, expected in zip(generator.parameters(), expected_generator.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
    assert all(logs[f"s{k}/pixel_adv_lambda"] == 0 and logs[f"s{k}/pixel_adv_grad_norm"] == 0 for k in (1, 2))


def test_norm_sum_happens_before_sqrt_and_does_not_rescale_generator_gradients():
    generator, encoder, critic, inputs = _problem()
    expected_generator = copy.deepcopy(generator)
    _, expected_norms = _naive(expected_generator, encoder, critic, inputs)
    generate, encode, terms, _ = _callbacks(generator, encoder, critic)
    reduced = []

    def reduce_two_identical_ranks(squares):
        assert squares.dtype == torch.float64 and squares.shape == (2, 2)
        reduced.append(squares.clone())
        return 2 * squares
    _, logs = pixel_gradcache_backward(list(inputs.split(3)), generate, encode, terms, _balance,
                                       norm_reduce_fn=reduce_two_identical_ranks)
    assert len(reduced) == 1
    for k in (1, 2):
        assert logs[f"s{k}/pixel_mmd_grad_norm"] == pytest.approx(2 ** 0.5 * expected_norms[k][0], rel=2e-5)
    for actual, expected in zip(generator.parameters(), expected_generator.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)


def test_torch_rng_replay_covers_generator_and_encoder_without_advancing_global_rng_twice():
    generator = nn.Linear(2, 2)
    inputs = [torch.ones(2, 2), torch.zeros(1, 2)]
    recorded = []

    def generate(x):
        return [generator(x) + torch.rand_like(x)]
    def encode(images, ci):
        result = images[0] * torch.rand_like(images[0])
        recorded.append(result.detach().clone())
        return {"s1/value": result}
    def terms(leaves):
        return {1: (leaves["s1/value"].mean().square(), None)}, {}
    start = torch.get_rng_state()
    with torch.no_grad():
        for ci in inputs:
            encode(generate(ci), ci)
    expected_state = torch.get_rng_state()
    recorded.clear()
    torch.set_rng_state(start)
    _, logs = pixel_gradcache_backward(inputs, generate, encode, terms, _balance)
    assert torch.equal(torch.get_rng_state(), expected_state)
    assert torch.equal(recorded[0], recorded[2]) and torch.equal(recorded[1], recorded[3])
    assert logs["pixel_gc/generator_replay_max_abs"] == 0


def test_no_encoder_or_critic_grad_written_even_if_parameters_require_grad():
    generator, encoder, critic, inputs = _problem()
    encoder.requires_grad_(True)
    critic.requires_grad_(True)
    # Preserve pre-existing (e.g. D-update) parameter gradients rather than clearing them.
    for p in critic.parameters():
        p.grad = torch.full_like(p, 0.37)
    old_grads = [p.grad.clone() for p in critic.parameters()]
    generate, encode, terms, _ = _callbacks(generator, encoder, critic)
    pixel_gradcache_backward(list(inputs.split(3)), generate, encode, terms, _balance)
    assert all(p.grad is None for p in encoder.parameters())
    for p, expected in zip(critic.parameters(), old_grads):
        assert torch.equal(p.grad, expected)
    assert all(p.grad is not None for p in generator.parameters())


def test_constant_feature_coordinate_and_constant_loss_need_no_pixel_gradient():
    parameter = nn.Parameter(torch.tensor(2.0))
    def generate(x):
        return [parameter * x]
    def encode(images, ci):
        return {"s1/constant": torch.full_like(images[0], 3.0)}
    def terms(leaves):
        return {1: (leaves["s1/constant"].square().mean(), None)}, {}
    loss, logs = pixel_gradcache_backward([torch.ones(3, 2)], generate, encode, terms, _balance)
    assert loss == 9 and logs["s1/pixel_mmd_grad_norm"] == 0
    assert parameter.grad == 0


def test_feature_keys_must_have_an_explicit_step_and_empty_chunks_fail():
    with pytest.raises(ValueError, match="empty"):
        pixel_gradcache_backward([], None, None, None, None)
    with pytest.raises(ValueError, match="must start"):
        pixel_gradcache_backward([torch.ones(1, 2)], lambda x: [x],
                                 lambda images, ci: {"unknown": images[0]}, None, None)


def test_middle_d_update_precedes_g_adv_and_g_does_not_write_d_gradients():
    torch.manual_seed(101)
    generator = nn.Linear(2, 2)
    critic = nn.Linear(2, 1)
    expected_generator, expected_critic = copy.deepcopy(generator), copy.deepcopy(critic)
    inputs = torch.randn(5, 2)
    observed = {}

    def make_terms(disc, record):
        optimizer = torch.optim.SGD(disc.parameters(), lr=0.3)
        def terms(leaves):
            record["calls"] = record.get("calls", 0) + 1
            value = leaves["s1/value"]
            disc.requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            d_loss = ((disc(value.detach()) + 1).square().mean() +
                      (disc(torch.full_like(value, 0.6)) - 1).square().mean())
            d_loss.backward()
            optimizer.step()
            record["d_grad"] = [p.grad.clone() for p in disc.parameters()]
            disc.requires_grad_(False)
            return {1: (value.mean(0).square().sum(), -disc(value).mean())}, {}
        return terms

    image = expected_generator(inputs)
    features = image.square() + 0.1 * image
    expected_terms, _ = make_terms(expected_critic, {})({"s1/value": features})
    force, adv = expected_terms[1]
    mmd = torch.autograd.grad(force, image, retain_graph=True)[0]
    adv_grad = torch.autograd.grad(adv, image, retain_graph=True)[0]
    weight, _ = _balance(1, float(mmd.double().norm()), float(adv_grad.double().norm()))
    (force + weight * adv).backward()

    pixel_gradcache_backward(list(inputs.split(2)), lambda x: [generator(x)],
                             lambda images, ci: {"s1/value": images[0].square() + 0.1 * images[0]},
                             make_terms(critic, observed), _balance)
    assert observed["calls"] == 1
    for actual, expected in zip(generator.parameters(), expected_generator.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=2e-5)
    for actual, expected, old_gradient in zip(critic.parameters(), expected_critic.parameters(), observed["d_grad"]):
        torch.testing.assert_close(actual, expected)
        assert torch.equal(actual.grad, old_gradient)


def test_cross_step_loss_or_encoder_dependency_is_rejected():
    parameter = nn.Parameter(torch.tensor(1.0))
    generate = lambda x: [parameter * x, parameter * x.square()]
    separate = lambda images, ci: {"s1/a": images[0], "s2/a": images[1]}
    cross_loss = lambda leaves: ({1: ((leaves["s1/a"] + leaves["s2/a"]).square().mean(), None)}, {})
    with pytest.raises(ValueError, match="objective depends"):
        pixel_gradcache_backward([torch.ones(2, 3)], generate, separate, cross_loss, _balance)
    cross_encoder = lambda images, ci: {"s1/a": images[0] + images[1], "s2/a": images[1]}
    separate_loss = lambda leaves: ({1: (leaves["s1/a"].square().mean(), None)}, {})
    with pytest.raises(ValueError, match="encoder's step feature"):
        pixel_gradcache_backward([torch.ones(2, 3)], generate, cross_encoder, separate_loss, _balance)


@pytest.mark.parametrize('chunk_size', [1, 3, 7])
@pytest.mark.parametrize('warmup', [False, True])
def test_fixed_weights_match_full_graph_without_norm_or_balance(chunk_size, warmup):
    generator, encoder, critic, inputs = _problem()
    expected = copy.deepcopy(generator)
    weights = {1: 5e-5, 2: 1e-4}
    first = expected(inputs).tanh()
    images = [first, first.square() + 0.2 * first]
    features = {}
    for k, image in enumerate(images, 1):
        f = encoder(image).sin()
        features[f's{k}/global'] = torch.cat([f, torch.full_like(f[:, :1], .3)], 1)
        features[f's{k}/spatial'] = f.reshape(-1, 2, 1, 3)
    _, _, terms, _ = _callbacks(expected, encoder, critic)
    pairs, _ = terms(features, warmup=warmup)
    total = sum(force + (0 if adv is None else weights[k] * adv)
                for k, (force, adv) in pairs.items())
    total.backward()
    generate, encode, terms, _ = _callbacks(generator, encoder, critic)
    def forbidden(*args):
        raise AssertionError('Fixed weights must never call the gradient balancing/reducer')
    loss, logs = pixel_gradcache_backward(list(inputs.split(chunk_size)), generate, encode,
        lambda f: terms(f, warmup=warmup), forbidden, fixed_weights=weights,
        norm_reduce_fn=forbidden, diagnose_replay=False)
    assert loss == pytest.approx(float(total.detach()), abs=2e-6)
    for actual, oracle in zip(generator.parameters(), expected.parameters()):
        torch.testing.assert_close(actual.grad, oracle.grad, rtol=2e-5, atol=2e-6)
    assert not any('pixel_mmd_grad_norm' in key or 'pixel_adv_grad_norm' in key for key in logs)
    assert logs['s1/gan/lambda'] == (0 if warmup else weights[1])
