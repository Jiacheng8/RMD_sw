"""Two-level GradCache for objectives with incompatible global/spatial feature axes.

1. Generate and encode without graphs, caching raw pixels on CPU and feature leaves.
2. Differentiate each step's MMD/adversarial terms at the feature leaves. Replay only
   frozen encoders on cached independent pixel leaves, obtaining both pixel VJPs.
3. Balance the globally summed pixel-gradient norms, then replay only the generator
   and inject the combined pixel cotangent. Generator and encoder each run twice.

Loss callbacks retain ownership of the existing batch/rank/step/encoder weighting.
This module does not gather features or divide by chunks, accumulation, or ranks.
Each s{k}/ feature must depend only on raw image k (one-based); generator images may
depend on earlier generator steps, which phase 3's ordinary backward handles.
"""
from __future__ import annotations

import contextlib
import math
import re

import torch
import torch.distributed as dist


def step_for_feature(name):
    match = re.match(r"^s([1-9][0-9]*)/", name)
    if not match:
        raise ValueError(f"Feature key {name!r} must start with s{{positive step}}/")
    return int(match.group(1))


def _rng_state(enabled):
    if not enabled:
        return None
    # A rank normally uses only its current CUDA device. Do not initialize/touch other
    # visible GPUs merely to capture RNG state. Explicit torch.Generator objects must
    # remain deterministic in chunk_inputs; their private RNG is not global state.
    cuda = {}
    if torch.cuda.is_initialized():
        device = torch.cuda.current_device()
        cuda[device] = torch.cuda.get_rng_state(device)
    return torch.get_rng_state(), cuda


@contextlib.contextmanager
def _replay_rng(state):
    if state is None:
        yield
        return
    cpu, cuda = state
    with torch.random.fork_rng(devices=list(cuda)):
        torch.set_rng_state(cpu)
        for device, value in cuda.items():
            torch.cuda.set_rng_state(value, device)
        yield


def _images(value, expected_steps=None):
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("generate_fn must return a nonempty list of raw image tensors")
    if expected_steps is not None and len(value) != expected_steps:
        raise ValueError("generate_fn changed the number of image steps during replay")
    if any(not isinstance(v, torch.Tensor) or v.ndim == 0 or not v.is_floating_point() for v in value):
        raise ValueError("Raw images must be floating-point tensors with a leading batch axis")
    if any(v.shape[0] != value[0].shape[0] or v.shape[0] == 0 for v in value):
        raise ValueError("All image steps must share a nonempty chunk batch axis")
    return list(value)


def _features(value, batch, schema=None):
    if not isinstance(value, dict) or not value:
        raise ValueError("encode_images_fn must return a nonempty feature dictionary")
    if schema is not None and set(value) != set(schema):
        raise ValueError("encode_images_fn changed feature keys during replay")
    for name, feature in value.items():
        if not isinstance(feature, torch.Tensor) or feature.ndim == 0 or not feature.is_floating_point():
            raise ValueError(f"Feature {name!r} must be a floating tensor with a batch axis")
        if feature.shape[0] != batch:
            raise ValueError(f"Feature {name!r} batch differs from its raw image chunk")
        if schema is not None and tuple(feature.shape[1:]) != schema[name]:
            raise ValueError(f"Feature {name!r} changed its non-batch shape during replay")
    return value


def _log_values(value):
    """Detach diagnostics so callback logs cannot retain the encoder/critic graphs."""
    if isinstance(value, torch.Tensor):
        return float(value.detach()) if value.numel() == 1 else value.detach().cpu()
    if isinstance(value, dict):
        return {key: _log_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_log_values(item) for item in value)
    return value


def _feature_cotangent(loss, leaves, names):
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ValueError("Every force/adv loss must be a scalar tensor (or adv=None)")
    if not loss.requires_grad:
        return {}
    all_names = list(leaves)
    grads = torch.autograd.grad(loss, [leaves[name] for name in all_names],
                                allow_unused=True, retain_graph=True)
    for name, grad in zip(all_names, grads):
        if name not in names and grad is not None and torch.count_nonzero(grad):
            raise ValueError("A step objective depends on feature leaves owned by another image step")
    return {name: grad.detach().to(device="cpu", dtype=torch.float32)
            for name, grad in zip(all_names, grads) if name in names and grad is not None}


def _pixel_vjp(features, cotangents, offset, batch, pixel_inputs, image_index):
    pixels = pixel_inputs[image_index]
    outputs, gradients = [], []
    for name, cached in cotangents.items():
        output = features[name]
        if output.requires_grad:
            outputs.append(output)
            gradients.append(cached[offset:offset + batch].to(device=output.device,
                                                              dtype=output.dtype))
    if not outputs:
        return None
    all_grads = torch.autograd.grad(outputs, pixel_inputs, grad_outputs=gradients,
                                    allow_unused=True, retain_graph=True)
    for index, grad in enumerate(all_grads):
        if index != image_index and grad is not None and torch.count_nonzero(grad):
            raise ValueError("An encoder's step feature depends on another step's raw pixel leaf")
    grad = all_grads[image_index]
    return None if grad is None else grad.detach().float()


def _reduce_norm_squares(squares, device, norm_reduce_fn):
    # NCCL reductions require CUDA tensors; Gloo/CPU tests retain the input's CPU device.
    totals = torch.tensor(squares, dtype=torch.float64, device=device)
    if norm_reduce_fn is not None:
        result = norm_reduce_fn(totals)
        if result is not None:
            totals = result
    elif dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if (not isinstance(totals, torch.Tensor) or totals.dtype != torch.float64 or
            tuple(totals.shape) != (len(squares), 2)):
        raise ValueError("norm_reduce_fn must preserve the float64 (steps,2) norm-square tensor")
    return totals.detach().cpu().double()


def pixel_gradcache_backward(chunk_inputs, generate_fn, encode_images_fn, loss_terms_fn, balance_fn,
                             *, step_for_feature=step_for_feature, preserve_rng_state=True,
                             diagnose_replay=True, replay_atol=None, norm_reduce_fn=None,
                             fixed_weights=None):
    """Accumulate exact first-order generator gradients; no optimizer step is taken.

    Args:
        generate_fn(ci): list of raw decoded images, one tensor per generator step.
        encode_images_fn(images,ci): per-step global/spatial feature tensors. Images are
            canonical float32 in both encoder calls. Text constants can be coupled here.
        loss_terms_fn(leaves): ({k: (force, adv_or_None)}, logs), k one-based. Update D
            internally before forming adv, then freeze D for G. Each loss k must use
            only features mapped to k. The caller supplies all existing loss weights.
        balance_fn(k,mmd_pixel_norm,adv_pixel_norm): (detached lambda,logs), with globally
            SUM-reduced L2 norms. Use the existing adaptive weight (e.g. 1.5) here;
            do not reduce these norms a second time. Never called for adv=None.
        fixed_weights: optional per-step fixed GAN coefficients. Combines feature
            cotangents before the encoder VJP; no separate gradient norms or balancing.
        norm_reduce_fn: optional SUM reducer for the float64 norm-square matrix. The
            default performs one distributed SUM when a process group is initialized.
        preserve_rng_state: replay global torch CPU/current-device CUDA RNG for both
            encoders and generator. Other/private RNG must be deterministic in ci.
        replay_atol: optional fail-fast raw pixel replay threshold; default only logs.

    Feature and image tensors may have arbitrary trailing dimensions. The function
    preserves pre-existing parameter gradients and adds the generator contribution.
    It does not clear gradients, step D/G, or add any /A, /R, or microbatch scaling.
    """
    chunks = list(chunk_inputs)
    if not chunks:
        raise ValueError("pixel_gradcache_backward: chunk_inputs is empty")
    reps, raw_cache, sizes = {}, [], []
    schema = None
    steps_count = None

    # Phase 1: no generator/encoder graph survives; raw decoded pixels live on CPU.
    with torch.no_grad():
        for ci in chunks:
            gen_rng = _rng_state(preserve_rng_state)
            raw = _images(generate_fn(ci), steps_count)
            steps_count = len(raw)
            images = [image.float() for image in raw]
            enc_rng = _rng_state(preserve_rng_state)
            features = _features(encode_images_fn(images, ci), images[0].shape[0], schema)
            if schema is None:
                schema = {name: tuple(value.shape[1:]) for name, value in features.items()}
            for name, value in features.items():
                reps.setdefault(name, []).append(value.detach().float())
            sizes.append(images[0].shape[0])
            raw_cache.append({"images": [value.detach().to(device="cpu", dtype=torch.float32, copy=True)
                                          for value in images],
                              "devices": [value.device for value in images],
                              "gen_rng": gen_rng, "enc_rng": enc_rng})
        del raw, images, features, value
    leaves = {name: torch.cat(values, dim=0).detach().requires_grad_(True)
              for name, values in reps.items()}
    reps.clear()
    ownership = {name: int(step_for_feature(name)) for name in leaves}
    if any(k < 1 or k > steps_count for k in ownership.values()):
        raise ValueError("Feature step does not correspond to a generated image")

    # Middle: feature cotangents, without writing encoder/critic parameter .grad.
    with torch.enable_grad():
        terms, logs = loss_terms_fn(leaves)
        logs = dict(_log_values(logs))
        if not isinstance(terms, dict) or not terms or any(
                not isinstance(k, int) or k < 1 or k > steps_count for k in terms):
            raise ValueError("loss_terms_fn must return one-based generated step keys")
        steps = sorted(terms)
        feature_grads, values, has_adv = {}, {}, {}
        for k in steps:
            names = [name for name in leaves if ownership[name] == k]
            if not names:
                raise ValueError(f"Step {k} has losses but no feature leaves")
            force, adv = terms[k]
            if fixed_weights is not None:
                weight = float(fixed_weights[k]) if adv is not None else 0.0
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError('Fixed GAN weights must be finite and nonnegative')
                combined = force if adv is None else force + weight * adv
                feature_grads[k] = (_feature_cotangent(combined, leaves, names), {})
                del combined
            else:
                feature_grads[k] = (_feature_cotangent(force, leaves, names),
                                    {} if adv is None else _feature_cotangent(adv, leaves, names))
            values[k] = (float(force.detach()), 0.0 if adv is None else float(adv.detach()))
            has_adv[k] = adv is not None
        del terms, force, adv, leaves

    # Phase 2: independent raw pixel leaves; no student forward/graph in encoder VJPs.
    pixel_cache = []
    squares = [[0.0, 0.0] for _ in steps]
    offset = 0
    for ci, saved, batch in zip(chunks, raw_cache, sizes):
        with torch.enable_grad(), _replay_rng(saved["enc_rng"]):
            pixels = [value.to(device=device, dtype=torch.float32, copy=True).requires_grad_(True)
                      for value, device in zip(saved["images"], saved["devices"])]
            features = _features(encode_images_fn(pixels, ci), batch, schema)
            cached = {}
            for pos, k in enumerate(steps):
                pair = []
                for which in ((0,) if fixed_weights is not None else (0, 1)):
                    grad = _pixel_vjp(features, feature_grads[k][which], offset, batch, pixels, k - 1)
                    # An absent warmup adv needs no full-resolution GPU zero allocation.
                    cpu_grad = (torch.zeros_like(saved["images"][k - 1]) if grad is None else
                                grad.to(device="cpu", dtype=torch.float32, copy=True))
                    pair.append(cpu_grad)
                    if fixed_weights is None:
                        squares[pos][which] += float(cpu_grad.double().square().sum())
                    del grad
                cached[k] = pair
            pixel_cache.append(cached)
            del features, pixels, pair, cached
        offset += batch
    del feature_grads

    # Balancing happens in shared raw-pixel coordinates, after a single cross-rank SUM.
    totals = (_reduce_norm_squares(squares, raw_cache[0]["devices"][0], norm_reduce_fn)
              if fixed_weights is None else None)
    logs["pixel_gc/balance_mode"] = "fixed" if fixed_weights is not None else "gradient_ratio"
    lambdas = {}
    loss_float = 0.0
    for pos, k in enumerate(steps):
        if fixed_weights is not None:
            weight = float(fixed_weights[k]) if has_adv[k] else 0.0
            logs[f"s{k}/gan/lambda"] = weight
        elif has_adv[k]:
            mmd_norm, adv_norm = (float(value.sqrt()) for value in totals[pos])
            weight, sublogs = balance_fn(k, mmd_norm, adv_norm)
            weight = float(weight.detach()) if isinstance(weight, torch.Tensor) else float(weight)
            if not math.isfinite(weight):
                raise ValueError(f"Step {k} returned a non-finite adaptive pixel weight")
            logs.update({f"s{k}/{name}": value for name, value in _log_values(sublogs).items()})
        else:
            weight = 0.0
        lambdas[k] = weight
        if fixed_weights is None:
            mmd_norm, adv_norm = (float(value.sqrt()) for value in totals[pos])
            logs[f"s{k}/pixel_mmd_grad_norm"] = mmd_norm
            logs[f"s{k}/pixel_adv_grad_norm"] = adv_norm
        logs[f"s{k}/pixel_adv_lambda"] = weight
        logs[f"s{k}/gan/g_active"] = has_adv[k]
        logs[f"s{k}/loss_mmd_force"] = values[k][0]
        logs[f"s{k}/loss_g_adv_raw"] = values[k][1]
        logs[f"s{k}/loss_g_adv_weighted"] = weight * values[k][1]
        loss_float += values[k][0] + weight * values[k][1]

    # Phase 3: generator-only replay; pixel cotangents replace the encoder graph.
    replay_max_abs = 0.0
    for ci, saved, cached in zip(chunks, raw_cache, pixel_cache):
        with torch.enable_grad(), _replay_rng(saved["gen_rng"]):
            raw = _images(generate_fn(ci), steps_count)
            if any(tuple(image.shape) != tuple(old.shape) for image, old in zip(raw, saved["images"])):
                raise ValueError("generate_fn changed raw image shapes during replay")
            if diagnose_replay or replay_atol is not None:
                for image, old in zip(raw, saved["images"]):
                    difference = float((image.detach().float().cpu() - old).abs().max())
                    replay_max_abs = max(replay_max_abs, difference) if math.isfinite(difference) else difference
                if replay_atol is not None and (not math.isfinite(replay_max_abs) or replay_max_abs > replay_atol):
                    raise RuntimeError(f"Raw generator replay differs by {replay_max_abs}, limit {replay_atol}")
            surrogate = None
            for k in steps:
                pair = cached[k]
                gradient = (pair[0] if fixed_weights is not None else
                            pair[0] + lambdas[k] * pair[1]).to(device=raw[k - 1].device)
                term = (raw[k - 1].float() * gradient).sum()
                surrogate = term if surrogate is None else surrogate + term
            if surrogate.requires_grad:
                surrogate.backward()
            elif any(torch.count_nonzero(pair[0] if fixed_weights is not None else
                                         pair[0] + lambdas[k] * pair[1]) for k, pair in cached.items()):
                raise RuntimeError("Generator replay has nonzero pixel cotangents but no autograd graph")
            del raw, surrogate, term, gradient
    logs["pixel_gc/generator_replay_max_abs"] = replay_max_abs if diagnose_replay or replay_atol is not None else None
    logs["pixel_gc/chunks"] = len(chunks)
    return loss_float, logs
