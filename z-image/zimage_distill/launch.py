"""Build and run the Z-Image few-step SW-LMMD trainer.

    PYTHONPATH=z-image torchrun --nproc_per_node=4 -m zimage_distill.launch \
        z-image/configs/zimage_2step_1024_h100.yaml [--set steps=5 ...]
    (or: GPUS=4 bash z-image/scripts/train.sh)

Same parsing and the same training loop as :mod:`rdm.sw_lmmd.launch` (window, memory, gan,
lr_schedule blocks; checkpoints, resume, drift probe and refresh); what differs is the model
(:class:`zimage_distill.model.ZImageStudent` + VAE + Qwen3 prompt encoder), the store (prompt
strings, :class:`zimage_distill.store.ZImageReferenceStore`), and the trainer
(:class:`zimage_distill.trainer.MultiStepTrainer`: one stream, one cache and one critic per
student step).

The ``student:`` block:

    num_steps             S, the student's steps (2)
    sigmas                null = the Turbo schedule for S steps ([1.0, 0.75] for S=2), or a list
    renoise               true: step k+1 starts from sg(x0_hat_k) re-noised to sigma_{k+1}
    detach_between_steps  true: step k's loss reaches only step k's forward
    step_weights          w_k per step (null = all 1)
"""
from __future__ import annotations

import dataclasses
import logging

import torch

from rdm.sw_lmmd.cache import GeneratedWindowCache
from rdm.sw_lmmd.launch import (_block, apply_override, cache_from_config, gan_from_config,
                                lr_schedule_from_config, memory_from_config, resolve_batching,
                                build_optimizer, shard_generator, train, warn_gan_pipeline,
                                window_from_config)
from rdm.sw_lmmd.window_schedule import SlidingWindowSchedule

from .model import ZIMAGE_REPO, QwenPromptEncoder, ZImageGenerator, ZImageStudent, ZImageVAE, \
    turbo_sigmas
from .store import ZImageReferenceStore, stream_names
from .trainer import MultiStepTrainer

logger = logging.getLogger("rdm")


def student_from_config(cfg) -> dict:
    """The ``student:`` block -> sampling args (``sigmas`` resolved) + step weights."""
    raw = _block(cfg, "student")
    known = {"num_steps", "sigmas", "renoise", "detach_between_steps", "step_weights"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"student: unknown key(s) {unknown} (known: {sorted(known)})")
    num_steps = int(raw.get("num_steps", 2))
    sigmas = raw.get("sigmas") or turbo_sigmas(num_steps)
    sigmas = [float(s) for s in sigmas]
    if len(sigmas) != num_steps:
        raise ValueError(f"student.sigmas has {len(sigmas)} entries for {num_steps} steps")
    if sigmas[0] != 1.0 or any(b >= a for a, b in zip(sigmas, sigmas[1:])) or sigmas[-1] <= 0:
        raise ValueError(f"student.sigmas must start at 1.0 and decrease, staying > 0: {sigmas}")
    weights = raw.get("step_weights") or [1.0] * num_steps
    if len(weights) != num_steps:
        raise ValueError(f"student.step_weights has {len(weights)} entries for {num_steps} steps")
    return {"num_steps": num_steps,
            "sampling_args": {"sigmas": sigmas, "renoise": bool(raw.get("renoise", True)),
                              "detach_between_steps": bool(raw.get("detach_between_steps", True))},
            "step_weights": [float(w) for w in weights]}


def load_student_weights(model: ZImageStudent, path: str) -> None:
    """``{"model": state_dict}`` (a step checkpoint or resume.pth) or a bare state dict."""
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=True)
    logger.info("[zimage] student weights <- %s", path)


def resolve_feature_pool(store: ZImageReferenceStore, img_size: int, setting="auto") -> int:
    """The average pool that turns a decoded ``img_size`` image into what the references were
    encoded from.

    The process pipeline encodes each kept 1024 px PNG through ``rdm.data.imagenet.
    center_crop_arr`` at ``--img-size`` (512): ADM's repeated 2x BOX downsample, then a resize to
    exactly that size and a centre crop, both no-ops for a square power-of-two ratio. So the
    references saw a ``r x r`` box average, ``r = img_size / feature size``, and the student's
    images must too. ``auto`` reads the size from ``encoder_features/<enc>.done.json``.
    """
    if setting != "auto":
        return int(setting)
    size = store.feature_image_size()
    if size is None:
        return 1
    ratio, rem = divmod(img_size, size)
    if rem or ratio < 1 or ratio & (ratio - 1):
        raise ValueError(f"references were encoded at {size} px: a {img_size} px image reaches that "
                         f"by an exact box average only for a power-of-two ratio; set "
                         f"zimage.feature_pool explicitly if you know what the crop did")
    return ratio


def check_render_config(store: ZImageReferenceStore, img_size: int) -> None:
    """The teacher renders must be ``img_size`` square; warn about a one-render-per-prompt store."""
    if store.render_root is None:
        return
    path = store.render_root / "render_config.json"
    if not path.exists():
        return
    import json
    rc = json.loads(path.read_text())
    if (int(rc.get("height", img_size)), int(rc.get("width", img_size))) != (img_size, img_size):
        raise ValueError(f"teacher renders are {rc.get('height')}x{rc.get('width')}, img_size is "
                         f"{img_size}")
    logger.info("[zimage] teacher: %s steps, %s candidates -> keep %s (%s), guidance %s",
                rc.get("steps"), rc.get("candidates"), rc.get("keep"), rc.get("curate"),
                rc.get("guidance_scale"))
    if int(rc.get("keep", 0)) == 1:
        logger.warning("[zimage] the reference holds ONE teacher render per prompt: each prompt's "
                       "reference distribution is a single image, so the paired force pulls "
                       "every seed of that prompt toward it (the FLUX store keeps 4)")


def build_trainer(cfg, device: str = "cuda") -> MultiStepTrainer:
    """Assemble the full trainer from a parsed config (needs the weights and a built store)."""
    from rdm.representation.battery import Battery
    from rdm.representation.registry import by_name

    window = window_from_config(cfg)
    if window.group_by_prompt:
        raise ValueError("the Z-Image trainer is ungrouped (as K128 · GAN)")
    policy = memory_from_config(cfg)
    batching = resolve_batching(cfg, policy, window)
    student = student_from_config(cfg)
    S = student["num_steps"]
    zcfg = _block(cfg, "zimage")
    repo = zcfg.get("repo", ZIMAGE_REPO)
    logger.info("[zimage] %d-step student, sigmas %s, renoise %s, detach %s, step weights %s | "
                "K=%d B=%d | world=%d micro_batch=%d grad_accum=%d", S,
                [round(s, 4) for s in student["sampling_args"]["sigmas"]],
                student["sampling_args"]["renoise"], student["sampling_args"]["detach_between_steps"],
                student["step_weights"], window.size, window.stride, batching["world_size"],
                batching["micro_batch"], batching["grad_accum"])

    names = list(cfg.encoders)
    store = ZImageReferenceStore(cfg.reference_root, names)
    img_size = int(getattr(cfg, "img_size", 1024))
    feature_pool = resolve_feature_pool(store, img_size, zcfg.get("feature_pool", "auto"))
    context = str(zcfg.get("context", "auto"))
    if context not in ("auto", "store", "online"):
        raise ValueError(f"zimage.context must be auto | store | online, got {context!r}")
    if context == "store" and not store.has_context:
        raise ValueError(f"zimage.context: store, but {cfg.reference_root} has no z_context.json")
    use_store_context = store.has_context and context != "online"
    check_render_config(store, img_size)
    logger.info("[zimage] store %s: %d rows / %d prompts | condition: %s | features: decoded %d px "
                "%s", cfg.reference_root, store.num_rows, store.num_prompts,
                "precomputed Qwen3 contexts" if use_store_context else "Qwen3 encoder online",
                img_size, (f"-> PNG + ADM crop to {img_size // feature_pool} px (exact forward, "
                           f"{feature_pool}x{feature_pool} average-pool gradient)"
                           if zcfg.get("feature_quantize", True) else
                           f"-> {feature_pool}x{feature_pool} average pool")
                if feature_pool > 1 else "as is")
    schedule = SlidingWindowSchedule(store.row_order(window.order_seed), window_size=window.size,
                                     stride=window.stride, cyclic=window.cyclic)

    model = ZImageStudent(image_resolution=img_size, repo=repo,
                          param_dtype=policy.param_dtype,
                          gradient_checkpointing=bool(getattr(cfg, "grad_checkpointing", True)))
    if getattr(cfg, "load_from", None):
        load_student_weights(model, cfg.load_from)
    if policy.shard != "fsdp":
        model = model.to(device)
    generator = ZImageGenerator(model, vae=None, text=None,
                                sampling_args=student["sampling_args"])
    # Shard before anything else loads: unsharded, every rank holds the whole fp32 student.
    clip_module = shard_generator(generator, model, policy, device)
    generator.vae = ZImageVAE(repo, device=device,
                              gradient_checkpointing=bool(zcfg.get("vae_checkpointing", True)))
    if not use_store_context:
        generator.text = QwenPromptEncoder(
            repo, device=device, max_sequence_length=int(zcfg.get("max_sequence_length", 512)),
            cache_size=int(zcfg.get("text_cache", 512)))

    battery = Battery([by_name(n) for n in names], device=device)
    if policy.battery_bf16:
        for enc in battery.encoders.values():
            if not getattr(enc, "has_logits", False):
                enc.to(torch.bfloat16)

    gan, critics = gan_from_config(cfg), None
    if gan.enabled:
        from rdm.sw_lmmd.adversarial import build_critic
        if gan.critic != "row":
            raise ValueError("only gan.critic: row is wired per step")
        warn_gan_pipeline(gan, policy)
        # one critic per step, each from its own fixed seed (the same on every rank)
        critics = [build_critic(dataclasses.replace(gan, seed=gan.seed + k), store, names,
                                joint=getattr(cfg, "joint", True), device=device)
                   for k in range(S)]
        logger.info("[zimage] %d critics (one per step): heads %s | %.1f M params each | weight %g "
                    "(%s) | %s loss | lr %g | generator term from step %d", S, critics[0].names,
                    sum(p.numel() for p in critics[0].critic.parameters()) / 1e6, gan.weight,
                    "adaptive" if gan.adaptive else "fixed", gan.loss, gan.lr, gan.g_start_step)

    lr_schedule = lr_schedule_from_config(cfg)
    cache_cfg = cache_from_config(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    return MultiStepTrainer(
        generator, battery, store, schedule, build_optimizer(params, cfg, policy),
        encoders=names, num_steps=S, step_weights=student["step_weights"], critics=critics,
        feature_pool=feature_pool, use_store_context=use_store_context,
        feature_quantize=bool(zcfg.get("feature_quantize", True)),
        noise_shape=(S, model.in_channels, model.input_size, model.input_size),
        cache=GeneratedWindowCache(stream_names(S, names), store_dtype=cache_cfg.store_dtype),
        micro_batch=policy.micro_batch, grad_clip=getattr(cfg, "grad_clip", 4.0),
        kernel_block=int(_block(cfg, "loss").get("kernel_block_size", 256)),
        seed=getattr(cfg, "seed", 0), joint=getattr(cfg, "joint", True),
        grad_reduce=policy.grad_reduce, monitor_every=getattr(cfg, "monitor_every", 1),
        cache_cfg=cache_cfg, clip_module=clip_module, device=device, lr_schedule=lr_schedule)


def main() -> None:
    """``torchrun --nproc_per_node=4 -m zimage_distill.launch <config.yaml> [--set k=v ...]``"""
    import argparse

    from rdm.train.launch import load_config
    from rdm.utils.distributed import setup_distributed
    from rdm.utils.logging import setup_logging
    from rdm.utils.seed import fix_random_seeds

    ap = argparse.ArgumentParser(description="Z-Image few-step SW-LMMD training")
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for override in args.set:
        apply_override(cfg, override)
    if getattr(cfg, "method", None) != "sw_lmmd" or getattr(cfg, "mode", None) != "zimage":
        raise SystemExit(f"{args.config} is not a Z-Image SW-LMMD config (method: sw_lmmd, "
                         f"mode: zimage)")
    student_from_config(cfg)                         # fail on a bad student block before loading

    rank, world, local_rank = setup_distributed()
    setup_logging(rank=rank)
    fix_random_seeds(getattr(cfg, "seed", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    train(cfg, device=device, trainer=build_trainer(cfg, device=device))


if __name__ == "__main__":
    main()
