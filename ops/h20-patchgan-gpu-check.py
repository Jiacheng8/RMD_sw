#!/usr/bin/env python3
"""Single-GPU, real-weight 1024px spatial PatchGAN gradient/contract oracle.

No training job, dataset writer or download is started. The default input is a
synthetic BF16 VAE decode with an exact device-side PNG integer round trip.
Optional existing image manifests are read only, with partial access explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
for directory in (REPO, REPO / "z-image"):
    sys.path.insert(0, str(directory))
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EXPECTED_SHAPES = {"qwen3vl_4b": (1, 2560, 32, 32), "dinov3_l": (1, 1024, 64, 64)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/rdm-workspace/rdm-sets"))
    parser.add_argument("--qwen-snapshot", type=Path)
    parser.add_argument("--qwen-receipt", type=Path)
    parser.add_argument("--dino-checkpoint", type=Path)
    parser.add_argument("--dino-receipt", type=Path)
    parser.add_argument("--image-manifest", type=Path, help="Existing immutable manifest; never opens a live writer")
    parser.add_argument("--allow-partial-images", action="store_true", help="Single-image pilot only")
    parser.add_argument("--image-row", type=int)
    parser.add_argument("--text-features", type=Path)
    parser.add_argument("--text-row", type=int, default=0)
    parser.add_argument("--critic-hidden", type=int, default=1024)
    parser.add_argument("--critic-depth", type=int, default=2)
    parser.add_argument("--critic-heads", type=int, default=8)
    parser.add_argument("--loss", choices=("hinge", "ns"), default="hinge")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-initial-used-mib", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def resolve_paths(args):
    qreceipt = args.qwen_receipt or args.root / "control/patchgan-qwen3vl-prefetch-verified.json"
    receipt = json.loads(qreceipt.read_text())
    require(receipt.get("repo") == "Qwen/Qwen3-VL-4B-Instruct" and receipt.get("revision") == REVISION
            and receipt.get("all_official_hashes_verified") is True, "complete pinned Qwen receipt required")
    snapshot = args.qwen_snapshot or Path(receipt["snapshot"])
    dreceipt = args.dino_receipt or args.root / "control/fast-prefetch-timm--vit_large_patch16_dinov3.lvd1689m-verified.json"
    dino_receipt = json.loads(dreceipt.read_text())
    candidates = [row for row in dino_receipt["files"] if Path(row["path"]).name in
                  ("model.safetensors", "pytorch_model.bin")]
    if args.dino_checkpoint is not None:
        matching = [row for row in candidates if Path(row["path"]).resolve() == args.dino_checkpoint.resolve()]
        require(len(matching) == 1, "explicit DINO checkpoint must match one official verified receipt file")
        checkpoint, evidence = args.dino_checkpoint, matching[0]
    else:
        require(len(candidates) == 1, "provide --dino-checkpoint when the DINO receipt has multiple weight files")
        checkpoint, evidence = Path(candidates[0]["path"]), candidates[0]
    expected = evidence.get("sha256", evidence.get("digest"))
    require(isinstance(expected, str) and len(expected) == 64 and checkpoint.stat().st_size == evidence["bytes"],
            "DINO official weight size/SHA evidence missing")
    require(sha256(checkpoint) == expected, "DINO checkpoint official SHA-256 mismatch")
    return snapshot, checkpoint, {"qwen_receipt": str(qreceipt), "qwen_receipt_sha256": sha256(qreceipt),
            "qwen_snapshot": str(snapshot), "dino_receipt": str(dreceipt),
            "dino_checkpoint": str(checkpoint), "dino_sha256": expected}


def gradient_summary(torch, value):
    require(value is not None, "gradient is absent")
    require(bool(torch.isfinite(value).all()), "gradient is nonfinite")
    norm = float(value.float().norm())
    require(norm > 0, "gradient is identically zero")
    return {"norm": norm, "max_abs": float(value.float().abs().max()),
            "nonzero": int(torch.count_nonzero(value)), "finite": True}


def main(argv=None):
    args = arguments(argv)
    output = args.output or args.root / "control/patchgan-gpu-check.json"
    report = {"passed": False, "status": "running", "scope": "single_image_real_weights_gradient_oracle",
              "formal_training_iteration": False, "started_epoch": time.time(), "gates": {},
              "script_sha256": sha256(__file__), "critic_options": {"hidden": args.critic_hidden,
              "depth": args.critic_depth, "heads": args.critic_heads, "spectral_norm": True},
              "actual_hidden1024": args.critic_hidden == 1024, "loss_kind": args.loss}
    atomic_json = None
    started = time.monotonic()
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "4", "require exactly CUDA_VISIBLE_DEVICES=4")
        require(args.root.is_absolute() and args.root.is_dir() and output.is_absolute(), "existing absolute remote root/output required")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        import torch
        import torch.nn.functional as F
        from rdm.sw_lmmd.reference_images import atomic_json, ReferenceImageStore
        from zimage_distill.model import reference_pixels
        from rdm.representation.backbones.qwen3_vl_spatial import (
            Qwen3VLSpatialEncoder, INSTRUCTION_SHA256, _load_processor, FIXED_INSTRUCTION,
        )
        from rdm.representation.backbones.dinov3_spatial import DINOv3SpatialBackbone
        from rdm.sw_lmmd.spatial_adversarial import (
            SpatialFeatureCritic, spatial_discriminator_loss, spatial_generator_loss,
        )

        def stage(name, **values):
            torch.cuda.synchronize()
            report["current_stage"] = name
            report["elapsed_seconds"] = time.monotonic() - started
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            atomic_json(output, report)
            print(json.dumps({"event": name, "elapsed_seconds": report["elapsed_seconds"],
                              "peak_allocated_bytes": report["peak_allocated_bytes"], **values}), flush=True)

        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one visible CUDA device required")
        require(torch.__version__.split("+")[0].startswith("2.8."), "validated torch2.8 runtime required")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        require(total - free <= args.max_initial_used_mib * 1024**2, "physicalGPU4 already has memory users")
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(args.seed)
        report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
             "cudnn": torch.backends.cudnn.version(), "physical_gpu": 4,
             "device_name": torch.cuda.get_device_properties(device).name, "total_bytes": total,
             "initial_used_bytes": total-free, "autocast": "bfloat16", "critic_dtype": "float32",
             "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "packages": {name: importlib.metadata.version(name) for name in ("transformers", "timm", "Pillow")}}
        stage("verify_prepared_weights")
        snapshot, dino_path, evidence = resolve_paths(args)
        report["weights"] = evidence

        raw = torch.randn((1, 3, 1024, 1024), device=device).clamp(-1, 1).to(torch.bfloat16)
        decoded = (raw.float() * 0.5 + 0.5).clamp(0, 1).detach().requires_grad_()
        native_uint8 = torch.round(255 * (raw / 2 + 0.5).clamp(0, 1).float()).to(torch.uint8)
        # The uint8 CPU trip models PNG storage; division occurs only after transfer.
        png_pixels = native_uint8.cpu().to(device=device).float().div(255)
        fake_pixels = reference_pixels(decoded, factor=1, quantize=True)
        require(torch.equal(png_pixels, fake_pixels), "GPU PNG/fake pixel forward differs by at least one bit")
        pixel_gradient = torch.autograd.grad(fake_pixels.sum(), decoded)[0]
        require(torch.equal(pixel_gradient, torch.ones_like(pixel_gradient)), "PNG quantization STE is not identity")
        report["gates"]["pixel_oracle"] = {"bitwise_equal": True, "decoded_raw_dtype": str(raw.dtype),
             "uint8_to_device_before_float_divide": True, "factor": 1, "size": [1024, 1024],
             "ste_identity": True, "uint8_sha256": hashlib.sha256(native_uint8.cpu().numpy().tobytes()).hexdigest()}
        real_pixels = png_pixels.detach()
        if args.image_manifest:
            images = ReferenceImageStore(args.image_manifest, allow_partial=args.allow_partial_images)
            row = args.image_row if args.image_row is not None else next(iter(images.available_row_ids))
            pixels_uint8 = images.read_uint8([row])
            real_pixels = pixels_uint8.to(device=device).float().div(255)
            report["image_source"] = {"manifest": str(images.manifest_path), "complete": images.is_complete,
                                      "row": row, "record": images.records([row])[0]}
            images.close()
        else:
            report["image_source"] = {"synthetic": True, "contract": "BF16 native decode to PNG integers"}
        del raw, decoded, native_uint8, fake_pixels, pixel_gradient
        stage("load_qwen_spatial24")
        qwen = Qwen3VLSpatialEncoder(snapshot_dir=snapshot, device=device,
                                    parameter_dtype=torch.bfloat16, attention_backend=args.attention_backend)
        stage("load_dino_spatial1024")
        dino = DINOv3SpatialBackbone(device=device, checkpoint_path=dino_path)
        encoders = {"qwen3vl_4b": qwen, "dinov3_l": dino}
        contracts = {name: model.feature_contract() for name, model in encoders.items()}
        qc, dc = contracts["qwen3vl_4b"], contracts["dinov3_l"]
        require(qc["instruction_sha256"] == INSTRUCTION_SHA256 and qc["instruction_before_image"]
            and qc["decoder_blocks"] == 24 and qc["original_decoder_blocks"] == 36
            and qc["deepstack_decoder_layers"] == [0, 1, 2] and qc["image_grid_thw"] == [1, 64, 64]
            and qc["image_tokens"] == 1024 and qc["input_res"] == 1024
            and qc["output_shape"] == [2560, 32, 32] and qc["final_norm"] is False
            and qc["lm_head"] is False and qc["weight_dtype"] == "torch.bfloat16", "Qwen spatial contract mismatch")
        require(dc["input_size"] == 1024 and dc["grid_size"] == 64 and dc["channels"] == 1024
                and dc["weight_dtype"] == "torch.float32" and dc["prefix_tokens_removed"] >= 1,
                "DINO1024/prefix/FP32 contract mismatch")
        require(len(qwen.model.language_model.layers) == 24
                and isinstance(qwen.model.language_model.norm, torch.nn.Identity)
                and not hasattr(qwen.model, "lm_head"), "Qwen decoder did not stop before final norm at24")
        require(all(not parameter.requires_grad for model in encoders.values() for parameter in model.parameters()),
                "encoder parameter unexpectedly trainable")
        processor = _load_processor(snapshot)
        template = processor.apply_chat_template([{"role": "user", "content": [
            {"type": "text", "text": FIXED_INSTRUCTION}, {"type": "image"}]}],
            tokenize=False, add_generation_prompt=False)
        oracle = processor(text=[template], images=[(real_pixels * 255).round().to(torch.uint8)[0].cpu()],
                           return_tensors="pt", add_special_tokens=False)
        for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
            require(torch.equal(oracle[name], getattr(qwen, name).cpu()), "official processor sequence mismatch: " + name)
        require(torch.equal(oracle["image_grid_thw"], torch.tensor([[1, 64, 64]])), "official processor image grid mismatch")
        image_mask = qwen.input_ids.eq(qwen.image_token_id)
        require(int(image_mask.sum()) == 1024 and torch.equal(qwen.mm_token_type_ids.bool(), image_mask),
                "native image mask does not select exactly1024 image tokens")
        report["gates"]["contracts"] = {"encoders": contracts, "official_processor_sequence_equal": True,
                                           "image_mask_count": int(image_mask.sum())}
        del processor, oracle

        deepstack = []
        language = qwen.model.language_model
        require(callable(getattr(language, "_deepstack_process", None)), "native DeepStack injection API absent")
        original_deepstack = language._deepstack_process

        def observe_deepstack(*args, **kwargs):
            bound = inspect.signature(original_deepstack).bind(*args, **kwargs).arguments
            hidden, mask = bound["hidden_states"], bound["visual_pos_masks"]
            before = hidden.detach().clone()
            result = original_deepstack(*args, **kwargs)
            with torch.no_grad():
                require(int(mask.sum()) == 1024, "DeepStack image mask count mismatch")
                delta = result.detach() - before
                require(bool(torch.isfinite(delta).all()) and float(delta[mask].abs().max()) > 0,
                        "DeepStack did not inject finite nonzero visual features")
                require(torch.equal(result.detach()[~mask], before[~mask]), "DeepStack changed non-image positions")
                deepstack.append({"image_positions": int(mask.sum()), "image_delta_max": float(delta[mask].abs().max()),
                                  "text_positions_unchanged": True})
            return result

        language._deepstack_process = observe_deepstack
        baseline, raw_baseline = {}, {}
        try:
            for name, model in encoders.items():
                stage("no_grad_" + name)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    value = model(real_pixels)
                require(tuple(value.shape) == EXPECTED_SHAPES[name] and bool(torch.isfinite(value).all()),
                        name + " spatial shape/nonfinite failure")
                raw_baseline[name] = value.detach()
                baseline[name] = value.detach().to(torch.bfloat16).float()
        finally:
            language._deepstack_process = original_deepstack
        require(len(deepstack) == 3, "native DeepStack must inject once after each of its first3 language blocks")
        report["gates"]["deepstack"] = {"native_injection_calls": deepstack, "count": len(deepstack)}

        stage("build_actual_spatial_critic")
        with torch.device(device):
            critic = SpatialFeatureCritic(hidden=args.critic_hidden, depth=args.critic_depth,
                heads=args.critic_heads, text_dim=2560, spectral_norm=True)
        text_path = args.text_features or args.root / "zimage/zimage-ref-v1/reference_store_qwen/text_features.npy"
        import numpy as np
        text_array = np.load(text_path, mmap_mode="r")
        tau = torch.from_numpy(np.asarray(text_array[args.text_row:args.text_row+1], dtype=np.float32).copy()).to(device)
        require(tuple(tau.shape) == (1, 2560) and bool(torch.isfinite(tau).all()), "caption tau must be finite Bx2560")
        tau.requires_grad_()
        report["text_condition"] = {"path": str(text_path), "row": args.text_row, "shape": list(tau.shape)}
        records = {name: [] for name in critic.names}
        # SpatialFeatureCritic uses head.aggregate()/head.f() directly. The final
        # readout module is the observable logit boundary on both API paths.
        hooks = [head.readout.register_forward_hook(lambda module, inputs, result, name=name: records[name].append(result))
                 for name, head in critic.heads.items()]
        # A distinct synthetic feature pair avoids the zero D gradient of identical
        # real/fake logits. These losses are routing oracles, not quality metrics.
        real_leaves = {name: value.detach().clone().requires_grad_() for name, value in baseline.items()}
        fake_leaves = {name: (-value).detach().requires_grad_() for name, value in baseline.items()}
        critic.train()
        d_losses = spatial_discriminator_loss(critic, real_leaves, fake_leaves, tau, kind=args.loss)
        for hook in hooks:
            hook.remove()
        d_report = {}
        for name, loss in d_losses.items():
            real_logit, fake_logit = records[name]
            require(tuple(real_logit.shape) == (1, 64, 1) and tuple(fake_logit.shape) == (1, 64, 1), "critic output must be B64x1")
            reference = (torch.relu(1-real_logit).mean() + torch.relu(1+fake_logit).mean()
                         if args.loss == "hinge" else
                         F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean())
            require(torch.equal(loss, reference) and bool(torch.isfinite(loss)),
                    f"per-region D {args.loss} reduction mismatch")
            d_report[name] = {"loss": float(loss.detach()), "loss_kind": args.loss,
                              "logit_shape": list(real_logit.shape)}
        sum(d_losses.values()).backward()
        require(all(value.grad is None for value in (*real_leaves.values(), *fake_leaves.values())) and tau.grad is None,
                "D phase leaked gradients through detached encoder/text features")
        for name, head in critic.heads.items():
            d_report[name]["parameter_gradient"] = gradient_summary(torch, torch.cat([
                parameter.grad.detach().float().reshape(-1) for parameter in head.parameters() if parameter.grad is not None]))
            with torch.no_grad():
                regions = head.aggregate(baseline[name])
            require(tuple(regions.shape) == (1, 64, args.critic_hidden), "learned aggregation must produce B64xD")
            d_report[name]["region_shape"] = list(regions.shape)
        report["gates"]["D"] = {"branches": d_report, "loss_kind": args.loss, "input_gradients_detached": True,
                                    "synthetic_fake_feature_pair": "negative baseline; routing oracle only"}
        critic.zero_grad(set_to_none=True)
        del d_losses, records, real_leaves, fake_leaves
        report["gates"]["G"] = {}
        for name, model in encoders.items():
            stage("grad_and_frozen_G_" + name)
            # This leaf already holds the exact PNG-forward values. The decoded
            # ->PNG STE's identity derivative was verified above; do not apply
            # BF16 postprocessing to an already quantized PNG a second time.
            generated = real_pixels.detach().clone().requires_grad_()
            pixels = generated
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = model(pixels)
            boundary = value.to(torch.bfloat16).float()
            require(torch.equal(boundary.detach(), baseline[name]), name + " no_grad/grad BF16 feature boundary differs")
            versions = {key: parameter._version for key, parameter in critic.named_parameters()}
            buffers = {key: buffer.detach().clone() for key, buffer in critic.named_buffers()}
            observed, logits = [], {}

            def frozen_pre(module, inputs):
                require(not module.training and all(not p.requires_grad for p in critic.parameters()), "G critic was not frozen/eval")
                observed.append(True)

            pre = critic.register_forward_pre_hook(frozen_pre)
            hooks = [head.readout.register_forward_hook(lambda module, inputs, result, key=key: logits.__setitem__(key, result))
                     for key, head in critic.heads.items()]
            try:
                maps = dict(baseline)
                maps[name] = boundary
                g_losses = spatial_generator_loss(critic, maps, tau, kind=args.loss)
                reference = (-logits[name].mean() if args.loss == "hinge" else
                             F.softplus(-logits[name]).mean())
                require(observed and torch.equal(g_losses[name], reference),
                        f"G {args.loss} loss is not the per-region Bx64 mean")
                g_losses[name].backward()
            finally:
                pre.remove()
                for hook in hooks:
                    hook.remove()
            require(all(p.grad is None for p in critic.parameters()) and tau.grad is None, "G backward accumulated critic/text gradients")
            require(all(p._version == versions[key] for key, p in critic.named_parameters()), "G changed critic weights")
            require(all(torch.equal(buffer, buffers[key]) for key, buffer in critic.named_buffers()), "G changed critic buffers")
            require(all(p.grad is None and not p.requires_grad for encoder in encoders.values() for p in encoder.parameters()),
                    "frozen encoder accumulated parameter gradients")
            report["gates"]["G"][name] = {"loss": float(g_losses[name].detach()), "loss_kind": args.loss,
                "pixel_gradient": gradient_summary(torch, generated.grad), "spatial_shape": list(value.shape),
                "gradient_leaf": "exact PNG-forward FP32 values; quantizer identity STE checked separately",
                "raw_no_grad_grad_max_difference": float((value.detach().float()-raw_baseline[name].float()).abs().max()),
                "BF16_boundary_bitwise_equal": True, "critic_gradients_absent": True,
                "encoder_gradients_absent": True, "critic_parameter_versions_unchanged": True,
                "critic_buffers_bitwise_unchanged": True}
            del value, boundary, generated, pixels, g_losses, logits, maps, buffers
        stage("all_single_image_gates_verified")
        report.update(passed=True, status="complete", finished_epoch=time.time(), elapsed_seconds=time.monotonic()-started)
        atomic_json(output, report)
        print(json.dumps({"event": "passed", "receipt": str(output), "peak_allocated_bytes": report["peak_allocated_bytes"]}), flush=True)
        return 0
    except Exception as error:
        report.update(passed=False, status="failed", failed_epoch=time.time(), error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        if atomic_json is not None:
            atomic_json(output, report)
        traceback.print_exc()  # Preserve the actual exception and full call stack.
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
