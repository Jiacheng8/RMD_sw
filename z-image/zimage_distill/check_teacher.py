#!/usr/bin/env python
"""M0: check the sampler against ``ZImagePipeline`` and look at the teacher's few-step baselines.

    PYTHONPATH=z-image python -m zimage_distill.check_teacher --prompts <file> --out <dir>

For the first ``--n`` prompts (seed 0 each), at 1024 px:

1. **Parity.** The pipeline at ``num_inference_steps=9`` against :class:`ZImageStudent` run as an
   ODE on ``turbo_sigmas(9)`` from the same latent, at the pipeline's precision (bf16 input, no
   autocast): the final latents must be identical. This is what licenses rendering the
   references with the pipeline and training with the student's own loop. (The bf16 transformer
   is sensitive: an fp32-ulp change of ``t`` moves one forward by ~4 %, so any precision change,
   autocast included, gives a visibly different but equally valid sample.)
2. **NFE.** Transformer forwards per pipeline call, counted with a hook (the model card says 9
   inference steps are 8 forwards; this prints what this diffusers version does).
3. **Baselines.** The teacher run directly with 9, 4 and 2 steps (``turbo_sigmas``), and the
   2-step re-noised sampler the student trains with, saved side by side, plus PickScore,
   seconds per image and peak memory.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()

    from diffusers import ZImagePipeline

    from rdm.eval.pickscore_eval import PickScorer
    from rdm.utils.io import save_uint8_png
    from .model import (ZIMAGE_REPO, ZImageStudent, ZImageVAE, noise_for, resolve_snapshot,
                        turbo_sigmas)
    from .store import read_prompts

    os.makedirs(args.out, exist_ok=True)
    prompts = read_prompts(args.prompts)[:args.n]
    pipe = ZImagePipeline.from_pretrained(resolve_snapshot(args.repo or ZIMAGE_REPO),
                                          torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    # encode every prompt up front, then move Qwen3 off the GPU: the rest fits a 24 GB card
    caps = [[c.detach() for c in pipe.encode_prompt(p, device="cuda",
                                                    do_classifier_free_guidance=False)[0]]
            for p in prompts]
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    student = ZImageStudent(args.img_size, transformer=pipe.transformer,
                            param_dtype=torch.bfloat16, gradient_checkpointing=False).eval()
    vae = ZImageVAE(vae=pipe.vae, device="cuda", gradient_checkpointing=False)
    scorer = PickScorer(device="cuda")
    side = args.img_size // 8

    calls = {"n": 0}
    pipe.transformer.register_forward_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))

    report = {"prompts": len(prompts), "parity": [], "pickscore": {}, "seconds_per_image": {}}
    variants = {"teacher_9": dict(sigmas=turbo_sigmas(9), renoise=False),
                "teacher_4": dict(sigmas=turbo_sigmas(4), renoise=False),
                "teacher_2": dict(sigmas=turbo_sigmas(2), renoise=False),
                "renoise_2": dict(sigmas=turbo_sigmas(2), renoise=True)}
    scores = {k: [] for k in variants}
    seconds = {k: 0.0 for k in variants}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            latent = noise_for(i, 0, (16, side, side), "cuda")[None]
            calls["n"] = 0
            z_pipe = pipe(prompt_embeds=caps[i], height=args.img_size, width=args.img_size,
                          num_inference_steps=9, guidance_scale=0.0, latents=latent.clone(),
                          output_type="latent").images.float()
            nfe = calls["n"]
            cap = caps[i]
            row = {}
            for name, sa in variants.items():
                noise = torch.stack([latent[0]] + [noise_for(i, 1000 + k, latent.shape[1:], "cuda")
                                                   for k in range(len(sa["sigmas"]) - 1)], 0)[None]
                torch.cuda.synchronize()
                t = time.time()
                # teacher_9 at the pipeline's precision (bf16 input, no autocast): must match it
                # exactly; the rest under autocast, as in training
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=name != "teacher_9"):
                    z = student(noise, cap, {**sa, "detach_between_steps": True})[-1]
                pixels = vae.decode(z)
                torch.cuda.synchronize()
                seconds[name] += time.time() - t
                scores[name].append(float(scorer.score(pixels, [prompt])[0]))
                save_uint8_png(pixels[0], os.path.join(args.out, f"{i:03d}_{name}.png"))
                row[name] = z
            diff_z = (row["teacher_9"] - z_pipe).abs()
            pix_pipe, pix_ours = vae.decode(z_pipe), vae.decode(row["teacher_9"])
            mse = float(((pix_pipe - pix_ours) ** 2).mean())
            report["parity"].append({"prompt": i, "nfe_pipeline": nfe,
                                     "latent_max_abs": float(diff_z.max()),
                                     "latent_rel_l2": float(diff_z.norm() / z_pipe.norm()),
                                     "pixel_psnr": float(10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))))})
            print(json.dumps(report["parity"][-1]), flush=True)
    report["pickscore"] = {k: sum(v) / len(v) for k, v in scores.items()}
    report["seconds_per_image"] = {k: v / len(prompts) for k, v in seconds.items()}
    report["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("pickscore", "seconds_per_image", "peak_gb")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
