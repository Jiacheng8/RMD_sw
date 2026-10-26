#!/usr/bin/env python
"""Render a prompt list with a trained few-step student (or the untrained Turbo weights).

    PYTHONPATH=z-image python -m zimage_distill.sample --ckpt <run>/step_0002000.pth \
        --prompts <file> --out <dir> [--seeds 1] [--steps-from-ckpt]

Writes ``{prompt:05d}_s{seed}.png`` (1024 px, the last step's image; ``--all-steps`` also writes
every earlier step's ``x0_hat`` as ``..._k{k}.png``). The sampler is the student's own:
``sigmas`` / ``renoise`` from the checkpoint (written by the trainer), or from the flags.
Seeds map to latents as a pure function of (prompt, seed) (``noise_for``), and the
re-noise draws are seeded per (prompt, seed, step), so a run is reproducible.
"""
from __future__ import annotations

import argparse
import os

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="a step_*.pth / resume.pth; none = Turbo weights")
    ap.add_argument("--sigmas", default=None, help="comma list; default: the checkpoint's, else "
                                                   "the Turbo schedule for --num-steps")
    ap.add_argument("--num-steps", type=int, default=2)
    ap.add_argument("--no-renoise", action="store_true", help="Euler ODE between steps")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--all-steps", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()

    from rdm.utils.io import save_uint8_png
    from .model import (ZIMAGE_REPO, QwenPromptEncoder, ZImageStudent, ZImageVAE, noise_for,
                        turbo_sigmas)
    from .store import read_prompts

    repo = args.repo or ZIMAGE_REPO
    sampling = {"sigmas": turbo_sigmas(args.num_steps), "renoise": not args.no_renoise,
                "detach_between_steps": True}
    model = ZImageStudent(args.img_size, repo=repo, param_dtype=torch.bfloat16,
                          gradient_checkpointing=False)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(state.get("model", state), strict=True)
        sampling.update(state.get("sampling_args") or {})
        del state
    if args.sigmas:
        sampling["sigmas"] = [float(s) for s in args.sigmas.split(",")]
    model = model.to("cuda").eval()
    vae = ZImageVAE(repo, device="cuda", gradient_checkpointing=False)
    text = QwenPromptEncoder(repo, device="cuda", cache_size=args.batch)
    print(f"[sample] {args.ckpt or 'Turbo weights'} | sigmas {sampling['sigmas']} | renoise "
          f"{sampling['renoise']}", flush=True)

    prompts = read_prompts(args.prompts)
    side, S = args.img_size // 8, len(sampling["sigmas"])
    jobs = [(p, s) for p in range(len(prompts)) for s in range(args.seeds)][args.rank::args.world]
    os.makedirs(args.out, exist_ok=True)
    with torch.no_grad():
        for lo in range(0, len(jobs), args.batch):
            sub = [j for j in jobs[lo:lo + args.batch]
                   if not os.path.exists(os.path.join(args.out, f"{j[0]:05d}_s{j[1]}.png"))]
            if not sub:
                continue
            noise = torch.stack([torch.stack([noise_for(p, s, (16, side, side), "cuda")] +
                                             [noise_for(p, 1000 * (k + 1) + s, (16, side, side), "cuda")
                                              for k in range(S - 1)]) for p, s in sub])
            cap = text([prompts[p] for p, _ in sub])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = model(noise, cap, sampling)
            for k, z in enumerate(outs, 1):
                if k < S and not args.all_steps:
                    continue
                pixels = vae.decode(z)
                for (p, s), img in zip(sub, pixels):
                    tag = "" if k == S else f"_k{k}"
                    save_uint8_png(img, os.path.join(args.out, f"{p:05d}_s{s}{tag}.png"))
    print(f"[sample] {len(jobs)} images -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
