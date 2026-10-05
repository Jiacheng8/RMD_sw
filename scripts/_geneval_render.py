#!/usr/bin/env python
"""One rank of the GenEval-block teacher render (step G2): 4-step klein-4B on the GenEval prompts.

Renders ``--seeds`` candidates for every GenEval prompt this rank owns, in the layout the
official scorer reads, so step G3 can judge every candidate and step G4 keeps the correct ones:

    <out>/shard_<rank>/<prompt:05d>/{metadata.jsonl, samples/<seed:04d>.png}

Prompt ``p`` belongs to rank ``p % world``; each rank's shard is scored by its own scorer
process. Noise seeds start at ``SEED_BASE`` -- far from the evaluation protocol's base
(46,000,000 + idx*100 + j), so no training reference shares a latent with an eval sample.
Existing PNGs are skipped, so an interrupted run resumes by relaunching.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

SEED_BASE = 910_000_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", required=True, help="qwen3 context of the GenEval prompts (553, L, d)")
    ap.add_argument("--prompts", required=True, help="assets/geneval_prompts.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=128, help="candidates rendered per prompt")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help=">0 renders only the first N prompts")
    args = ap.parse_args()

    from rdm.representation.generators import build_generator
    from rdm.representation.generators.flux_generator import Flux2AdapterModel, Flux2VAETokenizer
    from rdm.utils.io import save_uint8_png

    meta = [json.loads(line) for line in open(args.prompts) if line.strip()]
    ctx_arr = np.load(args.ctx, mmap_mode="r")
    if ctx_arr.shape[0] != len(meta):
        raise SystemExit(f"context has {ctx_arr.shape[0]} rows for {len(meta)} prompts")
    n_prompts = len(meta) if not args.limit else min(args.limit, len(meta))
    mine = [p for p in range(n_prompts) if p % args.world == args.rank]
    shard = os.path.join(args.out, f"shard_{args.rank}")

    todo = []
    for p in mine:
        pdir = os.path.join(shard, f"{p:05d}")
        os.makedirs(os.path.join(pdir, "samples"), exist_ok=True)
        with open(os.path.join(pdir, "metadata.jsonl"), "w") as f:
            json.dump(meta[p], f)
        todo += [(p, s) for s in range(args.seeds)
                 if not os.path.exists(os.path.join(pdir, "samples", f"{s:04d}.png"))]
    if not todo:
        print(f"[geneval render {args.rank}/{args.world}] shard complete", flush=True)
        return 0

    model = Flux2AdapterModel(image_resolution=args.img_size, param_dtype=torch.bfloat16,
                              gradient_checkpointing=False).to("cuda").eval()
    generator = build_generator("flux", model, {"num_steps": args.steps}, args=None,
                                tokenizer=Flux2VAETokenizer(device="cuda"))
    t0 = time.time()
    with torch.no_grad():
        for lo in range(0, len(todo), args.batch):
            sub = todo[lo:lo + args.batch]
            ctx = torch.stack([torch.from_numpy(np.ascontiguousarray(ctx_arr[p]))
                               for p, _ in sub]).to("cuda", torch.bfloat16)
            noise = torch.stack([
                torch.randn(model.in_channels, model.input_size, model.input_size,
                            generator=torch.Generator(device="cuda").manual_seed(
                                SEED_BASE + p * 10_000 + s), device="cuda")
                for p, s in sub]).to(torch.bfloat16)
            out = generator.sample(noise, ctx)                     # (b,3,H,W) in [0,1]
            for (p, s), img in zip(sub, out):
                save_uint8_png(img.float().clamp(0, 1),
                               os.path.join(shard, f"{p:05d}", "samples", f"{s:04d}.png"))
            done = lo + len(sub)
            if done % 500 < args.batch:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"[geneval render {args.rank}/{args.world}] {done}/{len(todo)} "
                      f"({rate:.2f} img/s, ~{(len(todo) - done) / rate / 60:.0f} min left)",
                      flush=True)
    print(f"[geneval render {args.rank}/{args.world}] rendered {len(todo)} in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
