#!/usr/bin/env python
"""One rank of the teacher render + PickScore curation (step 02).

Reproduces the paper's COCO reference block: render ``--seeds`` 4-step teacher samples per
caption, score them with PickScore, keep the top ``--keep``. That curation is not a detail --
it is why the reference is *better* than the teacher's average output, and therefore why a
one-step student trained against it can match or exceed the four-step teacher it came from.
Training against uncurated renders targets the teacher's mean and inherits its failures.

Curation is **streamed**: a chunk of prompts is rendered into RAM, scored, and only the kept
images are written. The paper's 24-seed recipe over 82,783 captions is ~2M renders, which
would be ~795 GB if every candidate hit the disk before being deleted; here the peak is one
chunk (~1 GB of uint8) and only 3 images per prompt are ever written.

Output is ``{prompt:08d}_k{rank}.png`` with rank ``0..keep-1`` in descending PickScore, so
downstream sees a uniform ``keep`` rows per prompt whether or not curation ran. A manifest
records the seed and score behind every kept image, so the selection is auditable and a rerun
is reproducible.

Chunks are assigned round-robin by rank and skipped when already complete, so an interrupted
run resumes by relaunching.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch


def _chunk_done(out_dir: str, prompts: list[int], keep: int) -> bool:
    return all(os.path.exists(os.path.join(out_dir, f"{p:08d}_k{k}.png"))
               for p in prompts for k in range(keep))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", required=True, help="qwen3 context .npy (prompt-indexed)")
    ap.add_argument("--captions", required=True, help="coco_pairs.npz (PickScore text side)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=24, help="candidates rendered per prompt")
    ap.add_argument("--keep", type=int, default=3, help="kept per prompt after scoring")
    ap.add_argument("--curate", default="pickscore", choices=["pickscore", "none"])
    ap.add_argument("--chunk-prompts", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from rdm.data.coco import load_coco_pairs
    from rdm.representation.generators import build_generator
    from rdm.representation.generators.flux_generator import Flux2AdapterModel, Flux2VAETokenizer
    from rdm.utils.io import save_uint8_png

    if args.curate == "none" and args.keep != args.seeds:
        raise SystemExit(f"--curate none keeps every candidate, so --keep ({args.keep}) must "
                         f"equal --seeds ({args.seeds})")

    ctx_arr = np.load(args.ctx, mmap_mode="r")
    captions = load_coco_pairs(args.captions)["captions"]
    n_prompts = ctx_arr.shape[0] if not args.limit else min(args.limit, ctx_arr.shape[0])
    if len(captions) < n_prompts:
        raise SystemExit(f"{len(captions)} captions for {n_prompts} context rows")
    os.makedirs(args.out, exist_ok=True)

    model = Flux2AdapterModel(image_resolution=args.img_size, param_dtype=torch.bfloat16,
                              gradient_checkpointing=False).to("cuda").eval()
    generator = build_generator("flux", model, {"num_steps": args.steps}, args=None,
                                tokenizer=Flux2VAETokenizer(device="cuda"))
    scorer = None
    if args.curate == "pickscore":
        from rdm.eval.pickscore_eval import PickScorer
        scorer = PickScorer(device="cuda")

    starts = list(range(0, n_prompts, args.chunk_prompts))[args.rank::args.world]
    manifest_path = os.path.join(args.out, f"manifest_rank{args.rank}.jsonl")
    t0, n_kept, n_rendered = time.time(), 0, 0

    for start in starts:
        prompts = list(range(start, min(start + args.chunk_prompts, n_prompts)))
        if _chunk_done(args.out, prompts, args.keep):
            continue

        # ---- render every candidate of this chunk into RAM (uint8, never to disk) ----
        pairs = [(p, s) for p in prompts for s in range(args.seeds)]
        imgs = torch.empty(len(pairs), 3, args.img_size, args.img_size, dtype=torch.uint8)
        with torch.no_grad():
            for lo in range(0, len(pairs), args.batch):
                sub = pairs[lo:lo + args.batch]
                ctx = torch.stack([torch.from_numpy(np.ascontiguousarray(ctx_arr[p]))
                                   for p, _ in sub]).to("cuda", torch.bfloat16)
                noise = torch.stack([
                    torch.randn(model.in_channels, model.input_size, model.input_size,
                                generator=torch.Generator(device="cuda").manual_seed(
                                    int(s) * 1_000_003 + int(p)), device="cuda")
                    for p, s in sub]).to(torch.bfloat16)
                out = generator.sample(noise, ctx)                     # (b,3,H,W) in [0,1]
                imgs[lo:lo + len(sub)] = (out.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu()
                n_rendered += len(sub)

        # ---- score, then keep the best `keep` per prompt ----
        scores = torch.zeros(len(pairs))
        if scorer is not None:
            with torch.no_grad():
                for lo in range(0, len(pairs), 32):
                    sub = pairs[lo:lo + 32]
                    batch = imgs[lo:lo + len(sub)].float() / 255.0
                    scores[lo:lo + len(sub)] = scorer.score(
                        batch.to("cuda"), [str(captions[p]) for p, _ in sub]).float().cpu()

        rows = []
        for pi, p in enumerate(prompts):
            lo = pi * args.seeds
            sc = scores[lo:lo + args.seeds]
            order = torch.argsort(sc, descending=True)[:args.keep].tolist()
            for rank_k, j in enumerate(order):
                save_uint8_png(imgs[lo + j].float() / 255.0,
                               os.path.join(args.out, f"{p:08d}_k{rank_k}.png"))
                rows.append({"prompt": int(p), "rank": rank_k, "seed": int(pairs[lo + j][1]),
                             "pickscore": float(sc[j]) if scorer is not None else None})
                n_kept += 1
        with open(manifest_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        del imgs

    dt = max(time.time() - t0, 1e-9)
    print(f"[render rank {args.rank}/{args.world}] rendered {n_rendered}, kept {n_kept} in "
          f"{dt/60:.1f} min ({n_rendered/dt:.2f} img/s) | peak "
          f"{torch.cuda.max_memory_reserved()/1e9:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
