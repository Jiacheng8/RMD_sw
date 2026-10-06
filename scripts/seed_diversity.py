#!/usr/bin/env python
"""Seed diversity of a GenEval render dir: how different the images of ONE prompt are.

GenEval renders 4 seeds per prompt (<dir>/<idx:05d>/samples/NNNN.png). For every prompt this
averages a distance over all pairs of its images, then averages over prompts -- higher means the
seeds of a prompt look less alike (mode collapse drives it down). Measures:

  pixel     mean |a - b| of the images as 64x64 grayscale (bilinear) in [0, 1]: layout/composition.
            The metric of the earlier reports (klein-4B 4-step teacher: 0.1905).
  <encoder> 1 - cosine similarity of the frozen encoder's embedding (rdm.representation.registry),
            e.g. dreamsim (perceptual; teacher 0.231) or dinov3_l (semantic). An encoder whose
            weights are unavailable is skipped with a warning, never fails the evaluation.

Writes <out> (default <dir>/../seed_diversity.json): {"pixel": .., "<encoder>": .., "prompts": N}.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np


def prompt_groups(render_dir: str) -> list[list[str]]:
    groups = []
    for d in sorted(glob.glob(os.path.join(render_dir, "[0-9]*"))):
        imgs = sorted(glob.glob(os.path.join(d, "samples", "[0-9]*.png")))
        if len(imgs) >= 2:
            groups.append(imgs)
    return groups


def pixel_diversity(groups: list[list[str]]) -> float:
    from PIL import Image

    def load(p):
        return np.asarray(Image.open(p).convert("L").resize((64, 64), Image.BILINEAR),
                          dtype=np.float32) / 255.0

    per_prompt = []
    for g in groups:
        ims = [load(p) for p in g]
        per_prompt.append(np.mean([np.abs(a - b).mean() for a, b in itertools.combinations(ims, 2)]))
    return float(np.mean(per_prompt))


def embedding_diversity(groups: list[list[str]], encoder: str, batch: int, workers: int) -> float:
    import torch

    from rdm.data.imagenet import ImageListDataset
    from rdm.refprep.extract_features import extract_features
    from rdm.representation.registry import by_name

    paths = [p for g in groups for p in g]
    feats = extract_features(by_name(encoder), ImageListDataset(paths, img_size=512),
                             batch_size=batch, num_workers=workers, device="cuda")
    feats = torch.nn.functional.normalize(feats.float(), dim=1).numpy()
    per_prompt, i = [], 0
    for g in groups:
        f = feats[i:i + len(g)]
        i += len(g)
        per_prompt.append(np.mean([1.0 - float(f[a] @ f[b])
                                   for a, b in itertools.combinations(range(len(g)), 2)]))
    return float(np.mean(per_prompt))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("render_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--encoders", default="dreamsim,dinov3_l", help="comma list ('' = pixel only)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    groups = prompt_groups(args.render_dir)
    if not groups:
        raise SystemExit(f"no <idx>/samples/*.png with >= 2 images per prompt under {args.render_dir}")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.render_dir)), "seed_diversity.json")
    res = {"prompts": len(groups), "images_per_prompt": len(groups[0]), "pixel": pixel_diversity(groups)}
    print(f"[diversity] pixel     {res['pixel']:.4f}  ({len(groups)} prompts)", flush=True)
    for enc in [e.strip() for e in args.encoders.split(",") if e.strip()]:
        try:
            res[enc] = embedding_diversity(groups, enc, args.batch, args.workers)
            print(f"[diversity] {enc:<9} {res[enc]:.4f}", flush=True)
        except Exception as e:                                   # missing weights, no network, ...
            print(f"[diversity] WARNING: {enc} skipped ({type(e).__name__}: {str(e)[:200]})",
                  file=sys.stderr, flush=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"[diversity] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
