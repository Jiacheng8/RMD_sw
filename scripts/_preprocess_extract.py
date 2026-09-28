#!/usr/bin/env python
"""One rank of the reference feature extraction (step 03): encoders -> per-row features.

Runs an assigned subset of the frozen battery over the teacher renders and writes, per
encoder, a ``(num_rows, d_img)`` fp16 array plus its fixed RBF bandwidth. Encoders are
independent, so the shell step shards them across GPUs and this worker handles its share.

Row order is the pipeline's contract and is fixed here:

    row r  <->  prompt r // keep , kept-rank r % keep   (file ``{prompt:08d}_k{rank}.png``)

so ``prompt_ids[r] = r // keep`` maps a reference row back to the prompt-indexed text table
and context pool. The same ordering is rebuilt identically in step 04; a mismatch would pair
generated samples with another prompt's reference and never surface as an error.

The bandwidth follows :func:`rdm.refprep.build_joint_reference.build_joint_one` exactly --
``sigma = sigma_scale * median(phi)`` on the image features, and ``beta = sigma / s_txt``.
With ``s_txt = 1.0`` this reproduces the released bundles' ``beta == sigma``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch


def row_paths(teacher_dir: str, num_prompts: int, keep: int) -> list[str]:
    """Canonical row order: prompt-major, kept-rank-minor (rank 0 = best PickScore)."""
    return [os.path.join(teacher_dir, f"{p:08d}_k{k}.png")
            for p in range(num_prompts) for k in range(keep)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--out", required=True, help="store directory (encoder_features/ written here)")
    ap.add_argument("--encoders", required=True, help="comma list for THIS rank")
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--keep", type=int, default=3, help="reference rows per prompt")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sigma-scale", type=float, default=0.25)
    ap.add_argument("--s-txt", type=float, default=1.0)
    args = ap.parse_args()

    from rdm.compare.median_heuristic import median_bandwidth
    from rdm.data.imagenet import ImageListDataset
    from rdm.refprep.extract_features import extract_features
    from rdm.representation.registry import by_name

    paths = row_paths(args.teacher_dir, args.num_prompts, args.keep)
    missing = [p for p in paths[:2000] if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} of the first 2000 renders are missing "
                         f"(e.g. {missing[0]}) -- step 02 did not finish")
    n_rows = len(paths)
    feat_dir = os.path.join(args.out, "encoder_features")
    bw_dir = os.path.join(args.out, "bandwidths_parts")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(bw_dir, exist_ok=True)

    for name in [e.strip() for e in args.encoders.split(",") if e.strip()]:
        out_path = os.path.join(feat_dir, f"{name}.npy")
        bw_path = os.path.join(bw_dir, f"{name}.json")
        if os.path.exists(out_path) and os.path.exists(bw_path):
            print(f"[extract] {name}: present, skipping", flush=True)
            continue
        t0 = time.time()
        ds = ImageListDataset(paths, img_size=args.img_size)
        feats = extract_features(by_name(name), ds, batch_size=args.batch,
                                 num_workers=args.workers, device="cuda")
        if feats.shape[0] != n_rows:
            raise RuntimeError(f"{name}: got {feats.shape[0]} features for {n_rows} rows")
        # sigma on the raw fp32 features, before the fp16 store cast
        sigma = float(median_bandwidth(feats)) * args.sigma_scale
        np.save(out_path, feats.numpy().astype(np.float16))
        json.dump({"sigma": sigma, "beta": sigma / args.s_txt, "d_img": int(feats.shape[1]),
                   "sigma_scale": args.sigma_scale, "s_txt": args.s_txt}, open(bw_path, "w"))
        print(f"[extract] {name}: {n_rows} rows x {feats.shape[1]}d in "
              f"{(time.time()-t0)/60:.1f} min | sigma={sigma:.4f} -> {out_path}", flush=True)
        del feats
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
