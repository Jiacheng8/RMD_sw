#!/usr/bin/env python
"""Assemble the GenEval reference block (step G4) from the scored renders of steps G2-G3.

Keeps, for every GenEval prompt, the teacher renders the official scorer judged CORRECT --
the recipe behind the iRDM headline reference ("geALLcoco": every GenEval-correct teacher image
next to the COCO PickScore-top block). Per prompt the count is rounded down to a multiple of
``--group`` and capped at ``--max-per-prompt``, so the block tiles into prompt-grouped windows.

The block is a small, self-contained directory that a COCO store reads as an extension
(``ReferenceFeatureStore(..., extension=<block>)``); the COCO store itself is never modified:

    block/
      metadata.json              num_rows, num_prompts, group_size, per-task row counts
      encoder_features/<e>.npy   (R, d_img) fp16, rows prompt-major then seed
      text_features.npy          (553, d_txt)  tau(c) of the GenEval prompts (released table)
      qwen_context.npy           (553, L, d)   generator context of the GenEval prompts
      prompt_ids.npy             (R,)  row -> GenEval prompt index (0..552)
      group_ids.npy              (R,)  row -> group index; G consecutive rows of one prompt
      rows.jsonl                 provenance: prompt, seed, tag, source image

No bandwidth is computed: the block is compared under the COCO store's fixed per-encoder
sigma, exactly like every other reference row.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import shutil

import numpy as np
import torch


def correct_renders(results_files: list[str]) -> dict[int, list[tuple[int, str, str]]]:
    """``{prompt: [(seed, path, tag), ...]}`` of the renders the scorer judged correct."""
    keep: dict[int, list] = collections.defaultdict(list)
    pat = re.compile(r"/(\d+)/samples/(\d+)\.png$")
    for path in results_files:
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            m = pat.search(r["filename"])
            if m is None:
                raise SystemExit(f"unexpected filename in {path}: {r['filename']}")
            if r["correct"]:
                keep[int(m.group(1))].append((int(m.group(2)), r["filename"], r["tag"]))
    return {p: sorted(v) for p, v in keep.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render-dir", required=True, help="step G2 output (shard_*/ inside)")
    ap.add_argument("--prompts", required=True, help="assets/geneval_prompts.jsonl")
    ap.add_argument("--ctx", required=True, help="qwen3 context of the GenEval prompts")
    ap.add_argument("--tau-released", required=True, help="siglip2_text_geall_g32.npy")
    ap.add_argument("--tau-offset", type=int, default=82783,
                    help="first GenEval row of the released tau table (after the COCO captions)")
    ap.add_argument("--base-store", default=None,
                    help="the COCO store this block extends (optional: only for the read-back "
                         "check and the mass fraction; a fresh render machine has none)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoders", required=True)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--max-per-prompt", type=int, default=96)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from rdm.data.imagenet import ImageListDataset
    from rdm.refprep.extract_features import extract_features
    from rdm.representation.registry import by_name

    names = [e.strip() for e in args.encoders.split(",") if e.strip()]
    meta = [json.loads(line) for line in open(args.prompts) if line.strip()]
    base_meta = (json.load(open(os.path.join(args.base_store, "metadata.json")))
                 if args.base_store else None)
    if args.max_per_prompt % args.group:
        raise SystemExit(f"--max-per-prompt {args.max_per_prompt} must be a multiple of "
                         f"--group {args.group}")

    # ---- select: correct renders, rounded down to whole groups, capped ----
    results = sorted(glob.glob(os.path.join(args.render_dir, "shard_*_results.jsonl")))
    if not results:
        raise SystemExit(f"no shard_*_results.jsonl in {args.render_dir} -- step G3 has not run")
    correct = correct_renders(results)
    rows = []                                  # (prompt, seed, path, tag)
    for p in range(len(meta)):
        cand = correct.get(p, [])
        n = min(len(cand) // args.group * args.group, args.max_per_prompt)
        rows += [(p, s, path, tag) for s, path, tag in cand[:n]]
    if not rows:
        raise SystemExit("no prompt has a full group of correct renders")
    n_rows = len(rows)
    prompt_ids = np.array([r[0] for r in rows], dtype=np.int64)
    group_ids = np.arange(n_rows, dtype=np.int64) // args.group
    per_tag = collections.Counter(r[3] for r in rows)
    covered = len(set(prompt_ids.tolist()))

    os.makedirs(os.path.join(args.out, "encoder_features"), exist_ok=True)
    with open(os.path.join(args.out, "rows.jsonl"), "w") as f:
        for p, s, path, tag in rows:
            f.write(json.dumps({"prompt": p, "seed": s, "tag": tag, "image": path}) + "\n")
    np.save(os.path.join(args.out, "prompt_ids.npy"), prompt_ids)
    np.save(os.path.join(args.out, "group_ids.npy"), group_ids)

    # ---- prompt-indexed tables: tau(c) slice of the released table, generator context ----
    tau = np.load(args.tau_released, mmap_mode="r")
    block_tau = np.ascontiguousarray(tau[args.tau_offset:args.tau_offset + len(meta)])
    replica = tau[args.tau_offset + len(meta):args.tau_offset + 2 * len(meta)]
    if replica.shape[0] == len(meta) and not np.allclose(block_tau, replica):
        raise SystemExit(f"released tau rows {args.tau_offset}.. are not the GenEval prompts "
                         f"(the x32 replication does not repeat) -- check --tau-offset")
    if args.base_store:
        base_tau = np.load(os.path.join(args.base_store, "text_features.npy"), mmap_mode="r")
        if base_tau.shape[1] != block_tau.shape[1]:
            raise SystemExit(f"tau width {block_tau.shape[1]} != the COCO store's "
                             f"{base_tau.shape[1]}")
    np.save(os.path.join(args.out, "text_features.npy"), block_tau.astype(np.float32))
    ctx = np.load(args.ctx, mmap_mode="r")
    if ctx.shape[0] != len(meta):
        raise SystemExit(f"context has {ctx.shape[0]} rows for {len(meta)} prompts")
    shutil.copyfile(args.ctx, os.path.join(args.out, "qwen_context.npy"))

    # ---- image features, the same extractor and precision as the COCO store's step 03 ----
    dims = {}
    paths = [r[2] for r in rows]
    for name in names:
        out_path = os.path.join(args.out, "encoder_features", f"{name}.npy")
        if os.path.exists(out_path) and np.load(out_path, mmap_mode="r").shape[0] == n_rows:
            dims[name] = int(np.load(out_path, mmap_mode="r").shape[1])
            print(f"[geneval block] {name}: present, skipping", flush=True)
            continue
        feats = extract_features(by_name(name), ImageListDataset(paths, img_size=args.img_size),
                                 batch_size=args.batch, num_workers=args.workers, device="cuda")
        np.save(out_path, feats.numpy().astype(np.float16))
        dims[name] = int(feats.shape[1])
        print(f"[geneval block] {name}: {n_rows} rows x {dims[name]}d", flush=True)
        del feats
        torch.cuda.empty_cache()

    base_rows = int(base_meta["num_rows"]) if base_meta else None
    json.dump({"kind": "geneval_block", "num_rows": n_rows, "num_prompts": len(meta),
               "group_size": args.group, "max_per_prompt": args.max_per_prompt,
               "prompts_covered": covered, "rows_per_task": dict(per_tag),
               "encoder_feature_dims": dims, "feature_dtype": "float16",
               "text_dim": int(block_tau.shape[1]), "base_store_rows": base_rows,
               "mass_fraction": n_rows / (n_rows + base_rows) if base_rows else None,
               "tau_source": os.path.abspath(args.tau_released), "tau_offset": args.tau_offset,
               "render_dir": os.path.abspath(args.render_dir)},
              open(os.path.join(args.out, "metadata.json"), "w"), indent=2)

    # ---- read it back: through the real reader as the store's extension, or on its own ----
    if args.base_store:
        from rdm.sw_lmmd import ReferenceFeatureStore
        store = ReferenceFeatureStore(args.base_store, names, extension=args.out,
                                      require_context=False)
        tail = np.arange(store.num_rows - 8, store.num_rows)
        for name in names:
            store.reference_joint_features(name, tail)
        share = f", {n_rows / (n_rows + base_rows):.1%} of the combined reference"
    else:
        for name in names:
            f = np.load(os.path.join(args.out, "encoder_features", f"{name}.npy"), mmap_mode="r")
            assert f.shape == (n_rows, dims[name]) and np.isfinite(f[:64]).all(), name
        assert np.load(os.path.join(args.out, "qwen_context.npy"), mmap_mode="r").shape[0] == len(meta)
        share = ""
    print(f"[geneval block] OK  {args.out}")
    print(f"        {n_rows:,} rows over {covered}/{len(meta)} prompts{share}")
    print(f"        per task: {dict(per_tag)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
