#!/usr/bin/env python
"""Assemble the SW-LMMD reference store from the three steps' outputs (step 04).

Gathers what 01-03 produced into the directory layout
:class:`rdm.sw_lmmd.reference_store.ReferenceFeatureStore` reads, and writes the pieces that
are pure bookkeeping:

* ``text_features.npy`` -- rows 0..num_prompts of the RELEASED SigLIP2 tau(c) table, which is
  row-aligned to the COCO pairing (verified by re-encoding: matched-row cosine 0.99998). No
  text tower is run here.
* ``qwen_context.npy`` -- symlinked to step 01's output rather than copied (61 GB).
* ``prompt_ids.npy`` -- ``r // keep``, mapping each (prompt, seed) reference row back to the
  prompt-indexed text/context tables, so the 61 GB context is not replicated per seed.
* ``bandwidths.json`` -- merged from the per-encoder parts step 03 wrote.
* ``row_order.npy`` -- the seeded permutation the window schedule traverses.

Then it opens the store through the real reader and checks a window's worth of rows, so a
broken layout fails here rather than on the first training step.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True)
    ap.add_argument("--tau-released", required=True, help="siglip2_text_geall_g32.npy")
    ap.add_argument("--ctx", required=True, help="step 01 output (prompt-indexed)")
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--keep", type=int, default=3, help="reference rows per prompt")
    ap.add_argument("--encoders", required=True)
    ap.add_argument("--row-order-seed", type=int, default=3407)
    args = ap.parse_args()

    from rdm.sw_lmmd import ReferenceFeatureStore, build_row_order, order_hash

    names = [e.strip() for e in args.encoders.split(",") if e.strip()]
    store = args.store
    os.makedirs(store, exist_ok=True)
    n_rows = args.num_prompts * args.keep

    # ---- tau(c): slice the released table (its first num_prompts rows are the COCO captions)
    tau = np.load(args.tau_released, mmap_mode="r")
    if tau.shape[0] < args.num_prompts:
        raise SystemExit(f"released tau table has {tau.shape[0]} rows < {args.num_prompts}")
    np.save(os.path.join(store, "text_features.npy"),
            np.ascontiguousarray(tau[:args.num_prompts]).astype(np.float32))

    # ---- context: symlink (61 GB), never copy
    ctx_link = os.path.join(store, "qwen_context.npy")
    ctx_src = os.path.abspath(args.ctx)
    ctx_rows = np.load(ctx_src, mmap_mode="r").shape[0]
    if ctx_rows != args.num_prompts:
        raise SystemExit(f"context has {ctx_rows} rows, expected {args.num_prompts}")
    if os.path.lexists(ctx_link):
        os.remove(ctx_link)
    os.symlink(ctx_src, ctx_link)

    # ---- row -> prompt indirection, and the traversal order
    np.save(os.path.join(store, "prompt_ids.npy"),
            np.repeat(np.arange(args.num_prompts, dtype=np.int64), args.keep))
    order = build_row_order(n_rows, seed=args.row_order_seed)
    np.save(os.path.join(store, "row_order.npy"), order)

    # ---- bandwidths merged from the per-encoder parts
    bands = {}
    for part in sorted(glob.glob(os.path.join(store, "bandwidths_parts", "*.json"))):
        bands[os.path.splitext(os.path.basename(part))[0]] = json.load(open(part))
    missing = [n for n in names if n not in bands]
    if missing:
        raise SystemExit(f"no bandwidth recorded for {missing} -- step 03 did not finish")
    json.dump(bands, open(os.path.join(store, "bandwidths.json"), "w"), indent=2)

    dims = {}
    for n in names:
        p = os.path.join(store, "encoder_features", f"{n}.npy")
        if not os.path.exists(p):
            raise SystemExit(f"missing features for {n}: {p}")
        arr = np.load(p, mmap_mode="r")
        if arr.shape[0] != n_rows:
            raise SystemExit(f"{n}: {arr.shape[0]} feature rows, expected {n_rows}")
        dims[n] = int(arr.shape[1])

    json.dump({"num_rows": n_rows, "num_prompts": args.num_prompts, "rows_per_prompt": args.keep,
               "text_dim": int(tau.shape[1]), "encoder_feature_dims": dims,
               "reference_type": "flux2_klein_4step_teacher", "feature_dtype": "float16",
               "row_order_seed": args.row_order_seed, "row_order_hash": order_hash(order),
               "tau_source": os.path.abspath(args.tau_released), "context_source": ctx_src},
              open(os.path.join(store, "metadata.json"), "w"), indent=2)

    # ---- read it back through the real reader, on a window's worth of rows
    s = ReferenceFeatureStore(store, names)
    rows = s.row_order()[:1024]
    for n in names:
        j = s.reference_joint_features(n, rows)
        assert j.shape == (len(rows), dims[n] + int(tau.shape[1])), f"{n}: {tuple(j.shape)}"
    ctx = s.generator_context(rows[:4])
    pid = s.prompt_rows(rows[:4])
    print(f"[store] OK  {store}")
    print(f"        {n_rows:,} rows = {args.num_prompts:,} prompts x {args.keep} kept renders")
    print(f"        {len(names)} encoders, joint dim {dims[names[0]]}+{tau.shape[1]}, "
          f"ctx {tuple(ctx.shape)}")
    print(f"        row->prompt spot check: rows {rows[:4].tolist()} -> prompts {pid.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
