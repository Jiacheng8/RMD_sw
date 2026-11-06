#!/usr/bin/env python
"""A reference store whose text block tau(c) is Qwen3-4B instead of SigLIP2, for the rows a run reads.

    python -m zimage_distill.qwen_text_store --source <root>/reference_store --out <root>/reference_store_qwen

tau(c) is the prompt's stored Qwen3 context, ``(L_p, 2560)``, mean-pooled over its tokens and L2-normalized,
as SigLIP2's is. Nothing is encoded, so no GPU is needed. Used by the joint MMD and the critics; the student's
own condition stays the token-level context.

Every row is filled. With ``--steps N`` only the rows a run of N steps reads are: the first ``N * B + K - B``
positions of ``row_order`` and the critics' standardization sample (``reference_stats``, seeds ``--critic-seeds``).
All others are then NaN, so a run that goes past them fails its finite check instead of training on made-up text.

The text kernel keeps SigLIP2's scale: ``s_txt`` is the ratio of median pair distances on the same rows, and
``beta = sigma / s_txt``. Image features, row order and contexts are the source store's (symlinks).

``--from-hub REPO`` instead assembles the store from the tables a build uploaded, next to the source:

    python -m zimage_distill.qwen_text_store --source <pack>/reference_store \
        --out <pack>/reference_store_qwen --from-hub jiachengcui888/zimage-ref-qwen-text
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .store import ZImageReferenceStore

LINKED = ("encoder_features", "group_ids.npy", "prompt_ids.npy", "row_order.npy", "rows.jsonl")
BUILT = ("bandwidths.json", "metadata.json", "text_features.done.json", "text_features.npy")


def link_source(source: Path, tmp: Path) -> None:
    """What the new store shares with its source, relative so the pack can move as a whole."""
    (tmp / "z_context.json").write_text((source / "z_context.json").read_text())
    for name in LINKED:
        os.symlink(os.path.join("..", source.name, name), tmp / name)


def from_hub(repo: str, source: Path, out: Path) -> None:
    """Assemble ``out`` from the tables uploaded to ``repo``, for the prompts of ``source``."""
    from huggingface_hub import hf_hub_download

    tmp = out.with_name(out.name + ".partial")
    tmp.mkdir(exist_ok=True)
    for name in BUILT:
        hf_hub_download(repo, name, repo_type="dataset", local_dir=tmp)
    shutil.rmtree(tmp / ".cache", ignore_errors=True)
    built, base = (json.loads((d / "metadata.json").read_text()) for d in (tmp, source))
    if built["prompt_hash"] != base["prompt_hash"] or built["num_rows"] != base["num_rows"]:
        raise SystemExit(f"{repo} was built for other prompts than {source}")
    shape = np.load(tmp / "text_features.npy", mmap_mode="r").shape
    if shape != (base["num_prompts"], built["text_dim"]):
        raise SystemExit(f"{repo}: text_features.npy has shape {shape}")
    for name in LINKED:
        if (tmp / name).is_symlink():
            (tmp / name).unlink()
    link_source(source, tmp)
    tmp.rename(out)
    print(f"{out}: tau {shape} from {repo}")


def pool_context(ctx: torch.Tensor) -> np.ndarray:
    """``(L, d)`` token context -> ``(d,)`` unit vector."""
    v = ctx.float().mean(0)
    return (v / v.norm()).numpy()


def rows_read(order: np.ndarray, steps: int, window: int, stride: int) -> np.ndarray:
    """The row ids a run of ``steps`` sliding-window updates reads, starting at position 0."""
    return np.asarray(order)[:steps * stride + window - stride]


def stats_rows(num_rows: int, n_rows: int, seeds) -> np.ndarray:
    """The rows :func:`rdm.sw_lmmd.adversarial.reference_stats` samples for the critics of ``seeds``."""
    n = min(n_rows, num_rows)
    return np.unique(np.concatenate([np.random.default_rng(s).choice(num_rows, size=n, replace=False)
                                     for s in seeds]))


def median_pair_distance(x: np.ndarray) -> float:
    half = len(x) // 2
    return float(np.median(np.linalg.norm(x[:half] - x[half:2 * half], axis=1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", required=True, help="the SigLIP2 reference store")
    ap.add_argument("--out", required=True, help="new store, next to the source (z_context.json's root is '..')")
    ap.add_argument("--steps", type=int, default=0, help="fill only the rows this many steps read (default: all)")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--stats-rows", type=int, default=8192, help="gan.stats_rows")
    ap.add_argument("--critic-seeds", type=int, nargs="+", default=[0, 1], help="gan.seed + k - 1 per step")
    ap.add_argument("--from-hub", metavar="REPO", help="take the tables from this HF dataset instead of building them")
    ap.add_argument("--workers", type=int, default=16, help="threads reading the contexts")
    a = ap.parse_args()

    source, out = Path(a.source).absolute(), Path(a.out).absolute()
    if out.exists():
        raise SystemExit(f"{out} exists; remove it or choose another --out")
    if out.parent != source.parent:
        raise SystemExit(f"--out must be next to {source}: the contexts are found through '..'")
    if a.from_hub:
        from_hub(a.from_hub, source, out)
        return 0
    names = list(json.loads((source / "bandwidths.json").read_text()))
    store = ZImageReferenceStore(str(source), names)
    order = store.row_order()
    used = rows_read(order, a.steps or store.num_rows, a.window, a.stride)
    rows = np.unique(np.concatenate([used, stats_rows(store.num_rows, a.stats_rows, a.critic_seeds)]))
    prompts = np.unique(store.prompt_rows(rows))
    print(f"filling {len(rows)} of {store.num_rows} rows", flush=True)

    tmp = out.with_name(out.name + ".partial")
    tmp.mkdir()
    dim = int(store.generator_context(rows[:1])[0].shape[1])
    tau = np.lib.format.open_memmap(tmp / "text_features.npy", mode="w+", dtype=np.float32,
                                    shape=(len(store.text), dim))
    tau[:] = np.nan
    row_of_prompt = dict(zip(store.prompt_rows(rows).tolist(), rows.tolist()))

    def pooled(p: int) -> tuple[int, np.ndarray]:
        return p, pool_context(store.generator_context(np.array([row_of_prompt[p]]))[0])

    with ThreadPoolExecutor(a.workers) as pool:       # thousands of small files: the wait is the disk
        for i, (p, v) in enumerate(pool.map(pooled, prompts.tolist())):
            tau[p] = v
            if i % 2000 == 0:
                print(f"{i}/{len(prompts)}", flush=True)
    tau.flush()

    sample = store.prompt_rows(order)[:2000]
    siglip, qwen = np.asarray(store.text[sample], dtype=np.float32), np.asarray(tau[sample])
    if not np.isfinite(qwen).all() or not np.allclose(np.linalg.norm(qwen, axis=1), 1, atol=1e-3):
        raise SystemExit("invalid Qwen3 text features")
    d_siglip, d_qwen = median_pair_distance(siglip), median_pair_distance(qwen)
    s_txt = d_qwen / d_siglip
    bands = json.loads((source / "bandwidths.json").read_text())
    for b in bands.values():
        b.update(s_txt=s_txt, beta=b["sigma"] / s_txt)
    (tmp / "bandwidths.json").write_text(json.dumps(bands, indent=2) + "\n")
    meta = json.loads((source / "metadata.json").read_text())
    meta.update(text_dim=dim, text_encoder={"prompt_hash": meta["prompt_hash"], "model": "Qwen3-4B",
                                            "pooling": "token mean, L2 normalized"})
    (tmp / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (tmp / "text_features.done.json").write_text(json.dumps({
        "prompt_hash": meta["prompt_hash"], "model": "Qwen3-4B", "pooling": "token mean, L2 normalized",
        "steps": a.steps or None, "window": a.window, "stride": a.stride, "rows_filled": len(rows),
        "median_pair_distance": {"qwen3": d_qwen, "siglip2": d_siglip}}, indent=2) + "\n")
    link_source(source, tmp)
    tmp.rename(out)
    print(f"{out}: tau {tau.shape}, median pair distance Qwen3 {d_qwen:.4f}, SigLIP2 {d_siglip:.4f}, s_txt {s_txt:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
