#!/usr/bin/env python
"""The DMD2 baseline's real data: Z-Image-Turbo reference renders as VAE latents, + the CFG negative context.

    python -m zimage_distill.dmd2_latents encode   --out <dir> --shards 24 --rank R --world W   # one per GPU
    python -m zimage_distill.dmd2_latents finalize --out <dir> --world W

DMD2's GAN tells the student's samples from real ones on the fake score's features (Yin et al. 2024:
the SDXL recipe trains it on 500K LAION images as VAE latents). Here the real data is what ours trains
with -- the Z-Image-Turbo reference renders, one 1024 px image per prompt -- so the methods see the same
data. They live in the public WebDataset ``jiachengcui888/zimage-turbo-reference`` (87 tar shards of
~1,425 ``<image_id>.png`` + ``<image_id>.json``, ~2 GB each). ``--shards K`` takes K of them, evenly
spaced (24 -> ~34k images, ~48 GB downloaded, each tar deleted once encoded). Each image becomes the
latent the student outputs: ``(mode(E(x)) - shift) * scale``, (16, 128, 128) bf16, 0.5 MB.

Layout in ``<out>/latents/``:

    manifest.json      {"format": "zimage-latents-v1", "shape", "dtype", "num_prompts", "shards": [...]}
    shard_{r:02d}.bin  rank r's latents back to back, bf16 stored as int16
    index.npy          (P, 2) int64: shard, position per prompt id; -1 where not encoded
    available.npy      the encoded prompt ids

and ``<out>/negative_context.pt``: the Qwen3 context of the empty prompt, the unconditional branch of
the real score's classifier-free guidance (``ZImagePipeline``'s default negative prompt).
A rank writes ``done_{r}.json`` last; a relaunch skips finished ranks and redoes the others.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

FORMAT = "zimage-latents-v1"
SHAPE = (16, 128, 128)
HF_REPO = "jiachengcui888/zimage-turbo-reference"
NUM_TAR_SHARDS = 87


def chosen_shards(k: int, total: int = NUM_TAR_SHARDS) -> list[int]:
    """``k`` tar shards spread evenly over all of them (the prompts are ordered by the manifest)."""
    k = max(1, min(int(k), total))
    return sorted({round(i * (total - 1) / max(1, k - 1)) for i in range(k)}) if k > 1 else [0]


def iter_tar(path):
    """``(prompt_id, prompt, uint8 HWC image)`` per sample of one WebDataset tar."""
    from PIL import Image
    pending = {}
    with tarfile.open(path, "r|") as t:
        for m in t:
            if not m.isfile():
                continue
            key, ext = m.name.rsplit(".", 1)
            pending.setdefault(key, {})[ext] = t.extractfile(m).read()
            got = pending[key]
            if "png" in got and "json" in got:
                meta = json.loads(got["json"])
                img = np.asarray(Image.open(io.BytesIO(got["png"])).convert("RGB"))
                del pending[key]
                yield int(meta["prompt_id"]), meta["prompt"], img


def encode(out: Path, shards: int, rank: int, world: int, repo: str | None, batch: int,
           pack: Path | None, keep_tars: bool) -> None:
    from diffusers import AutoencoderKL
    from huggingface_hub import hf_hub_download

    from .model import ZIMAGE_REPO, resolve_snapshot
    from .store import read_prompts

    lat = out / "latents"
    lat.mkdir(parents=True, exist_ok=True)
    done = lat / f"done_{rank:02d}.json"
    if done.exists():
        print(f"[latents {rank}/{world}] already done", flush=True)
        return
    mine = chosen_shards(shards)[rank::world]
    prompts = read_prompts(pack / "prompts.jsonl") if pack else None
    vae = AutoencoderKL.from_pretrained(os.path.join(resolve_snapshot(repo or ZIMAGE_REPO), "vae"),
                                        torch_dtype=torch.bfloat16).to("cuda").eval()
    scale, shift = float(vae.config.scaling_factor), float(vae.config.shift_factor)
    tars = out / "tars"
    shard = lat / f"shard_{rank:02d}.bin"
    order, t0 = [], time.time()

    def flush(imgs, f):
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to("cuda", torch.bfloat16) / 127.5 - 1
        with torch.no_grad():
            z = (vae.encode(x).latent_dist.mode() - shift) * scale
        f.write(z.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().tobytes())

    with open(shard, "wb") as f:
        for s in mine:
            name = f"images/shard-{s:05d}.tar"
            path = hf_hub_download(HF_REPO, name, repo_type="dataset", local_dir=str(tars))
            imgs = []
            for pid, prompt, img in iter_tar(path):
                if img.shape[:2] != (SHAPE[1] * 8, SHAPE[2] * 8):
                    raise ValueError(f"prompt {pid}: render is {img.shape[:2]}, expected 1024 x 1024")
                if prompts is not None and prompts[pid] != prompt:
                    raise ValueError(f"prompt {pid}: the render's prompt is not the pack's -- "
                                     f"another prompt table?")
                imgs.append(img)
                order.append(pid)
                if len(imgs) == batch:
                    flush(imgs, f)
                    imgs = []
            if imgs:
                flush(imgs, f)
            if not keep_tars:
                os.remove(path)
            print(f"[latents {rank}/{world}] {name}: {len(order)} latents so far, "
                  f"{len(order) / max(time.time() - t0, 1e-9):.1f} img/s", flush=True)
    np.save(lat / f"order_{rank:02d}.npy", np.asarray(order, dtype=np.int64))
    done.write_text(json.dumps({"images": len(order), "tar_shards": mine,
                                "seconds": round(time.time() - t0, 1)}))
    print(f"[latents {rank}/{world}] done: {len(order)} latents in {time.time() - t0:.0f} s", flush=True)


def finalize(out: Path, world: int, num_prompts: int, repo: str | None) -> None:
    from .model import QwenPromptEncoder

    lat = out / "latents"
    missing = [r for r in range(world) if not (lat / f"done_{r:02d}.json").exists()]
    if missing:
        raise SystemExit(f"ranks {missing} have not finished; relaunch encode for them")
    index = np.full((num_prompts, 2), -1, dtype=np.int64)
    per = int(np.prod(SHAPE)) * 2
    for r in range(world):
        order = np.load(lat / f"order_{r:02d}.npy")
        if (lat / f"shard_{r:02d}.bin").stat().st_size != order.size * per:
            raise SystemExit(f"shard_{r:02d}.bin does not hold {order.size} latents")
        index[order] = np.stack([np.full(order.size, r), np.arange(order.size)], 1)
    available = np.flatnonzero(index[:, 0] >= 0)
    np.save(lat / "index.npy", index)
    np.save(lat / "available.npy", available)
    (lat / "manifest.json").write_text(json.dumps({
        "format": FORMAT, "shape": list(SHAPE), "dtype": "bfloat16", "num_prompts": num_prompts,
        "available": int(available.size), "source": HF_REPO,
        "shards": [f"shard_{r:02d}.bin" for r in range(world)]}, indent=2))
    neg = out / "negative_context.pt"
    if not neg.exists():
        ctx = QwenPromptEncoder(repo or "Tongyi-MAI/Z-Image-Turbo", device="cuda")([""])[0]
        torch.save(ctx.cpu(), neg)
    print(f"[latents] {available.size} of {num_prompts} prompts have a latent "
          f"({available.size * per / 1e9:.1f} GB); empty-prompt context -> {neg}", flush=True)


class Latents:
    """Read-only latents by prompt id (bf16 tensors of ``SHAPE``)."""

    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"{directory}: not a {FORMAT} directory")
        self.shape = tuple(self.manifest["shape"])
        self.index = np.load(directory / "index.npy")
        self.available = np.load(directory / "available.npy")
        # a rank that got no tar shard (fewer shards than GPUs) wrote an empty file: nothing to map
        self.shards = [np.memmap(directory / s, dtype=np.int16, mode="r").reshape(-1, *self.shape)
                       if (directory / s).stat().st_size else None
                       for s in self.manifest["shards"]]

    def get(self, prompt_ids) -> torch.Tensor:
        out = []
        for p in np.asarray(prompt_ids, dtype=np.int64):
            s, i = self.index[p]
            if s < 0:
                raise KeyError(f"prompt {p} has no latent")
            out.append(torch.from_numpy(np.array(self.shards[s][i])).view(torch.bfloat16))
        return torch.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["encode", "finalize"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=24, help="tar shards of the 87 (24 -> ~34k images)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--pack", type=Path, default=None,
                    help="the training pack (prompts.jsonl): every render's prompt is checked against it")
    ap.add_argument("--num-prompts", type=int, default=125808)
    ap.add_argument("--repo", default=None, help="Z-Image-Turbo snapshot dir or repo id (its VAE)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--keep-tars", action="store_true")
    a = ap.parse_args()
    if a.stage == "encode":
        encode(a.out, a.shards, a.rank, a.world, a.repo, a.batch, a.pack, a.keep_tars)
    else:
        finalize(a.out, a.world, a.num_prompts, a.repo)


if __name__ == "__main__":
    main()
