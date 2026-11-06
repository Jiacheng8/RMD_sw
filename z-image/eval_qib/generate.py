#!/usr/bin/env python
"""Render the 1000 Qwen-Image-Bench prompts with a Z-Image model: one 1024 px image per prompt.

    PYTHONPATH=z-image python -m eval_qib.generate --bench <bench dir> --out <run dir> \
        --ckpt <run>/step_0002000.pth [--rank R --world W]       # the 2-step student
    ... --teacher-steps 9                                         # Z-Image-Turbo itself (9 steps)
    ... --teacher-steps 2                                         # Turbo run with 2 steps
    PYTHONPATH=z-image python -m eval_qib.generate --bench <bench dir> --out <run dir> --finalize

A rank renders every ``world``-th prompt to ``<out>/images/{ID:04d}.png`` (an existing image is
kept, so a relaunch resumes). ``--finalize`` checks all 1000 are there and writes the judge's input,
``<out>/input.jsonl`` (``ID``, ``prompt``, ``image_path``), and ``<out>/generation.json``.

The sampler is :class:`zimage_distill.model.ZImageStudent` in bf16 without autocast, which is
``ZImagePipeline`` bit for bit on the teacher schedule (check_teacher.py). A checkpoint brings its
own ``sampling_args`` (sigmas, re-noise); the latent of prompt ``ID`` is ``noise_for(ID, seed)``
and step ``k``'s re-noise ``noise_for(ID, seed + 1000 k)``, so every model sees the same starts.
The prompts are encoded first and the Qwen3 encoder leaves the GPU before the DiT loads, which
keeps a rank inside 24 GB.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bench", required=True, help="the Qwen-Image-Bench release (eval_qib.bench)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="a step_*.pth / resume.pth of the student")
    ap.add_argument("--teacher-steps", type=int, default=0,
                    help="no --ckpt: Z-Image-Turbo on its own N-step schedule (Euler, no re-noise)")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"],
                    help="zh = the leaderboard's setting (see eval_qib/bench.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--repo", default=None, help="Z-Image-Turbo snapshot dir or repo id")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke tests)")
    a = ap.parse_args()

    from eval_qib.bench import bench_prompts
    out = Path(a.out)
    images = out / "images"
    prompts = bench_prompts(a.bench, a.lang)
    if a.limit:
        prompts = prompts[:a.limit]

    if a.finalize:
        missing = [i for i, _ in prompts if not (images / f"{i:04d}.png").exists()]
        if missing:
            raise SystemExit(f"{len(missing)} of {len(prompts)} images missing (e.g. ID {missing[0]}); "
                             f"relaunch the ranks")
        with open(out / "input.jsonl", "w", encoding="utf-8") as f:
            for i, p in prompts:
                f.write(json.dumps({"ID": i, "prompt": p,
                                    "image_path": str((images / f"{i:04d}.png").resolve())},
                                   ensure_ascii=False) + "\n")
        print(f"[generate] {len(prompts)} images -> {out / 'input.jsonl'}", flush=True)
        return 0

    if bool(a.ckpt) == bool(a.teacher_steps):
        raise SystemExit("pass exactly one of --ckpt and --teacher-steps")
    from zimage_distill.model import (ZIMAGE_REPO, QwenPromptEncoder, ZImageStudent, ZImageVAE,
                                      noise_for, turbo_sigmas)
    repo = a.repo or ZIMAGE_REPO
    mine = [(i, p) for i, p in prompts[a.rank::a.world] if not (images / f"{i:04d}.png").exists()]
    images.mkdir(parents=True, exist_ok=True)
    print(f"[generate {a.rank}/{a.world}] {len(mine)} prompts to render", flush=True)
    if not mine:
        return 0

    # 1. contexts, then the prompt encoder leaves the GPU
    text = QwenPromptEncoder(repo, device="cuda", cache_size=len(mine) + 1)
    caps = {}
    for lo in range(0, len(mine), 16):
        chunk = mine[lo:lo + 16]
        for (i, _), c in zip(chunk, text([p for _, p in chunk])):
            caps[i] = c.cpu()
    del text
    torch.cuda.empty_cache()

    # 2. the model and its sampler
    sampling = {"sigmas": turbo_sigmas(a.teacher_steps or 2), "renoise": not a.teacher_steps,
                "detach_between_steps": True}
    model = ZImageStudent(a.img_size, repo=repo, param_dtype=torch.bfloat16,
                          gradient_checkpointing=False)
    if a.ckpt:
        state = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(state.get("model", state), strict=True)
        sampling.update(state.get("sampling_args") or {})
        del state
    model = model.to("cuda").eval()
    vae = ZImageVAE(repo, device="cuda", gradient_checkpointing=False)
    meta = {"ckpt": str(Path(a.ckpt).resolve()) if a.ckpt else None,
            "teacher_steps": a.teacher_steps or None, "sampling_args": sampling, "lang": a.lang,
            "seed": a.seed, "img_size": a.img_size, "images_per_prompt": 1}
    if a.rank == 0:
        (out / "generation.json").write_text(json.dumps(meta, indent=2))
    print(f"[generate {a.rank}/{a.world}] {meta['ckpt'] or f'Turbo, {a.teacher_steps} steps'} | "
          f"sigmas {[round(s, 4) for s in sampling['sigmas']]} | renoise {sampling['renoise']}",
          flush=True)

    from rdm.utils.io import save_uint8_png
    side, S, t0 = a.img_size // 8, len(sampling["sigmas"]), time.time()
    with torch.no_grad():
        for lo in range(0, len(mine), a.batch):
            sub = mine[lo:lo + a.batch]
            noise = torch.stack([torch.stack([noise_for(i, a.seed + 1000 * k, (16, side, side), "cuda")
                                              for k in range(S)]) for i, _ in sub])
            z = model(noise, [caps[i].to("cuda") for i, _ in sub], sampling)[-1]
            for (i, _), img in zip(sub, vae.decode(z)):
                tmp = images / f".{i:04d}.png"
                save_uint8_png(img, str(tmp))
                os.replace(tmp, images / f"{i:04d}.png")
            done = lo + len(sub)
            if done % (a.batch * 25) == 0 or done == len(mine):
                print(f"[generate {a.rank}/{a.world}] {done}/{len(mine)} | "
                      f"{(time.time() - t0) / done:.2f} s/image", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
