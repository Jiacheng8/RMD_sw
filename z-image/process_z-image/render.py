"""One GPU worker: Z-Image-Turbo candidates -> bounded top-k pool -> atomic prompt commit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import math
import pickle
from pathlib import Path
import time

from common import (atomic_json, atomic_save, digest, load_prompts, lock, prompt_dir,
                    read_json, seed_for, selection)


def resolve_model(model, revision, offline):
    if Path(model).is_dir():
        if revision:
            raise ValueError("--revision applies to Hub IDs, not local model directories")
        return str(Path(model).resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(model, revision=revision, local_files_only=offline)


class Scorer:
    def __init__(self, kind, model_path, processor_path, device):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.device, self.kind = torch, device, kind
        self.processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)

    def embedding(self, output):
        return output if self.torch.is_tensor(output) else output.pooler_output

    def encode_text(self, text):
        torch = self.torch
        # Fixed model-native tokenizer geometry (SigLIP2=64, PickScore=77).
        limit = 64 if self.kind == "siglip2" else 77
        raw_length = len(self.processor.tokenizer(text)["input_ids"])
        tokens = self.processor(text=[text], padding="max_length", truncation=True,
                                max_length=limit, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            features = self.embedding(self.model.get_text_features(**tokens)).float()
            features = torch.nn.functional.normalize(features, dim=-1)
        return features, raw_length > limit

    def score(self, images, text_features):
        torch = self.torch
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            features = self.embedding(self.model.get_image_features(**inputs)).float()
            features = torch.nn.functional.normalize(features, dim=-1)
            # Cosine suffices: model logit scale/bias are constant within each prompt.
            return (features @ text_features.T).flatten().cpu().tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo")
    ap.add_argument("--revision", help="Use a commit hash for reproducible remote loading")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--max-sequence-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--curate", choices=["siglip2", "pickscore", "none"], default="siglip2")
    ap.add_argument("--scorer-model")
    ap.add_argument("--scorer-revision")
    ap.add_argument("--processor-model")
    ap.add_argument("--processor-revision")
    ap.add_argument("--score-device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--cpu-offload", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the transformer (~18%% faster on a 4090; small numeric differences)")
    ap.add_argument("--dry-run", action="store_true", help="No model import/download/output writes")
    a = ap.parse_args()
    if not 0 <= a.rank < a.world or not 1 <= a.keep <= a.candidates:
        ap.error("Require 0 <= rank < world and 1 <= keep <= candidates")
    if min(a.batch, a.steps, a.max_sequence_length, a.height, a.width) < 1:
        ap.error("Batch, steps, sequence length and dimensions must be positive")
    if a.height % 16 or a.width % 16:
        ap.error("Height and width must be multiples of 16")
    if a.curate == "none" and a.keep != a.candidates:
        ap.error("--curate none requires --keep == --candidates")
    rows, meta = load_prompts(a.root)
    indices = list(range(a.rank, len(rows), a.world))
    print(f"rank {a.rank}/{a.world}: {len(indices)} prompts, "
          f"{len(indices)*a.candidates} candidates, {len(indices)*a.keep} kept", flush=True)
    if a.dry_run or not indices:
        return
    import numpy as np
    import torch
    from diffusers import ZImagePipeline
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; use the CUDA PyTorch environment and check the NVIDIA driver")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This pipeline requires a GPU with bfloat16 support")
    model_path = resolve_model(a.model, a.revision, a.offline)
    scorer_id = a.scorer_model or ("google/siglip2-so400m-patch16-256" if a.curate != "pickscore"
                                   else "yuvalkirstain/PickScore_v1")
    scorer_path = processor_path = None
    if a.curate != "none":
        scorer_path = resolve_model(scorer_id, a.scorer_revision, a.offline)
        processor_id = a.processor_model or (scorer_id if a.curate == "siglip2"
                                              else "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        processor_path = (scorer_path if processor_id == scorer_id and not a.processor_revision
                          else resolve_model(processor_id, a.processor_revision, a.offline))
    config = {k: getattr(a, k) for k in ("candidates", "keep", "batch", "steps", "height", "width",
               "max_sequence_length", "seed", "curate", "score_device", "cpu_offload", "compile")}
    config.update(prompt_hash=meta["prompt_hash"], model_path=model_path, scorer_path=scorer_path,
                  processor_path=processor_path, guidance_scale=0.0,
                  versions={k: importlib.metadata.version(k) for k in
                            ("torch", "diffusers", "transformers", "numpy", "Pillow")})
    config_hash = digest(config)
    with lock(a.root / ".render-config.lock"):
        config_file = a.root / "render_config.json"
        if config_file.exists() and read_json(config_file) != config:
            raise SystemExit("Render config changed. Use a new output root to avoid mixed references.")
        atomic_json(config_file, config)
    # Lock the entire launch: separate workers in one launch share world, disjoint prompt IDs.
    # Per-prompt locks below also reject overlapping launches with a different world size.
    with lock(a.root / f".render-rank-{a.rank}.lock"):
        pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                              local_files_only=True)
        if a.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        if a.compile:  # dynamic shapes: prompt token lengths differ
            pipe.transformer = torch.compile(pipe.transformer, dynamic=True)
        pipe.set_progress_bar_config(disable=True)
        scorer = Scorer(a.curate, scorer_path, processor_path, a.score_device) if scorer_path else None
        start, completed = time.monotonic(), 0
        # PNG encoding + the commit run on one background thread while the GPU renders the next
        # prompt; at most one commit is pending, and the commit is still written after its images.
        saver, pending = ThreadPoolExecutor(max_workers=1), None

        def commit(folder, p, top, ctx, truncated):
            with lock(folder / ".lock"):
                kept = []
                for k, (score, candidate, seed, image) in enumerate(top):
                    atomic_save(folder / f"k{k}.png", lambda f, image=image: image.save(f, format="PNG"))
                    kept.append(dict(rank=k, candidate=candidate, seed=seed,
                                     score=score if scorer else None, file=f"k{k}.png"))
                atomic_json(folder / "selection.json", dict(prompt_id=p, image_id=rows[p]["image_id"],
                    config_hash=config_hash, scorer_text_truncated=truncated,
                    context_tokens=ctx.shape[0], context_dim=ctx.shape[1], images=kept))

        for p in indices:
            folder = prompt_dir(a.root, p)
            with lock(folder / ".lock"):
                done = selection(a.root, p, a.keep)
                if done and done.get("config_hash") == config_hash and (folder / "context.pt").is_file():
                    try:
                        cached = torch.load(folder / "context.pt", map_location="cpu", weights_only=True)
                        if (cached.ndim == 2 and list(cached.shape) ==
                                [done["context_tokens"], done["context_dim"]]
                                and torch.isfinite(cached).all() and cached.abs().max() > 0):
                            continue
                    except (OSError, ValueError, RuntimeError, EOFError, KeyError, pickle.UnpicklingError):
                        pass
                # A stale commit must never survive a failed regeneration.
                (folder / "selection.json").unlink(missing_ok=True)
                text = rows[p]["prompt"]
                with torch.inference_mode():
                    context, _ = pipe.encode_prompt([text], device=torch.device("cuda"),
                        do_classifier_free_guidance=False, max_sequence_length=a.max_sequence_length)
                ctx = context[0].detach().cpu().contiguous()  # preserve bf16 and actual token length
                atomic_save(folder / "context.pt", lambda f: torch.save(ctx, f))
                text_features, truncated = scorer.encode_text(text) if scorer else (None, False)
                if scorer:
                    atomic_save(folder / "text.npy", lambda f: np.save(f, text_features.cpu().numpy()[0]))
                top = []  # only keep+batch PIL images resident; candidates never all land on disk
                for lo in range(0, a.candidates, a.batch):
                    candidates = list(range(lo, min(lo + a.batch, a.candidates)))
                    seeds = [seed_for(rows[p]["image_id"], j, a.seed) for j in candidates]
                    with torch.inference_mode():
                        images = pipe(prompt_embeds=[ctx.to("cuda") for _ in candidates],
                            height=a.height, width=a.width, num_inference_steps=a.steps,
                            guidance_scale=0.0, generator=[torch.Generator("cpu").manual_seed(s) for s in seeds],
                            max_sequence_length=a.max_sequence_length).images
                    scores = scorer.score(images, text_features) if scorer else [0.0] * len(images)
                    if len(images) != len(candidates) or not all(math.isfinite(s) for s in scores):
                        raise RuntimeError("Invalid image batch or nonfinite selection score")
                    top.extend(zip(scores, candidates, seeds, images))
                    top.sort(key=lambda x: (-x[0], x[1]))  # stable ties by candidate index
                    del top[a.keep:]
            if pending is not None:
                pending.result()  # surfaces a failed save; bounds memory to one prompt in flight
            pending = saver.submit(commit, folder, p, top, ctx, truncated)
            completed += 1
            print(f"rank {a.rank}: prompt {p} rendered; {completed} prompts / "
                  f"{time.monotonic()-start:.1f}s; scorer_truncated={truncated}", flush=True)
        if pending is not None:
            pending.result()  # the last commit must land before the worker reports success
        saver.shutdown()


if __name__ == "__main__":
    main()
