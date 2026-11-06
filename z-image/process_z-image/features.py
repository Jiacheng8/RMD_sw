"""Stream frozen RDM encoder features and build a SW-LMMD reference store."""
import argparse
import json
from pathlib import Path
import sys

from common import (atomic_json, atomic_save, digest, load_prompts, lock, prompt_dir,
                    read_json, selection)


def validate_renders(root, rows, config, deep=False):
    import torch
    config_hash = digest(config)
    for p, row in enumerate(rows):
        data = selection(root, p, config["keep"], verify=deep)
        folder = prompt_dir(root, p)
        if (not data or data.get("config_hash") != config_hash or
                data.get("image_id") != row["image_id"] or not (folder / "context.pt").is_file()):
            raise ValueError(f"Prompt {p} has no complete render commit; resume render first")
        if deep:
            ctx = torch.load(folder / "context.pt", map_location="cpu", weights_only=True)
            if (ctx.ndim != 2 or list(ctx.shape) != [data["context_tokens"], data["context_dim"]]
                    or not torch.isfinite(ctx).all() or not ctx.abs().max() > 0):
                raise ValueError(f"Invalid Z-Image context for prompt {p}")


def atomic_npy(path, value):
    import numpy as np
    atomic_save(path, lambda f: np.save(f, value))


def extract(a, rows, render, names):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from rdm.data.imagenet import ImageListDataset
    from rdm.representation.registry import by_name
    from rdm.representation.backbones import build_backbone
    from rdm.compare.median_heuristic import median_bandwidth
    n = len(rows) * render["keep"]
    paths = [str(prompt_dir(a.root, p) / f"k{k}.png")
             for p in range(len(rows)) for k in range(render["keep"])]
    for name in names:
        by_name(name)  # validate before writing paths
        folder = a.root / "reference_store" / "encoder_features"
        folder.mkdir(parents=True, exist_ok=True)
        final = folder / f"{name}.npy"
        part = folder / f"{name}.partial.npy"
        progress = folder / f"{name}.progress.json"
        done = folder / f"{name}.done.json"
        contract = dict(render_hash=digest(render), encoder=name, rows=n, img_size=a.img_size,
                        sigma_scale=a.sigma_scale, s_txt=a.s_txt, seed=a.seed)
        with lock(folder / f".{name}.lock"):
            if done.exists():
                state = read_json(done)
                if state["contract"] != contract:
                    raise ValueError(f"Feature config changed for {name}; use a new root")
                arr = np.load(final, mmap_mode="r")
                if list(arr.shape) != state["shape"]:
                    raise ValueError(f"Broken feature artifact: {final}")
                print(f"{name}: complete", flush=True)
                continue
            offset = 0
            arr = None
            if progress.exists():
                state = read_json(progress)
                if state["contract"] != contract:
                    raise ValueError(f"Feature config changed for {name}")
                offset = state["completed"]
                source = part if part.exists() else final
                arr = np.load(source, mmap_mode="r+")
                if list(arr.shape) != state["shape"] or not 0 <= offset <= n:
                    raise ValueError(f"Invalid feature checkpoint for {name}")
            if offset < n:
                backbone = build_backbone(by_name(name), device="cuda")
                amp = not getattr(backbone, "has_logits", False)
                loader = DataLoader(ImageListDataset(paths[offset:], img_size=a.img_size),
                                    batch_size=a.batch, shuffle=False, num_workers=a.workers,
                                    pin_memory=True)
                with torch.inference_mode():
                    for imgs in loader:
                        imgs = imgs.to("cuda")
                        with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
                            value = backbone(imgs).float().cpu().numpy()
                        if not np.isfinite(value).all():
                            raise ValueError(f"Nonfinite features from {name}")
                        if arr is None:
                            arr = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16,
                                                           shape=(n, value.shape[1]))
                        cast = value.astype(np.float16)
                        if not np.isfinite(cast).all():
                            raise ValueError(f"fp16 feature overflow for {name}")
                        arr[offset:offset+len(value)] = cast
                        arr.flush()
                        offset += len(value)
                        atomic_json(progress, dict(contract=contract, completed=offset, shape=list(arr.shape)))
                        print(f"{name}: {offset}/{n}", flush=True)
                del backbone
                torch.cuda.empty_cache()
            # Deterministic bounded sample; no full fp32 feature pool in RAM.
            sample_ids = np.random.default_rng(a.seed).choice(n, min(n, 2000), replace=False)
            sample = torch.from_numpy(np.asarray(arr[sample_ids], dtype=np.float32))
            torch.manual_seed(a.seed)
            sigma = float(median_bandwidth(sample)) * a.sigma_scale
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError(f"Invalid bandwidth for {name}")
            shape = list(arr.shape)
            del arr
            if part.exists():
                part.replace(final)
            atomic_json(done, dict(contract=contract, shape=shape,
                                   bandwidth=dict(sigma=sigma, beta=sigma/a.s_txt,
                                                  sigma_scale=a.sigma_scale, s_txt=a.s_txt,
                                                  estimator="median over deterministic <=2000 fp16 stored rows")))


def finalize(a, rows, meta, render, names):
    import numpy as np
    import torch
    from rdm.representation.text_encoder import TextEncoder
    from rdm.sw_lmmd import ReferenceFeatureStore
    store = a.root / "reference_store"
    store.mkdir(exist_ok=True)
    with lock(store / ".finalize.lock"):
        # Remove the public completion marker before rebuilding or validating artifacts.
        (store / "metadata.json").unlink(missing_ok=True)
        bands, dims = {}, {}
        n = len(rows) * render["keep"]
        for name in names:
            done = read_json(store / "encoder_features" / f"{name}.done.json")
            if done["contract"]["render_hash"] != digest(render):
                raise ValueError(f"Stale features for {name}")
            arr = np.load(store / "encoder_features" / f"{name}.npy", mmap_mode="r")
            if arr.shape[0] != n or list(arr.shape) != done["shape"]:
                raise ValueError(f"Wrong feature shape for {name}")
            for lo in range(0, n, 8192):
                if not np.isfinite(arr[lo:lo+8192]).all():
                    raise ValueError(f"Nonfinite stored features for {name}")
            bands[name], dims[name] = done["bandwidth"], arr.shape[1]
        # New prompt set => encode tau afresh. The released COCO table is never reused.
        tau_path = store / "text_features.npy"
        tau_meta = store / "text_features.done.json"
        tau_contract = dict(prompt_hash=meta["prompt_hash"], model=a.text_model, pretrained=a.text_pretrained)
        if tau_meta.exists() and read_json(tau_meta) != tau_contract:
            raise ValueError("Text encoder changed; use a new reference root")
        if not tau_meta.exists():
            encoder = TextEncoder(model_name=a.text_model, pretrained=a.text_pretrained, device="cuda")
            tmp = store / "text_features.partial.npy"
            arr = None
            for lo in range(0, len(rows), a.batch):
                value = encoder.encode([r["prompt"] for r in rows[lo:lo+a.batch]], batch=a.batch).numpy()
                if not np.isfinite(value).all():
                    raise ValueError("Nonfinite text embeddings")
                if arr is None:
                    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                                   shape=(len(rows), value.shape[1]))
                arr[lo:lo+len(value)] = value
            arr.flush()
            del arr, encoder
            tmp.replace(tau_path)
            atomic_json(tau_meta, tau_contract)
        tau = np.load(tau_path, mmap_mode="r")
        if tau.shape[0] != len(rows) or not np.isfinite(tau).all() or not np.allclose(
                np.linalg.norm(tau, axis=1), 1, atol=1e-3):
            raise ValueError("Invalid text feature table")
        ids = np.repeat(np.arange(len(rows), dtype=np.int64), render["keep"])
        atomic_npy(store / "prompt_ids.npy", ids)
        atomic_npy(store / "group_ids.npy", ids)
        atomic_npy(store / "row_order.npy", np.random.default_rng(a.seed).permutation(n).astype(np.int64))
        atomic_json(store / "bandwidths.json", bands)
        atomic_json(store / "z_context.json", dict(format="ragged-bf16-torch-v1",
                    root="..", num_prompts=len(rows), prompt_hash=meta["prompt_hash"],
                    max_sequence_length=render["max_sequence_length"], model_path=render["model_path"]))
        atomic_save(store / "rows.jsonl", lambda f: f.write("".join(
            json.dumps(dict(row_id=p*render["keep"]+k, prompt_id=p, image_id=r["image_id"],
                            path=str((prompt_dir(a.root, p) / f"k{k}.png").relative_to(a.root))),
                       ensure_ascii=False) + "\n"
            for p, r in enumerate(rows) for k in range(render["keep"])).encode()))
        metadata = dict(num_rows=n, num_prompts=len(rows), rows_per_prompt=render["keep"],
            text_dim=tau.shape[1], encoder_feature_dims=dims, prompt_hash=meta["prompt_hash"],
            render_hash=digest(render), reference_type="z_image_turbo_teacher", feature_dtype="float16",
            context_format="ragged-bf16-torch-v1", text_encoder=tau_contract,
            requires_generator_adapter="ZImageReferenceStore; FLUX context adapters are incompatible")
        # Reader expects metadata; publish it only after every other artifact is validated.
        atomic_json(store / "metadata.json", metadata)
        try:
            reader = ReferenceFeatureStore(str(store), names, require_context=False)
            reader.row_order()
            for name in names:
                reader.reference_joint_features(name, [0, n-1])
            from reference_store import ZImageReferenceStore
            contexts = ZImageReferenceStore(store, names).generator_context([0, n-1])
            assert len(contexts) == 2 and all(torch.isfinite(c).all() for c in contexts)
        except Exception:
            (store / "metadata.json").unlink(missing_ok=True)
            raise
        print(f"Validated {n} rows / {len(rows)} prompts -> {store}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["extract", "finalize", "check"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--rdm-repo", type=Path, default=Path(__file__).resolve().parents[2])  # this checkout
    ap.add_argument("--encoders", default="dinov3_l,siglip2,aimv2_huge")
    ap.add_argument("--img-size", type=int, default=512, help="Same ADM crop as old feature pipeline")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sigma-scale", type=float, default=0.25)
    ap.add_argument("--s-txt", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--text-model", default="ViT-SO400M-16-SigLIP2-256")
    ap.add_argument("--text-pretrained", default="webli")
    a = ap.parse_args()
    if min(a.batch, a.img_size, a.sigma_scale, a.s_txt) <= 0 or a.workers < 0:
        ap.error("Invalid batch/crop/kernel/worker setting")
    sys.path.insert(0, str(a.rdm_repo.resolve()))
    rows, meta = load_prompts(a.root)
    render = read_json(a.root / "render_config.json")
    if render["prompt_hash"] != meta["prompt_hash"]:
        raise ValueError("Render/prompt checksum mismatch")
    names = a.encoders.split(",")
    if any(not n or not n.replace("_", "").isalnum() for n in names) or len(set(names)) != len(names):
        ap.error("Encoder names must be distinct registry names")
    validate_renders(a.root, rows, render, deep=a.stage in ("check", "finalize"))
    if a.stage == "extract":
        extract(a, rows, render, names)
    elif a.stage == "finalize":
        finalize(a, rows, meta, render, names)
    else:
        from reference_store import ZImageReferenceStore
        reader = ZImageReferenceStore(a.root / "reference_store", names)
        reader.row_order()
        print(f"OK: {reader.num_rows} rows, {reader.num_prompts} prompts; images and contexts checked")


if __name__ == "__main__":
    main()
