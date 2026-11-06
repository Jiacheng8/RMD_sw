#!/usr/bin/env python3
"""Retain all official teacher tars and build a canonical 1024px PNG offset index.

CPU-only. At most two independent aria2 transfers overlap indexing. No PNGs are
expanded to disk and no feature/model tensors are created. Each tar is committed
transactionally; only full unique coverage publishes manifest.json.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def legacy_helpers():
    spec = importlib.util.spec_from_file_location("patchgan_teacher_download_helpers",
                                               REPO / "ops/h20-qwen3vl-reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emit(event, **value):
    print(json.dumps({"event": event, "epoch": time.time(), **value}, sort_keys=True), flush=True)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/rdm-workspace/rdm-sets"))
    parser.add_argument("--output", type=Path, help="New index directory; never a shared feature directory")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--download-workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--limit-shards", type=int, help="Pilot only; never publishes a complete manifest")
    parser.add_argument("--reserve-gib", type=float, default=64,
                        help="Conservative disk headroom for models/checkpoints; root still rechecks before training")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if platform.system() == "Darwin":
        raise RuntimeError("teacher image/tar preparation is remote-only")
    if not args.root.is_absolute() or not args.root.is_dir() or args.reserve_gib < 0 \
            or args.limit_shards is not None and not 1 <= args.limit_shards <= 87:
        raise ValueError("invalid remote root, reserve or shard limit")
    old = legacy_helpers()
    args.endpoint = old.validate_endpoint(args.endpoint)
    from rdm.sw_lmmd.reference_images import (
        DATASET, REVISION, NUM_ROWS, TOTAL_TAR_BYTES, ReferenceImageIndexBuilder,
        ReferenceImageStore, atomic_json, canonical_digest, file_sha256, tar_stat,
    )
    pack = args.root / "zimage/zimage-ref-v1"
    source = pack / "reference_store_qwen"
    output = args.output or pack / "reference_images_patchgan"
    if not output.is_absolute() or output.is_symlink() or output.resolve() in (pack.resolve(), source.resolve()):
        raise ValueError("use an independent absolute real-image index directory")
    old.require_real_directory(output)
    with (output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prompts, metadata, render, mapping = old.load_base(pack, source)
        contract = {"dataset": DATASET, "revision": REVISION, "num_rows": NUM_ROWS,
                    "prompt_hash": metadata["prompt_hash"], "render_hash": metadata["render_hash"],
                    "canonical_rows_sha256": file_sha256(source / "rows.jsonl"),
                    "prompt_ids_sha256": file_sha256(source / "prompt_ids.npy"),
                    "row_order_sha256": file_sha256(source / "row_order.npy"),
                    "identity": "prompt_id/image_id/prompt/source/lang/model/seed/steps/guidance/1024_RGB",
                    "writer_policy": "record differences; never substitute identity",
                    "read_contract": "CPU uint8 B3x1024x1024; move uint8 to device before float32/255",
                    "builder_sha256": file_sha256(Path(__file__)),
                    "reader_sha256": file_sha256(REPO / "rdm/sw_lmmd/reference_images.py")}
        final = output / "manifest.json"
        if final.exists():
            completed = ReferenceImageStore(final)
            saved_contract = dict(completed.manifest["contract"])
            saved_contract.pop("official_inventory_digest", None)
            if saved_contract != contract:
                raise ValueError("completed image-index contract differs from current source/code")
            for tar in completed.tars.values():
                if tar_stat(tar["path"]) != tar["stat"]:
                    raise ValueError("completed retained tar changed")
            completed.close()
            emit("already_complete", manifest=str(final), rows=NUM_ROWS)
            return 0
        import huggingface_hub as hf
        info = hf.HfApi(endpoint="https://huggingface.co", token=False).repo_info(
            DATASET, repo_type="dataset", revision=REVISION, files_metadata=True)
        shards = old.official_inventory(info, REVISION)
        if sum(shard["bytes"] for shard in shards) != TOTAL_TAR_BYTES:
            raise ValueError("fixed teacher revision disagrees with the approved 173074565120-byte inventory")
        official = {"dataset": DATASET, "revision": REVISION, "shards": shards,
                    "total_tar_bytes": TOTAL_TAR_BYTES}
        contract["official_inventory_digest"] = canonical_digest(official)
        atomic_json(output / "official-shards.json", official)
        atomic_json(output / "contract.json", contract)
        builder = ReferenceImageIndexBuilder(output, contract, NUM_ROWS)
        try:
            selected = shards[:args.limit_shards] if args.limit_shards is not None else shards
            remaining = [shard for shard in selected if builder.verify_committed(shard) is None]
            staging = args.root / ".cache/qwen3vl-teacher-staging"
            # Committed tars are already charged to free space; uncommitted full
            # files may also be reused after SHA verification by download_tar.
            required = 0
            for shard in remaining:
                candidate = staging / REVISION / shard["name"]
                if not candidate.is_file() or candidate.stat().st_size != shard["bytes"]:
                    required += shard["bytes"]
            free = shutil.disk_usage(args.root).free
            reserve = int(args.reserve_gib * 1024**3)
            emit("disk_preflight", free_bytes=free, new_tar_bytes_upper_bound=required,
                 total_retained_tar_bytes=TOTAL_TAR_BYTES, reserve_bytes=reserve,
                 projected_free_bytes=free - required, all_tars_retained=True)
            if free < required + reserve:
                raise RuntimeError("insufficient free disk for retained tars plus explicit model/checkpoint reserve")
            iterator = iter(remaining)
            with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
                queued = []

                def enqueue():
                    shard = next(iterator, None)
                    if shard is not None:
                        emit("teacher_tar_queued", shard=shard["name"], bytes=shard["bytes"])
                        queued.append((shard, pool.submit(old.download_tar, shard, staging, REVISION, args.endpoint)))

                for _ in range(args.download_workers):
                    enqueue()
                while queued:
                    shard, future = queued.pop(0)
                    tar = future.result()
                    emit("teacher_tar_download_verified", shard=shard["name"], bytes=shard["bytes"], retained=True)
                    enqueue()  # Keep at most two CPU downloads ahead of the indexer.
                    receipt = builder.index_tar(tar, shard, prompts, mapping, render)
                    progress = builder.progress()
                    atomic_json(output / "progress.json", progress)
                    emit("teacher_tar_index_committed", shard=shard["name"],
                         rows=receipt["rows"], writer_differences=receipt["auxiliary_writer_differences"], **progress)
            if args.limit_shards is not None:
                progress = builder.progress()
                partial = builder.publish_partial()
                emit("pilot_complete", **progress, final_manifest_published=False,
                     partial_manifest=str(output / "manifest.partial.json"),
                     available_rows=partial["available_rows"], allow_partial_required=True)
                return 0
            manifest = builder.finalize(shards)
            emit("real_images_complete", manifest=str(final), rows=manifest["num_rows"],
                 tars=len(manifest["tars"]), total_tar_bytes=TOTAL_TAR_BYTES, all_tars_retained=True)
            return 0
        finally:
            builder.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        import re
        message = re.sub(r"https?://\S+|hf_[A-Za-z0-9]+", "[redacted]", str(error))
        emit("failed", type=type(error).__name__, message=message[:1500])
        raise SystemExit(1)
