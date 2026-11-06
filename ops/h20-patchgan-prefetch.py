#!/usr/bin/env python3
"""Pinned Qwen3-VL support files and both shards needed by vision + decoder24.

Public official metadata defines all hashes. JSON/tokenizer support files reuse
HF cache; weight payloads use independent aria2 x16 staging and are never loaded.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import time
from urllib.parse import quote
import uuid

REPO = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
SHARDS = {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}


def helpers():
    path = Path(__file__).with_name("h20-qwen3vl-prefetch.py")
    spec = importlib.util.spec_from_file_location("patchgan_model_download_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory(info, old):
    old.official_inventory(info)  # Includes the approved exact shard2 SHA/size pin.
    result = []
    for entry in info.siblings:
        name = old.field(entry, "rfilename")
        if not isinstance(name, str) or not (name in SHARDS or name.endswith((".json", ".txt", ".model", ".jinja"))):
            continue
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("unsafe support/weight filename")
        lfs, size = old.field(entry, "lfs"), old.field(entry, "size")
        digest = old.field(lfs, "sha256") if lfs else old.field(entry, "blob_id")
        algorithm = "sha256" if lfs else "git_blob_sha1"
        if not isinstance(size, int) or size < 1 or not isinstance(digest, str) \
                or re.fullmatch("[0-9a-f]{%d}" % (64 if lfs else 40), digest) is None:
            raise ValueError("official model support/weight hash is missing")
        result.append({"name": name, "bytes": size, "algorithm": algorithm, "digest": digest})
    if {row["name"] for row in result if row["name"].endswith(".safetensors")} != SHARDS:
        raise ValueError("official revision must provide exactly the expected two model shards")
    return sorted(result, key=lambda row: row["name"])


def needed_shards(index):
    mapping = index["weight_map"]
    visual = [key for key in mapping if key.startswith("model.visual.")]
    embed = "model.language_model.embed_tokens.weight"
    if len(visual) != 315 or embed not in mapping:
        raise ValueError("model index is missing the expected visual tower/token embedding")
    selected = set(visual + [embed])
    for layer in range(24):
        keys = [key for key in mapping if key.startswith(f"model.language_model.layers.{layer}.")]
        if not keys:
            raise ValueError(f"model index is missing language decoder block {layer}")
        selected.update(keys)
    shards = {mapping[key] for key in selected}
    if shards != SHARDS:
        raise ValueError("vision + decoder24 index does not resolve to the approved two shards")
    return shards, sorted(selected)


def download_weight(row, root, endpoint, old):
    folder = root / ".cache/patchgan-vl-staging" / REVISION
    for directory in (root / ".cache", folder.parent, folder):
        old.real_directory(directory)
    payload = folder / row["name"]
    bitmap = payload.with_name(payload.name + ".aria2")
    log_path = folder / (row["name"] + ".download.log")
    exit_path = folder / (row["name"] + ".exit.json")
    for path in (payload, bitmap, log_path):
        if path.is_symlink():
            raise ValueError("refusing linked model staging/control file")
    with (folder / (row["name"] + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if payload.exists() and payload.stat().st_size == row["bytes"]:
            try:
                return payload, old.verify_file(payload, row)
            except ValueError:
                if not bitmap.exists():
                    raise ValueError("full-size wrong-SHA model staging retained; no resumable bitmap")
        # Reuse an independently verified previous vision shard, after re-hashing
        # its actual bytes. It stays in place; no old download/cache is touched.
        if row["name"] == old.SHARD:
            receipt_path = root / "control" / old.RECEIPT
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt.get("repo") == REPO and receipt.get("revision") == REVISION \
                        and receipt.get("all_official_hashes_verified") is True:
                    candidate = Path(receipt["snapshot"]) / row["name"]
                    if candidate.is_file():
                        verified = old.verify_file(candidate, row)
                        old.emit("existing_vision_shard_verified", name=row["name"], bytes=row["bytes"])
                        return candidate, verified
        url = endpoint + "/" + REPO + "/resolve/" + REVISION + "/" + quote(row["name"])
        command = ["aria2c", "-c", "-x16", "-s16", "-k4M", "--all-proxy=", "--http-proxy=",
                   "--https-proxy=", "--auto-file-renaming=false", "--file-allocation=none",
                   "--allow-overwrite=false", "--show-console-readout=false", "--human-readable=false",
                   "--enable-color=false", "--summary-interval=5", "--console-log-level=warn",
                   "--max-tries=5", "--retry-wait=5", "--connect-timeout=30", "--timeout=120",
                   "--dir=" + str(folder), "--out=" + payload.name, url]
        child_env = {key: value for key, value in os.environ.items()
                     if key.lower() not in ("http_proxy", "https_proxy", "all_proxy", "no_proxy")}
        child_env["NO_PROXY"] = "*"
        started, tail = time.time(), []
        old.atomic_json(folder / (row["name"] + ".command.json"), {"command": command, "started_epoch": started})
        old.emit("model_aria2_start", name=row["name"], expected_bytes=row["bytes"], connections=16, log=str(log_path))
        with log_path.open("a") as log:
            log.write(json.dumps({"command": command, "started_epoch": started}) + "\n")
            log.flush()
            try:
                process = subprocess.Popen(command, env=child_env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors="replace")
            except BaseException as error:
                old.atomic_json(exit_path, {"command": command, "exit": None, "spawn_error": type(error).__name__})
                raise
            try:
                for line in process.stdout:
                    safe = old.redact(line.rstrip())
                    log.write(safe + "\n"); log.flush()
                    tail = (tail + [safe])[-30:]
                    progress = old.parse_aria_progress(safe)
                    if progress is not None:
                        old.emit("model_aria2_progress", name=row["name"], expected_bytes=row["bytes"],
                                 progress_source="aria2_console", hashes_verified=False, **progress)
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise
            finally:
                process.stdout.close()
                old.atomic_json(exit_path, {"command": command, "exit": process.returncode, "last30": tail,
                                          "started_epoch": started, "finished_epoch": time.time()})
                log.write(json.dumps({"exit": process.returncode}) + "\n"); log.flush()
        if code != 0:
            raise RuntimeError(f"aria2 exit {code}; exact command/exit/sanitized tail at {exit_path}")
        verified = old.verify_file(payload, row)
        old.emit("model_shard_verified", name=row["name"], completed_bytes=row["bytes"], hashes_verified=True)
        return payload, verified


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/rdm-workspace/rdm-sets"))
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args(argv)
    if platform.system() == "Darwin" or not args.root.is_absolute() or not args.root.is_dir():
        raise RuntimeError("model payload preparation requires an existing remote Linux root")
    from urllib.parse import urlsplit
    parsed = urlsplit(args.endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password \
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("transport endpoint must be a plain HTTPS origin")
    endpoint = args.endpoint.rstrip("/")
    old = helpers()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import huggingface_hub as hf
    import httpx
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    info = hf.HfApi(endpoint="https://huggingface.co", token=False).repo_info(
        REPO, repo_type="model", revision=REVISION, files_metadata=True)
    rows = inventory(info, old)
    hf.set_client_factory(lambda: httpx.Client(trust_env=False, follow_redirects=True, timeout=120))
    support = [row for row in rows if row["name"] not in SHARDS]
    with old.CacheProgress(args.root / "hf/hub", support, 5):
        source = Path(hf.snapshot_download(REPO, repo_type="model", revision=REVISION, token=False,
                      allow_patterns=[row["name"] for row in support], cache_dir=args.root / "hf/hub",
                      endpoint=endpoint, max_workers=4))
    if source.name != REVISION:
        raise ValueError("support snapshot revision mismatch")
    verified, sources = {}, {}
    for row in support:
        sources[row["name"]] = source / row["name"]
        verified[row["name"]] = old.verify_file(sources[row["name"]], row)
    required, keys = needed_shards(json.loads((source / "model.safetensors.index.json").read_text()))
    for row in rows:
        if row["name"] in required:
            sources[row["name"]], verified[row["name"]] = download_weight(row, args.root, endpoint, old)
    base = args.root / "models/qwen3vl-4b-patchgan"
    for directory in (args.root / "models", base):
        old.real_directory(directory)
    final = base / REVISION
    if final.exists() or final.is_symlink():
        if final.is_symlink() or not final.is_dir():
            raise ValueError("existing PatchGAN model snapshot is not a real directory")
        final_verified = [old.verify_file(final / row["name"], row) for row in rows]
    else:
        temporary = base / ("." + REVISION + ".assembling." + uuid.uuid4().hex)
        temporary.mkdir()
        for row in rows:
            destination = temporary / row["name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if row["name"] in SHARDS:
                os.link(sources[row["name"]], destination)
            else:
                os.symlink(str(sources[row["name"]].resolve()), destination)
        # Check indexed required keys against both actual safetensors headers.
        import struct
        actual = set()
        for name in sorted(required):
            with (temporary / name).open("rb") as shard:
                length = struct.unpack("<Q", shard.read(8))[0]
                if not 1 <= length <= 64 * 1024**2:
                    raise ValueError("invalid model safetensors header size")
                actual.update(json.loads(shard.read(length)))
        if not set(keys).issubset(actual):
            raise ValueError("vision/decoder24 indexed keys are absent from verified shard headers")
        os.rename(temporary, final)
        final_verified = [{**verified[row["name"]], "path": str(final / row["name"]),
                           "source_path": str(sources[row["name"]])} for row in rows]
    receipt = {"repo": REPO, "revision": REVISION, "official_endpoint": "https://huggingface.co",
               "transport_endpoint": endpoint, "payload_transport": "aria2_x16",
               "snapshot": str(final), "files": final_verified, "required_shards": sorted(required),
               "decoder_blocks": 24, "required_key_count": len(keys), "model_tensors_loaded": False,
               "all_official_hashes_verified": True, "verified_epoch": time.time(), "receipt_version": 1}
    receipt_path = args.root / "control/patchgan-qwen3vl-prefetch-verified.json"
    old.atomic_json(receipt_path, receipt)
    old.emit("patchgan_model_complete", snapshot=str(final), receipt=str(receipt_path),
             total_bytes=sum(row["bytes"] for row in rows), decoder_blocks=24, all_official_hashes_verified=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"event": "failed", "type": type(error).__name__,
                          "message": re.sub(r"https?://\S+|hf_[A-Za-z0-9]+", "[redacted]", str(error))[:1500]}), flush=True)
        raise SystemExit(1)
