#!/usr/bin/env python3
"""Prefetch the pinned Qwen3-VL-4B vision-containing shard on the remote node.

Official metadata defines every expected byte count/digest. A mirror may only
transport these exact files; no first shard or complete language model is loaded.
HF's normal cache locks and incomplete files provide resumable transfers.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

REPO = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
OFFICIAL = "https://huggingface.co"
SHARD = "model-00002-of-00002.safetensors"
SHARD_BYTES = 3908490048
SHARD_SHA256 = "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6"
VISUAL_KEYS = 315
REQUIRED_JSON = {"config.json", "preprocessor_config.json", "model.safetensors.index.json"}
RECEIPT = "qwen3vl-4b-vision-prefetch-verified.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **fields):
    print(json.dumps({"event": event, "time": now(), **fields}, sort_keys=True), flush=True)


def field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def official_inventory(info):
    """Validate pinned official metadata before any payload request."""
    if field(info, "sha") != REVISION:
        raise ValueError("official metadata did not resolve the pinned revision")
    rows = []
    for entry in field(info, "siblings", []):
        name = field(entry, "rfilename")
        if not isinstance(name, str) or not (name.endswith(".json") or name == SHARD):
            continue
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("unsafe filename in official metadata")
        size = field(entry, "size")
        lfs = field(entry, "lfs")
        algorithm = "sha256" if lfs else "git_blob_sha1"
        digest = field(lfs, "sha256") if lfs else field(entry, "blob_id")
        width = 64 if lfs else 40
        if not isinstance(size, int) or size < 1 or not isinstance(digest, str) \
                or re.fullmatch(r"[0-9a-f]{%d}" % width, digest) is None:
            raise ValueError(f"missing official size/hash for {name}")
        rows.append({"name": name, "bytes": size, "algorithm": algorithm, "digest": digest})
    by_name = {row["name"]: row for row in rows}
    if len(by_name) != len(rows) or not REQUIRED_JSON.issubset(by_name) or SHARD not in by_name:
        raise ValueError("official inventory has duplicate or missing required files")
    shard = by_name[SHARD]
    if (shard["bytes"], shard["algorithm"], shard["digest"]) != \
            (SHARD_BYTES, "sha256", SHARD_SHA256):
        raise ValueError("official shard metadata disagrees with the approved pin")
    return sorted(rows, key=lambda row: row["name"])


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("w") as output:
            json.dump(data, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_file(path, expected):
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"size mismatch: {expected['name']}")
    sha256 = hashlib.sha256()
    official = sha256 if expected["algorithm"] == "sha256" else hashlib.sha1()
    if expected["algorithm"] == "git_blob_sha1":
        official.update(f"blob {expected['bytes']}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            sha256.update(block)
            if official is not sha256:
                official.update(block)
    if official.hexdigest() != expected["digest"]:
        raise ValueError(f"official hash mismatch: {expected['name']}")
    return {**expected, "path": str(path), "sha256": sha256.hexdigest(), "verified": True}


def verify_visual_inventory(directory):
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    mapping = index.get("weight_map", {})
    visual = {key for key in mapping if key.startswith("model.visual.")}
    if len(visual) != VISUAL_KEYS or any(mapping[key] != SHARD for key in visual):
        raise ValueError("official index does not place all 315 visual keys in the selected shard")
    # Header inspection is CPU-only and never materializes any model tensors.
    with (directory / SHARD).open("rb") as source:
        length = source.read(8)
        if len(length) != 8:
            raise ValueError("truncated safetensors header")
        header_bytes = struct.unpack("<Q", length)[0]
        if not 1 <= header_bytes <= 64 * 1024**2:
            raise ValueError("invalid safetensors header length")
        header = json.loads(source.read(header_bytes))
    actual = {key for key in header if key.startswith("model.visual.")}
    if actual != visual:
        raise ValueError("safetensors visual keys disagree with the verified official index")
    return sorted(visual)


def redact(value):
    return re.sub(r"https?://\S+|hf_[A-Za-z0-9]+", "[redacted]", str(value))


def real_directory(path):
    if path.is_symlink():
        raise ValueError(f"refusing symlink staging/snapshot directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"not a directory: {path}")


def parse_aria_progress(line):
    """Read actual completed bytes from aria2, never its sparse file length."""
    match = re.search(
        r"\[#[0-9a-f]+\s+(\d+)B?/(\d+)B?(?:\([^)]*\))?[^\]]*?\bDL:(\d+)B?\b", line,
        re.IGNORECASE)
    if match is None:
        return None
    return {"completed_bytes": int(match[1]), "aria_total_bytes": int(match[2]),
            "download_speed_bytes_per_second": int(match[3])}


def aria2_shard(root, expected, endpoint, progress_seconds):
    """Download into separate resumable staging; never open HF payload blobs."""
    staging = root / ".cache/qwen3vl-model-aria2"
    folder = staging / REVISION
    for directory in (root / ".cache", staging, folder):
        real_directory(directory)
    payload = folder / SHARD
    bitmap = payload.with_name(payload.name + ".aria2")
    log_path = folder / "aria2.download.log"
    exit_path = folder / "aria2.exit.json"
    for path in (payload, bitmap, log_path):
        if path.is_symlink():
            raise ValueError(f"refusing symlink aria2 payload/control: {path}")
    # Avoid two fast-mode processes changing the same staging payload. The
    # existing HF transfer owns different files and is not affected by this lock.
    with (folder / "aria2.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if payload.exists() and payload.stat().st_size == expected["bytes"]:
            try:
                result = verify_file(payload, expected)
            except ValueError:
                if not bitmap.exists():
                    raise ValueError("full-size aria2 staging has a wrong official SHA and no bitmap; retained")
            else:
                emit("aria2_verified_reuse", completed_bytes=expected["bytes"],
                     expected_bytes=expected["bytes"], hashes_verified=True)
                return payload, result, {"aria2_log": str(log_path), "aria2_exit": str(exit_path),
                                         "staging_payload": str(payload), "reused_verified_payload": True}
        url = endpoint.rstrip("/") + "/" + REPO + "/resolve/" + REVISION + "/" + quote(SHARD)
        command = ["aria2c", "-c", "-x16", "-s16", "-k4M", "--all-proxy=", "--http-proxy=",
                   "--https-proxy=", "--auto-file-renaming=false", "--file-allocation=none",
                   "--allow-overwrite=false", "--show-console-readout=false", "--human-readable=false",
                   "--enable-color=false", "--summary-interval=" + str(max(1, int(progress_seconds))),
                   "--console-log-level=warn", "--max-tries=5", "--retry-wait=5", "--connect-timeout=30",
                   "--timeout=120", "--dir=" + str(folder), "--out=" + payload.name, url]
        child_env = {key: value for key, value in os.environ.items()
                     if key.lower() not in ("http_proxy", "https_proxy", "all_proxy", "no_proxy")}
        child_env["NO_PROXY"] = "*"
        started = now()
        atomic_json(folder / "aria2.command.json", {"command": command, "started_at": started})
        tail, interrupted = [], None
        emit("aria2_start", name=SHARD, expected_bytes=expected["bytes"], connections=16,
             log=str(log_path), completed_bytes=None, progress_source="aria2_console")
        with log_path.open("a") as log:
            log.write(json.dumps({"command": command, "started_at": started}) + "\n")
            log.flush()
            try:
                process = subprocess.Popen(command, env=child_env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors="replace")
            except BaseException as error:
                atomic_json(exit_path, {"command": command, "exit": None, "last30": [],
                                       "started_at": started, "finished_at": now(),
                                       "spawn_error": type(error).__name__})
                raise
            try:
                for line in process.stdout:
                    safe = redact(line.rstrip())
                    log.write(safe + "\n")
                    log.flush()
                    tail = (tail + [safe])[-30:]
                    progress = parse_aria_progress(safe)
                    if progress is not None:
                        emit("aria2_progress", name=SHARD, expected_bytes=expected["bytes"],
                             hashes_verified=False, progress_source="aria2_console", **progress)
                code = process.wait()
            except BaseException as error:
                interrupted = type(error).__name__
                process.terminate()  # Only this newly created aria2 child.
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                code = process.returncode
                raise
            finally:
                process.stdout.close()
                outcome = {"command": command, "exit": process.returncode, "last30": tail,
                           "started_at": started, "finished_at": now(), "interrupted": interrupted}
                atomic_json(exit_path, outcome)
                log.write(json.dumps({"exit": process.returncode, "finished_at": now()}) + "\n")
                log.flush()
        if code != 0:
            raise RuntimeError(f"aria2 exit {code}; command, exit and sanitized tail retained at {exit_path}")
        # Exit0 alone and full sparse length alone are insufficient.
        emit("hash_start", name=SHARD, bytes=expected["bytes"])
        result = verify_file(payload, expected)
        emit("aria2_complete", name=SHARD, completed_bytes=expected["bytes"],
             expected_bytes=expected["bytes"], hashes_verified=True, aria2_exit=0)
        return payload, result, {"aria2_log": str(log_path), "aria2_exit": str(exit_path),
                                 "staging_payload": str(payload), "reused_verified_payload": False}


def assemble_aria_snapshot(root, json_snapshot, payload, rows, verified_sources):
    """Publish a separate view, leaving all active HF payload cache files alone."""
    base = root / "models/qwen3vl-4b-vision"
    for directory in (root / "models", base):
        real_directory(directory)
    final = base / REVISION
    if final.exists() or final.is_symlink():
        if final.is_symlink() or not final.is_dir():
            raise ValueError("existing independent snapshot is not a real directory")
        verified = [verify_file(final / row["name"], row) for row in rows]
        return final, verified, verify_visual_inventory(final)
    temporary = base / ("." + REVISION + ".assembling." + uuid.uuid4().hex)
    temporary.mkdir()
    for row in rows:
        destination = temporary / row["name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if row["name"] == SHARD:
            os.link(payload, destination)  # Same ROOT filesystem; no second payload copy.
        else:
            os.symlink(str((json_snapshot / row["name"]).resolve()), destination)
    visual = verify_visual_inventory(temporary)
    os.rename(temporary, final)
    verified = [{**verified_sources[row["name"]], "path": str(final / row["name"]),
                 "source_path": verified_sources[row["name"]]["path"]} for row in rows]
    return final, verified, visual


def observed_blob_bytes(blob):
    """Observe one blob, including HF's digest.<etag>.incomplete naming."""
    candidates = [blob, blob.with_name(blob.name + ".incomplete")]
    # The separating dot keeps another blob with a longer digest prefix out.
    candidates.extend(blob.parent.glob(blob.name + ".*.incomplete"))
    available = 0
    for candidate in candidates:
        try:
            available = max(available, candidate.stat().st_size)
        except FileNotFoundError:
            pass  # The downloader may rename a completed partial during polling.
    return available


class CacheProgress:
    """Report observed cache bytes, including resumable .incomplete files."""
    def __init__(self, cache, rows, interval):
        self.cache, self.rows, self.interval = cache, rows, interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.previous = 0

    def report(self):
        files = []
        base = self.cache / ("models--" + REPO.replace("/", "--")) / "blobs"
        for row in self.rows:
            blob = base / row["digest"]
            available = observed_blob_bytes(blob)
            files.append({"name": row["name"], "observed_bytes": available,
                          "expected_bytes": row["bytes"]})
        total = sum(row["observed_bytes"] for row in files)
        emit("cache_bytes", observed_bytes=total, delta_bytes=total - self.previous,
             expected_bytes=sum(row["bytes"] for row in self.rows), files=files,
             hashes_verified=False)
        self.previous = total

    def run(self):
        while not self.stop.wait(self.interval):
            self.report()

    def __enter__(self):
        self.report()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.report()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/rdm-workspace/rdm-sets"))
    parser.add_argument("--endpoint", default="https://hf-mirror.com",
                        help="Payload transport only; metadata always uses official HF")
    parser.add_argument("--transport", choices=("hf", "aria2"), default="hf",
                        help="aria2 uses separate 16-connection model staging; JSON still uses HF")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--progress-seconds", type=float, default=5)
    parser.add_argument("--trust-env-transport", action="store_true",
                        help="Use existing proxy variables for payload requests")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if platform.system() == "Darwin":
        raise RuntimeError("payload downloads are remote-only; refusing to run on Mac")
    if not args.root.is_absolute() or not args.root.is_dir():
        raise ValueError("--root must be an existing absolute remote directory")
    if args.max_workers < 1 or args.progress_seconds <= 0:
        raise ValueError("worker count and progress interval must be positive")
    endpoint = urlsplit(args.endpoint)
    if endpoint.scheme != "https" or not endpoint.netloc or endpoint.username \
            or endpoint.password or endpoint.query or endpoint.fragment or endpoint.path not in ("", "/"):
        raise ValueError("transport endpoint must be an HTTPS origin without credentials")
    # Public repo: no credential is needed or sent to the transport mirror.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import httpx
    import huggingface_hub as hf
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    emit("official_metadata", repo=REPO, revision=REVISION, endpoint=OFFICIAL)
    info = hf.HfApi(endpoint=OFFICIAL, token=False).repo_info(
        REPO, repo_type="model", revision=REVISION, files_metadata=True)
    rows = official_inventory(info)
    cache = args.root / "hf/hub"
    control = args.root / "control"
    inventory = {"repo": REPO, "revision": REVISION, "official_endpoint": OFFICIAL,
                 "transport_endpoint": args.endpoint.rstrip("/"), "files": rows,
                 "payload_transport": args.transport,
                 "metadata_checked_at": now(), "first_shard_requested": False}
    atomic_json(control / "qwen3vl-4b-prefetch-inventory.json", inventory)
    # Only the process-local transport client changes after official metadata.
    hf.set_client_factory(lambda: httpx.Client(
        trust_env=args.trust_env_transport, follow_redirects=True, timeout=120))
    emit("fetch", repo=REPO, revision=REVISION, selected_files=[row["name"] for row in rows],
         expected_bytes=sum(row["bytes"] for row in rows), cache=str(cache), transport=args.transport)
    hf_rows = rows if args.transport == "hf" else [row for row in rows if row["name"] != SHARD]
    with CacheProgress(cache, hf_rows, args.progress_seconds):
        hf_directory = Path(hf.snapshot_download(
            REPO, repo_type="model", revision=REVISION,
            allow_patterns=[row["name"] for row in hf_rows], cache_dir=cache,
            max_workers=args.max_workers, endpoint=args.endpoint.rstrip("/"), token=False))
    if hf_directory.name != REVISION:
        raise ValueError("download snapshot directory does not match the pinned revision")
    source_verified = {}
    for row in hf_rows:
        emit("hash_start", name=row["name"], bytes=row["bytes"])
        source_verified[row["name"]] = verify_file(hf_directory / row["name"], row)
        emit("hash_verified", name=row["name"], bytes=row["bytes"], algorithm=row["algorithm"])
    transfer_details = {}
    if args.transport == "aria2":
        shard = next(row for row in rows if row["name"] == SHARD)
        payload, result, transfer_details = aria2_shard(
            args.root, shard, args.endpoint, args.progress_seconds)
        source_verified[SHARD] = result
        directory, verified, visual = assemble_aria_snapshot(
            args.root, hf_directory, payload, rows, source_verified)
    else:
        directory = hf_directory
        verified = [source_verified[row["name"]] for row in rows]
        visual = verify_visual_inventory(directory)
    receipt = {**inventory, "files": verified, "snapshot": str(directory),
               "verified_at": now(), "all_official_hashes_verified": True,
               "visual_key_count": len(visual), "visual_keys": visual,
               "model_tensors_loaded": False, "first_shard_requested": False,
               "receipt_version": 1, **transfer_details}
    atomic_json(control / RECEIPT, receipt)
    emit("complete", receipt=str(control / RECEIPT), snapshot=str(directory),
         bytes=sum(row["bytes"] for row in rows), visual_key_count=len(visual),
         all_official_hashes_verified=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Avoid echoing request URLs/headers or possible tokens from dependency errors.
        message = re.sub(r"https?://\S+", "[request URL omitted]", str(error))
        message = re.sub(r"hf_[A-Za-z0-9]+", "[token omitted]", message)
        emit("failed", error_type=type(error).__name__, message=message[:1200])
        raise SystemExit(1)
