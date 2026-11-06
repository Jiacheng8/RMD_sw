#!/usr/bin/env python3
"""CPU-only, pinned stage-4 reference prefetch; no work happens on import.

Run only on H20 after sourcing the existing control/env.sh and setting
HF_HUB_DISABLE_XET=1. Payload and partial files stay below the remote zimage root.
This script never acquires the experiment lock or changes A/B control files.
"""

import datetime as dt
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import uuid


ROOT = Path("/share-evpfs/shaoshitong/rdm-sets")
BASE = ROOT / "zimage"
CONTROL = ROOT / "control"
REPO = "jiachengcui888/zimage-ref-v1"
REVISION = "94742caca158951278495ff3d9d35e7b3fc31994"
PREFIX = "zimage-ref-v1/"
PACK = BASE / "zimage-ref-v1"
STAGED = PACK / ".staged"
EXPECTED_FILES = 18
EXPECTED_BYTES = 1_625_900_176
METADATA = "zimage-ref-v1/reference_store/metadata.json"


class PrefetchError(Exception):
    """A controlled, credential-free error description."""


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(event, **fields):
    print(json.dumps({"utc": utc(), "event": event, **fields}, sort_keys=True), flush=True)


def atomic_write(path, data):
    """Publish a complete small record only after its contents are fsynced."""
    temp = path.with_name(path.name + ".tmp-" + str(os.getpid()) + "-" + uuid.uuid4().hex)
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        # Only our own small temporary record, never a model/Hub partial file.
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def child_path(relative):
    if not isinstance(relative, str) or "\\" in relative or "\0" in relative:
        raise PrefetchError("invalid_repository_path")
    raw = relative.split("/")
    pure = PurePosixPath(relative)
    if (pure.is_absolute() or not relative.startswith(PREFIX) or
            any(part in {"", ".", ".."} for part in raw)):
        raise PrefetchError("repository_path_outside_reference_prefix")
    candidate = BASE.joinpath(*raw)
    current = BASE
    for part in raw:
        current = current / part
        if current.is_symlink():
            raise PrefetchError("symlink_in_reference_payload_path")
    try:
        candidate.resolve().relative_to(BASE.resolve())
    except ValueError:
        raise PrefetchError("resolved_payload_path_escapes_zimage")
    return candidate


def ensure_layout():
    if not ROOT.is_dir() or not CONTROL.is_dir():
        raise PrefetchError("required_remote_root_or_control_directory_missing")
    if BASE.is_symlink():
        raise PrefetchError("zimage_root_must_not_be_a_symlink")
    BASE.mkdir(exist_ok=True)
    try:
        BASE.resolve().relative_to(ROOT.resolve())
    except ValueError:
        raise PrefetchError("zimage_root_escapes_remote_root")
    # snapshot_download(local_dir=...) also writes local Hub metadata here.
    current = BASE
    for part in [".cache", "huggingface"]:
        current = current / part
        if current.is_symlink():
            raise PrefetchError("symlink_in_local_hub_metadata_path")
    if current.is_dir():
        for existing in current.rglob("*"):
            if existing.is_symlink():
                raise PrefetchError("symlink_in_existing_local_hub_metadata_tree")
    if os.environ.get("HF_HUB_DISABLE_XET") != "1":
        raise PrefetchError("launcher_must_set_HF_HUB_DISABLE_XET_1")


def official_entries():
    proxy = os.environ.get("HTTP_PROXY")
    if proxy != "http://127.0.0.1:18791":
        raise PrefetchError("expected_task_streaming_proxy_not_configured")
    url = ("https://huggingface.co/api/datasets/" + REPO +
           "/revision/" + REVISION + "?blobs=true")
    try:
        result = subprocess.run(
            ["curl", "--silent", "--show-error", "--fail", "--location",
             "--proxy", proxy, "--connect-timeout", "10", "--max-time", "30",
             "--max-filesize", "1048576", "--write-out", "\n%{http_code}", url],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=35)
    except subprocess.TimeoutExpired:
        raise PrefetchError("official_api_request_timed_out")
    if result.returncode != 0:
        # curl stderr can contain URLs. Do not print it or an exception string.
        raise PrefetchError("official_api_request_failed_curl_exit_" + str(result.returncode))
    body, separator, code = result.stdout.rpartition("\n")
    if not separator or code.strip() != "200":
        raise PrefetchError("official_api_did_not_return_HTTP_200")
    try:
        response = json.loads(body)
    except ValueError:
        raise PrefetchError("official_api_response_is_not_JSON")
    if not isinstance(response, dict) or response.get("sha") != REVISION:
        raise PrefetchError("official_api_revision_mismatch")
    siblings = response.get("siblings")
    if not isinstance(siblings, list):
        raise PrefetchError("official_api_missing_file_inventory")
    entries, names = [], set()
    for item in siblings:
        if not isinstance(item, dict):
            raise PrefetchError("official_api_invalid_inventory_entry")
        name = item.get("rfilename")
        if not isinstance(name, str):
            raise PrefetchError("official_api_invalid_filename")
        if not name.startswith(PREFIX):
            continue
        child_path(name)
        if name in names:
            raise PrefetchError("official_api_duplicate_filename")
        names.add(name)
        size = item.get("size")
        if type(size) is not int or size < 0:
            raise PrefetchError("official_api_missing_file_size")
        lfs = item.get("lfs")
        if lfs is not None:
            if not isinstance(lfs, dict) or lfs.get("size") != size:
                raise PrefetchError("official_api_LFS_size_mismatch")
            digest = lfs.get("sha256") or lfs.get("oid")
            if isinstance(digest, str) and digest.startswith("sha256:"):
                digest = digest[len("sha256:"):]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise PrefetchError("official_api_missing_LFS_SHA256")
            algorithm = "sha256"
        else:
            digest = item.get("blobId")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{40}", digest):
                raise PrefetchError("official_api_missing_Git_blob_SHA1")
            algorithm = "git_blob_sha1"
        entries.append({"path": name, "size": size, "algorithm": algorithm, "digest": digest})
    if len(entries) != EXPECTED_FILES or sum(item["size"] for item in entries) != EXPECTED_BYTES:
        raise PrefetchError("official_api_reference_inventory_count_or_size_mismatch")
    if METADATA not in names:
        raise PrefetchError("official_api_reference_metadata_missing")
    return sorted(entries, key=lambda item: item["path"])


def verify_file(entry):
    path = child_path(entry["path"])
    if not path.is_file() or path.stat().st_size != entry["size"]:
        raise PrefetchError("downloaded_file_missing_or_wrong_size: " + entry["path"])
    digest = hashlib.sha256() if entry["algorithm"] == "sha256" else hashlib.sha1()
    if entry["algorithm"] == "git_blob_sha1":
        digest.update(("blob " + str(entry["size"])).encode() + b"\0")
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    if digest.hexdigest() != entry["digest"]:
        raise PrefetchError("downloaded_file_hash_mismatch: " + entry["path"])
    log("reference_file_verified", path=entry["path"], bytes=entry["size"],
        algorithm=entry["algorithm"])


def perform():
    ensure_layout()
    if STAGED.exists():
        raise PrefetchError("preexisting_staged_marker_requires_review")
    entries = official_entries()
    log("official_reference_inventory_verified", revision=REVISION,
        files=len(entries), bytes=sum(entry["size"] for entry in entries))
    # Import only inside the remote execution path; never access HF credentials.
    # Library retry/debug loggers can include signed URLs. Silence third-party
    # Python logging only in this helper process; our own UTC events use print.
    logging.disable(logging.CRITICAL)
    from huggingface_hub import snapshot_download
    log("reference_download_started", repo=REPO, revision=REVISION, max_workers=4)
    downloaded = snapshot_download(
        repo_id=REPO, repo_type="dataset", revision=REVISION,
        allow_patterns=[PREFIX + "*"], local_dir=str(BASE), max_workers=4,
        endpoint="https://huggingface.co", token=False)
    if Path(downloaded).resolve() != BASE.resolve():
        raise PrefetchError("snapshot_download_returned_unexpected_local_directory")
    log("reference_download_returned_verifying_files")
    for entry in entries:
        verify_file(entry)
    try:
        with child_path(METADATA).open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, ValueError):
        raise PrefetchError("reference_metadata_JSON_parse_failed")
    if not isinstance(metadata, dict):
        raise PrefetchError("reference_metadata_JSON_must_be_an_object")
    manifest = {"repo_id": REPO, "repo_type": "dataset", "revision": REVISION,
                "verified_utc": utc(), "files_count": len(entries),
                "total_bytes": sum(entry["size"] for entry in entries),
                "local_dir": str(BASE), "metadata_JSON_valid": True, "files": entries}
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    atomic_write(CONTROL / "reference-prefetch.manifest.json", payload)
    if STAGED.exists():
        raise PrefetchError("staged_marker_appeared_during_prefetch_requires_review")
    # All 18 payload files and the manifest are complete before the marker exists.
    atomic_write(STAGED, (REVISION + "\n").encode())
    log("reference_prefetch_complete", files=len(entries), bytes=EXPECTED_BYTES,
        revision=REVISION, manifest_sha256=hashlib.sha256(payload).hexdigest())


def interrupted(signum, unused):
    raise PrefetchError("reference_prefetch_interrupted_by_signal_" + str(signum))


def main():
    if not ROOT.is_dir() or not CONTROL.is_dir():
        log("reference_prefetch_error", reason="required_remote_root_or_control_directory_missing")
        return 1
    lock_fd = os.open(CONTROL / "reference-prefetch.lock",
                      os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(lock_fd, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("reference_prefetch_skipped", reason="another_reference_prefetch_holds_lock")
            return 3
        if (CONTROL / "reference-prefetch.exit").exists():
            log("reference_prefetch_skipped", reason="prior_exit_exists_no_automatic_restart")
            return 4
        code = 1
        try:
            atomic_write(CONTROL / "reference-prefetch.pid", (str(os.getpid()) + "\n").encode())
            atomic_write(CONTROL / "reference-prefetch.started", (utc() + "\n").encode())
            signal.signal(signal.SIGTERM, interrupted)
            signal.signal(signal.SIGINT, interrupted)
            perform()
            code = 0
        except PrefetchError as exc:
            log("reference_prefetch_error", reason=str(exc))
        except BaseException as exc:
            # HF/client errors may contain signed URLs. Never emit their text.
            log("reference_prefetch_error", reason="reference_prefetch_failed",
                error_type=type(exc).__name__)
        finally:
            atomic_write(CONTROL / "reference-prefetch.exit", (str(code) + "\n").encode())
            atomic_write(CONTROL / "reference-prefetch.finished", (utc() + "\n").encode())
        return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            raise
        log("reference_prefetch_error", reason="control_record_or_process_failure",
            error_type=type(exc).__name__)
        sys.exit(1)
