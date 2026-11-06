"""Canonical, row-addressed RGB PNGs inside retained uncompressed teacher tars.

The index stores offsets and provenance, never decoded images/spatial features.
``read_uint8`` preserves request order. Float conversion happens on the requested
device so real/fake PNG rounding can use the identical device arithmetic.
"""
from __future__ import annotations

import hashlib
import io
import json
import operator
import os
import sqlite3
import tarfile
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

DATASET = "jiachengcui888/zimage-turbo-reference"
REVISION = "3195cc9e58d83dfe17a6b5d89f9886121872586f"
FORMAT = "rdm-reference-images-v1"
TEACHER = "Tongyi-MAI/Z-Image-Turbo"
NUM_ROWS = 125808
TOTAL_TAR_BYTES = 173074565120


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w") as output:
        json.dump(value, output, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def tar_stat(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino, "device": stat.st_dev}


def decode_rgb_png(png):
    from PIL import Image
    with Image.open(io.BytesIO(png)) as image:
        if image.format != "PNG" or image.size != (1024, 1024) or image.mode != "RGB":
            raise ValueError("teacher sample must be an RGB 8-bit 1024x1024 PNG")
        image.load()
        return image.copy()


def validate_teacher_sample(meta, png, prompts, row_for_prompt, render):
    """Match all identity/render fields; writer attribution is recorded separately."""
    pid = meta.get("prompt_id")
    if type(pid) is not int or pid not in row_for_prompt:
        raise ValueError("unknown/invalid teacher prompt_id")
    expected = prompts[pid]
    for key in ("image_id", "prompt", "source", "lang"):
        if key in expected and meta.get(key) != expected[key]:
            raise ValueError(f"prompt{pid}: teacher {key} mismatch")
    seed = int(canonical_digest([expected["image_id"], 0, render["seed"]])[:15], 16)
    contract = {"model": TEACHER, "height": 1024, "width": 1024, "steps": render["steps"],
                "guidance_scale": render["guidance_scale"], "seed": seed}
    for key, value in contract.items():
        if meta.get(key) != value:
            raise ValueError(f"prompt{pid}: teacher {key} render mismatch")
    decode_rgb_png(png)  # Full decode, not merely an IHDR/size check.
    return {"row_id": row_for_prompt[pid], "prompt_id": pid,
            "image_id": expected["image_id"], "seed": seed,
            "prompt_sha256": hashlib.sha256(expected["prompt"].encode()).hexdigest(),
            "png_sha256": hashlib.sha256(png).hexdigest(), "teacher_metadata": meta,
            "auxiliary_writer": {"base": expected.get("writer"), "teacher": meta.get("writer"),
                                 "different": expected.get("writer") != meta.get("writer")}}


class ReferenceImageIndexBuilder:
    """One atomic SQLite transaction per verified tar; safe to resume after a crash."""

    def __init__(self, directory, contract, num_rows=NUM_ROWS):
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError("index directory cannot be a symlink")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "index.sqlite"
        if self.path.is_symlink():
            raise ValueError("index cannot be a symlink")
        self.num_rows, self.contract = int(num_rows), contract
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tars (
                name TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL, stat_json TEXT NOT NULL, committed_epoch REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS images (
                row_id INTEGER PRIMARY KEY, prompt_id INTEGER UNIQUE NOT NULL,
                image_id TEXT UNIQUE NOT NULL, tar_name TEXT NOT NULL REFERENCES tars(name),
                png_name TEXT NOT NULL, data_offset INTEGER NOT NULL, bytes INTEGER NOT NULL,
                png_sha256 TEXT NOT NULL, prompt_sha256 TEXT NOT NULL, seed INTEGER NOT NULL,
                json_name TEXT NOT NULL, json_offset INTEGER NOT NULL, json_bytes INTEGER NOT NULL,
                json_sha256 TEXT NOT NULL, teacher_metadata TEXT NOT NULL,
                auxiliary_writer TEXT NOT NULL, UNIQUE(tar_name,png_name));
        """)
        expected = json.dumps({"format": FORMAT, "num_rows": self.num_rows, "contract": contract},
                              ensure_ascii=False, sort_keys=True)
        previous = self.db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if previous is not None and previous[0] != expected:
            self.close()
            raise ValueError("canonical image contract changed; preserve existing partial index")
        if previous is None:
            self.db.execute("INSERT INTO metadata VALUES ('contract',?)", (expected,))
            self.db.commit()

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    def committed(self):
        return {row["name"]: dict(row) for row in self.db.execute("SELECT * FROM tars ORDER BY name")}

    def receipt(self, name):
        tar = self.db.execute("SELECT * FROM tars WHERE name=?", (name,)).fetchone()
        if tar is None:
            return None
        rows = list(self.db.execute("SELECT row_id,png_sha256,auxiliary_writer FROM images "
                                    "WHERE tar_name=? ORDER BY row_id", (name,)))
        return {"name": name, "path": tar["path"], "bytes": tar["bytes"], "sha256": tar["sha256"],
                "stat": json.loads(tar["stat_json"]), "rows": len(rows),
                "row_ids": [row["row_id"] for row in rows],
                "row_png_digest": canonical_digest([(row["row_id"], row["png_sha256"]) for row in rows]),
                "auxiliary_writer_differences": sum(json.loads(row["auxiliary_writer"])["different"] for row in rows),
                "committed_epoch": tar["committed_epoch"]}

    def verify_committed(self, shard):
        receipt = self.receipt(shard["name"])
        if receipt is None:
            return None
        if (receipt["bytes"], receipt["sha256"]) != (shard["bytes"], shard["sha256"]) \
                or tar_stat(receipt["path"]) != receipt["stat"]:
            raise ValueError("committed teacher tar changed; preserve index and investigate")
        return receipt

    def index_tar(self, path, shard, prompts, row_for_prompt, render, max_pending=16384):
        existing = self.verify_committed(shard)
        if existing is not None:
            return existing
        path = Path(path)
        if path.is_symlink() or path.stat().st_size != shard["bytes"] \
                or file_sha256(path) != shard["sha256"]:
            raise ValueError("teacher tar failed official byte-size/SHA256 verification")
        original_stat, pending, member_names = tar_stat(path), {}, set()
        try:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("INSERT INTO tars VALUES (?,?,?,?,?,?)", (shard["name"], str(path.resolve()),
                            shard["bytes"], shard["sha256"], json.dumps(original_stat), time.time()))
            count = 0
            # r: intentionally rejects compressed tars: offsets must address raw PNG bytes.
            with tarfile.open(path, "r:") as archive:
                for member in archive:
                    name = PurePosixPath(member.name)
                    if name.is_absolute() or ".." in name.parts or "\\" in member.name:
                        raise ValueError("unsafe teacher tar member")
                    if name.suffix not in (".png", ".json"):
                        continue
                    if not member.isfile() or member.name in member_names \
                            or not 0 < member.size <= 32 * 1024**2:
                        raise ValueError("duplicate/nonregular/oversize teacher PNG/JSON member")
                    member_names.add(member.name)
                    key, extension = str(name.with_suffix("")), name.suffix
                    pair = pending.setdefault(key, {})
                    pair[extension] = member
                    if len(pending) > max_pending:
                        raise ValueError("unpaired teacher member count exceeded bounded index limit")
                    if ".png" not in pair or ".json" not in pair:
                        continue
                    png_member, json_member = pair[".png"], pair[".json"]
                    png = archive.extractfile(png_member).read()
                    raw_json = archive.extractfile(json_member).read()
                    if len(png) != png_member.size or len(raw_json) != json_member.size:
                        raise ValueError("truncated teacher tar member")
                    meta = json.loads(raw_json)
                    proof = validate_teacher_sample(meta, png, prompts, row_for_prompt, render)
                    if not 0 <= proof["row_id"] < self.num_rows:
                        raise ValueError("teacher row outside canonical range")
                    self.db.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                        proof["row_id"], proof["prompt_id"], proof["image_id"], shard["name"],
                        png_member.name, png_member.offset_data, png_member.size, proof["png_sha256"],
                        proof["prompt_sha256"], proof["seed"], json_member.name, json_member.offset_data,
                        json_member.size, hashlib.sha256(raw_json).hexdigest(),
                        json.dumps(meta, ensure_ascii=False, sort_keys=True),
                        json.dumps(proof["auxiliary_writer"], ensure_ascii=False, sort_keys=True)))
                    del pending[key]
                    count += 1
            if pending or count == 0:
                raise ValueError("teacher tar has unpaired or no canonical images")
            if tar_stat(path) != original_stat:
                raise ValueError("teacher tar changed while indexing")
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        receipt = self.receipt(shard["name"])
        atomic_json(self.directory / "receipts" / (Path(shard["name"]).stem + ".json"), receipt)
        return receipt

    def progress(self):
        return {"completed_shards": self.db.execute("SELECT count(*) FROM tars").fetchone()[0],
                "completed_rows": self.db.execute("SELECT count(*) FROM images").fetchone()[0],
                "total_rows": self.num_rows, "manifest_complete": False}

    def finalize(self, shards):
        committed = self.committed()
        if set(committed) != {shard["name"] for shard in shards}:
            raise ValueError("not all official teacher tars have been committed")
        count, low, high = self.db.execute("SELECT count(*),min(row_id),max(row_id) FROM images").fetchone()
        if count != self.num_rows or low != 0 or high != self.num_rows - 1:
            raise ValueError("canonical real-image row coverage is incomplete")
        receipts = [self.verify_committed(shard) for shard in shards]
        if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("real-image index integrity check failed")
        return self._publish_manifest(receipts, complete=True)

    def publish_partial(self):
        """Explicit pilot receipt; it never creates manifest.json or complete=True."""
        names = sorted(self.committed())
        if not names:
            raise ValueError("cannot publish an empty partial real-image index")
        receipts = [self.receipt(name) for name in names]
        return self._publish_manifest(receipts, complete=False)

    def _publish_manifest(self, receipts, complete):
        available_ids = [row[0] for row in self.db.execute("SELECT row_id FROM images ORDER BY row_id")]
        index_path = self.path
        if not complete:
            # A small immutable index snapshot keeps a pilot readable while the
            # main index resumes further tar commits in the background.
            temporary = self.directory / ("index.partial-" + uuid.uuid4().hex + ".sqlite")
            snapshot = sqlite3.connect(temporary)
            try:
                self.db.backup(snapshot)
            finally:
                snapshot.close()
            checksum = file_sha256(temporary)
            index_path = self.directory / ("index.partial-" + checksum + ".sqlite")
            if index_path.exists():
                if file_sha256(index_path) != checksum:
                    raise ValueError("existing pilot index snapshot has changed")
                temporary.unlink()
            else:
                os.rename(temporary, index_path)
        self.close()
        manifest = {"format": FORMAT, "complete": complete, "num_rows": self.num_rows,
                    "available_rows": len(available_ids),
                    "dataset": DATASET, "revision": REVISION, "image_size": [1024, 1024],
                    "storage": "retained_uncompressed_tars", "pixel_storage": "uint8_RGB",
                    "float_conversion": "uint8.to(device).float()/255; no resize",
                    "contract": self.contract, "index": {"path": index_path.name,
                    "bytes": index_path.stat().st_size, "sha256": file_sha256(index_path)},
                    "tars": receipts, "completed_epoch": time.time()}
        if not complete:
            manifest["available_row_ids"] = available_ids
            manifest["pilot_only"] = True
        atomic_json(self.directory / ("manifest.json" if complete else "manifest.partial.json"), manifest)
        return manifest


class ReferenceImageStore:
    """Read-only canonical real images, with a bounded tar-FD cache and no image cache."""

    def __init__(self, manifest, *, verify_png=True, verify_index=True, max_open_tars=8,
                 allow_partial=False, expected_num_rows=NUM_ROWS):
        path = Path(manifest)
        self.manifest_path = path / "manifest.json" if path.is_dir() else path
        if path.is_dir() and allow_partial and not self.manifest_path.exists():
            self.manifest_path = path / "manifest.partial.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        self.is_complete = self.manifest.get("complete") is True
        if self.manifest.get("format") != FORMAT or not self.is_complete and not allow_partial \
                or self.manifest.get("revision") != REVISION or self.manifest.get("dataset") != DATASET:
            raise ValueError("a complete canonical real-image manifest is required")
        self.num_rows = int(self.manifest["num_rows"])
        if self.num_rows != expected_num_rows:
            raise ValueError("real-image manifest does not describe the expected full canonical row count")
        self.available_rows = int(self.manifest.get("available_rows", self.num_rows))
        if self.is_complete and self.available_rows != self.num_rows:
            raise ValueError("complete real-image manifest has incomplete row coverage")
        if not self.is_complete and (self.manifest.get("complete") is not False
                                    or self.manifest.get("pilot_only") is not True):
            raise ValueError("partial real-image access requires an explicit pilot manifest")
        self.available_row_ids = range(self.num_rows) if self.is_complete else tuple(self.manifest["available_row_ids"])
        self._allowed_ids = None if self.is_complete else set(self.available_row_ids)
        if not self.is_complete and (self.available_rows != len(self._allowed_ids)
                or len(self.available_row_ids) != len(self._allowed_ids)
                or any(type(index) is not int or not 0 <= index < self.num_rows for index in self.available_row_ids)):
            raise ValueError("invalid explicit partial-image row coverage")
        self.index = self.manifest_path.parent / self.manifest["index"]["path"]
        if self.index.stat().st_size != self.manifest["index"]["bytes"] \
                or verify_index and file_sha256(self.index) != self.manifest["index"]["sha256"]:
            raise ValueError("canonical real-image index size/hash mismatch")
        self.tars = {row["name"]: row for row in self.manifest["tars"]}
        self.verify_png = bool(verify_png)
        self.max_open_tars = operator.index(max_open_tars)
        if self.max_open_tars < 1:
            raise ValueError("max_open_tars must be positive")
        self._descriptors, self._pid, self._lock = OrderedDict(), os.getpid(), threading.RLock()
        with self._connect() as db:
            if db.execute("SELECT count(*) FROM images").fetchone()[0] != self.available_rows \
                    or {row[0] for row in db.execute("SELECT name FROM tars")} != set(self.tars):
                raise ValueError("real-image manifest/index coverage mismatch")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.index.resolve().as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    def records(self, row_ids):
        ids = []
        for value in row_ids:
            if isinstance(value, bool):
                raise IndexError("boolean is not a canonical row id")
            index = operator.index(value)
            if not 0 <= index < self.num_rows:
                raise IndexError("canonical real-image row id is out of range")
            if self._allowed_ids is not None and index not in self._allowed_ids:
                raise KeyError(f"row {index} is not committed in this explicit partial image index")
            ids.append(index)
        found = {}
        with self._connect() as db:
            unique = list(dict.fromkeys(ids))
            for start in range(0, len(unique), 500):
                chunk = unique[start:start + 500]
                query = "SELECT * FROM images WHERE row_id IN (" + ",".join("?" for _ in chunk) + ")"
                found.update({row["row_id"]: dict(row) for row in db.execute(query, chunk)})
        if len(found) != len(set(ids)):
            raise ValueError("requested canonical row is absent from the completed index")
        return [found[index] for index in ids]

    def _read(self, record):
        with self._lock:
            if self._pid != os.getpid():
                self.close()
                self._pid = os.getpid()
            name = record["tar_name"]
            receipt = self.tars[name]
            if name not in self._descriptors:
                path = Path(receipt["path"])
                if path.is_symlink() or tar_stat(path) != receipt["stat"]:
                    raise ValueError("retained teacher tar changed after manifest verification")
                descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                if os.fstat(descriptor).st_size != receipt["bytes"]:
                    os.close(descriptor)
                    raise ValueError("teacher tar size changed during open")
                self._descriptors[name] = descriptor
                while len(self._descriptors) > self.max_open_tars:
                    _, old = self._descriptors.popitem(last=False)
                    os.close(old)
            descriptor = self._descriptors.pop(name)
            self._descriptors[name] = descriptor
            offset, size = record["data_offset"], record["bytes"]
            if offset < 0 or size <= 0 or offset + size > receipt["bytes"]:
                raise ValueError("invalid PNG offset/size in canonical index")
            png = os.pread(descriptor, size, offset)
        if len(png) != size or self.verify_png and hashlib.sha256(png).hexdigest() != record["png_sha256"]:
            raise ValueError("canonical PNG bytes changed or are truncated")
        return png

    def read_pngs(self, row_ids):
        return [self._read(record) for record in self.records(row_ids)]

    def read_uint8(self, row_ids):
        import numpy as np
        import torch
        images = [torch.from_numpy(np.asarray(decode_rgb_png(png), dtype=np.uint8).copy()).permute(2, 0, 1)
                  for png in self.read_pngs(row_ids)]
        return torch.stack(images) if images else torch.empty((0, 3, 1024, 1024), dtype=torch.uint8)

    def tensor(self, row_ids, *, device="cpu", dtype=None):
        import torch
        if dtype is not None and dtype != torch.float32:
            raise ValueError("real-image integer conversion must use float32")
        return self.read_uint8(row_ids).to(device=device).to(torch.float32).div(255)

    def close(self):
        for descriptor in self._descriptors.values():
            os.close(descriptor)
        self._descriptors.clear()

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key not in ("_descriptors", "_lock")}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._descriptors, self._pid, self._lock = OrderedDict(), os.getpid(), threading.RLock()
