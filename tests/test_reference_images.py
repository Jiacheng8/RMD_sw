"""Tiny retained-tar fixtures; no model downloads, feature cache or GPU required."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sqlite3
import tarfile
from pathlib import Path

import pytest
import torch
from PIL import Image

# The reader/builder has no package-init dependency. Direct loading also lets
# these CPU fixtures run on the Mac's Python3.9 while the repository requires3.10.
_spec = importlib.util.spec_from_file_location("reference_images_under_test",
    Path(__file__).parents[1] / "rdm/sw_lmmd/reference_images.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
TEACHER = _module.TEACHER
ReferenceImageIndexBuilder = _module.ReferenceImageIndexBuilder
ReferenceImageStore = _module.ReferenceImageStore
canonical_digest, file_sha256 = _module.canonical_digest, _module.file_sha256


def fixtures():
    prompts = [{"image_id": "image_a", "prompt": "red object", "source": "toy", "lang": "en", "writer": "old"},
               {"image_id": "image_b", "prompt": "blue object", "source": "toy", "lang": "en", "writer": "old"}]
    render = {"seed": 3407, "steps": 9, "guidance_scale": 0.0}
    return prompts, render, {0: 1, 1: 0}


def make_tar(path, pids, *, bad=None, png_size=1024):
    prompts, render, _ = fixtures()
    with tarfile.open(path, "w") as archive:
        for pid in pids:
            prompt = prompts[pid]
            color = (7, 63, 255) if pid == 0 else (191, 29, 127)
            out = io.BytesIO()
            Image.new("RGB", (png_size, png_size), color).save(out, format="PNG")
            meta = {**prompt, "prompt_id": pid, "model": TEACHER, "height": 1024, "width": 1024,
                    "steps": 9, "guidance_scale": 0.0,
                    "seed": int(canonical_digest([prompt["image_id"], 0, 3407])[:15], 16),
                    "writer": "new"}
            if bad:
                meta.update(bad)
            # JSON preceding PNG exercises pairing independent of arrival order.
            for suffix, data in (("json", json.dumps(meta).encode()), ("png", out.getvalue())):
                member = tarfile.TarInfo(prompt["image_id"] + "." + suffix)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    return {"name": "images/" + path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def build_complete(tmp_path):
    path = tmp_path / "shard-00000.tar"
    shard = make_tar(path, [1, 0])
    prompts, render, mapping = fixtures()
    builder = ReferenceImageIndexBuilder(tmp_path / "index", {"toy": True}, num_rows=2)
    receipt = builder.index_tar(path, shard, prompts, mapping, render)
    manifest = builder.finalize([shard])
    return path, shard, receipt, manifest, tmp_path / "index"


def test_canonical_order_duplicate_requests_uint8_and_device_conversion(tmp_path):
    path, _, receipt, manifest, directory = build_complete(tmp_path)
    assert receipt["row_ids"] == [0, 1]
    assert receipt["auxiliary_writer_differences"] == 2
    assert manifest["complete"] and path.exists()
    assert not list(tmp_path.rglob("*.png"))
    store = ReferenceImageStore(directory, expected_num_rows=2)
    rows = [1, 0, 1]
    pixels = store.read_uint8(rows)
    assert pixels.shape == (3, 3, 1024, 1024) and pixels.dtype == torch.uint8
    assert pixels[:, :, 0, 0].tolist() == [[7, 63, 255], [191, 29, 127], [7, 63, 255]]
    torch.testing.assert_close(store.tensor(rows, device="cpu"), pixels.to("cpu").float() / 255,
                               rtol=0, atol=0)
    assert [row["row_id"] for row in store.records(rows)] == rows
    assert json.loads(store.records([1])[0]["teacher_metadata"])["prompt_id"] == 0
    assert store.read_uint8([]).shape == (0, 3, 1024, 1024)
    for bad in ([2], [-1], [True]):
        with pytest.raises(IndexError):
            store.read_uint8(bad)
    with pytest.raises(ValueError, match="float32"):
        store.tensor([0], dtype=torch.float16)
    store.close()


def test_tar_transaction_resume_and_bounded_file_descriptors(tmp_path):
    prompts, render, mapping = fixtures()
    one, two = tmp_path / "shard-00000.tar", tmp_path / "shard-00001.tar"
    shards = [make_tar(one, [0]), make_tar(two, [1])]
    directory = tmp_path / "index"
    builder = ReferenceImageIndexBuilder(directory, {"toy": True}, num_rows=2)
    builder.index_tar(one, shards[0], prompts, mapping, render)
    with pytest.raises(ValueError, match="not all"):
        builder.finalize(shards)
    assert not (directory / "manifest.json").exists()
    builder.close()
    builder = ReferenceImageIndexBuilder(directory, {"toy": True}, num_rows=2)
    assert builder.verify_committed(shards[0])["rows"] == 1
    builder.index_tar(two, shards[1], prompts, mapping, render)
    builder.finalize(shards)
    store = ReferenceImageStore(directory, max_open_tars=1, expected_num_rows=2)
    store.read_uint8([1, 0, 1])
    assert len(store._descriptors) == 1
    store.close()


@pytest.mark.parametrize("bad", [{"prompt": "wrong"}, {"seed": 1}, {"steps": 8},
                                  {"image_id": "wrong"}, {"model": "wrong"}])
def test_identity_render_mismatch_rolls_back_entire_tar(tmp_path, bad):
    path = tmp_path / "shard-00000.tar"
    shard = make_tar(path, [0], bad=bad)
    prompts, render, mapping = fixtures()
    builder = ReferenceImageIndexBuilder(tmp_path / "index", {}, num_rows=2)
    with pytest.raises(ValueError, match="mismatch"):
        builder.index_tar(path, shard, prompts, mapping, render)
    assert builder.progress()["completed_rows"] == 0
    assert builder.progress()["completed_shards"] == 0
    builder.close()


def test_duplicate_canonical_rows_across_tars_rollback(tmp_path):
    prompts, render, mapping = fixtures()
    one, two = tmp_path / "shard-00000.tar", tmp_path / "shard-00001.tar"
    first, duplicate = make_tar(one, [0]), make_tar(two, [0])
    builder = ReferenceImageIndexBuilder(tmp_path / "index", {}, num_rows=2)
    builder.index_tar(one, first, prompts, mapping, render)
    with pytest.raises(sqlite3.IntegrityError):
        builder.index_tar(two, duplicate, prompts, mapping, render)
    assert builder.progress()["completed_rows"] == 1
    assert builder.progress()["completed_shards"] == 1
    builder.close()


def test_adm512_images_are_rejected(tmp_path):
    path = tmp_path / "shard-00000.tar"
    shard = make_tar(path, [0], png_size=512)
    prompts, render, mapping = fixtures()
    builder = ReferenceImageIndexBuilder(tmp_path / "index", {}, num_rows=2)
    with pytest.raises(ValueError, match="1024"):
        builder.index_tar(path, shard, prompts, mapping, render)
    assert builder.progress()["completed_rows"] == 0
    builder.close()


def test_cached_open_tar_still_checks_each_png_digest(tmp_path):
    path, _, _, _, directory = build_complete(tmp_path)
    store = ReferenceImageStore(directory, expected_num_rows=2)
    original = store.read_pngs([0])[0]
    record = store.records([0])[0]
    with path.open("r+b") as payload:
        payload.seek(record["data_offset"] + 100)
        payload.write(bytes([original[100] ^ 1]))
    with pytest.raises(ValueError, match="changed"):
        store.read_pngs([0])
    store.close()


def test_wrong_official_tar_hash_and_wrong_index_hash_rejected(tmp_path):
    path = tmp_path / "shard-00000.tar"
    shard = make_tar(path, [0])
    prompts, render, mapping = fixtures()
    builder = ReferenceImageIndexBuilder(tmp_path / "partial", {}, num_rows=2)
    with pytest.raises(ValueError, match="SHA256"):
        builder.index_tar(path, {**shard, "sha256": "0" * 64}, prompts, mapping, render)
    builder.close()
    complete = tmp_path / "complete"
    complete.mkdir()
    _, _, _, _, directory = build_complete(complete)
    with (directory / "index.sqlite").open("ab") as index:
        index.write(b"corrupt")
    with pytest.raises(ValueError, match="index size/hash"):
        ReferenceImageStore(directory, expected_num_rows=2)


def test_explicit_partial_pilot_only_reads_committed_rows(tmp_path):
    prompts, render, mapping = fixtures()
    path = tmp_path / "shard-00000.tar"
    shard = make_tar(path, [0])
    directory = tmp_path / "index"
    builder = ReferenceImageIndexBuilder(directory, {}, num_rows=2)
    builder.index_tar(path, shard, prompts, mapping, render)
    partial = builder.publish_partial()
    assert partial["complete"] is False and partial["pilot_only"]
    assert not (directory / "manifest.json").exists()
    with pytest.raises(ValueError, match="complete"):
        ReferenceImageStore(directory / "manifest.partial.json", expected_num_rows=2)
    with pytest.raises(FileNotFoundError):
        ReferenceImageStore(directory, expected_num_rows=2)
    store = ReferenceImageStore(directory, allow_partial=True, expected_num_rows=2)
    assert store.is_complete is False and store.available_row_ids == (1,)
    assert store.read_uint8([1]).shape == (1, 3, 1024, 1024)
    with pytest.raises(KeyError, match="not committed"):
        store.read_uint8([0])
    store.close()
    with pytest.raises(ValueError, match="full canonical row count"):
        ReferenceImageStore(directory, allow_partial=True)  # Default requires125808 total rows.
    # A full preparation may resume without invalidating the immutable pilot DB.
    second = tmp_path / "shard-00001.tar"
    second_shard = make_tar(second, [1])
    builder = ReferenceImageIndexBuilder(directory, {}, num_rows=2)
    builder.index_tar(second, second_shard, prompts, mapping, render)
    builder.finalize([shard, second_shard])
    old_pilot = ReferenceImageStore(directory / "manifest.partial.json", allow_partial=True, expected_num_rows=2)
    assert old_pilot.available_row_ids == (1,)
    assert old_pilot.read_uint8([1]).shape[0] == 1
    old_pilot.close()
