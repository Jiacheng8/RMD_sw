"""Reference extension blocks (``reference_extension:``), e.g. the GenEval block.

The contract: a store opened with an extension reads every base row exactly as before, and the
block's rows follow at ``num_rows..`` with their own prompts (text, context) and groups, so
prompt-grouped windows tile base and block alike.
"""
import importlib.util
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rdm.sw_lmmd import ReferenceFeatureStore, SlidingWindowSchedule, build_grouped_row_order
from rdm.sw_lmmd.reference_store import write_reference_store
from sw_lmmd_fixtures import CTX_DIM, D_TXT, DIMS, NAMES, build_trainer

G = 4
BASE_PROMPTS, EXT_PROMPTS, EXT_ROWS = 8, 3, 12          # block: prompts 0,0,0,0,1,1,1,1,2,...


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("ext")
    pid = np.repeat(np.arange(BASE_PROMPTS), G)
    base = write_reference_store(
        str(root / "reference_store"), prompt_ids=pid, row_order=np.arange(pid.size),
        text_features=rng.standard_normal((BASE_PROMPTS, D_TXT)).astype(np.float32),
        context=rng.standard_normal((BASE_PROMPTS, CTX_DIM)).astype(np.float32),
        encoder_features={n: rng.standard_normal((pid.size, d)).astype(np.float32)
                          for n, d in zip(NAMES, DIMS)},
        bandwidths={n: {"sigma": 2.0, "beta": 1.0} for n in NAMES})

    block = root / "geneval_block"
    (block / "encoder_features").mkdir(parents=True)
    for n, d in zip(NAMES, DIMS):                          # row r's features are all == 100 + r
        np.save(block / "encoder_features" / f"{n}.npy",
                (100 + np.arange(EXT_ROWS))[:, None].repeat(d, 1).astype(np.float16))
    np.save(block / "text_features.npy",
            (50 + np.arange(EXT_PROMPTS))[:, None].repeat(D_TXT, 1).astype(np.float32))
    np.save(block / "qwen_context.npy",
            (70 + np.arange(EXT_PROMPTS))[:, None].repeat(CTX_DIM, 1).astype(np.float32))
    np.save(block / "prompt_ids.npy", np.arange(EXT_ROWS) // G)
    np.save(block / "group_ids.npy", np.arange(EXT_ROWS) // G)
    (block / "metadata.json").write_text(json.dumps({"kind": "geneval_block",
                                                     "num_rows": EXT_ROWS}))
    return base, str(block)


def test_base_rows_are_unchanged_and_block_rows_follow(stores):
    base_root, block = stores
    base = ReferenceFeatureStore(base_root, NAMES)
    both = ReferenceFeatureStore(base_root, NAMES, extension=block)
    n_base = base.num_rows
    assert both.num_rows == n_base + EXT_ROWS
    assert both.extension["first_row"] == n_base and both.extension["rows"] == EXT_ROWS

    rows = np.arange(n_base)
    for n in NAMES:
        torch.testing.assert_close(both.reference_joint_features(n, rows),
                                   base.reference_joint_features(n, rows))
    torch.testing.assert_close(both.generator_context(rows), base.generator_context(rows))

    ext = np.array([n_base + 5, n_base, n_base + 11])      # any order, mixed with base rows
    feats = both.reference_image_features(NAMES[0], ext)
    assert feats[:, 0].tolist() == [105.0, 100.0, 111.0]
    assert both.prompt_rows(ext).tolist() == [BASE_PROMPTS + 1, BASE_PROMPTS, BASE_PROMPTS + 2]
    assert both.text_features(ext)[:, 0].tolist() == [51.0, 50.0, 52.0]
    assert both.generator_context(ext)[:, 0].tolist() == [71.0, 70.0, 72.0]
    mixed = np.array([3, n_base + 4])
    torch.testing.assert_close(both.reference_image_features(NAMES[1], mixed)[0],
                               base.reference_image_features(NAMES[1], mixed[:1])[0])


def test_groups_tile_base_and_block(stores):
    base_root, block = stores
    both = ReferenceFeatureStore(base_root, NAMES, extension=block)
    gid = both.grouping_ids()
    assert gid.shape[0] == both.num_rows
    assert len(set(gid[:BASE_PROMPTS * G]) & set(gid[BASE_PROMPTS * G:])) == 0
    order, group = build_grouped_row_order(gid, seed=1)
    assert group == G and np.array_equal(np.sort(order), np.arange(both.num_rows))
    sched = SlidingWindowSchedule(order, window_size=8, stride=4, group_size=G)
    for _ in range(10):
        w = sched.next()
        blocks = gid[w.all_ids].reshape(-1, G)
        assert (blocks == blocks[:, :1]).all()


def test_row_order_is_rebuilt_for_the_combined_rows(stores):
    base_root, block = stores
    both = ReferenceFeatureStore(base_root, NAMES, extension=block)
    order = both.row_order(seed=3)
    assert np.array_equal(np.sort(order), np.arange(both.num_rows))
    with pytest.raises(FileNotFoundError):
        both.row_order()                                     # base row_order.npy is too short


def test_extension_must_cover_every_encoder(stores, tmp_path):
    base_root, block = stores
    import shutil
    partial = tmp_path / "partial"
    shutil.copytree(block, partial)
    os.remove(partial / "encoder_features" / f"{NAMES[1]}.npy")
    with pytest.raises(FileNotFoundError, match=NAMES[1]):
        ReferenceFeatureStore(base_root, NAMES, extension=str(partial))


def test_relative_extension_resolves_next_to_the_store(stores):
    from rdm.sw_lmmd.launch import resolve_reference_extension
    base_root, block = stores
    cfg = SimpleNamespace(reference_root=base_root, reference_extension="geneval_block")
    assert resolve_reference_extension(cfg) == os.path.abspath(block)
    assert resolve_reference_extension(SimpleNamespace(reference_root=base_root)) is None
    with pytest.raises(FileNotFoundError):
        resolve_reference_extension(SimpleNamespace(reference_root=base_root,
                                                    reference_extension="missing"))


def test_grouped_trainer_trains_on_the_combined_reference(stores):
    base_root, block = stores
    store = ReferenceFeatureStore(base_root, NAMES, extension=block)
    trainer = build_trainer(base_root, window=8, stride=4, micro_batch=4, lr=1e-2)
    trainer.store = store                                    # the extended view
    order, group = build_grouped_row_order(store.grouping_ids(), seed=0)
    trainer.schedule = SlidingWindowSchedule(order, window_size=8, stride=4, group_size=group)
    trainer.bootstrap()
    seen = set()
    for _ in range(store.num_rows // 4):                     # one full lap
        logs = trainer.step()
        seen |= set(trainer.cache.entries[NAMES[0]].row_ids[-4:].tolist())
        assert np.isfinite(logs["force"])
    assert seen == set(range(store.num_rows))                # block rows were trained on too


def test_block_builder_keeps_only_correct_renders(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "gb", os.path.join(os.path.dirname(__file__), "..", "scripts", "_geneval_block_build.py"))
    gb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gb)
    res = tmp_path / "shard_0_results.jsonl"
    with open(res, "w") as f:
        for p, s, ok in [(3, 7, True), (3, 2, True), (3, 5, False), (12, 0, True)]:
            f.write(json.dumps({"filename": f"/x/shard_0/{p:05d}/samples/{s:04d}.png",
                                "tag": "counting", "correct": ok}) + "\n")
    got = gb.correct_renders([str(res)])
    assert sorted(got) == [3, 12]
    assert [s for s, _, _ in got[3]] == [2, 7]               # sorted by seed, the wrong one dropped
