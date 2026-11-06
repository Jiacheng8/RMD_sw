"""Offline tests of coverage, immutable snapshots, atomic commits and row/context alignment."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import atomic_json, digest, load_prompts, prompt_dir, seed_for, selection
from prepare import collect, parse_shards


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build = self.root / "prompts"
        (self.build / "shards").mkdir(parents=True)
        with (self.build / "manifest.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "path", "source"])
            writer.writeheader()
            writer.writerows(dict(image_id=i, path=f"/missing/{i}.png", source="coco") for i in ("b", "a", "c"))
        (self.build / "shards" / "shard_0000.txt").write_text("a\nb\nc\n")
        self.input = self.root / "input.jsonl"

    def write(self, records, path=None):
        path = path or self.input
        path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records))
        return path

    def records(self):
        return [dict(image_id="a", prompt="雪地上的树。", lang="zh"),
                dict(image_id="b", prompt="A quiet room.", lang="en"),
                dict(image_id="c", skipped="unusable")]

    def command(self, script, *args, check=True):
        return subprocess.run([sys.executable, str(Path(__file__).parent / script), *map(str, args)],
                              capture_output=True, text=True, check=check)

    def prepare(self):
        self.write(self.records())
        self.out = self.root / "out"
        self.command("prepare.py", "--prompt-build", self.build, "--inputs", self.input,
                     "--out", self.out, "--shards", "0")
        return self.out

    def test_coverage_skips_dedup_and_manifest_order(self):
        self.write(self.records()+[self.records()[0]])
        rows, report = collect(self.build, [self.input], [0])
        self.assertEqual([r["image_id"] for r in rows], ["b", "a"])
        self.assertEqual(report["missing"], 0)
        self.assertEqual(report["skipped"], 1)

    def test_conflicting_prompt_rejected(self):
        self.write(self.records()+[dict(image_id="a", prompt="Changed", lang="en")])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            collect(self.build, [self.input], [0])

    def test_pending_prompts_fail_without_output(self):
        self.write(self.records()[:1])
        result = self.command("prepare.py", "--prompt-build", self.build, "--inputs", self.input,
                              "--out", self.root / "out", "--shards", "0", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "out").exists())

    def test_half_written_json_fails(self):
        self.input.write_text('{"image_id":"a",')
        with self.assertRaises(json.JSONDecodeError):
            collect(self.build, [self.input], [0])

    def test_snapshot_immutable_and_checksum(self):
        root = self.prepare()
        rows, _ = load_prompts(root)
        self.assertEqual(rows[0]["image_id"], "b")
        self.write([dict(image_id="a", prompt="Changed", lang="en"), *self.records()[1:]])
        result = self.command("prepare.py", "--prompt-build", self.build, "--inputs", self.input,
                              "--out", root, "--shards", "0", check=False)
        self.assertNotEqual(result.returncode, 0)
        with (root / "prompts.jsonl").open("a") as f:
            f.write('{}\n')
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_prompts(root)

    def test_atomic_commit_requires_all_images(self):
        from PIL import Image
        folder = prompt_dir(self.root, 0)
        folder.mkdir(parents=True)
        Image.new("RGB", (16,16)).save(folder / "k0.png")
        data = dict(prompt_id=0, images=[dict(rank=k, file=f"k{k}.png") for k in range(2)])
        atomic_json(folder / "selection.json", data)
        self.assertIsNone(selection(self.root, 0, 2))
        Image.new("RGB", (16,16)).save(folder / "k1.png")
        self.assertEqual(selection(self.root, 0, 2), data)
        (folder / "k1.png").write_bytes(b"corrupt")
        self.assertIsNone(selection(self.root, 0, 2))

    def test_launch_dry_run_without_models_or_writes(self):
        root = self.prepare()
        before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
        result = self.command("launch.py", "all", "--root", root, "--gpu-ids", "2,5", "--dry-run",
                              "--", "--candidates", "8", "--keep", "4")
        self.assertIn("CUDA_VISIBLE_DEVICES=2", result.stdout)
        self.assertIn("--rank 1 --world 2", result.stdout)
        self.assertIn("--candidates 8 --keep 4", result.stdout)
        self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob("*")))

    def test_seeds_and_shards_are_stable(self):
        self.assertEqual(parse_shards("0,2-4,3"), [0,2,3,4])
        seeds = [seed_for("image_a", i, 3407) for i in range(24)]
        self.assertEqual(len(set(seeds)), 24)
        self.assertNotEqual(seeds[0], seed_for("image_b", 0, 3407))
        for world in (1,2,5):
            ids = [p for rank in range(world) for p in range(rank, 17, world)]
            self.assertEqual(sorted(ids), list(range(17)))

    def test_reader_preserves_ragged_context_and_prompt_indirection(self):
        import numpy as np
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this RDM checkout
        from reference_store import ZImageReferenceStore
        root = self.prepare()
        store = root / "reference_store"
        (store / "encoder_features").mkdir(parents=True)
        np.save(store / "text_features.npy", np.eye(2, dtype=np.float32))
        np.save(store / "encoder_features" / "fake.npy", np.arange(8).reshape(4,2))
        np.save(store / "prompt_ids.npy", [0,0,1,1])
        atomic_json(store / "bandwidths.json", dict(fake=dict(sigma=1., beta=1.)))
        atomic_json(store / "metadata.json", dict(num_rows=4, prompt_hash="abc", encoder_feature_dims=dict(fake=2)))
        atomic_json(store / "z_context.json", dict(root="..", prompt_hash="abc"))
        for p, length in enumerate((3,7)):
            folder = prompt_dir(root, p)
            folder.mkdir(parents=True)
            torch.save(torch.full((length,4), float(p+1), dtype=torch.bfloat16), folder / "context.pt")
        reader = ZImageReferenceStore(store, ["fake"])
        contexts = reader.generator_context([3,0,2])
        self.assertEqual([c.shape[0] for c in contexts], [7,3,7])
        self.assertEqual(contexts[0].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(reader.text_features([3,0]), torch.tensor([[0.,1.],[1.,0.]])))

    def test_render_topk_resume_and_corruption_recovery(self):
        import types
        import torch
        from PIL import Image
        import render
        root = self.prepare()
        calls = []
        class FakePipe:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()
            def to(self, device):
                return self
            def set_progress_bar_config(self, **kwargs):
                pass
            def encode_prompt(self, prompts, **kwargs):
                return [torch.ones((3,4), dtype=torch.bfloat16)], []
            def __call__(self, **kwargs):
                seeds = [g.initial_seed() for g in kwargs["generator"]]
                calls.extend(seeds)
                return types.SimpleNamespace(images=[Image.new("RGB", (16,16), (s % 255,0,0)) for s in seeds])
        class FakeScorer:
            def __init__(self, *args):
                pass
            def encode_text(self, text):
                return torch.ones((1,2)), False
            def score(self, images, text_features):
                return [float(image.getpixel((0,0))[0]) for image in images]
        native_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if args and args[0] == "cuda":
                args = ("cpu", *args[1:])
            return native_to(tensor, *args, **kwargs)
        argv = ["render.py", "--root", str(root), "--candidates", "5", "--keep", "2", "--batch", "2"]
        with patch.dict(sys.modules, {"diffusers": types.SimpleNamespace(ZImagePipeline=FakePipe)}), \
             patch.object(render, "Scorer", FakeScorer), \
             patch.object(render, "resolve_model", return_value="/fake/snapshot"), \
             patch("importlib.metadata.version", return_value="test"), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "is_bf16_supported", return_value=True), \
             patch.object(torch.Tensor, "to", cpu_to), patch.object(sys, "argv", argv):
            render.main()
            self.assertEqual(len(calls), 10)
            committed = selection(root, 0, 2)
            self.assertEqual([r["score"] for r in committed["images"]],
                sorted([float(seed_for("b", j, 3407) % 255) for j in range(5)], reverse=True)[:2])
            render.main()
            self.assertEqual(len(calls), 10)  # complete prompts skipped
            (prompt_dir(root, 0) / "k1.png").write_bytes(b"broken")
            render.main()
            self.assertEqual(len(calls), 15)  # only the broken prompt regenerated
            self.assertEqual(selection(root, 0, 2), committed)


if __name__ == "__main__":
    unittest.main()
