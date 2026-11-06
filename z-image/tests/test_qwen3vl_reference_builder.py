"""CPU-only reference-builder invariants; no network, real tensors or model loading."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[2] / 'ops/h20-qwen3vl-reference.py'
spec = importlib.util.spec_from_file_location('qwen3vl_reference_builder', SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
REVISION = 'a' * 40


def inventory():
    return {'sha': REVISION, 'siblings': [
        {'rfilename': f'images/shard-{i:05d}.tar', 'size': 9,
         'lfs': {'sha256': 'b' * 64}} for i in range(87)]}


class ReferenceBuilderTests(unittest.TestCase):
    def test_inventory_requires_pinned_complete_hashed_set(self):
        self.assertEqual(len(builder.official_inventory(inventory(), REVISION)), 87)
        broken = inventory(); broken['siblings'].pop()
        with self.assertRaises(ValueError):
            builder.official_inventory(broken, REVISION)
        with self.assertRaises(ValueError):
            builder.official_inventory(inventory(), 'c' * 40)

    def test_real_writer_null_vs_claude_hand_is_auxiliary(self):
        # The first source row has writer=null plus a local origin audit label;
        # public teacher metadata reports claude-code/hand for the same image.
        prompt = {'image_id': 'fixture-image-0', 'prompt': '完整提示词', 'source': 'pilot',
                  'lang': 'zh', 'writer': None, 'origin': 'prompt_build/pilot_claude.jsonl:1'}
        render = {'steps': 9, 'guidance_scale': 0.0, 'seed': 3407}
        meta = {**prompt, 'writer': 'claude-code/hand', 'prompt_id': 0,
                'model': builder.TEACHER, 'height': 1024, 'width': 1024,
                'steps': 9, 'guidance_scale': 0.0,
                'seed': builder.seed_for(prompt['image_id'], 0, render['seed'])}
        buf = io.BytesIO(); Image.new('RGB', (1024, 1024)).save(buf, format='PNG')
        rid, image, proof = builder.check_sample(meta, buf.getvalue(), [prompt], {0: 11}, render)
        self.assertEqual((rid, image.size), (11, (1024, 1024)))
        self.assertTrue(proof['auxiliary_writer']['different'])
        meta['prompt'] += '!'
        with self.assertRaises(ValueError):
            builder.check_sample(meta, buf.getvalue(), [prompt], {0: 11}, render)

    def test_prefetch_never_loads_base_or_models_or_publishes_metadata(self):
        fake_hf = types.SimpleNamespace(HfApi=mock.Mock(return_value=types.SimpleNamespace(
            repo_info=mock.Mock(return_value=inventory()))))
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(builder.platform, 'system', return_value='Linux'), \
                mock.patch.dict('sys.modules', {'huggingface_hub': fake_hf}), \
                mock.patch.object(builder, 'load_base', side_effect=AssertionError('base accessed')), \
                mock.patch.object(builder, 'load_models', side_effect=AssertionError('models loaded')):
            root = Path(tmp); payload = root / 'fixture.tar'; payload.write_bytes(b'x')
            with mock.patch.object(builder, 'download_tar', return_value=payload) as download:
                builder.main(['--root', str(root), '--dataset-revision', REVISION,
                              '--prefetch-only', '--limit-shards', '1'])
            self.assertEqual(download.call_count, 1)
            work = root / 'zimage/zimage-ref-v1/reference_store_qwen3vl.prefetch'
            self.assertTrue((work / 'official-shards.json').is_file())
            self.assertFalse((work / 'metadata.json').exists())
            self.assertTrue(payload.exists())

    def test_aria2_command_disables_child_proxy_and_verifies_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = b'teacher tar fixture'
            shard = {'name': 'images/shard-00000.tar', 'bytes': len(data),
                     'sha256': hashlib.sha256(data).hexdigest()}
            observed = {}
            def fake_process(command, **kwargs):
                observed.update(command=command, env=kwargs['env'])
                folder = Path(next(a[6:] for a in command if a.startswith('--dir=')))
                (folder / 'shard-00000.tar').write_bytes(data)
                return types.SimpleNamespace(stdout=io.StringIO('https://signed.example/?token=secret\n'),
                                             wait=lambda: 0)
            with mock.patch.dict(os.environ, {'HTTP_PROXY': 'secret-proxy', 'all_proxy': 'secret'}), \
                    mock.patch.object(builder.subprocess, 'Popen', side_effect=fake_process):
                tar = builder.download_tar(shard, root / 'staging', REVISION, 'https://hf-mirror.com')
            self.assertEqual(tar.read_bytes(), data)
            for arg in ('-c', '-x16', '-s16', '-k4M', '--all-proxy=', '--http-proxy=', '--https-proxy='):
                self.assertIn(arg, observed['command'])
            self.assertEqual(observed['env']['NO_PROXY'], '*')
            self.assertNotIn('HTTP_PROXY', observed['env']); self.assertNotIn('all_proxy', observed['env'])
            self.assertNotIn('signed.example', tar.with_suffix('.download.log').read_text())
            self.assertTrue(tar.with_suffix('.verified.json').exists())

    def test_exit0_is_not_sha_success_and_keeps_bad_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); shard = {'name': 'images/shard-00000.tar', 'bytes': 2, 'sha256': 'b' * 64}
            def fake_process(command, **kwargs):
                folder = Path(next(a[6:] for a in command if a.startswith('--dir=')))
                (folder / 'shard-00000.tar').write_bytes(b'xx')
                return types.SimpleNamespace(stdout=io.StringIO(''), wait=lambda: 0)
            with mock.patch.object(builder.subprocess, 'Popen', side_effect=fake_process):
                with self.assertRaises(ValueError):
                    builder.download_tar(shard, root, REVISION, 'https://hf-mirror.com')
            tar = root / REVISION / shard['name']
            self.assertTrue(tar.exists()); self.assertFalse(tar.with_suffix('.verified.json').exists())
            self.assertEqual(json.loads(tar.with_suffix('.download-failed.json').read_text())['exit'], 0)

    def test_symlink_destination_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / 'source'; original.mkdir()
            link = Path(tmp) / 'target'; link.symlink_to(original, target_is_directory=True)
            with self.assertRaises(ValueError):
                builder.require_real_directory(link)
            self.assertEqual(list(original.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
