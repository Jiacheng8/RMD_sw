#!/usr/bin/env python3
"""Pinned official QIB inventory; existing direct mirror transport, full source hashes."""
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time

import httpx
import huggingface_hub as hf

ROOT = Path('/data/rdm-workspace/rdm-sets')
QIB = ROOT/'qib'
CONTROL = ROOT/'control'
REPO = 'Qwen/Qwen-Image-Bench'
BENCH_REV = 'd2493deb153b020cf169c7e3f57d15e4dd697038'
JUDGE_REV = '1b77ff83564ac4e4e8140769eacff7ff73f35f3c'


def write(name, data):
    path = CONTROL/name
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def verify(path, entry):
    assert path.stat().st_size == entry.size, str(path)
    digest = hashlib.sha256() if entry.lfs else hashlib.sha1()
    if not entry.lfs:
        digest.update(f'blob {entry.size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    expected = entry.lfs.sha256 if entry.lfs else entry.blob_id
    assert digest.hexdigest() == expected, 'Official hash mismatch: '+str(path)
    return dict(path=str(path), bytes=entry.size, digest=expected,
                algorithm='sha256' if entry.lfs else 'git_blob_sha1')


def main():
    QIB.mkdir(exist_ok=True)
    hf.set_client_factory(lambda: httpx.Client(trust_env=True, follow_redirects=True, timeout=30))
    api = hf.HfApi(endpoint='https://huggingface.co')
    inventories = []
    for kind, revision, patterns, folder in [
        ('dataset', BENCH_REV, ['*.py', 'metadata/*', 'qwen_image_bench_hf_v0518.jsonl', 'requirements.txt'], 'bench'),
        ('model', JUDGE_REV, ['*.safetensors', '*.json', '*.txt', '*.model', '*.jinja'], 'judge')]:
        info = api.repo_info(REPO, repo_type=kind, revision=revision, files_metadata=True)
        assert info.sha == revision
        files = [f for f in info.siblings if any(fnmatch.fnmatch(f.rfilename, p) for p in patterns)]
        assert files
        for entry in files:
            name = PurePosixPath(entry.rfilename)
            assert not name.is_absolute() and '..' not in name.parts
            # Official Python package markers may be valid zero-byte Git blobs.
            assert isinstance(entry.size, int) and entry.size >= 0
        inventories.append((kind, revision, files, folder))
    hf.set_client_factory(lambda: httpx.Client(trust_env=False, follow_redirects=True, timeout=120))
    aria = shutil.which('aria2c')
    assert aria, 'Existing aria2c is required for parallel ranged model download'
    for kind, revision, files, folder in inventories:
        destination = QIB/folder
        destination.mkdir(exist_ok=True)
        required = 0
        for f in files:
            p = destination/f.rfilename
            partial = p.with_name(p.name+'.download')
            allocated = partial.stat().st_blocks*512 if partial.exists() else 0
            if not p.exists():
                required += max(0, f.size-allocated)
        assert shutil.disk_usage(ROOT).free > required+8*1024**3, 'Insufficient remote space including environment/output margin'
        runtime = CONTROL/'qib350-bench-runtime.json'
        if folder == 'bench' and runtime.exists():
            saved = json.loads(runtime.read_text())
            assert saved['passed'] and saved['official_revision'] == revision
            backend = destination/'backends/ms_swift_backend.py'
            assert hashlib.sha256(backend.read_bytes()).hexdigest() == saved['patched_backend_sha256']
            for f in files:
                p = destination/f.rfilename
                if f.rfilename == 'backends/ms_swift_backend.py':
                    p = p.with_suffix('.py.official-original')
                verify(p, f)
            continue
        small = [f.rfilename for f in files if not f.rfilename.endswith('.safetensors')]
        hf.snapshot_download(REPO, repo_type=kind, revision=revision, allow_patterns=small,
            local_dir=destination, max_workers=8, endpoint='https://hf-mirror.com')
        records = []
        for entry in sorted(files, key=lambda f: f.rfilename):
            path = destination/entry.rfilename
            if entry.rfilename.endswith('.safetensors') and not path.exists():
                partial = path.with_name(path.name+'.download')
                url = f'https://hf-mirror.com/{REPO}/resolve/{revision}/{entry.rfilename}'
                command = [aria, '--all-proxy=', '--http-proxy=', '--https-proxy=', '--file-allocation=none', '--continue=true',
                    '--allow-overwrite=false', '--auto-file-renaming=false', '-x16', '-s16', '-k16M',
                    '--connect-timeout=15', '--timeout=30', '--max-tries=4', '--retry-wait=3',
                    '--lowest-speed-limit=64K', '--summary-interval=5', '--console-log-level=notice',
                    '--dir='+str(path.parent), '--out='+partial.name, url]
                print(json.dumps(dict(event='weight_download', name=entry.rfilename, bytes=entry.size)), flush=True)
                write('qib350-assets-current.json', dict(phase='download', path=str(partial),
                    expected_bytes=entry.size, command=command, started_epoch=time.time()))
                result = subprocess.run(command)
                assert result.returncode == 0, f'Weight download failed with actual exit{result.returncode}'
                verify(partial, entry)
                assert not path.exists()
                partial.replace(path)
            records.append(verify(path, entry))
            print(json.dumps(dict(event='official_file_verified', name=entry.rfilename, bytes=entry.size)), flush=True)
        proof = dict(passed=True, repository=REPO, kind=kind, revision=revision, files=records,
                     total_bytes=sum(f.size for f in files), verified_epoch=time.time())
        write(f'qib350-{folder}-verified.json', proof)
        if folder == 'bench':
            backend = destination/'backends/ms_swift_backend.py'
            original = backend.read_text()
            saved = backend.with_suffix('.py.official-original')
            assert not saved.exists()
            saved.write_text(original)
            old = 'TransformersEngine(model_path, max_batch_size=max_batch_size)'
            new = 'TransformersEngine(model_path, max_batch_size=max_batch_size, model_type="qwen3_5", template_type="qwen3_5")'
            assert old in original or new in original
            patched = original.replace(old, new)
            if 'self.engine.template.enable_thinking = True' not in patched:
                patched = patched.replace(new, new+'\n        self.engine.template.enable_thinking = True')
            compile(patched, str(backend), 'exec')
            backend.write_text(patched)
            write('qib350-bench-runtime.json', dict(passed=True, official_revision=revision,
                original_backend_sha256=hashlib.sha256(original.encode()).hexdigest(),
                patched_backend_sha256=hashlib.sha256(patched.encode()).hexdigest(),
                compatibility_patch='Same README model/template pin and explicit thinkingTrue; judge.py unchanged'))
    write('qib350-assets-complete.json', dict(passed=True, completed_epoch=time.time()))


if __name__ == '__main__':
    main()
