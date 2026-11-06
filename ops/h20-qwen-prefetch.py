"""Pin and verify the README's two public datasets; payload stays on H20."""
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download

ROOT = Path('/data/rdm-workspace/rdm-sets')
api = HfApi()
for repo, patterns, dest in [
    ('jiachengcui888/zimage-ref-v1', ['zimage-ref-v1/*'], ROOT / 'zimage'),
    ('jiachengcui888/zimage-ref-qwen-text',
     ['bandwidths.json', 'metadata.json', 'text_features.done.json', 'text_features.npy'],
     ROOT / 'zimage/qwen-text-download'),
]:
    info = api.dataset_info(repo, files_metadata=True)
    import fnmatch
    files = [f for f in info.siblings if any(fnmatch.fnmatch(f.rfilename, p) for p in patterns)]
    assert files, repo
    total = sum(f.size for f in files)
    print(json.dumps({'event': 'dataset_download_start', 'repo': repo, 'revision': info.sha,
                      'bytes': total, 'files': len(files)}), flush=True)
    snapshot_download(repo, repo_type='dataset', revision=info.sha,
                      allow_patterns=patterns, local_dir=dest, max_workers=3)
    checked = []
    for f in files:
        p = dest / f.rfilename
        assert p.is_file() and p.stat().st_size == f.size, str(p)
        digest = hashlib.sha256() if f.lfs else hashlib.sha1()
        if not f.lfs:
            digest.update(f'blob {f.size}\0'.encode())
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(block)
        expected = f.lfs.sha256 if f.lfs else f.blob_id
        assert digest.hexdigest() == expected, str(p)
        checked.append({'path': str(p), 'bytes': f.size, 'digest': digest.hexdigest()})
    receipt = {'repo': repo, 'revision': info.sha, 'verified_epoch': time.time(), 'files': checked}
    (ROOT / 'control' / (repo.split('/')[-1] + '-verified.json')).write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'event': 'dataset_verified', 'repo': repo, 'bytes': total}), flush=True)
print('BOTH_DATASETS_VERIFIED', flush=True)
