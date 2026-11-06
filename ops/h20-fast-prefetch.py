"""Use a reachable transport mirror, pinned and hashed against official HF metadata.

HF's normal cache/file locks coordinate with the original preparation downloader.
No active transfer is stopped or overwritten; completed cache blobs are reused.
"""
import fnmatch
import argparse
import hashlib
import json
from pathlib import Path
import httpx
import huggingface_hub as hf

ROOT = Path('/data/rdm-workspace/rdm-sets')
parser = argparse.ArgumentParser()
parser.add_argument('--repo', help='Prefetch one independent repo without waiting for another repo lock')
args = parser.parse_args()
patterns = ['*.safetensors', '*.json', '*.txt', '*.py', '*.model']
specs = [
    ('timm/vit_large_patch16_dinov3.lvd1689m', 'model', patterns, None),
    ('timm/vit_so400m_patch16_siglip_256.v2_webli', 'model', patterns, None),
    ('timm/aimv2_huge_patch14_224.apple_pt', 'model', patterns, None),
    ('Tongyi-MAI/Z-Image-Turbo', 'model', patterns, None),
    ('jiachengcui888/zimage-ref-qwen-text', 'dataset',
     ['bandwidths.json', 'metadata.json', 'text_features.done.json', 'text_features.npy'],
     ROOT / 'zimage/qwen-text-download'),
]
if args.repo:
    specs = [spec for spec in specs if spec[0] == args.repo]
    assert specs, args.repo
# This first client uses the existing proxy to query the official source.
api = hf.HfApi(endpoint='https://huggingface.co')
inventories = []
for repo, kind, wanted, dest in specs:
    info = api.repo_info(repo, repo_type=kind, files_metadata=True)
    files = [f for f in info.siblings if any(fnmatch.fnmatch(f.rfilename, p) for p in wanted)]
    assert files
    inventories.append((repo, kind, wanted, dest, info.sha, files))
hf.set_client_factory(lambda: httpx.Client(trust_env=False, follow_redirects=True, timeout=120))
receipts = []
for repo, kind, wanted, dest, revision, files in inventories:
    print(json.dumps({'event': 'mirror_fetch', 'repo': repo, 'revision': revision}), flush=True)
    directory = Path(hf.snapshot_download(
        repo, repo_type=kind, revision=revision, allow_patterns=wanted,
        cache_dir=ROOT / 'hf/hub', local_dir=dest, max_workers=4, endpoint='https://hf-mirror.com'))
    rows = []
    for f in files:
        path = directory / f.rfilename
        assert path.stat().st_size == f.size, str(path)
        h = hashlib.sha256() if f.lfs else hashlib.sha1()
        if not f.lfs:
            h.update(f'blob {f.size}\0'.encode())
        with path.open('rb') as inp:
            for block in iter(lambda: inp.read(8 * 1024**2), b''):
                h.update(block)
        assert h.hexdigest() == (f.lfs.sha256 if f.lfs else f.blob_id), str(path)
        rows.append({'path': str(path), 'bytes': f.size, 'digest': h.hexdigest()})
    receipts.append({'repo': repo, 'revision': revision, 'files': rows})
    (ROOT / 'control' / ('fast-prefetch-' + repo.replace('/', '--') + '-verified.json')).write_text(
        json.dumps(receipts[-1], indent=2))
    if not args.repo:
        (ROOT / 'control/fast-prefetch-verified.json').write_text(json.dumps(receipts, indent=2))
    print(json.dumps({'event': 'official_hashes_verified', 'repo': repo,
                      'bytes': sum(f.size for f in files)}), flush=True)
