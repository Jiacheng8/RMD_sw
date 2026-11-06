#!/usr/bin/env python3
"""Stream pinned teacher PNGs into a new Qwen3VL image-feature reference view.

Remote-only; one tar in flight. --audit-only is CPU PNG/identity validation.
--limit-shards performs a resumable pilot and never publishes final metadata.
Source stores, their encoder_features directory, prompts and contexts are read-only.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tarfile
import time
from urllib.parse import quote, urlsplit
import uuid

REPO = Path(__file__).resolve().parents[1]
for directory in (REPO, REPO / 'z-image'):
    sys.path.insert(0, str(directory))
DATASET = 'jiachengcui888/zimage-turbo-reference'
TEACHER = 'Tongyi-MAI/Z-Image-Turbo'
ENCODER = 'qwen3vl_4b'
DIM, NROWS = 2560, 125808
OLD = ('dinov3_l', 'aimv2_huge')
FAILURE_DIR = None


def redact(value):
    return re.sub(r'https?://\S+|hf_[A-Za-z0-9]+', '[redacted]', str(value))


def require_real_directory(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'Refusing symlink build/staging directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f'Not a directory: {path}')


def validate_endpoint(value):
    parsed = urlsplit(value)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password \
            or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
        raise ValueError('Payload endpoint must be an HTTPS origin without credentials/path/query')
    return value.rstrip('/')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('w') as output:
        json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        output.write('\n'); output.flush(); os.fsync(output.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)


def field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def official_inventory(info, revision):
    if field(info, 'sha') != revision:
        raise ValueError('Official metadata did not resolve the requested immutable revision')
    rows = []
    for item in field(info, 'siblings', []):
        name = field(item, 'rfilename')
        if not isinstance(name, str) or re.fullmatch(r'images/shard-\d{5}\.tar', name) is None:
            continue
        lfs, size = field(item, 'lfs'), field(item, 'size')
        checksum = field(lfs, 'sha256')
        if not isinstance(size, int) or size <= 0 or not isinstance(checksum, str) \
                or re.fullmatch('[0-9a-f]{64}', checksum) is None:
            raise ValueError(f'Missing official size/SHA256: {name}')
        rows.append({'name': name, 'bytes': size, 'sha256': checksum})
    rows.sort(key=lambda row: row['name'])
    if [r['name'] for r in rows] != [f'images/shard-{i:05d}.tar' for i in range(87)]:
        raise ValueError('Expected exactly the canonical87 teacher tar shards')
    return rows


def seed_for(image_id, candidate, base_seed):
    return int(digest([image_id, candidate, base_seed])[:15], 16)


def check_sample(meta, png, prompts, row_for_prompt, render):
    from PIL import Image
    pid = meta.get('prompt_id')
    if type(pid) is not int or pid not in row_for_prompt:
        raise ValueError('Unknown or invalid prompt_id')
    expected = prompts[pid]
    for key in ('image_id', 'prompt', 'source', 'lang'):
        if key in expected and meta.get(key) != expected[key]:
            raise ValueError(f'prompt{pid}: teacher {key} mismatch')
    contract = {'model': TEACHER, 'height': 1024, 'width': 1024,
                'steps': render['steps'], 'guidance_scale': render['guidance_scale'],
                'seed': seed_for(expected['image_id'], 0, render['seed'])}
    for key, value in contract.items():
        if meta.get(key) != value:
            raise ValueError(f'prompt{pid}: teacher {key} does not match base render contract')
    with Image.open(io.BytesIO(png)) as image:
        if image.format != 'PNG' or image.size != (1024, 1024):
            raise ValueError(f'prompt{pid}: not a1024x1024 PNG')
        image.load()
        rgb = image.convert('RGB').copy()
    return row_for_prompt[pid], rgb, {
        'row_id': row_for_prompt[pid], 'prompt_id': pid, 'image_id': expected['image_id'],
        'seed': contract['seed'], 'png_sha256': hashlib.sha256(png).hexdigest(),
        'prompt_sha256': hashlib.sha256(expected['prompt'].encode()).hexdigest(),
        'auxiliary_writer': {'base': expected.get('writer'), 'teacher': meta.get('writer'),
                             'different': expected.get('writer') != meta.get('writer')},
    }


def iter_samples(path, max_pending_bytes=64 * 1024**2):
    pending, memory = {}, 0
    with tarfile.open(path, 'r|*') as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('Unsafe tar member path')
            if name.suffix not in ('.png', '.json'):
                continue
            if member.size <= 0 or member.size > 32 * 1024**2:
                raise ValueError('Unexpected PNG/JSON member size')
            key, ext = str(name.with_suffix('')), name.suffix
            entry = pending.setdefault(key, {})
            if ext in entry:
                raise ValueError(f'Duplicate tar member: {member.name}')
            entry[ext] = archive.extractfile(member).read()
            memory += len(entry[ext])
            if memory > max_pending_bytes:
                raise ValueError('Unpaired tar members exceed bounded-memory limit')
            if '.png' in entry and '.json' in entry:
                del pending[key]
                memory -= sum(len(v) for v in entry.values())
                yield key, json.loads(entry['.json']), entry['.png']
    if pending:
        raise ValueError('Tar contains unpaired PNG/JSON members')


def feature_digest(array, ids):
    import numpy as np
    return hashlib.sha256(np.asarray(array[ids], dtype=np.float16).tobytes()).hexdigest()


def download_tar(shard, staging, revision, endpoint):
    """One resumable public payload; only a final official SHA commits verification."""
    folder = staging / revision / 'images'
    require_real_directory(staging)
    require_real_directory(staging / revision)
    require_real_directory(folder)
    tar = folder / Path(shard['name']).name
    if tar.is_symlink() or tar.with_suffix(tar.suffix + '.aria2').is_symlink():
        raise ValueError('Refusing symlink staging payload/control file')
    receipt_path = tar.with_suffix('.verified.json')
    def verified():
        return tar.exists() and tar.stat().st_size == shard['bytes'] and sha256(tar) == shard['sha256']
    if verified():
        atomic_json(receipt_path, {'dataset': DATASET, 'revision': revision, 'shard': shard,
                                  'verified_epoch': time.time()})
        return tar
    # A full-length corrupt file cannot be safely range-resumed without an aria2 bitmap.
    # Preserve it and require an explicit operator decision rather than truncate evidence.
    if tar.exists() and tar.stat().st_size >= shard['bytes'] and not Path(str(tar) + '.aria2').exists():
        raise ValueError('Existing full-length teacher tar has wrong official SHA; retained')
    url = endpoint + '/datasets/' + DATASET + '/resolve/' + revision + '/' + quote(shard['name'], safe='/')
    command = ['aria2c', '-c', '-x16', '-s16', '-k4M', '--all-proxy=', '--http-proxy=',
               '--https-proxy=', '--auto-file-renaming=false', '--file-allocation=none',
               '--allow-overwrite=false', '--show-console-readout=false', '--summary-interval=15',
               '--console-log-level=warn', '--max-tries=5', '--retry-wait=5', '--connect-timeout=30',
               '--timeout=120', '--dir=' + str(folder), '--out=' + tar.name, url]
    child_env = {k: v for k, v in os.environ.items()
                 if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy')}
    child_env['NO_PROXY'] = '*'
    log_path = tar.with_suffix('.download.log')
    last_lines = []
    with log_path.open('a') as log:
        log.write(json.dumps({'command': command, 'started_epoch': time.time()}) + '\n'); log.flush()
        process = subprocess.Popen(command, env=child_env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors='replace')
        try:
            for line in process.stdout:
                safe = redact(line.rstrip())
                log.write(safe + '\n'); log.flush()
                last_lines = (last_lines + [safe])[-30:]
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
        finally:
            process.stdout.close()
    if code != 0:
        failure = {'command': command, 'exit': code, 'last30': last_lines, 'shard': shard,
                   'failed_epoch': time.time()}
        atomic_json(tar.with_suffix('.download-failed.json'), failure)
        raise RuntimeError(f'aria2 failed with exit{code}; sanitized command/tail saved at{log_path}')
    if not verified():
        atomic_json(tar.with_suffix('.download-failed.json'), {'command': command, 'exit': code,
                    'last30': last_lines, 'shard': shard, 'error': 'Official size/SHA256 verification failed'})
        raise ValueError('aria2 exit0 but official teacher tar size/SHA mismatch; staging retained')
    with tar.open('rb') as handle:
        os.fsync(handle.fileno())
    atomic_json(receipt_path, {'dataset': DATASET, 'revision': revision, 'shard': shard,
                              'verified_epoch': time.time()})
    return tar


def load_base(pack, source):
    import numpy as np
    prompts = [json.loads(line) for line in (pack / 'prompts.jsonl').read_text().splitlines() if line]
    meta = json.loads((source / 'metadata.json').read_text())
    render = json.loads((pack / 'render_config.json').read_text())
    if meta['num_rows'] != NROWS or meta['num_prompts'] != NROWS or meta['rows_per_prompt'] != 1:
        raise ValueError('This builder supports only the complete125808-row keep1 reference')
    if len(prompts) != NROWS or digest(prompts) != meta['prompt_hash']:
        raise ValueError('Canonical frozen prompt hash/count mismatch')
    if digest(render) != meta['render_hash'] or render['keep'] != 1 or render['candidates'] != 1:
        raise ValueError('Require the original single-candidate teacher render contract')
    if (render['height'], render['width']) != (1024, 1024):
        raise ValueError('Base reference does not describe1024px teacher images')
    pid = np.load(source / 'prompt_ids.npy')
    rows = [json.loads(line) for line in (source / 'rows.jsonl').read_text().splitlines() if line]
    mapping = {}
    if len(rows) != NROWS or pid.shape != (NROWS,):
        raise ValueError('Invalid canonical row map')
    for index, row in enumerate(rows):
        if row['row_id'] != index or row['prompt_id'] != int(pid[index]):
            raise ValueError('rows.jsonl and prompt_ids.npy disagree')
        p = int(pid[index])
        if not 0 <= p < NROWS or p in mapping or row['image_id'] != prompts[p]['image_id']:
            raise ValueError('Duplicate or inconsistent canonical image/prompt identity')
        mapping[p] = index
    order = np.load(source / 'row_order.npy')
    if order.shape != (NROWS,) or not np.array_equal(np.sort(order), np.arange(NROWS)):
        raise ValueError('row_order is not the original complete permutation')
    return prompts, meta, render, mapping


def load_models(device):
    import torch
    from huggingface_hub import constants
    from rdm.representation.backbones import build_backbone
    from rdm.representation.registry import by_name
    previous = constants.HF_HUB_OFFLINE
    try:
        constants.HF_HUB_OFFLINE = True  # No moving-main refresh of cached old encoders.
        models = {name: build_backbone(by_name(name), device) for name in (ENCODER, *OLD)}
    finally:
        constants.HF_HUB_OFFLINE = previous
    if any(p.requires_grad or p.dtype != torch.float32 for model in models.values()
           for p in model.parameters()):
        raise ValueError('All frozen reference encoders require fp32 weights')
    return models


def old_feature_audit(models, pixels, ids, source, max_relative, min_cosine):
    import numpy as np
    import torch
    result = {}
    for name in OLD:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            actual = models[name](pixels).float()
        stored = np.load(source / 'encoder_features' / (name + '.npy'), mmap_mode='r')
        expected = torch.from_numpy(np.asarray(stored[ids], dtype=np.float32).copy()).to(actual.device)
        relative = (actual - expected).norm(dim=1) / expected.norm(dim=1).clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(actual, expected)
        if not torch.isfinite(actual).all() or float(relative.max()) > max_relative \
                or float(cosine.min()) < min_cosine:
            raise ValueError(f'{name}: same-row teacher feature audit failed; do not relax silently')
        result[name] = {'rows': list(map(int, ids)), 'max_relative_l2': float(relative.max()),
                        'min_cosine': float(cosine.min())}
    return result


def make_links(work, source):
    folder = work / 'encoder_features'
    if folder.is_symlink():
        raise ValueError('New encoder_features must be an independent directory')
    folder.mkdir(exist_ok=True)
    links = {name: source / name for name in ('text_features.npy', 'text_features.done.json',
              'prompt_ids.npy', 'group_ids.npy', 'row_order.npy', 'rows.jsonl')}
    for name in OLD:
        for suffix in ('.npy', '.done.json'):
            links['encoder_features/' + name + suffix] = source / 'encoder_features' / (name + suffix)
    for name, original in links.items():
        target = work / name
        if target.is_symlink():
            if target.resolve() != original.resolve():
                raise ValueError(f'Stale source link: {target}')
        elif target.exists():
            raise ValueError(f'Refusing to overwrite non-link source artifact: {target}')
        else:
            os.symlink(os.path.relpath(original, target.parent), target)
    zc = json.loads((source / 'z_context.json').read_text())
    if zc['root'] != '..':
        raise ValueError('New view must remain next to source with the same render parent')
    atomic_json(work / 'z_context.json', zc)


def finalize(work, output, source, array, contract, receipts):
    import numpy as np
    import torch
    from rdm.compare.median_heuristic import median_bandwidth
    from rdm.sw_lmmd.reference_store import ReferenceFeatureStore
    from rdm.sw_lmmd.adversarial import reference_stats
    from zimage_distill.qwen_text_store import rows_read, stats_rows
    for lo in range(0, NROWS, 8192):
        if not np.isfinite(array[lo:lo+8192]).all():
            raise ValueError('Nonfinite or incomplete new image table')
    bands = json.loads((source / 'bandwidths.json').read_text())
    scales = [float(bands[name]['sigma']) / float(bands[name]['beta']) for name in (*OLD, 'siglip2')]
    if not all(math.isfinite(s) and s > 0 and math.isclose(s, scales[0], rel_tol=1e-6) for s in scales):
        raise ValueError('Existing Qwen text-kernel scales disagree')
    sample_ids = np.random.default_rng(0).choice(NROWS, 2000, replace=False)
    torch.manual_seed(0)
    sigma = float(median_bandwidth(torch.from_numpy(np.asarray(array[sample_ids], dtype=np.float32)))) * 0.25
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('Invalid new image bandwidth')
    make_links(work, source)
    feature = work / 'encoder_features' / f'{ENCODER}.npy'
    part = work / 'encoder_features' / f'{ENCODER}.partial.npy'
    if part.exists():
        os.replace(part, feature)
    bandwidth = {'sigma': sigma, 'beta': sigma / scales[0], 's_txt': scales[0], 'sigma_scale': 0.25,
                 'estimator': 'median over deterministic2000 fp16 stored rows',
                 'sample_ids_sha256': hashlib.sha256(sample_ids.tobytes()).hexdigest()}
    atomic_json(work / 'encoder_features' / f'{ENCODER}.done.json',
                {'contract': contract, 'shape': [NROWS, DIM], 'bandwidth': bandwidth,
                 'feature_sha256': sha256(feature)})
    atomic_json(work / 'bandwidths.json', {OLD[0]: bands[OLD[0]], ENCODER: bandwidth, OLD[1]: bands[OLD[1]]})
    metadata = json.loads((source / 'metadata.json').read_text())
    metadata.update(encoder_feature_dims={OLD[0]: 1024, ENCODER: DIM, OLD[1]: 1536},
                    image_feature_provenance=contract)
    atomic_json(work / 'metadata.json', metadata)  # Still inside unpublished .partial directory.
    names = [OLD[0], ENCODER, OLD[1]]
    store = ReferenceFeatureStore(str(work), names, require_context=False)
    required = np.unique(np.concatenate([rows_read(store.row_order(), 500, 128, 32),
                                        stats_rows(NROWS, 8192, [0, 1])]))
    text = store.text_features(required)
    if not torch.isfinite(text).all() or not torch.allclose(text.norm(dim=1),
            torch.ones(len(required)), atol=1e-3, rtol=0):
        raise ValueError('Qwen text table does not cover planned500-step/critic rows')
    stats_audit = {}
    for seed in (0, 1):
        image, text_stats = reference_stats(store, names, 8192, seed, with_text=True)
        entry = {}
        for name, (mean, scale) in {**image, 'text': text_stats}.items():
            if not torch.isfinite(mean).all() or not torch.isfinite(scale) or not scale > 0:
                raise ValueError('Nonfinite critic reference statistics')
            entry[name] = {'mean_dim': mean.numel(), 'mean_sha256': hashlib.sha256(mean.numpy().tobytes()).hexdigest(),
                           'rms': float(scale)}
        stats_audit[str(seed)] = entry
    atomic_json(work / 'critic-stat-audit.json', stats_audit)
    atomic_json(work / 'image-row-manifest.json', {'complete': True, 'rows': NROWS,
                'shards': receipts, 'contract': contract, 'text_coverage_steps': 500,
                'identity_evidence': 'fixed official tar hashes + full row/settings/seed checks + stratified DINO/AIM audit; not original PNG byte-hash proof'})
    from zimage_distill.store import ZImageReferenceStore
    view = ZImageReferenceStore(str(work), names)
    if view.feature_image_size() != 512 or not view.has_context:
        raise ValueError('Shared ADM/context contract does not read back')
    for name in names:
        if not torch.isfinite(view.reference_joint_features(name, required[:128])).all():
            raise ValueError('New reference view does not read finite joint rows')
    fsync_dir(work)
    os.rename(work, output)
    fsync_dir(output.parent)


def main(argv=None):
    global FAILURE_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/rdm-workspace/rdm-sets'))
    parser.add_argument('--dataset-revision', required=True, help='immutable official40-hex dataset commit')
    parser.add_argument('--endpoint', default='https://hf-mirror.com', help='payload transport only')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit-shards', type=int)
    parser.add_argument('--audit-only', action='store_true', help='CPU identity/PNG audit; no features or final metadata')
    parser.add_argument('--prefetch-only', action='store_true', help='CPU official manifest and at most2 verified tars; always retained')
    parser.add_argument('--keep-tars', action='store_true')
    parser.add_argument('--audit-max-relative-l2', type=float, default=0.03)
    parser.add_argument('--audit-min-cosine', type=float, default=0.999)
    args = parser.parse_args(argv)
    if platform.system() == 'Darwin':
        raise RuntimeError('Teacher images and feature tensors are remote-only')
    if not re.fullmatch('[0-9a-f]{40}', args.dataset_revision) or args.batch < 1 \
            or args.limit_shards is not None and not 1 <= args.limit_shards <= 87:
        raise ValueError('Invalid fixed revision/batch/shard limit')
    args.endpoint = validate_endpoint(args.endpoint)
    if args.prefetch_only and (args.audit_only or args.limit_shards not in (1, 2)):
        raise ValueError('--prefetch-only requires --limit-shards1 or2 and excludes --audit-only')
    if not args.root.is_absolute() or not args.root.is_dir():
        raise ValueError('Use the existing absolute remote root')
    if not 0 < args.audit_max_relative_l2 < 1 or not 0 < args.audit_min_cosine <= 1:
        raise ValueError('Invalid explicit audit tolerances')
    pack = args.root / 'zimage/zimage-ref-v1'
    source, output = pack / 'reference_store_qwen', pack / 'reference_store_qwen3vl'
    if output.exists():
        raise ValueError('Final view exists; refusing overwrite')
    work = output.with_name(output.name + ('.prefetch' if args.prefetch_only else
                                         '.audit' if args.audit_only else '.partial'))
    require_real_directory(work)
    require_real_directory(work / 'encoder_features')
    FAILURE_DIR = work
    lock = (work / '.build.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import huggingface_hub as hf
    info = hf.HfApi(endpoint='https://huggingface.co', token=False).repo_info(
        DATASET, repo_type='dataset', revision=args.dataset_revision, files_metadata=True)
    shards = official_inventory(info, args.dataset_revision)
    manifest = {'dataset': DATASET, 'revision': args.dataset_revision, 'shards': shards}
    selected = shards[:args.limit_shards] if args.limit_shards is not None else shards
    staging = args.root / '.cache/qwen3vl-teacher-staging'
    if args.prefetch_only:
        existing_manifest = work / 'official-shards.json'
        if existing_manifest.exists() and json.loads(existing_manifest.read_text()) != manifest:
            raise ValueError('Prefetch manifest changed; preserve previous staging')
        atomic_json(existing_manifest, manifest)
        for shard in selected:
            tar = download_tar(shard, staging, args.dataset_revision, args.endpoint)
            print(json.dumps({'event': 'teacher_tar_verified', 'shard': shard['name'],
                              'bytes': tar.stat().st_size, 'retained': True}), flush=True)
        print('PREFETCH_COMPLETE; no models loaded; final metadata NOT published', flush=True)
        lock.close()
        return
    import numpy as np
    import importlib.metadata as package_metadata
    prompts, metadata, render, mapping = load_base(pack, source)
    source_names = ['metadata.json', 'bandwidths.json', 'row_order.npy', 'prompt_ids.npy',
                    'group_ids.npy', 'rows.jsonl', 'text_features.npy', 'text_features.done.json', 'z_context.json']
    source_names += ['encoder_features/' + name + ext for name in OLD for ext in ('.npy', '.done.json')]
    contract = {'encoder': ENCODER, 'rows': NROWS, 'dim': DIM, 'img_size': 512,
                'model_input_size': 256, 'render_hash': metadata['render_hash'],
                'prompt_hash': metadata['prompt_hash'], 'source_dataset': DATASET,
                'source_revision': args.dataset_revision, 'shard_manifest_sha256': digest(manifest),
                'builder_sha256': sha256(__file__),
                'adapter_sha256': sha256(REPO / 'rdm/representation/backbones/qwen3_vl_backbone.py'),
                'source_artifacts': {n: sha256(source / n) for n in source_names},
                'package_versions': {name: package_metadata.version(name) for name in
                                     ('torch', 'numpy', 'transformers', 'Pillow')},
                'attention_backend': os.environ.get('RDM_QWEN3VL_ATTN_IMPLEMENTATION', 'sdpa'),
                'storage_dtype': 'float16', 'autocast': 'bfloat16', 'audit_only': args.audit_only,
                'audit_max_relative_l2': args.audit_max_relative_l2, 'audit_min_cosine': args.audit_min_cosine}
    if (work / 'contract.json').exists() and json.loads((work / 'contract.json').read_text()) != contract:
        raise ValueError('Build contract changed; preserve this partial and choose a new output')
    atomic_json(work / 'contract.json', contract)
    atomic_json(work / 'official-shards.json', manifest)
    receipts = {}
    visited = np.zeros(NROWS, dtype=bool)
    array = models = None
    if not args.audit_only:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        index = int(args.device.split(':')[-1])
        if not args.device.startswith('cuda:') or visible not in (['4'], ['5'], ['4', '5']) or index >= len(visible):
            raise ValueError('Only explicitly visible physical GPU4/5 may be used')
        idle = subprocess.check_output(['nvidia-smi', '-i', visible[index],
                '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        if any(int(x) > y for x, y in zip(idle.strip().split(','), [32, 0])):
            raise RuntimeError('Selected GPU is occupied; do not overlap capacity tests')
        folder = work / 'encoder_features'
        part, final = folder / f'{ENCODER}.partial.npy', folder / f'{ENCODER}.npy'
        if part.is_symlink() or final.is_symlink() or part.exists() and final.exists():
            raise ValueError('Refusing linked/ambiguous new feature table')
        target = part if part.exists() else final if final.exists() else part
        is_new = not target.exists()
        array = np.lib.format.open_memmap(target, mode='w+' if is_new else 'r+',
                                         dtype=np.float16, shape=(NROWS, DIM))
        if array.shape != (NROWS, DIM) or array.dtype != np.float16:
            raise ValueError('Existing feature table shape/dtype mismatch')
        if is_new:
            array[:] = np.nan; array.flush()
        models = load_models(args.device)
        model_contract = models[ENCODER].feature_contract()
        if model_contract['weight_dtype'] != 'torch.float32' or model_contract['dim'] != DIM \
                or model_contract['l2_normalize']:
            raise ValueError('Unexpected Qwen3VL feature contract')
        if (work / 'image-encoder-contract.json').exists() and \
                json.loads((work / 'image-encoder-contract.json').read_text()) != model_contract:
            raise ValueError('Encoder runtime contract changed on resume')
        atomic_json(work / 'image-encoder-contract.json', model_contract)
    receipt_dir = work / 'receipts'; require_real_directory(receipt_dir)
    expected_shards = {s['name']: s for s in shards}
    for path in sorted(receipt_dir.glob('shard-*.json')):
        receipt = json.loads(path.read_text()); ids = receipt['row_ids']
        shard_name = receipt['shard']['name']
        if receipt['shard'] != expected_shards.get(shard_name) or shard_name in receipts \
                or path.stem != Path(shard_name).stem or not ids \
                or receipt['contract_sha256'] != digest(contract) or len(ids) != len(set(ids)) \
                or any(type(i) is not int or not 0 <= i < NROWS or visited[i] for i in ids):
            raise ValueError('Inconsistent resumed shard receipt')
        proof_ids = [p['row_id'] for p in receipt['image_rows']]
        if len(proof_ids) != len(ids) or sorted(proof_ids) != ids:
            raise ValueError('Receipt image provenance differs from committed rows')
        if array is not None and feature_digest(array, ids) != receipt['feature_rows_sha256']:
            raise ValueError('Committed feature rows changed')
        visited[ids] = True; receipts[receipt['shard']['name']] = receipt
    audit_ids = set(np.random.default_rng(0).choice(NROWS, 256, replace=False).tolist())
    for shard in selected:
        if shard['name'] in receipts:
            continue
        tar = download_tar(shard, staging, args.dataset_revision, args.endpoint)
        pending, provenance, local_ids, audits = [], [], set(), []
        def flush():
            if not pending or array is None:
                return
            import torch
            from rdm.data.imagenet import default_transform
            ids = [row[0] for row in pending]
            pixels = torch.stack([default_transform(512)(row[1]) for row in pending]).to(args.device)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                features = models[ENCODER](pixels).float().cpu().numpy()
            cast = features.astype(np.float16)
            if features.shape != (len(ids), DIM) or not np.isfinite(features).all() or not np.isfinite(cast).all():
                raise ValueError('Invalid or fp16-overflowing Qwen3VL features')
            array[ids] = cast
            sample = [i for i, row in enumerate(pending) if row[2]]
            if sample:
                audits.append(old_feature_audit(models, pixels[sample], [ids[i] for i in sample],
                             source, args.audit_max_relative_l2, args.audit_min_cosine))
            pending.clear()
        for key, meta, png in iter_samples(tar):
            rid, image, proof = check_sample(meta, png, prompts, mapping, render)
            if visited[rid] or rid in local_ids:
                raise ValueError('Duplicate teacher row across/within shards')
            do_audit = not local_ids or rid in audit_ids
            local_ids.add(rid); proof['member_key'] = key; provenance.append(proof)
            if array is not None:
                pending.append((rid, image, do_audit))
                if len(pending) >= args.batch:
                    flush()
        flush()
        ids = sorted(local_ids)
        if not ids:
            raise ValueError('Empty teacher shard')
        if array is not None:
            array.flush()
            with Path(array.filename).open('rb') as handle:
                os.fsync(handle.fileno())
        receipt = {'shard': shard, 'row_ids': ids, 'contract_sha256': digest(contract),
                   'feature_rows_sha256': feature_digest(array, ids) if array is not None else None,
                   'same_image_audits': audits, 'image_rows': provenance, 'committed_epoch': time.time(),
                   'auxiliary_writer_differences': sum(p['auxiliary_writer']['different'] for p in provenance)}
        atomic_json(receipt_dir / (Path(shard['name']).stem + '.json'), receipt)
        receipts[shard['name']] = receipt; visited[ids] = True
        atomic_json(work / 'progress.json', {'completed_shards': len(receipts),
                    'completed_rows': int(visited.sum()), 'total_rows': NROWS, 'metadata_published': False})
        print(json.dumps({'event': 'shard_committed', 'shard': shard['name'],
                          'completed_rows': int(visited.sum()), 'audit_only': args.audit_only}), flush=True)
        if array is not None and not args.keep_tars:
            tar.unlink()
            # A crash after all bytes arrived can leave a valid tar and old aria2
            # bitmap; remove only after this tar's durable feature receipt exists.
            Path(str(tar) + '.aria2').unlink(missing_ok=True)
            fsync_dir(tar.parent)
    if args.audit_only or args.limit_shards is not None:
        print('PILOT_COMPLETE; final metadata NOT published', flush=True)
        lock.close()
        return
    if not visited.all() or len(receipts) != 87:
        raise ValueError('Full reference coverage is incomplete')
    finalize(work, output, source, array, contract, list(receipts.values()))
    print(json.dumps({'event': 'reference_complete', 'path': str(output), 'rows': NROWS, 'dim': DIM}), flush=True)
    lock.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Do not emit request headers, signed URLs or tokens from dependency exceptions.
        failure = {'event': 'failed', 'type': type(error).__name__,
                   'message': redact(error)[:1500], 'failed_epoch': time.time()}
        if FAILURE_DIR is not None:
            # Failed final validation must not leave a completion marker, even in .partial.
            marker = FAILURE_DIR / 'metadata.json'
            if marker.is_file() and not marker.is_symlink():
                marker.unlink(); fsync_dir(FAILURE_DIR)
            atomic_json(FAILURE_DIR / 'failed.json', failure)
        print(json.dumps(failure), flush=True)
        raise SystemExit(1)
