#!/usr/bin/env python3
"""Offload completed H20 reference tars and resume with full final-name SHA checks.

No training launches, foreign files, or checkpoints are deleted. Original provenance
is retained; a new SQLite/manifest view changes storage locations only. Execute on
H20 with system Python, through an existing control channel. Restartable receipts.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

OLD = Path('/data/shaoshitong')
NEW = Path('/data/rdm-workspace')
BASE = NEW if NEW.exists() else OLD
ROOT = BASE/'rdm-sets'
OBJECT = Path('/mnt/airs-business/rdm-experiments')
CONTROL = ROOT/'control/storage-relocation-20261106'
EXP = 'zimage-2step-1024-dino-qwenvl-mmd2-from0'
REV = '3195cc9e58d83dfe17a6b5d89f9886121872586f'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def stat(path):
    s=path.stat()
    return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)


def preflight():
    assert sys.platform=='linux' and ROOT.is_dir() and OBJECT.parent.is_mount()
    final=json.loads((ROOT/'control/dino-qwenvl-mmd2-from0-final-integrity.json').read_text())
    assert final['passed'] and final['step']==500 and final['actual_formal_exit']==0 and final['actual_workflow_exit']==0
    # Do not move a directory used by a running process. Never kill anything.
    busy=[]
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name)==os.getpid():continue
        try:
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd=os.readlink(proc/'cwd')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if str(BASE)+'/' in cmd or cwd.startswith(str(BASE)+'/'):busy.append(dict(pid=int(proc.name),cwd=cwd))
    assert not busy, 'Active project processes: '+str(busy)
    OBJECT.mkdir(exist_ok=True)
    probe=OBJECT/('.storage-probe-'+str(os.getpid()))
    with probe.open('xb') as f:f.write(b'rdm-storage-probe\n');f.flush();os.fsync(f.fileno())
    assert probe.read_bytes()==b'rdm-storage-probe\n';probe.unlink()
    return final


def transfer(source, destination, expected_size, expected_sha, receipt_name):
    receipt=CONTROL/'files'/receipt_name
    # An explicit controlled-resume mode may reuse a previously completed full
    # final-name SHA check. The object namespace is unchanged by local renaming;
    # a later object mtime or a mismatched receipt forces a fresh full read.
    if os.environ.get('H20_REUSE_VERIFIED_OBJECTS')=='1' and receipt.is_file() and destination.exists():
        previous=json.loads(receipt.read_text())
        st=destination.stat()
        if previous.get('passed') is True and previous.get('final_named_SHA_readback') is True \
                and previous.get('destination')==str(destination) and previous.get('sha256')==expected_sha \
                and previous.get('bytes')==expected_size and st.st_size==expected_size \
                and st.st_mtime<=previous.get('verified_epoch',0):
            previous.update(source=str(source),reused_previously_verified_object=True,
                metadata_rechecked_epoch=time.time(),unchanged_object_mtime=st.st_mtime)
            write(receipt,previous)
            print(json.dumps(dict(event='verified_object_reused',file=source.name,bytes=expected_size)),flush=True)
            return previous
    if destination.exists():
        assert destination.stat().st_size==expected_size and sha(destination)==expected_sha
    else:
        assert source.is_file() and not source.is_symlink() and source.stat().st_size==expected_size
        before=stat(source);temporary=destination.with_name(destination.name+'.partial-'+str(os.getpid()))
        destination.parent.mkdir(parents=True,exist_ok=True)
        assert not temporary.is_symlink()
        digest=hashlib.sha256();copied=0
        with source.open('rb') as inp,temporary.open('wb') as out:
            for block in iter(lambda:inp.read(8*1024**2),b''):
                assert out.write(block)==len(block);digest.update(block);copied+=len(block)
            out.flush();os.fsync(out.fileno())
        assert stat(source)==before and copied==expected_size and digest.hexdigest()==expected_sha
        temporary.replace(destination)
        assert destination.stat().st_size==expected_size and sha(destination)==expected_sha
    value=dict(passed=True,source=str(source),destination=str(destination),bytes=expected_size,
               sha256=expected_sha,final_named_SHA_readback=True,verified_epoch=time.time())
    write(receipt,value)
    print(json.dumps(dict(event='file_verified',file=source.name,bytes=expected_size)),flush=True)
    return value


def object_image_view(original, proofs):
    manifest=json.loads((original/'manifest.json').read_text())
    view=original.with_name('reference_images_patchgan_object_v1')
    view.mkdir(exist_ok=True)
    index=view/'index.sqlite'
    if index.exists():index.unlink()
    with sqlite3.connect(original/'index.sqlite') as source,sqlite3.connect(index) as target:source.backup(target)
    proof_by_name={Path(x['destination']).name:x for x in proofs}
    with sqlite3.connect(index) as db:
        assert db.execute('SELECT count(*) FROM images').fetchone()[0]==125808
        for row in manifest['tars']:
            proof=proof_by_name[Path(row['path']).name]
            assert (row['bytes'],row['sha256'])==(proof['bytes'],proof['sha256'])
            destination=Path(proof['destination']);row['path']=str(destination);row['stat']=stat(destination)
            db.execute('UPDATE tars SET path=?, stat_json=? WHERE name=?',(str(destination),json.dumps(row['stat']),row['name']))
        db.commit()
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    manifest['index']['sha256']=sha(index);manifest['index']['bytes']=index.stat().st_size
    manifest['storage_migration']=dict(original_manifest_sha256=sha(original/'manifest.json'),
        paths_only=True,all87_final_named_SHA_verified=True,epoch=time.time())
    write(view/'manifest.json',manifest)
    # Independently exercise the original pread+PNG SHA contract across EVERY tar,
    # preserving index rows/offsets and all image identities; no GPU or tensor load.
    samples=[]
    with sqlite3.connect(index) as db:
        db.row_factory=sqlite3.Row
        for tar in manifest['tars']:
            records=list(db.execute('SELECT * FROM images WHERE tar_name=? ORDER BY row_id LIMIT 2',(tar['name'],)))
            with Path(tar['path']).open('rb') as inp:
                for record in records:
                    inp.seek(record['data_offset']);data=inp.read(record['bytes'])
                    assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['png_sha256']
                    assert data[:8]==b'\x89PNG\r\n\x1a\n'
                    samples.append(record['row_id'])
    write(CONTROL/'object-image-view.json',dict(passed=True,view=str(view),manifest_sha256=sha(view/'manifest.json'),
        index_sha256=sha(index),canonical_rows=125808,sampled_pngs=len(samples),all87_tars_sampled=True))
    return view


def archive():
    final=preflight();CONTROL.mkdir(parents=True,exist_ok=True)
    assert not (CONTROL/'complete.json').exists(), 'Migration already complete; use receipts, do not repeat'
    before=shutil.disk_usage('/data').free
    manifest_path=ROOT/'zimage/zimage-ref-v1/reference_images_patchgan/manifest.json'
    manifest=json.loads(manifest_path.read_text());assert len(manifest['tars'])==87 and manifest['complete']
    specs=[]
    for row in manifest['tars']:
        name=Path(row['path']).name
        assert name.startswith('shard-') and name.endswith('.tar')
        source=ROOT/'.cache/qwen3vl-teacher-staging'/REV/'images'/name
        destination=OBJECT/'datasets'/('zimage-reference-'+REV)/'images'/name
        specs.append((source,destination,row['bytes'],row['sha256'],name+'.json'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(transfer,*spec) for spec in specs];proofs=[]
        for future in concurrent.futures.as_completed(futures):
            proofs.append(future.result())
            write(CONTROL/'progress.json',dict(phase='reference_tars',verified_files=len(proofs),total_files=87,
                verified_bytes=sum(x['bytes'] for x in proofs),epoch=time.time()))
    view=object_image_view(manifest_path.parent,proofs)
    # A durable remote receipt exists BEFORE removing any source payload.
    write(OBJECT/'manifests/reference-tars-relocated.json',dict(passed=True,files=proofs,view=str(view)))
    removed=0
    for source,_,size,digest,_ in specs:
        if source.exists():
            assert not source.is_symlink() and source.stat().st_size==size
            source.unlink();removed+=size
    source=ROOT/'zimage/work_dirs'/EXP/'resume.pth'
    proof=transfer(source,OBJECT/'checkpoints'/EXP/'resume-step500.pth',final['resume_bytes'],
                   final['resume_sha256'],'resume-step500.json')
    write(OBJECT/'manifests/resume500-relocated.json',proof)
    if not source.is_symlink():
        assert source.stat().st_size==proof['bytes'];source.unlink();source.symlink_to(proof['destination']);removed+=proof['bytes']
    result=dict(passed=True,phase='archive_complete',reference_tars=87,reference_bytes=sum(x['bytes'] for x in proofs),
        resume_bytes=proof['bytes'],payload_bytes_removed_from_data=removed,image_view=str(view),
        disk_free_before=before,disk_free_after=shutil.disk_usage('/data').free,epoch=time.time(),
        all_student_checkpoints_retained=True,all_final_named_SHA_verified=True)
    write(CONTROL/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':archive()
