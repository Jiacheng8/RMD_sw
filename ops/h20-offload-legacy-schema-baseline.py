#!/usr/bin/env python3
"""Preserve an inactive legacy schema-reference weight and release local staging.

Never touches the current model, resume checkpoint, optimizer, or training process.
"""
import gc, hashlib, json, os, pathlib, time, traceback
ROOT=pathlib.Path('/data/rdm-workspace/rdm-sets')
CONTROL=ROOT/'control'
SOURCE=CONTROL/'spatial-A-user-visualization/step_0000150-model.pth'
DEST=pathlib.Path('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints/legacy-spatial-A/step_0000150-model.pth')
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TMPDIR']=str(ROOT/'.cache/tmp')


def write(path,value):
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def assert_inactive():
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            assert str(SOURCE) not in (p/'maps').read_text(errors='replace'), 'Legacy reference is actively mapped; preserve local source'
        except (OSError,ProcessLookupError):pass


def main():
    import torch
    torch.set_num_threads(1)
    assert SOURCE.is_file() and not SOURCE.is_symlink() and not DEST.exists()
    assert_inactive();before=SOURCE.stat();started=time.monotonic()
    state=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True)
    model=state['model'];assert len(model)==521 and sum(v.numel() for v in model.values())==6154908736
    schema={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in model.items()}
    assert all(s['dtype']=='torch.float32' for s in schema.values())
    del state,model;gc.collect()
    DEST.parent.mkdir(parents=True,exist_ok=True)
    partial=DEST.with_name(DEST.name+'.partial')
    digest=hashlib.sha256();copied=0;reported=-1
    with SOURCE.open('rb') as src,partial.open('xb') as dst:
        for block in iter(lambda:src.read(8*1024**2),b''):
            dst.write(block);digest.update(block);copied+=len(block)
            if copied//1024**3!=reported:
                reported=copied//1024**3
                row=dict(phase='copy',bytes=copied,total=before.st_size,seconds=time.monotonic()-started)
                write(CONTROL/'legacy-baseline-offload-progress.json',row);print(json.dumps(row),flush=True)
        dst.flush();os.fsync(dst.fileno())
    assert copied==before.st_size and partial.stat().st_size==before.st_size
    os.replace(partial,DEST)
    verify=hashlib.sha256();read=0;reported=-1
    with DEST.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            verify.update(block);read+=len(block)
            if read//1024**3!=reported:
                reported=read//1024**3
                row=dict(phase='final_SHA_readback',bytes=read,total=before.st_size,seconds=time.monotonic()-started)
                write(CONTROL/'legacy-baseline-offload-progress.json',row);print(json.dumps(row),flush=True)
    assert read==before.st_size and verify.hexdigest()==digest.hexdigest()
    after=SOURCE.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    assert_inactive()
    receipt=dict(passed=True,source=str(SOURCE),destination=str(DEST),bytes=read,sha256=digest.hexdigest(),
        classification='inactive legacy experiment schema reference; NOT current twohead step150',
        verified_epoch=time.time(),seconds=time.monotonic()-started)
    write(CONTROL/'legacy-baseline150-schema.json',dict(passed=True,source_archive_sha256=digest.hexdigest(),model_schema=schema))
    write(CONTROL/'legacy-baseline-offload.json',receipt)
    link=SOURCE.with_name(SOURCE.name+'.verified-archive-link')
    link.symlink_to(DEST);os.replace(link,SOURCE)
    receipt['original_path_is_verified_archive_symlink']=True
    write(CONTROL/'legacy-baseline-offload.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:traceback.print_exc();code=1
    finally:(CONTROL/'legacy-baseline-offload.exit').write_text(str(code)+'\n')
    raise SystemExit(code)
