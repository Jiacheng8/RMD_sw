#!/usr/bin/env python3
"""Build MMD's two-encoder view from the already verified local canonical teacher PNGs.

Only a bounded3.5GiB visual encoder may share each evaluation GPU. Source stores
and the active350 evaluation are immutable. Metadata publishes after complete
image coverage, source alignment, pixel-gradient and fixed-bandwidth validation.
"""
import argparse
import concurrent.futures
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1]
ROOT=Path('/data/rdm-workspace/rdm-sets')
PACK=ROOT/'zimage/zimage-ref-v1'
SOURCE=PACK/'reference_store_qwen'
DEST=PACK/'reference_store_dino_qwenvl'
WORK=PACK/'reference_store_dino_qwenvl.partial'
IMAGE=PACK/'reference_images_patchgan'
NAMES=['dinov3_l','qwen3vl_4b']
sys.path[:0]=[str(CODE),str(CODE/'z-image')]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_name(path.name+f'.partial-{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def worker(rank):
    import numpy as np
    import torch
    from PIL import Image
    from rdm.data.imagenet import default_transform
    from rdm.representation.backbones.qwen3_vl_backbone import Qwen3VLBackbone
    from rdm.sw_lmmd.reference_images import ReferenceImageStore
    gpu=rank+4
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    torch.set_num_threads(2)
    budget=int(3.5*1024**3)
    torch.cuda.set_per_process_memory_fraction(budget/torch.cuda.get_device_properties(0).total_memory)
    table=np.load(WORK/'encoder_features/qwen3vl_4b.npy',mmap_mode='r+')
    count=len(table);start=count*rank//2;end=count*(rank+1)//2
    progress_path=WORK/f'rank{rank}-progress.json'
    prior=json.loads(progress_path.read_text()) if progress_path.exists() else None
    next_row=prior['next_row'] if prior else start
    assert start<=next_row<=end
    if prior:
        lo=prior['last_committed_start'];hi=prior['next_row']
        assert hashlib.sha256(np.asarray(table[lo:hi]).tobytes()).hexdigest()==prior['last_chunk_sha256']
    images=ReferenceImageStore(IMAGE)
    transform=default_transform(512)
    def decode(png):
        with Image.open(io.BytesIO(png)) as image:return transform(image.convert('RGB'))
    model=Qwen3VLBackbone(device='cuda')
    contract=model.feature_contract()
    assert contract['dim']==2560 and contract['weight_dtype']=='torch.float32' and not contract['l2_normalize']
    contract_path=WORK/f'rank{rank}-model-contract.json'
    if contract_path.exists():assert json.loads(contract_path.read_text())==contract
    write(contract_path,contract)
    if rank==0:
        pixels=decode(images.read_pngs([0])[0]).unsqueeze(0).to('cuda').requires_grad_(True)
        with torch.autocast('cuda',dtype=torch.bfloat16):value=model(pixels).float().square().mean()
        gradient=torch.autograd.grad(value,pixels)[0]
        assert bool(torch.isfinite(gradient).all()) and bool(gradient.abs().sum()>0)
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        write(WORK/'pixel-gradient-verified.json',dict(passed=True,finite_nonzero_input_gradient=True,frozen_weights=True))
        del pixels,value,gradient;gc.collect();torch.cuda.empty_cache()
    batch=8;limit_batch=32;committed=next_row
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool,torch.no_grad():
        while next_row<end:
            ids=list(range(next_row,min(end,next_row+batch)))
            decoded=list(pool.map(decode,images.read_pngs(ids)))
            pixels=torch.stack(decoded).to('cuda')
            torch.cuda.reset_peak_memory_stats()
            try:
                with torch.autocast('cuda',dtype=torch.bfloat16):values=model(pixels).float()
            except torch.cuda.OutOfMemoryError as error:
                error.__traceback__=None
                assert batch>1,'Reference encoder OOM at batch1 within its bounded budget'
                batch=max(1,batch//2);limit_batch=batch;del pixels,decoded;gc.collect();torch.cuda.empty_cache();continue
            assert tuple(values.shape)==(len(ids),2560) and bool(torch.isfinite(values).all())
            table[ids]=values.cpu().numpy().astype(np.float16)
            peak=torch.cuda.max_memory_allocated();next_row+=len(ids)
            del values,pixels,decoded
            if next_row-committed>=256 or next_row==end:
                table.flush()
                proof=dict(rank=rank,physical_GPU=gpu,start=start,end=end,next_row=next_row,
                    completed=next_row-start,total=end-start,last_committed_start=committed,
                    last_chunk_sha256=hashlib.sha256(np.asarray(table[committed:next_row]).tobytes()).hexdigest(),
                    batch=batch,peak_allocated_bytes=peak,allocator_budget_bytes=budget,utc=time.time())
                write(progress_path,proof);print(json.dumps(proof),flush=True);committed=next_row
            if batch<limit_batch and peak*2<.8*budget:batch=min(limit_batch,batch*2)
    images.close()
    assert next_row==end
    write(WORK/f'rank{rank}-complete.json',dict(passed=True,rank=rank,start=start,end=end,rows=end-start,
        feature_sha256=hashlib.sha256(np.asarray(table[start:end]).tobytes()).hexdigest()))


def dino_audit():
    import numpy as np
    import torch
    from PIL import Image
    from rdm.data.imagenet import default_transform
    from rdm.representation.backbones import build_backbone
    from rdm.representation.registry import by_name
    from rdm.sw_lmmd.reference_images import ReferenceImageStore
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4'
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(3.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    images=ReferenceImageStore(IMAGE);transform=default_transform(512)
    model=build_backbone(by_name('dinov3_l'),'cuda')
    stored=np.load(SOURCE/'encoder_features/dinov3_l.npy',mmap_mode='r')
    ids=np.random.default_rng(0).choice(len(stored),16,replace=False).tolist();results=[]
    for lo in range(0,len(ids),4):
        chosen=ids[lo:lo+4];pixels=[]
        for png in images.read_pngs(chosen):
            with Image.open(io.BytesIO(png)) as image:pixels.append(transform(image.convert('RGB')))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=model(torch.stack(pixels).to('cuda')).float()
        expected=torch.from_numpy(np.asarray(stored[chosen],dtype=np.float32).copy()).to('cuda')
        relative=(actual-expected).norm(dim=1)/expected.norm(dim=1).clamp_min(1e-12)
        cosine=torch.nn.functional.cosine_similarity(actual,expected)
        assert bool(torch.isfinite(actual).all()) and float(relative.max())<=.03 and float(cosine.min())>=.999
        results.append(dict(rows=chosen,max_relative_l2=float(relative.max()),min_cosine=float(cosine.min())))
    images.close();write(WORK/'dino-source-alignment.json',dict(passed=True,checks=results))


def parent():
    import numpy as np
    import torch
    from rdm.sw_lmmd.reference_images import ReferenceImageStore
    if DEST.exists():
        assert json.loads((DEST/'reference-verified.json').read_text())['passed'];return
    torch.set_num_threads(2)
    source=json.loads((SOURCE/'metadata.json').read_text());image_manifest=json.loads((IMAGE/'manifest.json').read_text())
    assert source['num_rows']==image_manifest['num_rows']==125808 and source['text_dim']==2560
    assert image_manifest['complete'] and image_manifest['available_rows']==125808
    assert source['prompt_hash']==image_manifest['contract']['prompt_hash'] and source['render_hash']==image_manifest['contract']['render_hash']
    for name in ['prompt_ids','row_order']:
        assert sha(SOURCE/(name+'.npy'))==image_manifest['contract'][name+'_sha256']
    ReferenceImageStore(IMAGE).close()
    (WORK/'encoder_features').mkdir(parents=True,exist_ok=True)
    build_contract=dict(builder_sha256=sha(__file__),
        adapter_sha256=sha(CODE/'rdm/representation/backbones/qwen3_vl_backbone.py'),
        preprocessing_sha256=sha(CODE/'rdm/data/imagenet.py'),canonical_index_sha256=image_manifest['index']['sha256'],
        source_metadata_sha256=sha(SOURCE/'metadata.json'),names=NAMES,rows=125808,dim=2560,storage='float16 raw image features')
    if (WORK/'build-contract.json').exists():assert json.loads((WORK/'build-contract.json').read_text())==build_contract,'Changed partial build contract; preserve old partial before a new attempt'
    write(WORK/'build-contract.json',build_contract)
    path=WORK/'encoder_features/qwen3vl_4b.npy'
    if not path.exists():
        table=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(125808,2560));table[:]=np.nan;table.flush();del table
    command_dir=WORK/'commands';command_dir.mkdir(exist_ok=True)
    children=[]
    def stop(signum,frame):raise KeyboardInterrupt('Reference preparation stopped by operator')
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        for rank in (0,1):
            if (WORK/f'rank{rank}-complete.json').exists():continue
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(rank+4)
            command=[sys.executable,__file__,'--worker',str(rank)]
            with (WORK/f'rank{rank}.log').open('ab') as stream:
                p=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            write(command_dir/f'rank{rank}.json',dict(command=command,pid=p.pid,code_commit=(CODE/'SOURCE_COMMIT').read_text().strip()))
            children.append((rank,p))
        for rank,p in children:
            code=p.wait();write(command_dir/f'rank{rank}.exit.json',dict(actual_exit=code));assert code==0,f'Reference rank{rank} failed; preserve its command/log/exit'
    finally:
        for rank,p in children:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                write(command_dir/f'rank{rank}.exit.json',dict(actual_exit=p.returncode,operator_or_parent_cleanup=True))
    table=np.load(path,mmap_mode='r');assert table.shape==(125808,2560) and table.dtype==np.float16
    for lo in range(0,len(table),4096):assert np.isfinite(table[lo:lo+4096]).all()
    contracts=[json.loads((WORK/f'rank{k}-model-contract.json').read_text()) for k in (0,1)];assert contracts[0]==contracts[1]
    for k in (0,1):
        proof=json.loads((WORK/f'rank{k}-complete.json').read_text());assert proof['passed']
        assert hashlib.sha256(np.asarray(table[proof['start']:proof['end']]).tobytes()).hexdigest()==proof['feature_sha256']
    assert json.loads((WORK/'pixel-gradient-verified.json').read_text())['passed']
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='4'
    command=[sys.executable,__file__,'--dino-audit']
    with (WORK/'dino-audit.log').open('ab') as stream:p=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    write(command_dir/'dino-audit.exit.json',dict(actual_exit=p.returncode));assert p.returncode==0
    for name in ['text_features.npy','text_features.done.json','prompt_ids.npy','group_ids.npy','row_order.npy','rows.jsonl']:
        target=WORK/name
        if not target.exists():target.symlink_to(os.path.relpath(SOURCE/name,target.parent))
    for suffix in ['.npy','.done.json']:
        target=WORK/'encoder_features'/('dinov3_l'+suffix)
        if not target.exists():target.symlink_to(os.path.relpath(SOURCE/'encoder_features'/target.name,target.parent))
    context=json.loads((SOURCE/'z_context.json').read_text());assert context['root']=='..';write(WORK/'z_context.json',context)
    from rdm.compare.median_heuristic import median_bandwidth
    bands=json.loads((SOURCE/'bandwidths.json').read_text());scale=float(bands['dinov3_l']['sigma'])/float(bands['dinov3_l']['beta'])
    sample=np.random.default_rng(0).choice(len(table),2000,replace=False);torch.manual_seed(0)
    sigma=float(median_bandwidth(torch.from_numpy(np.asarray(table[sample],dtype=np.float32))))*.25
    assert np.isfinite(sigma) and sigma>0
    bandwidth=dict(sigma=sigma,beta=sigma/scale,s_txt=scale,sigma_scale=.25,
        estimator='median over deterministic2000 fp16 stored raw image rows',sample_ids_sha256=hashlib.sha256(sample.tobytes()).hexdigest())
    write(WORK/'bandwidths.json',{'dinov3_l':bands['dinov3_l'],'qwen3vl_4b':bandwidth})
    contract=dict(encoder='qwen3vl_4b',rows=125808,dim=2560,img_size=512,model_input_size=256,
        input_images='complete canonical PNG index; every row PNG SHA verified by reader',canonical_index_sha256=image_manifest['index']['sha256'],
        model=contracts[0],render_hash=source['render_hash'],prompt_hash=source['prompt_hash'],storage_dtype='float16',autocast='bfloat16')
    write(WORK/'encoder_features/qwen3vl_4b.done.json',dict(contract=contract,shape=[125808,2560],bandwidth=bandwidth,feature_sha256=sha(path)))
    source.update(encoder_feature_dims={'dinov3_l':1024,'qwen3vl_4b':2560},image_feature_provenance=contract)
    write(WORK/'metadata.json',source)
    from zimage_distill.store import ZImageReferenceStore
    from zimage_distill.qwen_text_store import rows_read,stats_rows
    from rdm.sw_lmmd.adversarial import reference_stats
    store=ZImageReferenceStore(str(WORK),NAMES);assert store.has_context and store.feature_image_size()==512
    required=np.unique(np.concatenate([rows_read(store.row_order(),500,128,32),stats_rows(125808,8192,[0,1])]))
    text=store.text_features(required);assert bool(torch.isfinite(text).all()) and torch.allclose(text.norm(dim=1),torch.ones(len(required)),atol=1e-3,rtol=0)
    for seed in (0,1):
        image_stats,text_stats=reference_stats(store,NAMES,8192,seed,with_text=True)
        for mean,rms in list(image_stats.values())+[text_stats]:assert bool(torch.isfinite(mean).all()) and bool(torch.isfinite(rms)) and float(rms)>0
    proof=dict(passed=True,MMD_encoders=NAMES,removed=['siglip2','aimv2_huge'],rows=125808,
        raw_image_features_finite=True,image_features_L2_normalized=False,required_text_rows=len(required),text_norm_verified=True,
        feature_sha256=sha(path),canonical_index_sha256=image_manifest['index']['sha256'],bandwidth=bandwidth,completed_epoch=time.time())
    write(WORK/'reference-verified.json',proof);WORK.rename(DEST)
    write(ROOT/'control/dino-qwenvl-mmd2-reference-verified.json',dict(proof,reference_root=str(DEST)))
    print(json.dumps(proof),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=int,choices=[0,1]);parser.add_argument('--dino-audit',action='store_true');args=parser.parse_args()
    if args.worker is not None:worker(args.worker)
    elif args.dino_audit:dino_audit()
    else:parent()
