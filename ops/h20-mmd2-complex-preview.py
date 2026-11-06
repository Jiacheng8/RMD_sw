#!/usr/bin/env python3
"""Exact new-MMD2 checkpoints0,50..450; eight multi-constraint stress cases/two seeds/native1024."""
import argparse
import datetime
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data/rdm-workspace/rdm-sets')
CODE=Path(__file__).resolve().parents[1]
TRAIN=Path('/data/rdm-workspace/RDM-dino-qwenvl-mmd2-v7')
TRAIN_SHA='ad5dddc428a32591fedf9b034bf5f1639ced0514'
EXPERIMENT='zimage-2step-1024-dino-qwenvl-mmd2-from0'
ARCHIVE=Path('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints')/EXPERIMENT
BASE=ROOT/'visualizations'/EXPERIMENT/'complex-cases-v1'
SAMPLING=dict(sigmas=[1.,.75],renoise=True,detach_between_steps=True)
STAGES={6:[0,450,50,100,150],7:[200,250,300,350,400]}


def write(path,value):
    temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(step):
    proof=ROOT/'control/dino-qwenvl-mmd2-from0-archive'/f'checkpoint-archive-{step}.json'
    receipt=json.loads(proof.read_text())
    path=ARCHIVE/f'step_{step:07d}.pth'
    assert receipt['passed'] and receipt['step']==step and receipt['final_named_sha_readback']
    assert receipt['destination']==str(path) and path.stat().st_size==receipt['bytes']
    assert receipt['model_tensors']==521 and receipt['model_numel']==6154908736
    return path,receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,choices=[6,7],required=True)
    args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(args.gpu)
    assert (TRAIN/'SOURCE_COMMIT').read_text().strip()==TRAIN_SHA
    BASE.mkdir(parents=True,exist_ok=True)
    cases_path=CODE/'ops/mmd2-complex-cases-v1.json'
    spec=json.loads(cases_path.read_text());cases=spec['cases']
    assert len(cases)==8 and spec['revision']=='mmd2-complex-v1'
    assert len({c['ID'] for c in cases})==8 and all(c['ID']>=9001 and len(c['checks'])>=8 for c in cases)
    with (ROOT.parent/f'rdm-preview-gpu{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        used=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
            '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert used<1024,'Preview GPU occupied: never touch foreign processes'
        sys.path[:0]=[str(TRAIN/'z-image'),str(TRAIN)]
        import torch
        from PIL import Image
        from zimage_distill.model import ZImageStudent,ZImageVAE,QwenPromptEncoder,noise_for
        from rdm.utils.io import save_uint8_png
        from h20_portrait_checkpoint import load_archived_state,validate_state
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.95)
        text=QwenPromptEncoder(device='cuda',cache_size=8,batch=2)
        token_counts=[]
        for c in cases:
            wrapped=text.tokenizer.apply_chat_template([dict(role='user',content=c['prompt_en'])],
                tokenize=False,add_generation_prompt=True,enable_thinking=True)
            length=len(text.tokenizer(wrapped,truncation=False).input_ids)
            assert length<=512,f"Case{c['ID']} would truncate"
            token_counts.append(dict(ID=c['ID'],tokens=length))
        caps=[c.cpu() for c in text([c['prompt_en'] for c in cases])]
        del text
        gc.collect();torch.cuda.empty_cache()
        model=ZImageStudent(1024,param_dtype=torch.bfloat16,gradient_checkpointing=False).to('cuda').eval()
        vae=ZImageVAE(device='cuda',gradient_checkpointing=False)
        jobs=[(i,seed) for i in range(8) for seed in [0,1]]
        for step in STAGES[args.gpu]:
            folder=BASE/f'step_{step:07d}'
            folder.mkdir(exist_ok=True)
            assert not (folder/'preview-verified.json').exists(),'Preserve already completed stage'
            receipt=None;path=None
            if step:
                path,receipt=checkpoint(step)
                def progress(done,total):
                    event=dict(step=step,phase='verified_checkpoint_RAM_stream',bytes=done,total_bytes=total,epoch=time.time())
                    write(folder/'progress.json',event)
                    print(json.dumps(event),flush=True)
                state,headroom=load_archived_state(path,receipt,progress)
                validate_state(state,step,archived=True)
                model.load_state_dict(state['model'],strict=True)
                del state
                gc.collect()
            # Stage0 precedes every load_state_dict on GPU6: official initial weights.
            assert step!=0 or args.gpu==6
            torch.cuda.reset_peak_memory_stats()
            started=time.time()
            with torch.no_grad():
                for lo in range(0,len(jobs),4):
                    batch=jobs[lo:lo+4]
                    noise=torch.stack([torch.stack([noise_for(cases[i]['ID'],seed+1000*k,
                        (16,128,128),'cuda') for k in range(2)]) for i,seed in batch])
                    latent=model(noise,[caps[i].to('cuda') for i,seed in batch],SAMPLING)[-1]
                    pixels=vae.decode(latent)
                    assert bool(torch.isfinite(pixels).all())
                    for (i,seed),image in zip(batch,pixels):
                        dest=folder/f"case{cases[i]['ID']:04d}-seed{seed}.png"
                        temp=folder/('.'+dest.name)
                        save_uint8_png(image,str(temp));temp.replace(dest)
                    del noise,latent,pixels
                    event=dict(step=step,phase='actual_images_saved',images=lo+len(batch),total=16,epoch=time.time())
                    write(folder/'progress.json',event);print(json.dumps(event),flush=True)
            files=[]
            for image in sorted(folder.glob('case*.png')):
                with Image.open(image) as im:im.load();assert im.mode=='RGB' and im.size==(1024,1024)
                files.append(dict(path=str(image),bytes=image.stat().st_size,sha256=sha(image)))
            assert len(files)==16
            proof=dict(passed=True,actual_step=step,checkpoint_step=step,experiment=EXPERIMENT,
                training_commit=TRAIN_SHA,preview_code_commit=(CODE/'SOURCE_COMMIT').read_text().strip(),
                checkpoint=str(path) if path else None,checkpoint_sha256=receipt['sha256'] if receipt else None,
                checkpoint_RAM_stream_fullSHA_verified=bool(step),
                initial_weights='official pretrained Z-Image-Turbo f332072aa78be7aecdf3ee76d5c247082da564a6' if step==0 else None,
                case_suite=spec['suite'],case_origin=spec['dataset'],cases_sha256=sha(cases_path),sampling=SAMPLING,image_size=1024,seeds=[0,1],batch_size=4,
                attention='native',sampler_autocast=False,decode_autocast=False,prompt_token_counts=token_counts,
                physical_gpu=args.gpu,files=files,seconds=time.time()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write(folder/'preview-verified.json',proof)
            print(json.dumps(dict(event='stage_complete',step=step,images=16)),flush=True)


if __name__=='__main__':
    main()
