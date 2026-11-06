#!/usr/bin/env python3
"""After clean500 only: full CPU resume integrity and exact final student equivalence.

No training hooks or per-layer telemetry. All weight reads stay on the remote host.
The final archived student is SHA-checked in RAM; the rolling resume is local mmap.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT=Path('/data/rdm-workspace/rdm-sets')
CONTROL=ROOT/'control'
PREFIX='dino-qwenvl-mmd2-from0'
EXP=ROOT/'zimage/work_dirs/zimage-2step-1024-dino-qwenvl-mmd2-from0'
NAMES=['dinov3_l','qwen3vl_4b']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def audit(source,source_sha):
    assert (CONTROL/(PREFIX+'.exit')).read_text().strip()=='0','Full audit requires actual successful workflow exit'
    assert json.loads((CONTROL/(PREFIX+'-formal.exit.json')).read_text())['actual_exit']==0
    complete=json.loads((CONTROL/(PREFIX+'-training-complete.json')).read_text())
    assert complete['passed'] and complete['steps']==500 and complete['source_commit']==source_sha
    assert (source/'SOURCE_COMMIT').read_text().strip()==source_sha
    command=json.loads((CONTROL/(PREFIX+'-formal.command.json')).read_text())
    assert command['source']==str(source) and command['source_commit']==source_sha and command['CUDA_VISIBLE_DEVICES']=='4,5'
    assert not any('resume_from' in x or 'load_from' in x for x in command['command'])
    assert str(source/'z-image/configs/zimage_2step_1024_dino_qwenvl_mmd2_from0.yaml') in command['command']
    import numpy as np
    import torch
    import yaml
    torch.set_num_threads(1)
    from h20_portrait_checkpoint import load_archived_state
    from rdm.sw_lmmd.adaptive_d_lr import AdaptiveDLRConfig,AdaptiveDLRController
    from rdm.sw_lmmd.window_schedule import SlidingWindowSchedule,build_row_order
    cfg=yaml.safe_load((source/'z-image/configs/zimage_2step_1024_dino_qwenvl_mmd2_from0.yaml').read_text())
    assert cfg['encoders']==NAMES and cfg['spatial_gan']['global_gan_encoders']==[]
    assert cfg['lr']==2.83e-5 and cfg['gan']['lr']==1e-5 and cfg['gan']['loss']=='ns'
    assert cfg['steps']==500 and cfg['save_freq']==cfg['resume_every']==50
    assert not cfg.get('load_from') and not cfg.get('resume_from')
    rows=[json.loads(s) for s in (EXP/'train_log.jsonl').read_text().splitlines() if s.strip()]
    assert [r['step'] for r in rows]==list(range(1,501))
    controller=AdaptiveDLRController(base_lr=1e-5,cfg=AdaptiveDLRConfig.from_mapping(cfg['adaptive_d_lr']),heads=NAMES)
    actions=[]
    for row in rows:
        assert row['skipped'] is False and not any(row[f's{k}/gan/d_skipped'] for k in [1,2])
        assert not any('siglip' in key or 'aimv2' in key for key in row)
        active=row['step']>50
        assert all(bool(row[f's{k}/gan/g_active'])==active for k in [1,2])
        heads={}
        for k in [1,2]:
            for name in NAMES:
                prefix=f's{k}/gan/{name}'
                heads[f's{k}/{name}']=dict(d_loss=row[prefix+'/d_loss'],
                    g_loss=row.get(prefix+'/g_loss') if active else None,g_proxy=row.get(prefix+'/g_proxy_pre_d_window'))
                assert math.isfinite(row[f's{k}/{name}/mmd2'])
        assert row['gan/d_lr_used']==controller.current_lr
        decision=controller.observe(row['step'],heads,generator_active=active,valid=True)
        assert decision==row['gan/d_lr_control'] and row['gan/d_lr_next']==controller.current_lr
        if decision['changed']:actions.append({k:v for k,v in decision.items() if k!='statistics'})
    log=(ROOT/'launch-logs'/(PREFIX+'-formal.log')).read_text(errors='replace')
    assert 'resume state (step 500)' in log and 'checkpoint -> '+str(EXP/'step_0000500.pth') in log
    assert 'done: 500 steps' in log
    archive_control=CONTROL/(PREFIX+'-archive')
    archived=json.loads((archive_control/'checkpoint-archive-complete.json').read_text())
    assert archived['passed'] and [r['step'] for r in archived['weights']]==list(range(50,501,50))
    for receipt in archived['weights']:
        assert receipt['passed'] and receipt['final_named_sha_readback'] and receipt['local_source_removed']
        assert Path(receipt['destination']).stat().st_size==receipt['bytes']
    resume_path=EXP/'resume.pth';before=resume_path.stat()
    assert before.st_mtime>=command['started_epoch'] and before.st_size>24_000_000_000
    resume=torch.load(resume_path,map_location='cpu',weights_only=False,mmap=True)
    proof=archived['weights'][-1]
    student,headroom=load_archived_state(Path(proof['destination']),proof)
    for value in [resume,student]:
        assert value['step']==value['schedule']['step']==500 and value['world_size']==2 and value['num_steps']==2
        assert value['sampling_args']==dict(sigmas=[1.,.75],renoise=True,detach_between_steps=True)
        assert len(value['model'])==521 and sum(v.numel() for v in value['model'].values())==6154908736
        assert all(v.dtype==torch.float32 for v in value['model'].values())
        assert value['adaptive_d_lr']==controller.state_dict()
    assert resume['model'].keys()==student['model'].keys()
    for name,value in resume['model'].items():
        other=student['model'][name];assert value.shape==other.shape
        assert torch.equal(value,other),'Final full resume/student weights differ: '+name
    del student
    assert len(resume['critics'])==2
    for critic in resume['critics']:
        contract=critic['contract'];assert critic['updates']==500
        assert contract['head_order']==NAMES and contract['global']==[] and contract['encoder_weight']==.5
        assert contract['loss']=='ns' and contract['lr']==1e-5
        opt=critic['optimizer'];assert opt['param_groups'] and opt['state']
        assert all(g['lr']==controller.current_lr for g in opt['param_groups'])
        assert all(int(float(s['step']))==500 for s in opt['state'].values())
    schedule=SlidingWindowSchedule(build_row_order(125808,seed=3407),window_size=128,stride=32)
    schedule.load_state_dict(resume['schedule']);expected_ids=schedule._window_at(499).all_ids
    expected={f's{k}/{name}':(128,dim) for k in [1,2] for name,dim in [('dinov3_l',3584),('qwen3vl_4b',5120)]}
    spatial={f's{k}/patch/{name}':(128,channels,grid,grid) for k in [1,2]
        for name,channels,grid in [('dinov3_l',1024,64),('qwen3vl_4b',2560,32)]}
    common=None
    for key,shapes,dtype in [('cache',expected,torch.float32),('spatial_cache',spatial,torch.bfloat16)]:
        assert set(resume[key])==set(shapes)
        for name,entry in resume[key].items():
            assert tuple(entry['features'].shape)==shapes[name] and entry['features'].dtype==dtype
            assert np.array_equal(entry['row_ids'],expected_ids)
            ages=entry['generated_at_steps'];assert tuple(ages.shape)==(128,) and 0<=int(ages.min())<=int(ages.max())<=500
            if common is None:common=ages
            assert torch.equal(common,ages)
    canonical=json.loads((ROOT/'zimage/zimage-ref-v1/reference_images_patchgan/manifest.json').read_text())
    assert resume['spatial_gan_contract']['reference']==canonical['contract']
    assert resume['spatial_gan_contract']['index_sha256']==canonical['index']['sha256']
    count=0
    def finite(value):
        nonlocal count
        if isinstance(value,torch.Tensor):
            assert value.device.type=='cpu'
            if value.is_floating_point():
                flat=value.reshape(-1)
                for lo in range(0,flat.numel(),1048576):assert bool(torch.isfinite(flat[lo:lo+1048576]).all())
                count+=1
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,(list,tuple)):
            for v in value:finite(v)
        elif isinstance(value,float):assert math.isfinite(value)
    finite(resume);finite(rows)
    digest=sha(resume_path);after=resume_path.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
    return dict(passed=True,step=500,actual_formal_exit=0,actual_workflow_exit=0,source=str(source),source_commit=source_sha,
        all10_weights_final_named_SHA_verified=True,final_student_SHA=proof['sha256'],resume_sha256=digest,resume_bytes=before.st_size,
        final_resume_and_student_exact_model_equality=True,model_tensors=521,model_numel=6154908736,
        full_saved_tensors_finite=count,critic_D_Adam_updates=500,cache_row_age_alignment=True,
        adaptive_controller_all500_decisions_replayed=True,adaptive_actions=actions,current_D_lr=controller.current_lr,
        G_base_lr=2.83e-5,MMD_encoders=NAMES,GAN_encoders=NAMES,generator_Adam_saved='optimizer' in resume,
        original_FSDP_G_Adam_resume_limitation_retained=True,remote_RAM_headroom_before_load=headroom,
        completed_epoch=time.time())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-source',type=Path,required=True);p.add_argument('--training-sha',required=True)
    args=p.parse_args();assert args.training_source.is_relative_to('/data/rdm-workspace')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TMPDIR']=str(ROOT/'.cache/tmp')
    import sys
    sys.path[:0]=[str(args.training_source),str(args.training_source/'z-image')]
    result=audit(args.training_source,args.training_sha)
    target=CONTROL/(PREFIX+'-final-integrity.json');tmp=target.with_suffix('.partial');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(target)
    print(json.dumps(result),flush=True)
