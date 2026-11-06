#!/usr/bin/env python3
"""Read-only CPU comparison of complete student weights and inference BF16 casts.

No model, optimizer, GPU, training file or preview image is modified. Full
per-tensor results stay remote; summaries contain only scalar diagnostics.
"""
from __future__ import annotations

from contextlib import ExitStack
import datetime as dt
import json
import math
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open

ROOT = Path('/data/rdm-workspace/rdm-sets')
CONTROL = ROOT/'control'
OUT = CONTROL/'spatial-A-model-update-audit'
BASE = ROOT/'hf/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/f332072aa78be7aecdf3ee76d5c247082da564a6/transformer'
RUN = ROOT/'zimage/work_dirs/zimage-2step-1024-dlr1e-5-spatial-patchgan-softplus'
PATHS = {150: CONTROL/'spatial-A-user-visualization/step_0000150-model.pth',
         200: CONTROL/'spatial-A-user-visualization/step_0000200-model.pth',
         300: RUN/'resume.pth'}
PAIRS = [(0,150), (0,200), (0,300), (150,200), (200,300)]
CHUNK = 2**20


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def empty():
    return dict(elements=0, changed_fp32=0, changed_bf16=0,
                sum_abs_delta=0., sum_sq_delta=0., max_abs_delta=0.)


def add(total, value):
    for name in total:
        if name == 'max_abs_delta': total[name] = max(total[name], value[name])
        else: total[name] += value[name]


def finish(stats, left_norm):
    count = stats['elements']
    return {**stats,
            'changed_fp32_fraction': stats['changed_fp32']/count,
            'changed_bf16_fraction': stats['changed_bf16']/count,
            'mean_abs_delta': stats['sum_abs_delta']/count,
            'rms_delta': math.sqrt(stats['sum_sq_delta']/count),
            'relative_L2_delta': math.sqrt(stats['sum_sq_delta']/left_norm) if left_norm else None}


def group_for(key):
    names = key.split('.')
    if len(names)>2 and names[1] in ['layers','noise_refiner','context_refiner']:
        return '.'.join(names[:3])
    return '.'.join(names[:2])


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Audit must be CPU-only'
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    index = json.loads((BASE/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    before = {n: (p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns) for n,p in PATHS.items()}
    states = {n: torch.load(p,map_location='cpu',weights_only=False,mmap=True) for n,p in PATHS.items()}
    assert all(s['step']==n and s['num_steps']==2 for n,s in states.items())
    models = {n: s['model'] for n,s in states.items()}
    keys = list(models[150])
    assert len(keys)==521 and sum(t.numel() for t in models[150].values())==6154908736
    assert all(set(m)==set(keys) for m in models.values())
    assert {k.removeprefix('transformer.') for k in keys}==set(index)
    total = {f'{a}->{b}': empty() for a,b in PAIRS}
    norms = {n:0. for n in [0,*models]}
    groups = {}; group_norms = {}; tensors = []; storage_dtypes = set()
    with ExitStack() as stack:
        files = {name:stack.enter_context(safe_open(BASE/name,framework='pt',device='cpu'))
                 for name in sorted(set(index.values()))}
        for number,key in enumerate(keys,1):
            official_key=key.removeprefix('transformer.')
            baseline=files[index[official_key]].get_tensor(official_key)
            storage_dtypes.add(str(baseline.dtype))
            assert all(t[key].shape==baseline.shape for t in models.values()), key
            values={0:baseline.reshape(-1), **{n:m[key].reshape(-1) for n,m in models.items()}}
            group=group_for(key)
            groups.setdefault(group,{f'{a}->{b}':empty() for a,b in PAIRS})
            group_norms.setdefault(group,{n:0. for n in norms})
            per={f'{a}->{b}':empty() for a,b in PAIRS}; per_norm={n:0. for n in norms}
            for lo in range(0,baseline.numel(),CHUNK):
                chunks={n:v[lo:lo+CHUNK].float() for n,v in values.items()}
                casts={n:v.bfloat16() for n,v in chunks.items()}
                for n,v in chunks.items():
                    assert bool(torch.isfinite(v).all()), (key,n)
                    per_norm[n]+=float(v.double().square().sum())
                for a,b in PAIRS:
                    delta=(chunks[b]-chunks[a]).double()
                    stat={'elements':delta.numel(), 'changed_fp32':int(torch.count_nonzero(delta)),
                          'changed_bf16':int(torch.count_nonzero(casts[a]!=casts[b])),
                          'sum_abs_delta':float(delta.abs().sum()),
                          'sum_sq_delta':float(delta.square().sum()),
                          'max_abs_delta':float(delta.abs().max())}
                    add(per[f'{a}->{b}'],stat)
            for n,v in per_norm.items():
                norms[n]+=v;group_norms[group][n]+=v
            row={'name':key,'shape':list(baseline.shape),'pairs':{}}
            for a,b in PAIRS:
                pair=f'{a}->{b}';add(total[pair],per[pair]);add(groups[group][pair],per[pair])
                row['pairs'][pair]=finish(per[pair],per_norm[a])
            tensors.append(row)
            if number%10==0 or number==len(keys):
                progress={'utc':utc(),'completed_tensors':number,'total_tensors':len(keys),
                          'elements_compared':total['0->300']['elements'],
                          'changed_fp32_so_far':total['0->300']['changed_fp32'],
                          'changed_bf16_so_far':total['0->300']['changed_bf16'],
                          'elapsed_seconds':time.monotonic()-started}
                atomic_json(OUT/'progress.json',progress)
                print(json.dumps(progress),flush=True)
    assert all(before[n]==(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns) for n,p in PATHS.items()), 'Input checkpoint replaced during audit'
    pair_results={f'{a}->{b}':finish(total[f'{a}->{b}'],norms[a]) for a,b in PAIRS}
    for pair,result in pair_results.items():
        result['changed_tensors']=sum(t['pairs'][pair]['changed_fp32']>0 for t in tensors)
        result['changed_tensors_after_bf16']=sum(t['pairs'][pair]['changed_bf16']>0 for t in tensors)
    summary={'utc':utc(),'passed':True,'CPU_only':True,'complete_tensor_count':len(keys),
             'complete_parameter_elements':total['0->300']['elements'],
             'baseline_transformer':str(BASE),'baseline_storage_dtypes':sorted(storage_dtypes),
             'checkpoints':{str(n):str(p) for n,p in PATHS.items()},
             'checkpoint_saved_steps':{str(n):states[n]['step'] for n in states},
             'pairs':pair_results,'model_L2_norm':{str(n):math.sqrt(v) for n,v in norms.items()},
             'groups':{g:{f'{a}->{b}':finish(stats[f'{a}->{b}'],group_norms[g][a]) for a,b in PAIRS}
                       for g,stats in groups.items()},
             'elapsed_seconds':time.monotonic()-started,'training_modified':False,
             'BF16_is_sampler_parameter_cast':True,
             'numerical_weight_change_is_not_quality_improvement':True}
    atomic_json(OUT/'tensor-details.json',tensors)
    atomic_json(OUT/'summary.json',summary)
    print(json.dumps({'complete':True,'summary':str(OUT/'summary.json'),'pairs':pair_results}),flush=True)


if __name__=='__main__':
    main()
