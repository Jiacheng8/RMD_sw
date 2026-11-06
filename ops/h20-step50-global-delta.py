#!/usr/bin/env python3
"""One-off CPU global FP32 displacement of checkpoint50 from official initialization.
No BF16 casts, per-layer statistics, training hooks, weight copies or GPU work.
"""
from contextlib import ExitStack
import datetime,json,math,os,pathlib,time
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from safetensors import safe_open
ROOT=pathlib.Path('/data/rdm-workspace/rdm-sets')
BASE=ROOT/'hf/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/f332072aa78be7aecdf3ee76d5c247082da564a6/transformer'
CKPT=ROOT/'zimage/work_dirs/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A/resume.pth'
OUT=ROOT/'control/twohead-lr10-step50-global-delta'
def write(name,value):
 p=OUT/name;t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(value,indent=2)+'\n');os.replace(t,p)
def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=False)
 started=time.monotonic();before=CKPT.stat()
 state=torch.load(CKPT,map_location='cpu',weights_only=False,mmap=True)
 assert state['step']==50 and state['num_steps']==2
 model=state['model'];index=json.loads((BASE/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
 assert len(model)==521 and sum(t.numel() for t in model.values())==6154908736
 assert {k.removeprefix('transformer.') for k in model}==set(index)
 n=changed=0;base_sq=delta_sq=abs_sum=max_abs=0.
 with ExitStack() as stack:
  shards={f:stack.enter_context(safe_open(BASE/f,framework='pt',device='cpu')) for f in set(index.values())}
  for number,(name,value) in enumerate(model.items(),1):
   key=name.removeprefix('transformer.');initial=shards[index[key]].get_tensor(key)
   assert initial.dtype==value.dtype==torch.float32 and initial.shape==value.shape
   a=initial.reshape(-1);b=value.reshape(-1)
   for lo in range(0,a.numel(),1048576):
    x=a[lo:lo+1048576];y=b[lo:lo+1048576]
    assert bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
    xd=x.double();d=(y-x).double()
    n+=d.numel();changed+=int(torch.count_nonzero(d))
    base_sq+=float(torch.dot(xd,xd));delta_sq+=float(torch.dot(d,d))
    abs_sum+=float(d.abs().sum());max_abs=max(max_abs,float(d.abs().max()))
   if number%50==0 or number==len(model):
    progress={'completed_tensors':number,'total_tensors':len(model),'elements_compared':n,'elapsed_seconds':time.monotonic()-started}
    write('progress.json',progress);print(json.dumps(progress),flush=True)
 after=CKPT.stat()
 assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'CPU_only':True,
  'checkpoint_step':50,'checkpoint':str(CKPT),'checkpoint_bytes':before.st_size,'baseline':str(BASE),
  'elements':n,'tensors':len(model),'changed_fp32_elements':changed,'changed_fp32_fraction':changed/n,
  'relative_L2_delta':math.sqrt(delta_sq/base_sq),'relative_L2_percent':100*math.sqrt(delta_sq/base_sq),
  'mean_abs_delta':abs_sum/n,'rms_delta':math.sqrt(delta_sq/n),'max_abs_delta':max_abs,
  'baseline_L2':math.sqrt(base_sq),'delta_L2':math.sqrt(delta_sq),'seconds':time.monotonic()-started,
  'comparison_scope':'accumulated checkpoint50 versus original FP32 initialization, not per-iteration update',
  'no_BF16_statistics':True,'no_per_layer_statistics':True,'training_modified':False}
 write('summary.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
