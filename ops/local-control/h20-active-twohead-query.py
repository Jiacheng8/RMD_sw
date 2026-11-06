# Bound read-only query for the authorized resume200-v3 trajectory.
# Execute remotely through the existing owned SSH helper; no state mutation.
import os
os.environ.update(H20_CONTROL_PREFIX='twohead-lr10-resume200-500-v3',H20_TRAIN_SOURCE='/data/rdm-workspace/RDM-twohead-resume200-v3',H20_TRAIN_SHA='1e06bf49569b4ee7847a417bc1ff1445bb8ba51b')
import contextlib,io,json,runpy
with contextlib.redirect_stdout(io.StringIO()):
 ns=runpy.run_path('/data/rdm-workspace/RDM-twohead-resume200-v3/ops/h20-observe-twohead-a.py',run_name='__main__')
v=ns['out'];rows=ns['rows']
fields=['completed_step','effective_from_step','action','reason','old_lr','new_lr','changed','valid_observations','invalid_observations','action_sequence','cooldown_remaining']
v['controller_latest']={k:rows[-1]['gan/d_lr_control'].get(k) for k in fields}
v['controller_window_steps']=rows[-1]['gan/d_lr_control'].get('statistics',{}).get('window_steps')
v['controller_decisions']=[{k:r['gan/d_lr_control'].get(k) for k in fields} for r in rows if r['gan/d_lr_control']['changed']]
v['D_lr_last3']=[{'step':r['step'],'used':r['gan/d_lr_used'],'next':r['gan/d_lr_next']} for r in rows[-3:]]
v['D_lr_transition_mismatches']=sum(r['gan/d_lr_used']!=rows[i-1]['gan/d_lr_next'] for i,r in enumerate(rows) if i)
v['controller_step_mismatches']=sum(r['gan/d_lr_control']['completed_step']!=r['step'] for r in rows)
v['G_metric_scope_mismatches']=sum(r.get(f's{k}/gan/g_rows_global')!=32 or r.get(f's{k}/gan/g_metric_scope')!='global_active_rows' for r in rows if r['step']>50 for k in [1,2])
a=ns['control']/'twohead-retain50-archive'
p=a/'checkpoint-final-readback.exit'
v['final_archived_weight_readback_exit']=p.read_text().strip() if p.exists() else None
v['final_archived_weight_verified_steps']=[step for step in range(200,501,50) if (a/f'checkpoint-final-readback-{step}.json').exists()]
p=a/'checkpoint-final-readback-progress.json'
v['final_archived_weight_readback_progress']=json.loads(p.read_text()) if p.exists() else None
pf='twohead-qib-portraits-retained'
p=ns['control']/(pf+'.pid')
pid=p.read_text().strip() if p.exists() else None
proc=ns['root'].__class__('/proc/'+pid+'/cmdline') if pid else None
p=ns['control']/(pf+'.exit')
progress=ns['control']/(pf+'-progress.json')
v['portrait_follower']={'pid':int(pid) if pid else None,'command':proc.read_bytes().replace(b'\x00',b' ').decode() if proc and proc.exists() else None,'exit':p.read_text().strip() if p.exists() else None,'progress':json.loads(progress.read_text()) if progress.exists() else None}
v['portrait_completed_steps']=[]
for step in [0,150,200,250,300,350,400,450,500]:
 p=ns['root']/f'visualizations/twohead-lr10-A/qib-portraits-v1/step_{step:07d}/preview-verified.json'
 ep=ns['control']/('twohead-qib-portraits-'+str(150 if step==0 else step)+'.exit')
 if p.exists() and ep.exists() and ep.read_text().strip()=='0':
  proof=json.loads(p.read_text())
  files=proof.get('files',[]);pages=proof.get('native_comparisons',[])
  if proof.get('passed') and proof.get('actual_step')==step and len(files)==16 and (step==0 or len(pages)==4):
   if all(ns['root'].__class__(r['path']).is_file() and ns['root'].__class__(r['path']).stat().st_size==r['bytes'] for r in files+pages):v['portrait_completed_steps'].append(step)
print(json.dumps(v))

