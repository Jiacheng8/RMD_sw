#!/usr/bin/env python3
"""Read-only one-shot observer for the bound two-head A run; no Torch/GPU kernels."""
import pathlib,json,subprocess,datetime,math,os
root=pathlib.Path('/data/rdm-workspace/rdm-sets');control=root/'control'
prefix=os.environ.get('H20_CONTROL_PREFIX','twohead-lr10-A-500')
source=pathlib.Path(os.environ.get('H20_TRAIN_SOURCE','/data/rdm-workspace/RDM-twohead-lr10-v1'))
exp=root/'zimage/work_dirs/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A'
def read(p):
 return p.read_text().strip() if p.exists() else None
pid=read(control/(prefix+'.pid'));exitcode=read(control/(prefix+'.exit'))
records=[];partial=0
p=exp/'train_log.jsonl'
if p.exists():
 for line in p.read_text(errors='replace').splitlines():
  try:records.append(json.loads(line))
  except ValueError:partial+=1
rows=[v for v in records if isinstance(v,dict) and 'step' in v and 'grad_norm' in v]
fields=['seconds','step','window_step','epoch','grad_norm','lr','skipped','force','raw_mmd2','loss_total','gan/d_lr_used','gan/d_lr_next','s1/gan/d_updates','s2/gan/d_updates','s1/gan/d_skipped','s2/gan/d_skipped','s1/gan/g_active','s2/gan/g_active','s1/gan/dinov3_l/d_loss','s1/gan/qwen3vl_4b/d_loss','s2/gan/dinov3_l/d_loss','s2/gan/qwen3vl_4b/d_loss','s1/gan/dinov3_l/g_loss','s1/gan/qwen3vl_4b/g_loss','s2/gan/dinov3_l/g_loss','s2/gan/qwen3vl_4b/g_loss','pixel_gc/generator_replay_max_abs']
def nonfinite(v):
 if isinstance(v,dict):return sum(nonfinite(a) for a in v.values())
 if isinstance(v,list):return sum(nonfinite(a) for a in v)
 return int(isinstance(v,float) and not math.isfinite(v))
log=root/'launch-logs'/ (prefix+'.log');lines=log.read_text(errors='replace').splitlines() if log.exists() else []
errors=[l for l in lines if any(k in l for k in ['Traceback (most recent','CUDA out of memory','ChildFailedError','SignalException','[rank0]: RuntimeError','[rank1]: RuntimeError'])]
save_lines=lines
handoff=control/'twohead-resume200-stop-verified.json'
if prefix!='twohead-lr10-A-500' and handoff.exists() and json.loads(handoff.read_text()).get('passed'):
 prior=root/'launch-logs/twohead-lr10-A-500.log'
 save_lines=(prior.read_text(errors='replace').splitlines() if prior.exists() else [])+lines
saves=[l for l in save_lines if any(k in l for k in ['Saved resume','Saved student','Saved checkpoint','saved resume','saved student','resume state (step','checkpoint ->']) and 'formal_A_output_preflight' not in l]
stats={}
for name in ['resume.pth','step_0000500.pth']:
 f=exp/name
 if f.exists():
  st=f.stat();stats[name]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
proc=None
if pid and pathlib.Path('/proc/'+pid+'/cmdline').exists():
 cmd=pathlib.Path('/proc/'+pid+'/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 proc={'pid':int(pid),'command':cmd,'expected_source':str(source) in cmd}
probe_history=[{k:v for k,v in row.items() if k=='step' or 'drift' in k or k in ['drift_probe_rows','refreshed_rows','spatial_refreshed_rows']} for row in rows if 'drift_probe_rows' in row or 'refreshed_rows' in row]
out={'probe_history':probe_history[-4:],'probe_steps':[r['step'] for r in probe_history if 'drift_probe_rows' in r],'refresh_steps':[r['step'] for r in probe_history if 'refreshed_rows' in r],'probe_row_count_mismatches':sum(r['drift_probe_rows']!=64 for r in probe_history if 'drift_probe_rows' in r),'refresh_row_count_mismatches':sum(r['refreshed_rows']!=96 or r.get('spatial_refreshed_rows')!=96 for r in probe_history if 'refreshed_rows' in r),'D_counter_mismatches':sum(r.get(f's{k}/gan/d_updates')!=r['step'] for r in rows for k in [1,2]),'G_GAN_phase_mismatches':sum(r.get(f's{k}/gan/g_active')!=(r['step']>50) for r in rows for k in [1,2]),'unexpected_AIM_GAN_rows':sum(any('/gan/aimv2_huge/' in key for key in r) for r in rows),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':read(source/'SOURCE_COMMIT'),'wrapper':proc,'exit':exitcode,'complete_iterations':len(rows),'last_step':rows[-1]['step'] if rows else None,'all_steps_contiguous':([v['step'] for v in rows]==list(range(1,len(rows)+1))),'last_two':[{k:v.get(k) for k in fields if k in v} for v in rows[-2:]],'G_skips':sum(bool(v.get('skipped')) for v in rows),'D_skips':sum(bool(v.get(f's{k}/gan/d_skipped')) for v in rows for k in [1,2]),'nonfinite_numeric_count':sum(nonfinite(v) for v in rows),'replay_nonzero_count':sum(v.get('pixel_gc/generator_replay_max_abs',0)!=0 for v in rows),'parse_incomplete_lines':partial,'errors':errors[-8:],'save_lines':saves[-12:],'artifacts':stats,'log_bytes':log.stat().st_size if log.exists() else 0,'log_tail':[(l[:700] if len(l)>700 else l) for l in lines[-8:]],'disk_free_bytes':__import__('shutil').disk_usage(root).free,'gpus':subprocess.check_output(['nvidia-smi','-i','4,5','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)}
for name in ['command.json','preflight.json']:
 p=control/(prefix+'.'+name)
 out[name]=json.loads(p.read_text()) if p.exists() else None
archive_control=control/'twohead-retain50-archive'
progress=archive_control/'checkpoint-archive-progress.json'
out['checkpoint_archive_progress']=json.loads(progress.read_text()) if progress.exists() else None
out['checkpoint_archive_exit']=read(archive_control/'checkpoint-archive-worker.exit')
out['archived_weights']={}
for step in range(200,501,50):
 p=archive_control/f'checkpoint-archive-{step}.json'
 if p.exists():
  proof=json.loads(p.read_text());target=pathlib.Path(proof['destination'])
  out['archived_weights'][str(step)]=dict(proof,destination_exists=target.exists(),current_bytes=target.stat().st_size if target.exists() else None)
print(json.dumps(out,ensure_ascii=False))
