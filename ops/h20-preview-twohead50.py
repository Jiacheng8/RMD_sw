#!/usr/bin/env python3
"""Actual step50 preview; bounded VRAM, unchanged trainer and published sampler."""
import argparse,datetime,fcntl,json,os,pathlib,runpy,subprocess,sys,time
ROOT=pathlib.Path('/data/rdm-workspace/rdm-sets')
SOURCE=pathlib.Path('/data/rdm-workspace/RDM-twohead-lr10-v1')
CKPT=ROOT/'zimage/work_dirs/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A/resume.pth'
OUT=ROOT/'visualizations/twohead-lr10-A/step_0000050'
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True,choices=[6,7]);ap.add_argument('--cap-gib',type=float,default=32.);a=ap.parse_args()
 assert sys.platform.startswith('linux') and os.environ['CUDA_VISIBLE_DEVICES']==str(a.gpu)
 assert 24<=a.cap_gib<=32
 assert (SOURCE/'SOURCE_COMMIT').read_text().strip()=='a25e4c42794b13d5c881d3f1994944929b019190'
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'preview-verified.json').exists()
 with (ROOT.parent/f'rdm-preview-gpu{a.gpu}.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);checks=[]
  for _ in range(2):
   used,total,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
   assert total-used>=int((a.cap_gib+16)*1024),'Insufficient VRAM headroom'
   checks.append({'used_mib':used,'total_mib':total,'utilization':util});time.sleep(2)
  import torch
  torch.set_num_threads(2)
  state=torch.load(CKPT,map_location='cpu',weights_only=False,mmap=True)
  assert state['step']==50 and state['num_steps']==2
  assert all(c['contract']['head_order']==['dinov3_l','qwen3vl_4b'] for c in state['critics'])
  sampling=state['sampling_args'];assert sampling['sigmas']==[1.0,.75] and sampling['renoise']
  before=CKPT.stat();del state
  props=torch.cuda.get_device_properties(0)
  torch.cuda.set_per_process_memory_fraction(a.cap_gib*1024**3/props.total_memory,device=0)
  torch.cuda.reset_peak_memory_stats()
  receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint':str(CKPT),'checkpoint_step':50,'checkpoint_inode':before.st_ino,'checkpoint_bytes':before.st_size,'training_commit':(SOURCE/'SOURCE_COMMIT').read_text().strip(),'physical_gpu':a.gpu,'allocator_cap_gib':a.cap_gib,'preflight':checks,'sharing_existing_compute':True,'sampling':sampling,'image_size':1024,'prompts':str(SOURCE/'ops/spatial-preview-prompts.jsonl'),'seeds':[0,1],'batch':2,'training_gpus_untouched':[4,5]}
  (OUT/'preview-command.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
  sys.path[:0]=[str(SOURCE/'z-image'),str(SOURCE)]
  sys.argv=['zimage_distill.sample','--prompts',str(SOURCE/'ops/spatial-preview-prompts.jsonl'),'--out',str(OUT),'--ckpt',str(CKPT),'--num-steps','2','--seeds','2','--batch','2','--img-size','1024']
  try:runpy.run_module('zimage_distill.sample',run_name='__main__')
  except SystemExit as e:
   if e.code not in (None,0):raise
  after=CKPT.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Input checkpoint changed'
  from PIL import Image,ImageDraw
  for i in range(12):
   for seed in (0,1):
    with Image.open(OUT/f'{i:05d}_s{seed}.png') as im:assert im.size==(1024,1024)
  baseline=ROOT/'visualizations/spatial-A/step_0000000'
  for seed in (0,1):
   sheet=Image.new('RGB',(1152,1680),'white');draw=ImageDraw.Draw(sheet);draw.text((10,12),f'Fresh two-head A / iteration50 / 2-step1024 / seed{seed}',fill='black')
   for i in range(12):
    x=(i%3)*384;y=40+(i//3)*410
    with Image.open(OUT/f'{i:05d}_s{seed}.png') as im:sheet.paste(im.convert('RGB').resize((384,384),Image.Resampling.LANCZOS),(x,y))
    draw.text((x+8,y+388),f'P{i+1:02d} / iter50',fill='black')
   sheet.save(OUT/f'contact-seed{seed}.jpg',quality=93)
   chosen=[0,1,3,6,9,11];compare=Image.new('RGB',(768,len(chosen)*412+40),'white');dr=ImageDraw.Draw(compare)
   dr.text((8,12),f'Same prompts/seeds / initial0 | fresh iteration50 / seed{seed}',fill='black')
   for row,i in enumerate(chosen):
    for col,(folder,tag) in enumerate([(baseline,'initial0'),(OUT,'iteration50')]):
     with Image.open(folder/f'{i:05d}_s{seed}.png') as im:compare.paste(im.convert('RGB').resize((384,384),Image.Resampling.LANCZOS),(col*384,row*412+40))
     dr.text((col*384+8,row*412+428),f'P{i+1:02d} / {tag}',fill='black')
   compare.save(OUT/f'compare-initial50-seed{seed}.jpg',quality=94)
  receipt.update({'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'images':24,'passed':True,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'no_checkpoint_copied_or_pinned':True})
  (OUT/'preview-verified.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
