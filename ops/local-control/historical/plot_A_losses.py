import json,pathlib,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path.cwd();d=json.loads((root/'outputs/h20-A-loss-metrics.json').read_text());rows=d['rows']
fig,axs=plt.subplots(3,2,figsize=(14,11),sharex=True)
colors=['#2266b1','#d65e22']; summary=[]
for col,(enc,label) in enumerate([('dinov3_l','DINOv3-L'),('siglip2','SigLIP2')]):
 for ri,(metric,title) in enumerate([('d_loss','Discriminator loss'),('g_loss','Generator adversarial loss (unweighted)'),('mmd2','MMD squared (monitor)')]):
  ax=axs[ri,col]
  for s,c in zip((1,2),colors):
   key=f's{s}/{enc}/mmd2' if metric=='mmd2' else f's{s}/gan/{enc}/{metric}'
   vals=[(r['step'],r[key]) for r in rows if key in r and np.isfinite(r[key])]
   x=np.array([v[0] for v in vals]);y=np.array([v[1] for v in vals]); sm=np.array([y[max(0,i-19):i+1].mean() for i in range(len(y))])
   ax.plot(x,y,color=c,alpha=.16,lw=.7);ax.plot(x,sm,color=c,lw=2,label=f's{s}: 20-step mean')
   # Compare equal windows after generator GAN activation.
   early=[r[key] for r in rows if 51<=r['step']<=70 and key in r]
   late=[r[key] for r in rows[-20:] if key in r]
   summary.append({'encoder':enc,'metric':metric,'denoise_step':s,'first_logged':int(x[0]),'early_51_70_mean':float(np.mean(early)),'latest20_mean':float(np.mean(late)),'last':float(y[-1]),'latest20_vs_51_70_percent':float((np.mean(late)/np.mean(early)-1)*100) if np.mean(early) else None})
  ax.set_title(f'{label} — {title}',fontsize=11);ax.grid(alpha=.2);ax.legend(fontsize=9);ax.set_ylabel(title.split(' (')[0])
  if ri==1:ax.axvline(51,color='#777',ls=':',lw=1)
  if ri==2:ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
for ax in axs[-1]:ax.set_xlabel('Completed training iteration')
fig.suptitle(f'A training loss history | iterations 1–{rows[-1]["step"]}',fontsize=17,fontweight='bold')
fig.text(.5,.025,'Faint: raw values. Solid: trailing 20 logged iterations. s1 / s2: denoising steps, not GPU ranks.\nGenerator GAN starts at iteration 51. MMD² is a monitoring statistic, not the complete optimized objective.',ha='center',fontsize=10,color='#444')
fig.tight_layout(rect=(0,.065,1,.955))
fig.savefig(root/'outputs/figures/A-loss-curves.png',dpi=160)
fig.savefig(root/'outputs/figures/A-loss-curves.pdf')
(root/'outputs/h20-A-loss-summary.json').write_text(json.dumps({'snapshot_utc':d['observed_utc'],'last_step':rows[-1]['step'],'series':summary},indent=2)+'\n')
print(json.dumps(summary,indent=2))
