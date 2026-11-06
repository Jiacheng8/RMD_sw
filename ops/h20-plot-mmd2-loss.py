#!/usr/bin/env python3
"""Export real fresh0 MMD2 losses, with actual encoder identities and active-G masking.

No remote access. The exporter only consumes validated existing scalar snapshots;
empty, mislabeled or historical three-encoder data cannot generate a new-run plot.
"""
import argparse
import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('mmd2_loss_data',Path(__file__).with_name('h20-mmd2-loss-data.py'))
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
COLORS={'dinov3_l':'#2563a6','qwen3vl_4b':'#d16b28'}


def plot(data,output,width=10):
    rows=contract.validate_payload(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
        'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(4,2,figsize=(12.5,14),layout='constrained')
    def series(ax,key,label,color,active=False):
        values=[r for r in rows if key in r and (not active or r['step']>=data['GAN_G_first_iteration'])]
        if not values:return False
        xs=[r['step'] for r in values];ys=[r[key] for r in values]
        ax.plot(xs,ys,color=color,alpha=.24,linewidth=.9)
        ax.plot(xs,contract.trailing_mean(ys,width),color=color,label=label,linewidth=1.8)
        if len(xs)==1:ax.scatter(xs,ys,s=14,color=color)
        return True
    for step in [1,2]:
        col=step-1
        for kind,row in [('d',0),('g',1)]:
            ax=axs[row,col];any_series=False
            for name in data['GAN_encoders']:
                any_series=series(ax,f's{step}/gan/{name}/{kind}_loss',contract.LABELS[name],COLORS[name],kind=='g') or any_series
            if any_series:ax.legend(frameon=False)
            else:ax.text(.5,.5,'GAN-G inactive until iteration 51',transform=ax.transAxes,ha='center')
            ax.set_title(f'Generation step {step}: '+('discriminator loss' if kind=='d' else 'actual adversarial G loss'))
            ax.set_ylabel('Softplus '+kind.upper()+' loss')
            if kind=='d':
                values=[r[f's{step}/gan/{name}/d_loss'] for r in rows for name in data['GAN_encoders']]
                if any(v==0 for v in values):
                    positive=[v for v in values if v>0]
                    ax.set_yscale('symlog',linthresh=min(positive)/10 if positive else 1)
                    ax.set_ylabel('Softplus D loss (symlog includes zero)')
                else:ax.set_yscale('log')
        ax=axs[2,col]
        for name in data['MMD_encoders']:
            series(ax,f's{step}/{name}/mmd2',contract.LABELS[name],COLORS[name])
        ax.set_title(f'Generation step {step}: MMD');ax.set_ylabel('Raw MMD squared')
        ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.legend(frameon=False)
    series(axs[3,0],'loss_total','G training loss_total','#303843')
    series(axs[3,0],'force','MMD-force contribution','#318564')
    axs[3,0].set_title('G training surrogate: both generation steps');axs[3,0].set_ylabel('Training loss');axs[3,0].legend(frameon=False)
    series(axs[3,1],'grad_norm','Existing global grad_norm','#303843')
    axs[3,1].set_title('Generator gradient norm (before clipping)');axs[3,1].set_ylabel('Original global gradient norm');axs[3,1].legend(frameon=False)
    last=rows[-1]['step']
    for ax in axs.flat:
        ax.set_xlim(.5,max(2,last+.5));ax.set_xlabel('Completed training iteration');ax.grid(axis='y',alpha=.16,linewidth=.7)
        ax.xaxis.get_major_locator().set_params(integer=True)
        if last>=51:ax.axvline(50.5,color='#8a9299',linestyle='--',linewidth=.9)
        for action in data['D_lr_actions']:
            ax.axvline(action['effective_from_step']-.5,color='#9950a2',linestyle=':',linewidth=.9)
    fig.suptitle('Fresh0 DINOv3 + Qwen-VL | iterations 1–'+str(last)+'\n'
        f'Thin: actual iterations; solid: trailing {width}-iteration mean. Dashed: GAN-G starts. Dotted: actual D LR change.\n'
        'GAN and MMD: DINOv3 + Qwen-VL. Source '+data['source_commit'][:12]+' | '+data['observed_utc'],fontsize=12)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    paths=[]
    for ext in ['png','pdf']:
        path=output.with_suffix('.'+ext);fig.savefig(path,dpi=240);paths.append(str(path.resolve()))
    plt.close(fig)
    result=dict(last_iteration=last,points=len(rows),MMD_encoders=data['MMD_encoders'],GAN_encoders=data['GAN_encoders'],
        source_commit=data['source_commit'],files=paths,actual_GAN_G_only=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('metrics',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--smooth',type=int,default=10)
    a=p.parse_args();assert a.smooth>0
    print(json.dumps(plot(json.loads(a.metrics.read_text()),a.output,a.smooth)))
