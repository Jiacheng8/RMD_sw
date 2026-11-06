#!/usr/bin/env python3
"""Retain inference weights and remove only verified, completed preview staging links."""
import hashlib
import json
import os
from pathlib import Path
import torch

ROOT=Path('/data/rdm-workspace/rdm-sets')
CONTROL=ROOT/'control/spatial-A-user-visualization'


def main():
    torch.set_num_threads(2)
    result=[]
    for step in [150,200]:
        preview=ROOT/f'visualizations/spatial-A/step_{step:07d}'
        assert (preview/'preview-verified.json').exists()
        assert (ROOT/f'control/spatial-A-preview-{step}.exit').read_text().strip()=='0'
        source=CONTROL/f'step_{step:07d}-resume.pth'
        target=CONTROL/f'step_{step:07d}-model.pth'
        if target.exists():
            assert (target.with_suffix('.verified.json')).exists()
            result.append(json.loads(target.with_suffix('.verified.json').read_text()))
            continue
        state=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        assert state['step']==step and state['num_steps']==2
        kept={k:state[k] for k in ['model','step','num_steps','sampling_args']}
        temporary=target.with_suffix('.pth.partial')
        assert not temporary.exists()
        torch.save(kept,temporary)
        loaded=torch.load(temporary,map_location='cpu',weights_only=False,mmap=True)
        assert loaded['step']==step and loaded['sampling_args']==state['sampling_args']
        assert loaded['model'].keys()==state['model'].keys()
        assert all(torch.equal(t,loaded['model'][name]) for name,t in state['model'].items())
        digest=hashlib.sha256()
        with temporary.open('rb') as handle:
            for chunk in iter(lambda:handle.read(16*1024*1024),b''):digest.update(chunk)
        os.replace(temporary,target)
        receipt={'step':step,'model_path':str(target),'bytes':target.stat().st_size,
                 'sha256':digest.hexdigest(),'all_model_tensors_equal_source':True,
                 'purpose':'Preserved historical inference checkpoint, not full training resume',
                 'source_preview_staging_link':str(source),'rolling_training_checkpoint_untouched':True}
        target.with_suffix('.verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
        # This is the staging hard link created by this preview task, never RUN/resume.pth.
        source.unlink()
        receipt['preview_staging_link_removed']=True
        (preview/'retained-checkpoint.json').write_text(json.dumps(receipt,indent=2)+'\n')
        result.append(receipt)
        del loaded,state,kept
    (CONTROL/'retained-models.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
