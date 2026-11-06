#!/usr/bin/env python3
"""CPU-only real200 archive/RAM/metadata validation; no inference or model updates."""
import gc, json, os, pathlib, time, traceback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from h20_portrait_checkpoint import archived_checkpoint, load_archived_state, validate_state

ROOT = pathlib.Path('/data/rdm-workspace/rdm-sets')
CONTROL = ROOT/'control'
(ROOT/'.cache/tmp').mkdir(parents=True, exist_ok=True)
os.environ['TMPDIR'] = str(ROOT/'.cache/tmp')


def write(name, value):
    p=CONTROL/name;tmp=p.with_name(p.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,p)


def main():
    import torch
    torch.set_num_threads(1)
    started=time.monotonic()
    path,proof=archived_checkpoint(200)
    def progress(read,total):
        row=dict(phase='RAM_read',step=200,read_bytes=read,total_bytes=total,
                 elapsed_seconds=time.monotonic()-started)
        write('portrait-archive-input-validation-progress.json',row)
        print(json.dumps(row),flush=True)
    state,headroom=load_archived_state(path,proof,progress)
    validate_state(state,200,archived=True)
    receipt=dict(passed=True,step=state['step'],model_tensors=len(state['model']),
        model_numel=sum(v.numel() for v in state['model'].values()),
        model_dtypes=sorted(set(str(v.dtype) for v in state['model'].values())),
        archive_path=str(path),archive_bytes=proof['bytes'],archive_sha256=proof['sha256'],
        remote_memory_headroom_before_load=headroom,seconds=time.monotonic()-started,
        no_GPU_kernels=True,no_inference=True,no_disk_weight_copy=True)
    write('portrait-archive-input-validation.json',receipt)
    print(json.dumps(receipt),flush=True)
    del state;gc.collect()


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:traceback.print_exc();code=1
    finally:(CONTROL/'portrait-archive-input-validation.exit').write_text(str(code)+'\n')
    raise SystemExit(code)
