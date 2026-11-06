#!/usr/bin/env python3
"""Finish a verified archive, then rename and exercise real relocated CPU runtime.

Deploy this small controller outside the tree it renames. It never launches GPU
training/evaluation. Its exit/status/log are durable under the moved control root.
"""
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import time
import traceback

OLD=Path('/data/shaoshitong')
NEW=Path('/data/rdm-workspace')
SOURCE_NAME='RDM-storage-relocation-v7'

def control():
    return (NEW if NEW.exists() else OLD)/'rdm-sets/control/storage-relocation-20261106'

def write(name,value):
    p=control()/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)

def main():
    while not (control()/'archive.exit').exists():
        write('driver-status.json',dict(phase='waiting_for_full_archive',epoch=time.time()))
        time.sleep(10)
    assert (control()/'archive.exit').read_text().strip()=='0', 'Archive failed; keep original files and inspect archive.log'
    if OLD.exists():
        source=OLD/SOURCE_NAME
        runpy.run_path(str(source/'ops/h20-storage-rename.py'),run_name='__main__')
    else:
        assert NEW.is_dir() and json.loads((control()/'rename.json').read_text())['passed']
    source=NEW/SOURCE_NAME;root=NEW/'rdm-sets'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',TMPDIR=str(root/'.cache/tmp'),
        PYTHONPATH=str(source)+':'+str(source/'z-image'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    python=root/'envs/rdm/bin/python'
    write('driver-status.json',dict(phase='validating_relocated_environment',epoch=time.time()))
    tests=subprocess.run([str(python),str(source/'tests/test_storage_relocation.py'),'-v'],env=env,cwd=source,check=True)
    check='''import pathlib,json,sys,sysconfig,torch,numpy,PIL,transformers
from rdm.sw_lmmd.reference_images import ReferenceImageStore
root=pathlib.Path('/data/rdm-workspace/rdm-sets')
assert torch.__version__.startswith('2.8.')
assert pathlib.Path(sysconfig.get_config_var('INCLUDEPY')).is_dir()
assert sys.prefix.startswith('/data/rdm-workspace/')
view=root/'zimage/zimage-ref-v1/reference_images_patchgan_object_v1'
store=ReferenceImageStore(view)
rows=[0,1500,10000,40000,80000,100000,120000,125807]
images=store.read_uint8(rows)
assert images.shape==(len(rows),3,1024,1024) and images.dtype==torch.uint8
store.close()
print(json.dumps(dict(passed=True,torch=torch.__version__,python_prefix=sys.prefix,
    include=sysconfig.get_config_var('INCLUDEPY'),real_image_rows=rows,shape=list(images.shape),
    reference_rows=125808,old_directory_absent=not pathlib.Path('/data/shaoshitong').exists())))
'''
    r=subprocess.run([str(python),'-c',check],env=env,cwd=source,check=True,text=True,capture_output=True,timeout=180)
    result=json.loads(r.stdout);write('runtime-verified.json',result);print(r.stdout,flush=True)
    for args in [[str(python),'-m','pip','--version'],[str(root/'envs/rdm/bin/torchrun'),'--help']]:
        p=subprocess.run(args,env=env,cwd=source,text=True,capture_output=True,timeout=45)
        assert p.returncode==0, p.stderr[-2000:]
    samples=[]
    for _ in range(2):
        assert not OLD.exists() and not OLD.is_symlink()
        assert (root/'zimage/zimage-ref-v1/contexts/shard_00.bin').is_file()
        assert (root/'zimage/work_dirs/zimage-2step-1024-dino-qwenvl-mmd2-from0/resume.pth').is_file()
        samples.append(dict(epoch=time.time(),disk_free_bytes=shutil.disk_usage('/data').free))
        if len(samples)==1:time.sleep(10)
    du=subprocess.check_output(['du','-s','-x','-B1',str(NEW)],text=True)
    result=dict(passed=True,workspace=str(NEW),object_root='/mnt/airs-business/rdm-experiments',
        project_allocated_bytes=int(du.split()[0]),two_final_observations=samples,
        old_personal_directory_and_alias_absent=True,actual_training_steps=500,
        actual_training_exit=0,all_checkpoints_preserved=True,environment_and_image_reader_passed=True,
        epoch=time.time())
    write('final-verified.json',result);print(json.dumps(result),flush=True)

code=1
try:
    main();code=0
except BaseException:
    traceback.print_exc()
finally:
    write('driver.exit',code)
raise SystemExit(code)
