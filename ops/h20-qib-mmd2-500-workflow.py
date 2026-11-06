#!/usr/bin/env python3
"""Full official Chinese1024 QIB for the exact new-MMD2 student500 on GPUs4..7."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import sys
import time
import traceback

CODE=Path(__file__).resolve().parents[1]
ROOT=Path('/data/rdm-workspace/rdm-sets')
CONTROL=ROOT/'control'
TRAIN=Path('/data/rdm-workspace/RDM-dino-qwenvl-mmd2-v7')
TRAIN_SHA='ad5dddc428a32591fedf9b034bf5f1639ced0514'
PREFIX='qib-mmd2-s500-gpu4567-v1'
RUN=ROOT/'qib/runs/mmd2-from0-s500-zh-1024-seed0-v1'
GPUS=[4,5,6,7]
SHA='7a189cf104b9c4b7f09ef7e27b090e1824ca520a35c9626ff734bba2d47bae32'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

base=module('qib500_shared',CODE/'ops/h20-qib-baselines-workflow.py')
runtime=base.runtime;runtime.RUN=RUN;runtime.CODE=CODE


def status(phase,**values):
    v=dict(phase=phase,epoch=time.time(),**values);runtime.write(CONTROL/(PREFIX+'-status.json'),v)
    print(json.dumps(v),flush=True)


def main():
    signal.signal(signal.SIGTERM,runtime.stop);signal.signal(signal.SIGINT,runtime.stop)
    assert (TRAIN/'SOURCE_COMMIT').read_text().strip()==TRAIN_SHA
    final=json.loads((CONTROL/'dino-qwenvl-mmd2-from0-final-integrity.json').read_text())
    assert final['passed'] and final['step']==500 and final['actual_formal_exit']==0 and final['final_student_SHA']==SHA
    assert final['MMD_encoders']==final['GAN_encoders']==['dinov3_l','qwen3vl_4b']
    assert not (CONTROL/(PREFIX+'-command.json')).exists(),'Preserve an existing run; recover explicitly'
    for gpu in GPUS:
        used=int(__import__('subprocess').check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert used<1024,f'GPU{gpu} occupied; never stop a foreign process'
    assert shutil.disk_usage(ROOT).free>32*10**9
    assert json.loads((CONTROL/'qib350-assets-complete.json').read_text())['passed']
    assert json.loads((ROOT/'qib/env/.ready').read_text())['passed']
    (RUN/'logs').mkdir(parents=True,exist_ok=True)
    runtime.write(CONTROL/(PREFIX+'-command.json'),dict(pid=os.getpid(),command=sys.argv,source=str(CODE),
        source_commit=(CODE/'SOURCE_COMMIT').read_text().strip(),training_commit=TRAIN_SHA,
        physical_GPUs=GPUS,checkpoint_step=500,checkpoint_sha256=SHA,run=str(RUN),started_epoch=time.time()))
    status('verifying_remote_RAM_models')
    module('student500_ram',CODE/'ops/h20-stage-mmd2-500-ram.py').stage()
    judge=module('judge500_ram',CODE/'ops/h20-stage-qib-judge-ram.py').stage()
    sys.path[:0]=[str(TRAIN/'z-image'),str(TRAIN)]
    from eval_qib.bench import bench_prompts
    prompts=bench_prompts(ROOT/'qib/bench','zh')
    metadata=json.loads((ROOT/'qib/bench/metadata/bench_metadata.json').read_text())
    assert len(prompts)==len(metadata)==1000 and {i for i,_ in prompts}=={int(r['ID']) for r in metadata}
    calls=sum(len({d.split('/')[0].strip() for d in r['dims_en'].split(';') if d.strip()}) for r in metadata)
    assert calls==4105
    proof=json.loads((CONTROL/'dino-qwenvl-mmd2-from0-archive/checkpoint-archive-500.json').read_text())
    generation=dict(ckpt=proof['destination'],checkpoint_step=500,checkpoint_sha256=SHA,
        training_commit=TRAIN_SHA,experiment='zimage-2step-1024-dino-qwenvl-mmd2-from0',teacher_steps=None,
        sampling_args=dict(sigmas=[1.,.75],renoise=True,detach_between_steps=True),lang='zh',seed=0,img_size=1024,images_per_prompt=1)
    runtime.write(RUN/'generation.json',generation)
    for gpu in GPUS:
        used=int(__import__('subprocess').check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert used<1024,f'GPU{gpu} became occupied during CPU staging'
    status('generating',physical_GPUs=GPUS,images_required=1000)
    generators=[]
    for rank,gpu in enumerate(GPUS):
        name=f'generate_{rank}';p=runtime.spawn(name,[runtime.PYTHON,CODE/'ops/h20-qib-mmd2-500-generate.py',
            '--rank',rank,'--gpu',gpu,'--out',RUN],gpu);generators.append((name,p))
    for name,p in generators:assert runtime.wait(name,p)==0,f'{name} failed; inspect actual exit/log'
    files=[]
    for ident,_ in prompts:
        p=RUN/'images'/f'{ident:04d}.png';raw=p.read_bytes()
        assert raw[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',raw[16:24])==(1024,1024)
        files.append(dict(ID=ident,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    assert len(list((RUN/'images').glob('[0-9]*.png')))==1000
    with (RUN/'input.jsonl').open('w') as out:
        for ident,prompt in prompts:out.write(json.dumps(dict(ID=ident,prompt=prompt,image_path=str(RUN/'images'/f'{ident:04d}.png')),ensure_ascii=False)+'\n')
    runtime.write(RUN/'generation-verified.json',dict(passed=True,images=files,generation=generation))
    utility=CODE/'z-image/eval_qib/judge_parallel.py'
    name='split';p=runtime.spawn(name,[runtime.JUDGE,utility,'split','--input',RUN/'input.jsonl','--parts',8,
        '--metadata',ROOT/'qib/bench/metadata/bench_metadata.json']);assert runtime.wait(name,p)==0
    for gpu in GPUS:
        used=int(__import__('subprocess').check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert used<1024,f'GPU{gpu} occupied before judging; preserve generated images'
    status('judging',images=1000,expected_dimension_answers=calls,physical_GPUs=GPUS)
    def worker(gpu):
        batch=80
        for part in range(gpu-4,8,4):
            dest=RUN/'shards'/f'part_{part}_judged.jsonl';attempt=0
            while not dest.exists():
                name=f'judge_part{part}_gpu{gpu}_b{batch}_attempt{attempt}'
                p=runtime.spawn(name,[runtime.JUDGE,CODE/'ops/h20-qib-mmd2-500-judge-part.py',
                    '--input',RUN/'shards'/f'part_{part}.jsonl','--batch',batch,'--model',judge],gpu)
                code=runtime.wait(name,p)
                if code==0:assert dest.exists();break
                log=(RUN/'logs'/f'{name}.log').read_text(errors='replace')
                assert 'out of memory' in log.lower() and batch>8,f'{name} exit{code}; see actual log'
                batch=max(8,batch//2);attempt+=1
            name=f'part{part}-thinking';p=runtime.spawn(name,[runtime.JUDGE,utility,'check-thinking','--judged',dest])
            assert runtime.wait(name,p)==0
            print(json.dumps(dict(event='actual_judged_part_complete',part=part,gpu=gpu)),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,GPUS))
    name='merge';p=runtime.spawn(name,[runtime.JUDGE,utility,'merge','--run',RUN,'--parts',8,'--bench',ROOT/'qib/bench']);assert runtime.wait(name,p)==0
    rows=[json.loads(s) for s in (RUN/'input_judged.jsonl').read_text().splitlines() if s.strip()]
    assert [int(r['ID']) for r in rows]==[i for i,_ in prompts]
    raw,parsed=base.validate_judged(rows,metadata)
    name='check-thinking';p=runtime.spawn(name,[runtime.JUDGE,utility,'check-thinking','--judged',RUN/'input_judged.jsonl']);assert runtime.wait(name,p)==0
    name='report';p=runtime.spawn(name,[runtime.PYTHON,'-m','eval_qib.report',RUN,'--out',RUN/'report.md']);assert runtime.wait(name,p)==0
    scores=json.loads((RUN/'input_bench_scores.json').read_text());base.archive.finite_scores(scores)
    complete=dict(passed=True,checkpoint_step=500,checkpoint_sha256=SHA,run=str(RUN),images=1000,
        raw_answers=raw,parsed_L1=parsed,judge_parts=8,scores=scores,source_commit=(CODE/'SOURCE_COMMIT').read_text().strip(),
        generation=generation,completed_epoch=time.time())
    runtime.write(CONTROL/(PREFIX+'-scores-complete.json'),complete)
    status('preserving_full_evaluation',scores=scores)
    complete['preservation']=base.preserve(RUN,complete,'/mnt/airs-business/rdm-experiments/evaluations')
    comparison=ROOT/'qib/runs/mmd2-s500-vs-turbo2-turbo9-old350-report.md'
    others=[ROOT/'qib/runs'/name for name in ['turbo2-zh-1024-seed0-v1','turbo9-zh-1024-seed0-v1','twohead-lr10-A-s350-zh-v1']]
    name='comparison-report';p=runtime.spawn(name,[runtime.PYTHON,'-m','eval_qib.report',RUN,*others,'--out',comparison]);assert runtime.wait(name,p)==0
    complete['comparison_report']=str(comparison);runtime.write(CONTROL/(PREFIX+'-complete.json'),complete)
    status('complete',scores=scores,comparison_report=str(comparison))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        runtime.write(CONTROL/(PREFIX+'-failure.json'),dict(error=str(e),traceback=traceback.format_exc(),epoch=time.time()))
        raise
    finally:runtime.cleanup_owned_children()
