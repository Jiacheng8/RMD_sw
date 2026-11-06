#!/usr/bin/env python3
"""Full QIB350: overlapped CPU preparation, two GPU generators, two judge workers."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path('/data/rdm-workspace/rdm-sets')
CODE = Path(__file__).resolve().parents[1]
TRAIN = Path('/data/rdm-workspace/RDM-twohead-resume200-v3')
RUN = ROOT/'qib/runs/twohead-lr10-A-s350-zh-v1'
CONTROL = ROOT/'control'
PYTHON = ROOT/'envs/rdm/bin/python'
JUDGE = ROOT/'qib/env/bin/python'
PREFIX = 'qib350-two-gpu'
CHILDREN = {}
LOCK = threading.Lock()


def write(path, value):
    tmp = path.with_name(path.name+f'.partial-{threading.get_ident()}')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def spawn(name, command, gpu=None):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
        TMPDIR=str(ROOT/'.cache/tmp'), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if name.startswith('generate'):
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if name.startswith('judge'):
        env.pop('PYTHONPATH', None)
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    path = RUN/'logs'/f'{name}.log'
    stream = path.open('ab')
    process = subprocess.Popen([str(x) for x in command], env=env, stdin=subprocess.DEVNULL,
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    stream.close()
    with LOCK:
        CHILDREN[process.pid] = process
    process.h20_name = name
    write(RUN/'logs'/f'{name}.command.json', dict(command=[str(x) for x in command],
        pid=process.pid, GPU=gpu, log=str(path), started_epoch=time.time(), code_commit=(CODE/'SOURCE_COMMIT').read_text().strip()))
    return process


def wait(name, process):
    code = process.wait()
    write(RUN/'logs'/f'{name}.exit.json', dict(actual_exit=code, completed_epoch=time.time()))
    with LOCK:
        CHILDREN.pop(process.pid, None)
    return code


def stop(signum, frame):
    with LOCK:
        pids = list(CHILDREN)
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise SystemExit(128+signum)


def cleanup_owned_children():
    with LOCK:
        processes = list(CHILDREN.values())
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        wait(process.h20_name, process)


def main():
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN/'logs').mkdir(exist_ok=True)
    stop_proof = json.loads((CONTROL/'twohead-user-stop-for-qib350-verified.json').read_text())
    assert stop_proof['passed'] and stop_proof['last_complete_iteration'] == 418
    assert all(not Path(f'/proc/{pid}').exists() for pid in [2174015,2174036,2174084,2174085])
    for gpu in (4,5):
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
            '--format=csv,noheader,nounits'],text=True).strip())
        assert used < 1024, f'GPU{gpu} is occupied; inspect ownership before full-memory evaluation'
    write(CONTROL/f'{PREFIX}-command.json', dict(command=sys.argv, pid=os.getpid(), code=str(CODE),
        code_commit=(CODE/'SOURCE_COMMIT').read_text().strip(), physical_GPUs=[4,5], checkpoint_step=350,
        run=str(RUN), latest_manual_stop_supersedes_old500=True, started_epoch=time.time()))
    prepare = []
    if not (CONTROL/'qib350-assets-complete.json').exists():
        prepare.append(('assets', spawn('assets',[PYTHON,CODE/'ops/h20-qib350-assets.py'])))
    if not (ROOT/'qib/env/.ready').exists():
        prepare.append(('environment',spawn('environment',['bash',CODE/'ops/h20-qib350-prepare-env.sh'])))
    started = time.monotonic()
    while not (CONTROL/'qib350-bench-runtime.json').exists():
        assets = next((p for n,p in prepare if n=='assets'), None)
        if assets is not None and assets.poll() is not None:
            assert wait('assets', assets) == 0, 'Benchmark preparation failed; see actual assets exit/log'
        assert time.monotonic()-started < 1800, 'No verified benchmark within30min; inspect transport'
        time.sleep(2)
    sys.path[:0]=[str(TRAIN/'z-image'),str(TRAIN)]
    from eval_qib.bench import bench_prompts
    prompts=bench_prompts(ROOT/'qib/bench','zh')
    official_metadata=json.loads((ROOT/'qib/bench/metadata/bench_metadata.json').read_text())
    assert len(official_metadata)==1000 and {int(r['ID']) for r in official_metadata}=={i for i,_ in prompts}
    calls=sum(len({d.split('/')[0].strip() for d in r['dims_en'].split(';') if d.strip()}) for r in official_metadata)
    assert calls==4105
    metadata = dict(ckpt='/mnt/airs-business/shaoshitong/rdm-sets/checkpoints/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A/step_0000350.pth',
        checkpoint_step=350, checkpoint_sha256='6428309c6fd2af6b1e1588428b44545524ee5d066c07bdaf1fdcb5dc87bb663b',
        teacher_steps=None, sampling_args=dict(sigmas=[1.,.75],renoise=True,detach_between_steps=True),
        lang='zh',seed=0,img_size=1024,images_per_prompt=1)
    generation = RUN/'generation.json'
    if generation.exists():
        assert json.loads(generation.read_text()) == metadata, 'Different run identity; refuse reuse'
    else:
        write(generation,metadata)
    generators=[]
    for rank,gpu in enumerate([4,5]):
        name=f'generate_{rank}'
        generators.append((name,spawn(name,[PYTHON,CODE/'ops/h20-qib350-generate.py',
            '--rank',rank,'--gpu',gpu,'--out',RUN],gpu)))
    for name,process in generators:
        assert wait(name,process)==0, f'{name} failed; inspect its actual exit/log'
    images=[]
    for ident,prompt in prompts:
        p=RUN/'images'/f'{ident:04d}.png'
        raw=p.read_bytes()
        import struct
        assert raw[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',raw[16:24])==(1024,1024)
        images.append(dict(ID=ident,path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    assert len(images)==1000
    with (RUN/'input.jsonl').open('w') as f:
        for ident,prompt in prompts:
            f.write(json.dumps(dict(ID=ident,prompt=prompt,image_path=str(RUN/'images'/f'{ident:04d}.png')),ensure_ascii=False)+'\n')
    write(RUN/'generation-verified.json',dict(passed=True,expected_images=1000,files=images,generation=metadata))
    print(json.dumps(dict(event='all_actual_images_verified',count=1000)),flush=True)
    for name,process in prepare:
        assert wait(name,process)==0, f'{name} failed; inspect actual preparation exit/log'
    assert json.loads((CONTROL/'qib350-assets-complete.json').read_text())['passed']
    assert json.loads((ROOT/'qib/env/.ready').read_text())['passed']
    utility=CODE/'z-image/eval_qib/judge_parallel.py'
    split=spawn('split',[JUDGE,utility,'split','--input',RUN/'input.jsonl','--parts',8,
        '--metadata',ROOT/'qib/bench/metadata/bench_metadata.json'])
    assert wait('split',split)==0
    def worker(gpu):
        batch=64
        for part in range(gpu-4,8,2):
            destination=RUN/'shards'/f'part_{part}_judged.jsonl'
            if destination.exists():
                continue
            attempt=0
            while True:
                name=f'judge_part{part}_gpu{gpu}_b{batch}_attempt{attempt}'
                process=spawn(name,[JUDGE,CODE/'ops/h20-qib350-judge-part.py','--input',
                    RUN/'shards'/f'part_{part}.jsonl','--batch',batch],gpu)
                peak=0
                while process.poll() is None:
                    current=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
                        '--format=csv,noheader,nounits'],text=True).strip())
                    peak=max(peak,current)
                    time.sleep(5)
                code=wait(name,process)
                log=(RUN/'logs'/f'{name}.log').read_text(errors='replace')
                write(RUN/'logs'/f'{name}.memory.json',dict(physical_gpu=gpu,peak_memory_MiB=peak,batch=batch,actual_exit=code))
                if code==0:
                    assert destination.exists(), 'Judge exit0 without completed part'
                    check=spawn(name+'-thinking',[JUDGE,utility,'check-thinking','--judged',destination])
                    assert wait(name+'-thinking',check)==0
                    print(json.dumps(dict(event='actual_judged_part_complete',part=part,gpu=gpu,batch=batch,peak_memory_MiB=peak)),flush=True)
                    if peak<110000 and batch<128:
                        batch=min(128,batch+16)
                    break
                assert 'out of memory' in log.lower() and batch>8, f'{name} failed with exit{code}; inspect preserved log'
                batch=max(8,batch//2);attempt+=1
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker,[4,5]))
    merge=spawn('merge',[JUDGE,utility,'merge','--run',RUN,'--parts',8,'--bench',ROOT/'qib/bench'])
    assert wait('merge',merge)==0
    judged=[json.loads(line) for line in (RUN/'input_judged.jsonl').read_text().splitlines() if line.strip()]
    assert [int(r['ID']) for r in judged]==[i for i,_ in prompts]
    raw_answers=sum(1 for row in judged for key,value in row.items()
                    if key.endswith('_judge_output') and isinstance(value,str) and value)
    assert raw_answers==calls, f'Only{raw_answers}/{calls} actual dimension answers'
    check=spawn('check-thinking',[JUDGE,utility,'check-thinking','--judged',RUN/'input_judged.jsonl'])
    assert wait('check-thinking',check)==0
    report=spawn('report',[PYTHON,'-m','eval_qib.report',RUN,'--out',RUN/'report.md'])
    assert wait('report',report)==0
    scores=json.loads((RUN/'input_bench_scores.json').read_text())
    write(CONTROL/f'{PREFIX}-complete.json',dict(passed=True,checkpoint_step=350,images=1000,
        judge_parts=8,actual_raw_dimension_answers=raw_answers,
        images_without_parsed_judgement=sum(not r.get('judge_model_output') for r in judged),
        scores=scores,run=str(RUN),completed_epoch=time.time()))
    print(json.dumps(dict(event='full_QIB350_complete',scores=scores)),flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup_owned_children()
