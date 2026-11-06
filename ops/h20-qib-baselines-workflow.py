#!/usr/bin/env python3
"""Full native Turbo2/Turbo9 QIB on physical6/7, independent of training4/5."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
ROOT = Path('/data/rdm-workspace/rdm-sets')
CONTROL = ROOT/'control'
PREFIX = 'qib-turbo2-turbo9-gpu67'
REVISION = 'f332072aa78be7aecdf3ee76d5c247082da564a6'
JUDGE_MODEL = Path('/dev/shm/rdm-qib-judge-1b77ff83564ac4e4e8140769eacff7ff73f35f3c')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runtime = module('qib_runtime', CODE/'ops/h20-qib350-workflow.py')
runtime.CODE = CODE
archive = module('qib_archive', CODE/'ops/h20-preserve-qib350.py')


def status(phase, **kwargs):
    runtime.write(CONTROL/(PREFIX+'-status.json'), dict(phase=phase, epoch=time.time(), **kwargs))
    print(json.dumps(dict(event=phase, **kwargs)), flush=True)


def validate_judged(rows, metadata):
    expected = {int(r['ID']): {d.split('/')[0].strip() for d in r['dims_en'].split(';') if d.strip()}
                for r in metadata}
    assert len(rows) == len(expected) == 1000
    assert len({int(r['ID']) for r in rows}) == 1000
    actual_raw = actual_parsed = 0
    for row in rows:
        wanted = expected[int(row['ID'])]
        parsed = row['judge_model_output']
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        assert isinstance(parsed, dict) and set(parsed) == wanted
        actual_parsed += len(parsed)
        for key, value in row.items():
            if key.endswith('_judge_output') and isinstance(value, str) and value:
                actual_raw += 1
    assert actual_raw == actual_parsed == 4105
    return actual_raw, actual_parsed


def preserve(run, completion, archive_root=None):
    """Stream only remote outputs; verify final-named full archive; retain source."""
    dest = Path(archive_root or '/mnt/airs-business/shaoshitong/rdm-sets/evaluations')/(run.name+'.tar.gz')
    dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists(), 'Never overwrite an existing benchmark archive'
    files = [(p, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(run.rglob('*')) if p.is_file()]
    assert all(not p.is_symlink() for p, _, _ in files)
    temp = dest.with_name(dest.name+f'.partial-{os.getpid()}')
    expected = hashlib.sha256()
    class Writer:
        def __init__(self, stream): self.stream = stream
        def write(self, block):
            n = self.stream.write(block)
            assert n == len(block)
            expected.update(block)
            return n
        def flush(self): self.stream.flush()
    with temp.open('xb') as stream:
        writer = Writer(stream)
        with archive.archive_stream(writer) as tar:
            for path, size, mtime in files:
                tar.add(path, arcname=str(Path(run.name)/path.relative_to(run)), recursive=False)
                assert (path.stat().st_size, path.stat().st_mtime_ns) == (size, mtime)
        writer.flush()
        os.fsync(stream.fileno())
    size = temp.stat().st_size
    temp.replace(dest)
    digest = hashlib.sha256()
    with dest.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): digest.update(block)
    assert digest.digest() == expected.digest() and dest.stat().st_size == size
    receipt = dict(passed=True, run=str(run), archive=str(dest), bytes=size,
                   sha256=digest.hexdigest(), files=len(files), source_retained=True,
                   scores=completion['scores'], completed_epoch=time.time())
    runtime.write(dest.with_suffix('.verified.json'), receipt)
    runtime.write(CONTROL/(run.name+'-preserved.json'), receipt)
    return receipt


def run_one(steps):
    from eval_qib.bench import bench_prompts
    from zimage_distill.model import turbo_sigmas
    run = ROOT/'qib/runs'/f'turbo{steps}-zh-1024-seed0-v1'
    runtime.RUN = run
    (run/'logs').mkdir(parents=True, exist_ok=True)
    prompts = bench_prompts(ROOT/'qib/bench', 'zh')
    metadata = json.loads((ROOT/'qib/bench/metadata/bench_metadata.json').read_text())
    assert len(prompts) == len(metadata) == 1000
    assert {i for i,_ in prompts} == {int(r['ID']) for r in metadata}
    generation = dict(ckpt=None, teacher_steps=steps, lang='zh', seed=0, img_size=1024,
        images_per_prompt=1, sampling_args=dict(sigmas=turbo_sigmas(steps), renoise=False, detach_between_steps=True))
    gpath = run/'generation.json'
    if gpath.exists(): assert json.loads(gpath.read_text()) == generation
    else: runtime.write(gpath, generation)
    status('generating', teacher_steps=steps, run=str(run))
    generators = []
    for rank, gpu in enumerate([6,7]):
        name = f'generate_{rank}'
        process = runtime.spawn(name, [runtime.PYTHON, CODE/'ops/h20-qib-baseline-generate.py',
            '--rank', rank, '--gpu', gpu, '--out', run, '--teacher-steps', steps], gpu)
        generators.append((name, process))
    for name, process in generators:
        assert runtime.wait(name, process) == 0, f'{name} failed: see preserved exit/log'
    verified = []
    for ident, _ in prompts:
        path = run/'images'/f'{ident:04d}.png'
        raw = path.read_bytes()
        assert raw[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', raw[16:24]) == (1024,1024)
        verified.append(dict(ID=ident, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    assert len(list((run/'images').glob('[0-9]*.png'))) == 1000
    with (run/'input.jsonl').open('w') as stream:
        for ident, prompt in prompts:
            stream.write(json.dumps(dict(ID=ident, prompt=prompt, image_path=str(run/'images'/f'{ident:04d}.png')), ensure_ascii=False)+'\n')
    runtime.write(run/'generation-verified.json', dict(passed=True, images=verified, generation=generation,
        official_model_revision=REVISION, source_commit=(CODE/'SOURCE_COMMIT').read_text().strip()))
    utility = CODE/'z-image/eval_qib/judge_parallel.py'
    split = runtime.spawn('split', [runtime.JUDGE, utility, 'split', '--input', run/'input.jsonl',
        '--parts', 8, '--metadata', ROOT/'qib/bench/metadata/bench_metadata.json'])
    assert runtime.wait('split', split) == 0
    status('judging', teacher_steps=steps, images=1000, run=str(run))
    def worker(gpu):
        batch = 80
        for part in range(gpu-6, 8, 2):
            destination = run/'shards'/f'part_{part}_judged.jsonl'
            attempt = 0
            while not destination.exists():
                name = f'judge_part{part}_gpu{gpu}_b{batch}_attempt{attempt}'
                process = runtime.spawn(name, [runtime.JUDGE, CODE/'ops/h20-qib-baseline-judge-part.py',
                    '--input', run/'shards'/f'part_{part}.jsonl', '--batch', batch, '--model', JUDGE_MODEL], gpu)
                code = runtime.wait(name, process)
                if code == 0:
                    assert destination.exists(), 'Judge exit0 without output'
                    break
                log = (run/'logs'/f'{name}.log').read_text(errors='replace')
                assert 'out of memory' in log.lower() and batch > 8, f'{name}: exit{code}, inspect actual log'
                batch = max(8, batch//2)
                attempt += 1
            check_name = f'part{part}-thinking'
            check = runtime.spawn(check_name, [runtime.JUDGE, utility, 'check-thinking', '--judged', destination])
            assert runtime.wait(check_name, check) == 0
            print(json.dumps(dict(event='actual_judged_part_complete', teacher_steps=steps, part=part, gpu=gpu)), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, [6,7]))
    merge = runtime.spawn('merge', [runtime.JUDGE, utility, 'merge', '--run', run, '--parts', 8, '--bench', ROOT/'qib/bench'])
    assert runtime.wait('merge', merge) == 0
    rows = [json.loads(s) for s in (run/'input_judged.jsonl').read_text().splitlines() if s.strip()]
    assert [int(r['ID']) for r in rows] == [i for i,_ in prompts]
    raw, parsed = validate_judged(rows, metadata)
    check = runtime.spawn('check-thinking', [runtime.JUDGE, utility, 'check-thinking', '--judged', run/'input_judged.jsonl'])
    assert runtime.wait('check-thinking', check) == 0
    report = runtime.spawn('report', [runtime.PYTHON, '-m', 'eval_qib.report', run, '--out', run/'report.md'])
    assert runtime.wait('report', report) == 0
    scores = json.loads((run/'input_bench_scores.json').read_text())
    archive.finite_scores(scores)
    completion = dict(passed=True, teacher_steps=steps, run=str(run), images=1000, judge_parts=8,
        raw_answers=raw, parsed_L1=parsed, scores=scores, source_commit=(CODE/'SOURCE_COMMIT').read_text().strip(),
        generation=generation, completed_epoch=time.time())
    runtime.write(CONTROL/(run.name+'-complete.json'), completion)
    status('preserving_full_run', teacher_steps=steps, scores=scores)
    completion['preservation'] = preserve(run, completion)
    return completion


def main():
    signal.signal(signal.SIGTERM, runtime.stop)
    signal.signal(signal.SIGINT, runtime.stop)
    runtime.write(CONTROL/(PREFIX+'-command.json'), dict(pid=os.getpid(), source=str(CODE),
        source_commit=(CODE/'SOURCE_COMMIT').read_text().strip(), physical_GPUs=[6,7],
        steps=[2,9], official_model_revision=REVISION, started_epoch=time.time(), no_training_mutation=True))
    status('staging_verified_judge_in_remote_RAM')
    stager = module('judge_ram_stage', CODE/'ops/h20-stage-qib-judge-ram.py')
    assert stager.stage() == JUDGE_MODEL
    for gpu in (6,7):
        used = int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True).strip())
        assert used < 1024, f'GPU{gpu} no longer idle; do not touch foreign processes'
    assert shutil.disk_usage(ROOT).free > 36*10**9, 'Keep rolling-training-checkpoint headroom'
    assert json.loads((CONTROL/'qib350-assets-complete.json').read_text())['passed']
    assert json.loads((ROOT/'qib/env/.ready').read_text())['passed']
    assert (ROOT/'hf/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots'/REVISION/'model_index.json').exists()
    runtime.write(CONTROL/(PREFIX+'-command.json'), dict(pid=os.getpid(), source=str(CODE),
        source_commit=(CODE/'SOURCE_COMMIT').read_text().strip(), physical_GPUs=[6,7],
        steps=[2,9], official_model_revision=REVISION, started_epoch=time.time(), no_training_mutation=True))
    completed = [run_one(steps) for steps in [2,9]]
    comparison = ROOT/'qib/runs/turbo2-turbo9-vs350-report.md'
    runs = [x['run'] for x in completed] + [str(ROOT/'qib/runs/twohead-lr10-A-s350-zh-v1')]
    report = runtime.spawn('comparison-report', [runtime.PYTHON, '-m', 'eval_qib.report', *runs, '--out', comparison])
    assert runtime.wait('comparison-report', report) == 0
    result = dict(passed=True, completed=completed, comparison_report=str(comparison), completed_epoch=time.time())
    runtime.write(CONTROL/(PREFIX+'-complete.json'), result)
    status('complete', scores=[x['scores'] for x in completed], comparison_report=str(comparison))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        import traceback
        runtime.write(CONTROL/(PREFIX+'-failure.json'), dict(error=str(error), traceback=traceback.format_exc(), epoch=time.time()))
        raise
    finally:
        runtime.cleanup_owned_children()
