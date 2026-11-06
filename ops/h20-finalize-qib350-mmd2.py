#!/usr/bin/env python3
"""Follow the existing350 evaluation and fresh0 successor; preserve/audit, never launch training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path('/data/rdm-workspace/rdm-sets')
CONTROL=ROOT/'control'
CODE=Path(__file__).resolve().parents[1]
PREFIX='qib350-mmd2-finalization'
CHILD=None
OWNS=False


def write(name,value):
    p=CONTROL/(PREFIX+'-'+name+'.json');t=p.with_name(p.name+f'.partial-{os.getpid()}')
    t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def wait_exit(prefix):
    path=CONTROL/(prefix+'.exit')
    write('status',dict(phase='waiting_real_exit',dependency=prefix,epoch=time.time()))
    while not path.exists():time.sleep(5)
    actual=int(path.read_text().strip());assert actual==0,f'{prefix} failed with real exit{actual}; preserve/inspect existing log'


def run(name,command):
    global CHILD
    log=ROOT/'launch-logs'/(PREFIX+'-'+name+'.log')
    with log.open('ab') as f:
        CHILD=subprocess.Popen([str(x) for x in command],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True,cwd=CODE)
    write(name+'-command',dict(command=[str(x) for x in command],pid=CHILD.pid,log=str(log),code=str(CODE),
        code_commit=(CODE/'SOURCE_COMMIT').read_text().strip(),epoch=time.time()))
    code=CHILD.wait();write(name+'-exit',dict(actual_exit=code,epoch=time.time()));CHILD=None
    if code:
        write('failure',dict(stage=name,actual_exit=code,last30=log.read_text(errors='replace').splitlines()[-30:]))
        raise RuntimeError(f'{name} failed; actual exit/command/last30 retained')


def stop(signum,frame):
    if CHILD is not None and CHILD.poll() is None:
        try:os.killpg(CHILD.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    raise SystemExit(128+signum)


def main():
    global OWNS
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--training-source',type=Path,required=True);ap.add_argument('--training-sha',required=True)
    a=ap.parse_args()
    assert a.training_source.is_relative_to('/data/rdm-workspace')
    assert (a.training_source/'SOURCE_COMMIT').read_text().strip()==a.training_sha
    lock=(CONTROL/(PREFIX+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (CONTROL/(PREFIX+'-status.json')).exists(),'Inspect existing finalizer rather than duplicate'
    OWNS=True;(CONTROL/(PREFIX+'.pid')).write_text(str(os.getpid())+'\n')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    os.environ.update(CUDA_VISIBLE_DEVICES='',TMPDIR=str(ROOT/'.cache/tmp'),OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    run('CPU-gates',[sys.executable,'-m','pytest','-q','tests/test_mmd2_handoff.py','tests/test_checkpoint_archive.py'])
    wait_exit('qib350-two-gpu')
    prior=CONTROL/'qib350-full-preservation-verified.json'
    if prior.exists():
        saved=json.loads(prior.read_text())
        assert saved['passed'] and saved['images']==1000 and saved['actual_raw_dimension_answers']==4105
        assert saved['source_commit']=='31bbce8253dfee657a6399e5e23d8946e9d65bd1'
        assert Path(saved['archive']).stat().st_size==saved['archive_bytes']
        write('preserve-benchmark-reused',dict(passed=True,receipt=str(prior),archive_sha256=saved['archive_sha256']))
    else:run('preserve-benchmark',[sys.executable,CODE/'ops/h20-preserve-qib350.py'])
    wait_exit('dino-qwenvl-mmd2-from0')
    run('final-resume-student-audit',[sys.executable,CODE/'ops/h20-audit-mmd2-final.py',
        '--training-source',a.training_source,'--training-sha',a.training_sha])
    write('status',dict(phase='artifacts_preserved_full500_integrity_verified',epoch=time.time(),
        benchmark_receipt=str(CONTROL/'qib350-full-preservation-verified.json'),
        training_receipt=str(CONTROL/'dino-qwenvl-mmd2-from0-final-integrity.json'),
        remaining='Read authoritative scores/audit and final curves/report; do not claim goal complete from PID/status alone'))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        traceback.print_exc();code=1
        if OWNS:write('status',dict(phase='failed_preserve_outputs',epoch=time.time()))
    finally:
        if CHILD is not None:
            if CHILD.poll() is None:
                try:os.killpg(CHILD.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            try:CHILD.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(CHILD.pid,signal.SIGKILL);CHILD.wait()
        if OWNS:(CONTROL/(PREFIX+'.exit')).write_text(str(code)+'\n')
    raise SystemExit(code)
