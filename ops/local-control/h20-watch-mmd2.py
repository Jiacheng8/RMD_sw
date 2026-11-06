#!/usr/bin/env python3
"""Local read-only follow of the one existing MMD2 source; never train/restart/modify remote files."""
import argparse,datetime,fcntl,hashlib,json,os
from pathlib import Path
import subprocess,sys,time,traceback


def write(path,value):
    tmp=path.with_name(path.name+f'.partial-{os.getpid()}');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ssh-helper',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--expected-source',required=True);p.add_argument('--expected-pid',type=int,required=True);p.add_argument('--period',type=float,default=60);p.add_argument('--hours',type=float,default=24)
    a=p.parse_args();assert a.period>=10 and a.hours>0
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);lock=(out/'h20-mmd2-observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'h20-mmd2-observer.command.json').exists(),'Inspect prior observer before duplicate'
    query=Path(__file__).with_name('h20-observe-mmd2.py').read_text()
    write(out/'h20-mmd2-observer.command.json',dict(command=sys.argv,pid=os.getpid(),query_sha256=hashlib.sha256(query.encode()).hexdigest(),expected_pid=a.expected_pid,expected_source=a.expected_source,read_only=True))
    history=[];deadline=time.monotonic()+a.hours*3600;previous=None;reason='deadline_requires_new_observation';code=0
    while time.monotonic()<deadline:
        r=subprocess.run([sys.executable,str(a.ssh_helper.resolve())],input=query,text=True,capture_output=True,timeout=125)
        if r.returncode:
            write(out/'h20-mmd2-observer-query-error.json',dict(actual_query_exit=r.returncode,stderr=r.stderr[-2000:]))
            reason='query_failure_is_not_training_exit';code=1;break
        v=json.loads(r.stdout);actor=v['pipeline'];assert actor['wrapper_pid']==a.expected_pid and actor['source_commit']==a.expected_source
        if actor['actual_exit'] is None:assert actor['live_command'],'Actor missing without authoritative exit; inspect, never restart on observation'
        write(out/'h20-mmd2-progress-latest.json',v)
        point=dict(utc=v['utc'],phase=v['status']['phase'] if v['status'] else None,formal_iteration=v['completed_formal_iterations'],smoke_iterations=len(v['smoke_iterations']),retained_weights=v['retained_weights'],actual_exit=actor['actual_exit'],final_integrity_passed=bool(v['final_integrity'] and v['final_integrity']['passed']))
        signature=json.dumps({k:x for k,x in point.items() if k!='utc'},sort_keys=True)
        if signature!=previous:
            history.append(point);write(out/'h20-mmd2-history.json',history);print(json.dumps(point),flush=True);previous=signature
        if actor['actual_exit'] not in (None,0) or v['G_skips'] or v['D_skips'] or v['nonfinite_numeric_paths']:
            reason='real_failure_or_invalid_update_requires_inspection';code=1;break
        if v['finalizer_exit'] not in (None,0):reason='final_integrity_stage_failed_requires_inspection';code=1;break
        if point['final_integrity_passed']:reason='actual500_integrity_ready_for_final_curves_and_report';break
        time.sleep(a.period)
    write(out/'h20-mmd2-observer.status.json',dict(reason=reason,observer_exit=code,goal_completion_proven=False));return code


if __name__=='__main__':
    try:rc=main()
    except BaseException:traceback.print_exc();rc=1
    args=sys.argv
    if '--output-dir' in args:
        (Path(args[args.index('--output-dir')+1])/'h20-mmd2-observer.exit').write_text(str(rc)+'\n')
    raise SystemExit(rc)
