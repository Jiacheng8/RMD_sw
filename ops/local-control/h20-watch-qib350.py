#!/usr/bin/env python3
"""Follow the exact existing QIB350 process read-only; never launch/restart evaluation."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def atomic(path,value):
    p=path.with_name(path.name+f'.partial-{os.getpid()}')
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');p.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-helper',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--expected-pid',type=int,required=True)
    parser.add_argument('--expected-source',required=True)
    parser.add_argument('--hours',type=float,default=12)
    parser.add_argument('--period',type=float,default=40)
    args=parser.parse_args();assert args.hours>0 and args.period>=10
    output=args.output_dir.resolve();output.mkdir(exist_ok=True,parents=True)
    lock=(output/'h20-qib350-observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=output/'h20-qib350-observer-runs'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'-{os.getpid()}');run.mkdir(parents=True)
    previous=run/'previous'
    for name in ['h20-qib350-observer.command.json','h20-qib350-observer.status.json',
                 'h20-qib350-observer.pid','h20-qib350-observer.exit','h20-qib350-history.json']:
        p=output/name
        if p.exists():previous.mkdir(exist_ok=True);p.replace(previous/name)
    query=Path(__file__).with_name('h20-observe-qib350.py').read_text()
    atomic(output/'h20-qib350-observer.command.json',dict(command=sys.argv,pid=os.getpid(),
        expected_pid=args.expected_pid,expected_source=args.expected_source,
        query_sha256=hashlib.sha256(query.encode()).hexdigest(),
        observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),run_dir=str(run),read_only=True))
    (output/'h20-qib350-observer.pid').write_text(str(os.getpid())+'\n')
    def stop(signum,frame):
        raise KeyboardInterrupt('Read-only observer stopped by operator signal'+str(signum))
    signal.signal(signal.SIGTERM,stop)
    history=[];started=last_advance=time.monotonic();previous_signature=None;last_notice=None
    reason='observer_exception_requires_inspection';code=1;data=None
    try:
        while time.monotonic()-started<args.hours*3600:
            data=None
            for attempt in range(3):
                process=subprocess.run([sys.executable,str(args.ssh_helper.resolve())],input=query,text=True,capture_output=True,timeout=120)
                if process.returncode==0:
                    data=json.loads(process.stdout);break
                atomic(run/'last-query-error.json',dict(actual_query_exit=process.returncode,stderr=process.stderr[-3000:]))
                time.sleep(10)
            if data is None:reason='query_unavailable_not_evaluation_exit';break
            pipeline=data['pipeline']
            assert pipeline['wrapper_pid']==args.expected_pid and pipeline['source_commit']==args.expected_source,'Evaluation identity differs; inspect actual actor'
            if pipeline['actual_exit'] is None:
                assert pipeline['live_command'],'Expected evaluation wrapper is absent; inspect rather than relaunch'
            atomic(output/'h20-qib350-progress-latest.json',data)
            point=dict(utc=data['utc'],actual_images=data['actual_valid_ID_PNGs'],
                completed_judge_parts=data['completed_judge_parts'],
                judge_batches=[dict(part=j['part'],gpu=j['gpu'],batch=j['batch'],done=j['completed_batches'],total=j['total_batches'],exit=j['actual_exit']) for j in data['judge_attempts']],
                assets=data['assets_actual_aria2_progress'],allocated_download_bytes=sum(f['allocated_bytes'] for f in data['assets_sparse_files']),
                gen_events=[data['stages'][f'generate_{k}'].get('latest_actual_event') for k in (0,1)],
                GPU4_5=data['GPU4_5'],actual_evaluation_exit=pipeline['actual_exit'],full_completion_proven=data['full_benchmark_completion_proven'])
            history.append(point);atomic(output/'h20-qib350-history.json',history)
            atomic(output/'h20-qib350-observer.status.json',dict(state='observing',latest=point,observations=len(history)))
            signature=json.dumps([point['actual_images'],point['judge_batches'],point['assets'],point['gen_events']],sort_keys=True)
            if signature!=previous_signature:last_advance=time.monotonic();previous_signature=signature
            notice=(point['actual_images']//100,tuple(point['completed_judge_parts']),bool(data['assets_verified']),bool(data['judge_environment_verified']))
            if notice!=last_notice:
                print(json.dumps(dict(event='verified_QIB350_progress',evidence=point)),flush=True);last_notice=notice
            if pipeline['actual_exit'] is not None:
                reason='full_QIB350_requires_final_acceptance' if data['full_benchmark_completion_proven'] else 'evaluation_terminal_needs_inspection'
                code=0 if data['full_benchmark_completion_proven'] else 1;break
            if time.monotonic()-last_advance>5400:reason='no_measured_progress90min_inspect_IO_and_judge_before_deciding';break
            time.sleep(args.period)
        else:reason='observation_window_ended_not_evaluation_exit';code=0
    except BaseException:
        (run/'exception.log').write_text(traceback.format_exc());raise
    finally:
        status=dict(reason=reason,observer_exit=code,observations=len(history),
            actual_evaluation_exit=data['pipeline']['actual_exit'] if data else None,run_dir=str(run),goal_completion_proven=False)
        atomic(output/'h20-qib350-observer.status.json',status);atomic(run/'status.json',status)
        (output/'h20-qib350-observer.exit').write_text(str(code)+'\n');(run/'exit').write_text(str(code)+'\n')
        print(json.dumps(status),flush=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
