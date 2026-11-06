#!/usr/bin/env python3
"""Observe the existing resume200-v3 job; never start, stop, or mutate training.

Targets are intermediate acceptance points. Even a successful500 observation
still requires the final full checkpoint/source audit before goal completion.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

TRAIN_SHA = '1e06bf49569b4ee7847a417bc1ff1445bb8ba51b'
FAIL_FIELDS = ['G_skips', 'D_skips', 'nonfinite_numeric_count', 'replay_nonzero_count',
    'probe_row_count_mismatches', 'refresh_row_count_mismatches', 'D_counter_mismatches',
    'G_GAN_phase_mismatches', 'unexpected_AIM_GAN_rows', 'D_lr_transition_mismatches',
    'controller_step_mismatches', 'G_metric_scope_mismatches']


def atomic(path, value):
    temporary = path.with_name(path.name+f'.partial-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def compact(data):
    return dict(utc=data['utc'], step=data['last_step'], exit=data['exit'], errors=data['errors'],
        checks={key: data[key] for key in FAIL_FIELDS},
        archive_verified=data['final_archived_weight_verified_steps'],
        portrait_verified=data['portrait_completed_steps'], disk_free_bytes=data['disk_free_bytes'],
        controller=data['controller_latest'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=int, choices=[400, 450, 500], required=True)
    parser.add_argument('--ssh-helper', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--owned-training-pid', type=int, default=2174015)
    parser.add_argument('--period', type=float, default=40)
    parser.add_argument('--deadline-hours', type=float, default=3)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    assert args.period >= 10 and args.deadline_hours > 0
    args.ssh_helper = args.ssh_helper.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir/f'h20-twohead-observer-{args.target}'
    lock = stem.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_dir = args.output_dir/f'h20-twohead-observer-{args.target}-runs'/(
        datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'-{os.getpid()}')
    run_dir.mkdir(parents=True)
    previous = run_dir/'previous'
    preserved = []
    prior_paths = [stem.with_suffix(suffix) for suffix in
                   ('.command.json', '.pid', '.status.json', '.exit')]
    prior_paths.append(args.output_dir/f'h20-twohead-observation-history-to{args.target}.json')
    for prior in prior_paths:
        if prior.exists():
            previous.mkdir(exist_ok=True)
            destination = previous/prior.name
            prior.replace(destination)
            preserved.append(str(destination))
    query_path = Path(__file__).with_name('h20-active-twohead-query.py')
    query = query_path.read_text()
    atomic(stem.with_suffix('.command.json'), dict(command=sys.argv, cwd=os.getcwd(),
        pid=os.getpid(), ssh_helper=str(args.ssh_helper), query=str(query_path.resolve()),
        run_dir=str(run_dir), preserved_previous_files=preserved,
        query_sha256=hashlib.sha256(query.encode()).hexdigest(),
        observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_source_commit=TRAIN_SHA, read_only=True,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    stem.with_suffix('.pid').write_text(str(os.getpid())+'\n')
    history, channel_errors = [], []
    started = last_advance = time.monotonic()
    previous = -1
    data = None
    code = 1
    reason = 'observer_exception_requires_inspection'
    try:
        while time.monotonic()-started < args.deadline_hours*3600:
            data = None
            for attempt in range(1, 4):
                try:
                    proc = subprocess.run([sys.executable, str(args.ssh_helper)], input=query,
                        text=True, capture_output=True, timeout=120)
                    if proc.returncode:
                        raise RuntimeError(f'SSH query exit{proc.returncode}: '+proc.stderr[-2000:])
                    data = json.loads(proc.stdout)
                    break
                except (subprocess.TimeoutExpired, RuntimeError, ValueError) as error:
                    channel_errors.append(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                               attempt=attempt, error=str(error)))
                    if attempt < 3:
                        time.sleep(10)
            if data is None:
                reason = 'observation_channel_unavailable_not_training_exit'
                break
            point = compact(data)
            history.append(point)
            atomic(args.output_dir/'h20-twohead-lr10-progress-latest.json', data)
            atomic(args.output_dir/f'h20-twohead-observation-history-to{args.target}.json',
                   dict(target_step=args.target, channel_errors=channel_errors, observations=history))
            atomic(stem.with_suffix('.status.json'), dict(state='observing', latest=point,
                                                        observations=len(history)))
            failed = [key for key in FAIL_FIELDS if data[key] != 0]
            if not data['all_steps_contiguous'] or data['source_commit'] != TRAIN_SHA:
                failed.append('trajectory_identity')
            if data['exit'] is None:
                wrapper = data.get('wrapper') or {}
                if wrapper.get('pid') != args.owned_training_pid or not wrapper.get('expected_source'):
                    failed.append('owned_live_training_identity')
            for key, value in [('training', data['exit']), ('archive', data['checkpoint_archive_exit']),
                              ('final_SHA', data['final_archived_weight_readback_exit']),
                              ('portraits', data['portrait_follower']['exit'])]:
                if value is not None and str(value) != '0':
                    failed.append(key+'_nonzero_exit')
            if failed or data['errors']:
                reason = 'pipeline_failure_requires_inspection'
                print(json.dumps(dict(event=reason, failed_checks=failed, evidence=point)), flush=True)
                break
            if data['last_step'] > previous:
                last_advance = time.monotonic()
                if previous < 0 or previous//10 != data['last_step']//10:
                    print(json.dumps(dict(event='actual_training_progress', **point)), flush=True)
                previous = data['last_step']
            if data['exit'] is not None:
                # Clean process absence at500 is expected, not a live-PID failure.
                reason, code = 'formal_exit_requires_final_acceptance', 0
                break
            if (data['last_step'] > args.target and
                    args.target in data['final_archived_weight_verified_steps'] and
                    args.target in data['portrait_completed_steps']):
                reason, code = 'target_archive_portraits_and_postsave_iteration_verified', 0
                break
            if time.monotonic()-last_advance > 1800:
                reason = 'no_iteration_advance30min_inspect_IO_before_deciding'
                break
            if args.once:
                reason, code = 'one_live_observation_not_goal_completion', 0
                break
            time.sleep(args.period)
        else:
            reason, code = 'observation_window_ended_not_training_exit', 0
        return code
    except BaseException:
        (run_dir/'exception.log').write_text(traceback.format_exc())
        raise
    finally:
        status = dict(reason=reason, observer_exit=code, observations=len(history),
                      actual_training_exit=data['exit'] if data else None,
                      last_step=data['last_step'] if data else None,
                      channel_errors=channel_errors[-3:], goal_completion_proven=False,
                      run_dir=str(run_dir),
                      exception_log=str(run_dir/'exception.log') if (run_dir/'exception.log').exists() else None)
        atomic(run_dir/'status.json', status)
        (run_dir/'exit').write_text(str(code)+'\n')
        atomic(stem.with_suffix('.status.json'), status)
        stem.with_suffix('.exit').write_text(str(code)+'\n')
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
