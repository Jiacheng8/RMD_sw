#!/usr/bin/env python3
"""Read-only A/B observations through the task's already owned SSH master.

This program never starts, stops, resumes, signals or writes to a remote task.
Observation expiry is distinct from experiment exit. No SSH fallback is allowed.
"""
import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time

SESSION_ROOT = Path('/Users/shaoshitong/Documents/Codex/2026-11-02/chat-thread-codex-threads-01a19ad1-3f77-2')
RELAY_STATE = Path('/Users/shaoshitong/Documents/Codex/2026-11-01/ben-d/work/rdm-relay-state.json')
SSH_HOST = 'jump_h20-1'
QUERY_TIMEOUT = 40

REMOTE_QUERY = r'''
import collections, datetime, json, math, os, pathlib, re, subprocess, sys
ROOT = pathlib.Path('/share-evpfs/shaoshitong/rdm-sets')
SETTING = sys.argv[1]
if SETTING not in ('A', 'B'):
    raise SystemExit(2)
EXP = {'A': 'zimage-2step-1024-dlr1e-4', 'B': 'zimage-2step-1024-dlr1e-4-stepw0.25'}[SETTING]
RUN = ROOT / 'zimage/work_dirs' / EXP
ERRORS = []

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def safe_path(path):
    path = pathlib.Path(path)
    path.resolve().relative_to(ROOT.resolve())
    return path

def text(path, limit=8192):
    try:
        with safe_path(path).open('rb') as handle:
            return handle.read(limit).decode('utf-8', 'replace').strip()
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        ERRORS.append({'file': str(path), 'error': 'read_failed_or_outside_root'})
        return None

def small_json(path):
    raw = text(path, 65536)
    if raw is None:
        return None
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else None
    except (ValueError, TypeError):
        ERRORS.append({'file': str(path), 'error': 'invalid_json'})
        return None

def file_stat(path):
    try:
        s = safe_path(path).stat()
        return {'bytes': s.st_size, 'mtime_utc': datetime.datetime.fromtimestamp(s.st_mtime, datetime.timezone.utc).isoformat()}
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        ERRORS.append({'file': str(path), 'error': 'stat_failed_or_outside_root'})
        return None

def redacted_command(raw):
    # Never return arbitrary process arguments: only recognize this task's fixed entrypoint.
    args = raw.rstrip(b'\0').split(b'\0')
    decoded = [a.decode('utf-8', 'replace') for a in args]
    known = {str(ROOT / 'control/run-one.sh'), str(ROOT / 'control/continue-a-v2.sh')}
    if len(decoded) in (2, 3) and pathlib.Path(decoded[0]).name in ('bash', 'sh') and decoded[1] in known:
        if len(decoded) == 2 or decoded[2] in ('A', 'B'):
            return ' '.join(decoded)
    return '[arguments withheld; command did not match fixed task entrypoints]'

def process(pid_raw):
    if not pid_raw or not re.fullmatch(r'[1-9][0-9]*', pid_raw):
        return {'live': False, 'reason': 'no_valid_pid'}
    proc = pathlib.Path('/proc') / pid_raw
    try:
        # Read no environ, open files, descriptors or unrelated process trees.
        status_text = (proc / 'status').read_text()
        status = {}
        for line in status_text.splitlines():
            key, _, value = line.partition(':')
            if key in ('Name', 'State', 'PPid'):
                status[key] = value.strip()
        cmd = redacted_command((proc / 'cmdline').read_bytes()[:8192])
        stat_fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'live': not status.get('State', '').startswith(('Z', 'X')), 'proc_status': status,
                'command': cmd, 'start_ticks': int(stat_fields[19]),
                'identity_scope': 'pid_file_current_proc_only_no_historical_start_tick_claim'}
    except FileNotFoundError:
        return {'live': False, 'reason': 'pid_absent'}
    except (OSError, ValueError, IndexError):
        return {'live': None, 'reason': 'proc_read_failed'}

def control(name):
    result = {k: text(ROOT / 'control' / (name + '.' + k), 128) for k in ('pid', 'exit', 'started', 'finished')}
    result.update(process(result['pid']))
    return result

BASE_KEYS = ('step', 'window_step', 'seconds', 'skipped', 'grad_norm', 'lr', 'force', 'raw_mmd2',
             'loss_total', 'cache_max_age', 'drift_probe_rows', 'drift_mean', 'refreshed_rows', 'refresh_drift_mean')
def selected(row):
    out = {k: row[k] for k in BASE_KEYS if k in row}
    for key, value in row.items():
        if re.fullmatch(r's[12]/gan/(lambda|g_loss|g_fake|grad_ratio|grad_ratio_raw|d_loss|d_skipped|d_acc)', key):
            out[key] = value
        elif re.fullmatch(r's[12]/gan/(dinov3_l|siglip2|aimv2_huge)/[gd]_loss', key):
            out[key] = value
        elif re.fullmatch(r'(drift_age_[0-9]+|s[12]/(dinov3_l|siglip2|aimv2_huge)/drift_(mean|max))', key):
            out[key] = value
    return out

def nonfinite(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(nonfinite(v) for v in value.values())
    if isinstance(value, list):
        return any(nonfinite(v) for v in value)
    return False

def train_summary():
    path = RUN / 'train_log.jsonl'
    out = {'record_count': 0, 'last_step': None, 'first_step': None, 'contiguous': None,
           'complete_line_sequence_contiguous': None, 'evidence_complete': False,
           'parse_errors': 0, 'invalid_step_records': 0, 'sequence_error_count': 0,
           'sequence_error_examples': [], 'partial_final_line_bytes': 0,
           'skipped_count': 0, 'skipped_steps': [], 'nonfinite_count': 0, 'nonfinite_steps': [],
           'critic_skipped_count': 0, 'critic_skipped_steps': [], 'train_recent': [],
           'missing_or_invalid_skipped_count': 0, 'missing_or_invalid_skipped_steps': [],
           'probe_records': [], 'GAN_generator_first_logged_step': None}
    out['train_log_stat'] = file_stat(path)
    if out['train_log_stat'] is None:
        return out
    recent, probes = collections.deque(maxlen=2), collections.deque(maxlen=6)
    previous = None
    try:
        with safe_path(path).open('rb') as handle:
            remaining = out['train_log_stat']['bytes']
            while remaining > 0:
                line = handle.readline(remaining)
                if not line:
                    break
                remaining -= len(line)
                if not line.endswith(b'\n'):
                    out['partial_final_line_bytes'] = len(line)
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError()
                except (ValueError, UnicodeError):
                    out['parse_errors'] += 1
                    continue
                out['record_count'] += 1
                step = row.get('step')
                if not isinstance(row.get('skipped'), bool):
                    out['missing_or_invalid_skipped_count'] += 1
                    if len(out['missing_or_invalid_skipped_steps']) < 20:
                        out['missing_or_invalid_skipped_steps'].append(step if isinstance(step, int) and not isinstance(step, bool) else None)
                if isinstance(step, bool) or not isinstance(step, int) or step < 1:
                    out['invalid_step_records'] += 1
                    continue
                if out['first_step'] is None:
                    out['first_step'] = step
                expected = 1 if previous is None else previous + 1
                if step != expected:
                    out['sequence_error_count'] += 1
                    if len(out['sequence_error_examples']) < 10:
                        out['sequence_error_examples'].append({'expected': expected, 'actual': step})
                previous = step
                out['last_step'] = step
                for condition, stem in ((bool(row.get('skipped')), 'skipped'),
                                        (nonfinite(row), 'nonfinite'),
                                        (any(bool(v) for k, v in row.items() if k.endswith('/gan/d_skipped') or k == 'gan/d_skipped'), 'critic_skipped')):
                    if condition:
                        out[stem + '_count'] += 1
                        if len(out[stem + '_steps']) < 20:
                            out[stem + '_steps'].append(step)
                if any(k.endswith('/gan/g_loss') for k in row) and out['GAN_generator_first_logged_step'] is None:
                    out['GAN_generator_first_logged_step'] = step
                recent.append(selected(row))
                if 'drift_probe_rows' in row or 'refreshed_rows' in row:
                    probes.append({k: row[k] for k in ('step', 'drift_probe_rows', 'drift_mean', 'refreshed_rows', 'refresh_drift_mean') if k in row})
    except (OSError, ValueError):
        ERRORS.append({'file': str(path), 'error': 'train_scan_failed'})
        out['scan_failed'] = True
    out['contiguous'] = bool(out['record_count']) and not any(out[k] for k in ('parse_errors', 'invalid_step_records', 'sequence_error_count')) and not out.get('scan_failed', False)
    out['complete_line_sequence_contiguous'] = out['contiguous']
    out['evidence_complete'] = out['contiguous'] and out['missing_or_invalid_skipped_count'] == 0 and out['partial_final_line_bytes'] == 0
    out['train_recent'], out['probe_records'] = list(recent), list(probes)
    return out

def log_errors(path):
    # Only classify the final 30 lines; never transmit raw log lines or exception messages.
    info = file_stat(path)
    if info is None:
        return {'stat': None, 'tail_lines_examined': 0, 'hits': []}
    try:
        with safe_path(path).open('rb') as handle:
            size = info['bytes']
            offset = max(0, size - 65536)
            handle.seek(offset)
            data = handle.read(size - offset)
        lines = data.decode('utf-8', 'replace').splitlines()
        if offset and lines:
            lines = lines[1:]
        tail = lines[-30:]
        patterns = {'traceback': r'\bTraceback\b', 'error': r'\berror\b|\w+Error:',
                    'exception': r'\bexception\b|\w+Exception:', 'cuda_oom': r'out of memory|CUDA.*OOM',
                    'nonfinite': r'\b(nan|inf|non.?finite)\b', 'failed': r'\b(failed|failure|fatal)\b'}
        hits = []
        for i, line in enumerate(tail):
            categories = [name for name, pattern in patterns.items() if re.search(pattern, line, re.I)]
            if categories:
                hits.append({'tail_index': i, 'categories': categories})
        stages = re.findall(r'stage ([1-7])/7: ([A-Za-z0-9 -]+)', data.decode('utf-8', 'replace'))
        return {'stat': info, 'tail_lines_examined': len(tail), 'hits': hits,
                'last_stage_in_tail': ('stage ' + stages[-1][0] + '/7: ' + stages[-1][1].strip()) if stages else None,
                'raw_lines_withheld': True}
    except (OSError, ValueError):
        return {'stat': info, 'error': 'tail_read_failed'}

def gpu():
    try:
        p = subprocess.run(['nvidia-smi', '-i', '4,5', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=12)
        if p.returncode:
            return {'ok': False, 'error': 'nvidia_smi_nonzero', 'returncode': p.returncode}
        rows = []
        for line in p.stdout.splitlines():
            values = [int(x.strip()) for x in line.split(',')]
            if len(values) != 4 or values[0] not in (4, 5):
                raise ValueError()
            rows.append(dict(zip(('index', 'memory_used_mib', 'memory_total_mib', 'utilization_percent'), values)))
        return {'ok': {r['index'] for r in rows} == {4, 5}, 'rows': rows}
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'nvidia_smi_timeout'}
    except (OSError, ValueError):
        return {'ok': False, 'error': 'nvidia_smi_unavailable_or_invalid'}

def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value

out = {'schema_version': 1, 'utc': utc(), 'setting': SETTING, 'read_only': True,
       'source_verification': {'performed': False, 'reason': 'observer_does_not_query_scientific_git_or_config'}}
for name in ('A', 'B', 'A-continuation'):
    out[name] = control(name)
out.update(train_summary())
out['log_tail_error_scan'] = log_errors(ROOT / 'launch-logs' / (SETTING + '.log'))
stage = out['log_tail_error_scan'].get('last_stage_in_tail')
out['stage'] = stage or ('training_records_observed' if out['last_step'] is not None else 'stage_not_in_bounded_log_tail')
out['checkpoint_stats'] = {}
for pattern in ('step_*.pth', 'step_*.pth.partial', 'resume.pth', 'resume.pth.partial'):
    for path in sorted(RUN.glob(pattern))[:20]:
        out['checkpoint_stats'][path.name] = file_stat(path)
wandb = small_json(RUN / 'wandb.json') or {}
out['wandb'] = {k: wandb[k] for k in ('id', 'last_step', 'evals_done') if k in wandb}
out['wandb']['online_sync_verified'] = False
out['offline_files'] = []
for path in sorted((RUN / 'wandb').glob('offline-run-*/run-*.wandb'))[:10]:
    item = file_stat(path)
    if item:
        out['offline_files'].append({'name': str(path.relative_to(RUN)), **item})
out['offline_bytes'] = sum(x['bytes'] for x in out['offline_files'])
out['result_table_stat'] = file_stat(RUN / 'qwen_image_bench.md')
out['A_reported'] = (ROOT / 'control/A.reported').is_file()
out['B_started'] = out['B']['started'] is not None
out['gpu4_5'] = gpu()
out['read_errors'] = ERRORS[:20]
out['observation_completed_utc'] = utc()
print(json.dumps(clean(out), separators=(',', ':'), allow_nan=False))
'''


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def observe(setting):
    try:
        state = json.loads(RELAY_STATE.read_text())
        sock = Path(state.get('control_socket', ''))
        if state.get('status') != 'owned':
            return {'ok': False, 'error': 'relay_not_owned'}
        sock.resolve().relative_to(RELAY_STATE.parent.resolve())
        if not stat.S_ISSOCK(sock.stat().st_mode):
            return {'ok': False, 'error': 'control_socket_not_socket'}
    except (OSError, ValueError, TypeError):
        return {'ok': False, 'error': 'relay_state_or_socket_unavailable'}
    args = ['ssh', '-F', '/dev/null', '-T', '-S', str(sock), '-oControlMaster=no',
            '-oProxyCommand=/usr/bin/false', '-oProxyJump=none', '-oBatchMode=yes',
            '-oConnectTimeout=10', '-oConnectionAttempts=1', SSH_HOST, 'python3 - ' + setting]
    try:
        completed = subprocess.run(args, input=REMOTE_QUERY, text=True, capture_output=True, timeout=QUERY_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'observation_ssh_timeout'}
    except OSError:
        return {'ok': False, 'error': 'observation_ssh_unavailable'}
    if completed.returncode:
        # Never dump SSH stderr: it could include configuration details.
        return {'ok': False, 'error': 'observation_ssh_nonzero', 'returncode': completed.returncode}
    if len(completed.stdout) > 262144:
        return {'ok': False, 'error': 'observation_payload_exceeds_limit'}
    try:
        observation = json.loads(completed.stdout)
        if observation.get('setting') != setting or observation.get('read_only') is not True:
            raise ValueError()
        return {'ok': True, 'observation': observation}
    except (ValueError, TypeError, AttributeError):
        return {'ok': False, 'error': 'invalid_observation_json'}


def save(result, setting, deadline_utc, reason, success_count, failure_count):
    # All observer instances serialize this shared read/merge/replace. Any separate
    # local progress writer must use the same advisory lock to avoid lost updates.
    lock_path = SESSION_ROOT / 'work/h20-progress.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _save_locked(result, setting, deadline_utc, reason, success_count, failure_count)


def _save_locked(result, setting, deadline_utc, reason, success_count, failure_count):
    latest_path = SESSION_ROOT / 'work' / ('h20-' + setting + '-observed-latest.json')
    progress_path = SESSION_ROOT / 'outputs/h20-progress-latest.json'
    stamp = utc()
    try:
        progress = json.loads(progress_path.read_text())
        if not isinstance(progress, dict):
            raise ValueError()
    except FileNotFoundError:
        progress = {}
    # Invalid existing JSON is never silently replaced and its evidence pointers are preserved.
    observation = result.get('observation')
    previous = progress.get(setting + '_latest_observation') or {}
    last_observation_utc = (observation or previous).get('observation_completed_utc') or (observation or previous).get('utc')
    metadata = {'status_updated_utc': stamp, 'last_observation_utc': last_observation_utc,
                'setting': setting, 'deadline_utc': deadline_utc,
                'stop_reason': reason, 'successful_queries': success_count, 'failed_queries': failure_count,
                'last_query_ok': result['ok'], 'source_current_verified': False}
    if result['ok']:
        snapshot = dict(observation)
        snapshot['observer'] = metadata
        atomic_json(latest_path, snapshot)
        progress['checked_utc'] = last_observation_utc
        progress['stage'] = observation['stage']
        progress[setting + '_steps'] = observation['last_step']
        progress[setting + '_training_steps'] = observation['last_step']
        progress[setting + '_exit'] = observation[setting]['exit']
        progress[setting + '_latest_observation'] = observation
        progress['B_started'] = observation['B_started']
        progress['latest_observation_path'] = str(latest_path)
        if observation['GAN_generator_first_logged_step'] is not None:
            progress['GAN_generator_first_logged_step'] = observation['GAN_generator_first_logged_step']
        # No new source check, phase acceptance, publication or online-sync assertion is made.
    else:
        try:
            snapshot = json.loads(latest_path.read_text())
            if not isinstance(snapshot, dict):
                raise ValueError()
        except FileNotFoundError:
            snapshot = {}
        snapshot['observer'] = {**metadata, 'error': result['error']}
        atomic_json(latest_path, snapshot)
        metadata['error'] = result['error']
    progress['observer'] = metadata
    progress['observer_deadline_utc'] = deadline_utc
    progress['observer_finished_normally'] = result['ok'] and reason in ('once_complete', 'observation_period_elapsed', 'experiment_exit_recorded')
    atomic_json(progress_path, progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=('A', 'B'), default='A')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--duration-seconds', type=float, default=3600)
    parser.add_argument('--interval-seconds', type=float, default=45)
    args = parser.parse_args()
    if not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
        parser.error('--duration-seconds must be positive and finite')
    if not math.isfinite(args.interval_seconds) or args.interval_seconds <= 0:
        parser.error('--interval-seconds must be positive and finite')
    started = time.monotonic()
    deadline = started + args.duration_seconds
    deadline_utc = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=args.duration_seconds)).isoformat()
    successes = failures = 0
    result = {'ok': False, 'error': 'not_observed'}
    try:
        while True:
            result = observe(args.setting)
            successes += int(result['ok'])
            failures += int(not result['ok'])
            experiment_exit = result.get('observation', {}).get(args.setting, {}).get('exit')
            if experiment_exit is not None and experiment_exit != '':
                reason = 'experiment_exit_recorded'
            elif args.once:
                reason = 'once_complete' if result['ok'] else 'once_observation_failed'
            elif time.monotonic() >= deadline:
                reason = 'observation_period_elapsed' if result['ok'] else 'observation_period_elapsed_after_query_failure'
            else:
                reason = 'observing'
            save(result, args.setting, deadline_utc, reason, successes, failures)
            print(json.dumps({'checked_utc': utc(), 'observer_stop_reason': reason, **result}, separators=(',', ':'), allow_nan=False), flush=True)
            if reason != 'observing':
                return 0 if result['ok'] else 1
            time.sleep(min(args.interval_seconds, max(0, deadline - time.monotonic())))
            if time.monotonic() >= deadline:
                expired_reason = 'observation_period_elapsed' if result['ok'] else 'observation_period_elapsed_after_query_failure'
                save(result, args.setting, deadline_utc, expired_reason, successes, failures)
                print(json.dumps({'status_updated_utc': utc(), 'observer_stop_reason': expired_reason,
                                  'last_observation_utc': result.get('observation', {}).get('observation_completed_utc'),
                                  'last_query_ok': result['ok'],
                                  'experiment_exit': experiment_exit, 'experiment_exit_observed': False,
                                  'note': 'Observation window ended; this does not assert experiment completion.'}), flush=True)
                return 0 if result['ok'] else 1
    except KeyboardInterrupt:
        save(result, args.setting, deadline_utc, 'observer_interrupted', successes, failures)
        print(json.dumps({'checked_utc': utc(), 'observer_stop_reason': 'observer_interrupted', 'remote_action': 'none'}), flush=True)
        return 130
    except (OSError, ValueError, TypeError):
        print(json.dumps({'checked_utc': utc(), 'observer_stop_reason': 'local_audit_write_failed',
                          'remote_action': 'none', 'error_details_withheld': True}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
