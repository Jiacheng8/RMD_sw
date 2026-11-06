"""Reap only this scan's ranks if its controller/elastic parent disappears.

The active scan is not changed. Each target must have the exact isolated benchmark
script and the unique scan output directory in argv, plus unchanged PID/start ticks.
This covers elastic workers which start their own sessions/process groups.
"""
import datetime as dt
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path('/data/rdm-workspace/rdm-sets')
STATE = ROOT / 'control/checkpoint-sweep-latest.json'
while not STATE.exists():
    time.sleep(1)
scan = json.loads(STATE.read_text())
SOURCE = Path(scan['source'])
OUTPUT = Path(scan['output'])
ENTRY = str(SOURCE / 'ops/h20-benchmark.py')
CONTROLLER = str(SOURCE / 'ops/h20-checkpoint-sweep.py')
RECORD = ROOT / 'control/checkpoint-sweep-guard.json'


def process_info(pid):
    try:
        p = Path('/proc') / str(pid)
        argv = (p / 'cmdline').read_bytes().split(b'\0')[:-1]
        stat = (p / 'stat').read_text().split(') ', 1)[1].split()
        return {'pid': int(pid), 'start_ticks': int(stat[19]),
                'argv': [os.fsdecode(a) for a in argv], 'state': stat[0],
                'cwd': os.readlink(p / 'cwd')}
    except (OSError, ValueError):
        return None


def inventory():
    controllers, parents, workers = [], [], []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        item = process_info(p.name)
        if not item or item['state'] == 'Z':
            continue
        args = item['argv']
        if any(a.endswith('ops/h20-checkpoint-sweep.py') and
               (Path(item['cwd']) / a).resolve() == Path(CONTROLLER).resolve() for a in args):
            controllers.append(item)
        if ENTRY not in args or '--out-dir' not in args:
            continue
        out = Path(args[args.index('--out-dir') + 1])
        if OUTPUT not in out.parents:
            continue
        item['case'] = str(out)
        if any(Path(a).name == 'torchrun' for a in args):
            parents.append(item)
        else:
            workers.append(item)
    return controllers, parents, workers


def stop_owned(item, sig):
    current = process_info(item['pid'])
    if not current or current['start_ticks'] != item['start_ticks']:
        return
    if current['argv'] != item['argv']:
        return
    try:
        os.kill(item['pid'], sig)
    except ProcessLookupError:
        pass


events = []
confirmed_controller = None
confirmation_deadline = time.monotonic() + 30
while True:
    controllers, parents, workers = inventory()
    if confirmed_controller is None:
        if len(controllers) == 1:
            confirmed_controller = (controllers[0]['pid'], controllers[0]['start_ticks'])
        elif time.monotonic() >= confirmation_deadline:
            raise RuntimeError('Controller identity was never confirmed; refusing any cleanup')
        else:
            time.sleep(1)
            continue
    # A transient argv/read failure is not proof of exit. Only the kernel's PID,
    # start ticks and zombie state establish that the confirmed owner is gone.
    owner = process_info(confirmed_controller[0])
    if owner is None and (Path('/proc') / str(confirmed_controller[0])).exists():
        time.sleep(1)
        continue
    owner_alive = owner is not None and owner['start_ticks'] == confirmed_controller[1] and owner['state'] != 'Z'
    controllers = [owner] if owner_alive else []
    targets = [] if owner_alive else parents + workers
    if targets:
        event = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                 'reason': 'confirmed controller PID exited',
                 'targets': [{k: x[k] for k in ('pid', 'start_ticks', 'case')} for x in targets]}
        events.append(event)
        for target in targets:
            stop_owned(target, signal.SIGTERM)
        time.sleep(10)
        for target in targets:
            stop_owned(target, signal.SIGKILL)
    record = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'scan_output': str(OUTPUT),
              'controllers': [x['pid'] for x in controllers], 'parents': [x['pid'] for x in parents],
              'workers': [x['pid'] for x in workers], 'cleanup_events': events}
    tmp = RECORD.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, RECORD)
    if not controllers:
        # One last inventory catches a child created while an elastic parent terminated.
        _, remaining_parents, remaining_workers = inventory()
        if not remaining_parents and not remaining_workers:
            break
    time.sleep(1)
