"""Replace only the task's transfer controller, retaining its partial file."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/share-evpfs/shaoshitong/rdm-sets')
CONTROL = ROOT / 'control'
SOCKET = '/data/rdm-workspace/rdm-tmux.sock'
PARTIAL = ROOT / 'qib/judge/model-00001-of-00002.safetensors.curl-part'


def command(pid):
    return (Path(f'/proc/{pid}') / 'cmdline').read_bytes().split(b'\0')[:-1]


def wait_gone(pid):
    for _ in range(100):
        if not Path(f'/proc/{pid}').exists():
            return
        time.sleep(0.1)
    raise RuntimeError(f'Owned process {pid} did not terminate')


lock = (CONTROL / 'transfer-maintenance.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert (CONTROL / 'A.exit').read_text().strip() == '1'
assert not (CONTROL / 'B.started').exists()
assert not (CONTROL / 'judge-prefetch.exit').exists()
assert PARTIAL.stat().st_size < 49825162976
assert not (ROOT / 'zimage/work_dirs/zimage-2step-1024-dlr1e-4/train_log.jsonl').exists()
transfer = int((CONTROL / 'judge-prefetch.pid').read_text())
continuation = int((CONTROL / 'A-continuation.pid').read_text())
assert command(transfer) == [b'bash', bytes(CONTROL / 'judge-prefetch.sh')]
assert command(continuation) == [b'bash', bytes(CONTROL / 'continue-a-v2.sh')]
assert hashlib.sha256((CONTROL / 'judge-prefetch-v2.sh').read_bytes()).hexdigest() == '908ba75c0f9da886feb5177a4918bd87f2e6888187205e272dabbc9ad3e984ce'
if '--apply' not in sys.argv:
    print(json.dumps({'check_only': True, 'transfer_pid': transfer, 'continuation_pid': continuation, 'bytes': PARTIAL.stat().st_size}))
    raise SystemExit(0)

stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
archive = CONTROL / 'history' / f'transfer-controller-{stamp}'
archive.mkdir(parents=True)
record = {'utc': stamp, 'reason': 'Controlled transfer maintenance: budget consecutive failures without payload progress, not total interrupted streams. No training has started.', 'manual_goal_reactivation_unix': 1793636551, 'old_transfer_pid': transfer, 'waiting_continuation_pid': continuation, 'bytes_before': PARTIAL.stat().st_size}
(archive / 'maintenance.json').write_text(json.dumps(record, indent=2) + '\n')
stopped_continuation = False
stopped_transfer = False
try:
    os.kill(continuation, signal.SIGSTOP)
    stopped_continuation = True
    os.kill(transfer, signal.SIGSTOP)
    stopped_transfer = True
    children = (Path(f'/proc/{transfer}/task/{transfer}/children')).read_text().split()
    for value in children:
        pid = int(value)
        try:
            args = command(pid)
        except FileNotFoundError:
            continue
        name = Path(os.fsdecode(args[0])).name
        assert name in ('curl', 'sleep'), args
        if name == 'curl':
            assert os.fsencode(PARTIAL) in args
        os.kill(pid, signal.SIGTERM)
        # The stopped parent cannot reap an exited child until continued.
        for _ in range(100):
            stat = Path(f'/proc/{pid}/stat')
            if not stat.exists() or stat.read_text().split(') ', 1)[1].split()[0] == 'Z':
                break
            time.sleep(0.1)
        else:
            raise RuntimeError('Owned transfer child did not stop')
    os.kill(transfer, signal.SIGTERM)
    os.kill(transfer, signal.SIGCONT)
    stopped_transfer = False
    wait_gone(transfer)
    for _ in range(50):
        if subprocess.run(['tmux', '-S', SOCKET, 'has-session', '-t', 'rdm-judge-prefetch'], capture_output=True).returncode != 0:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError('Old transfer pane did not exit')
    oldlog = Path((CONTROL / 'judge-prefetch.logpath').read_text().strip())
    for suffix in ['pid', 'started', 'exit', 'finished', 'logpath']:
        p = CONTROL / f'judge-prefetch.{suffix}'
        if p.exists():
            shutil.move(str(p), str(archive / p.name))
    shutil.move(str(oldlog), str(archive / oldlog.name))
    newlog = ROOT / 'launch-logs' / f'judge-prefetch-v2-{stamp}.log'
    (CONTROL / 'judge-prefetch.logpath').write_text(str(newlog) + '\n')
    subprocess.run(['tmux', '-S', SOCKET, 'new-session', '-d', '-s', 'rdm-judge-prefetch', f"bash '{CONTROL}/judge-prefetch-v2.sh' > '{newlog}' 2>&1"], check=True)
    for _ in range(50):
        p = CONTROL / 'judge-prefetch.pid'
        if p.exists() and int(p.read_text()) != transfer:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError('Replacement transfer has no PID')
    record.update({'bytes_after': PARTIAL.stat().st_size, 'new_transfer_pid': int((CONTROL / 'judge-prefetch.pid').read_text()), 'new_log': str(newlog)})
    assert record['bytes_after'] >= record['bytes_before']
    (archive / 'maintenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
finally:
    for pid, stopped in [(transfer, stopped_transfer), (continuation, stopped_continuation)]:
        if stopped and Path(f'/proc/{pid}').exists():
            os.kill(pid, signal.SIGCONT)
