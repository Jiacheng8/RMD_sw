#!/usr/bin/env python3
"""Retain the user-requested identical portraits at future existing saves.

Never changes, starts, or restarts training; preserves PNGs, never checkpoint copies.
"""
import datetime, json, os, pathlib, subprocess, time
from h20_portrait_checkpoint import archived_checkpoint

ROOT = pathlib.Path('/data/rdm-workspace/rdm-sets')
EXPERIMENT = ROOT/'zimage/work_dirs/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A'
OUT = ROOT/'visualizations/twohead-lr10-A/qib-portraits-v1'
REPO = pathlib.Path(__file__).resolve().parents[1]
PREFIX = os.environ.get('H20_CONTROL_PREFIX','twohead-lr10-A-500')
FOLLOW_PREFIX = os.environ.get('H20_PORTRAIT_FOLLOW_PREFIX','twohead-qib-portraits-follow')


def write(value):
    p = ROOT/'control'/f'{FOLLOW_PREFIX}-progress.json'
    value['observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tmp = p.with_suffix('.partial'); tmp.write_text(json.dumps(value, indent=2)+'\n'); os.replace(tmp, p)


def main():
    deadline = time.monotonic()+36*3600
    pending = list(range(200, 501, 50))
    results = []
    while pending and time.monotonic() < deadline:
        initial = OUT/'step_0000000/preview-verified.json'
        first_exit = ROOT/'control/twohead-qib-portraits-150.exit'
        if first_exit.exists() and first_exit.read_text().strip() != '0':
            raise RuntimeError('Initial0/150 preview failed; inspect retained command and log')
        target = pending[0]
        verified = archived_checkpoint(target)
        if initial.exists() and first_exit.exists() and first_exit.read_text().strip() == '0':
            proof_path = OUT/f'step_{target:07d}'/'preview-verified.json'
            exit_path = ROOT/f'control/twohead-qib-portraits-{target}.exit'
            if proof_path.exists() and exit_path.exists() and exit_path.read_text().strip() == '0':
                assert json.loads(proof_path.read_text())['passed']
                results.append(dict(step=target, status='verified_existing')); pending.pop(0)
            elif verified is not None:
                used, total = map(int, subprocess.check_output(['nvidia-smi','-i','6',
                    '--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'], text=True).strip().split(','))
                if total-used >= 48*1024:
                    command = ['bash', str(REPO/'ops/h20-run-qib-portrait-preview.sh'), str(target), '6',
                               '--checkpoint', str(verified[0])]
                    print(json.dumps(dict(event='preview_start', step=target, command=command)), flush=True)
                    proc = subprocess.run(command, timeout=1860)
                    if proc.returncode != 0:
                        write(dict(pending=pending, results=results, failed_step=target, exit=proc.returncode))
                        raise RuntimeError(f'Stage{target} portrait failed with true exit{proc.returncode}')
                    proof = json.loads(proof_path.read_text())
                    assert proof['passed'] and len(proof['files']) == 16 and len(proof['native_comparisons']) == 4
                    results.append(dict(step=target, status='verified', images=16)); pending.pop(0)
        write(dict(input_mode='verified-independent-archive', waiting_step=pending[0] if pending else None,
                   archive_ready=verified is not None, pending=pending, results=results))
        training_exit = ROOT/'control'/f'{PREFIX}.exit'
        if pending and training_exit.exists() and training_exit.read_text().strip() != '0':
            raise RuntimeError('Training exited nonzero; portrait follow-up stops without restart')
        if pending:
            time.sleep(20)
    assert not pending, 'Portrait follow-up exceeded36h deadline'
    write(dict(passed=True, pending=[], results=results))


if __name__ == '__main__':
    main()
