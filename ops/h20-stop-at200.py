#!/usr/bin/env python3
"""User-authorized one-time handoff after actual current-run step200 save."""
import datetime, json, os, pathlib, signal, subprocess, time

ROOT = pathlib.Path('/data/rdm-workspace/rdm-sets')
PREFIX = 'twohead-lr10-A-500'
EXPECTED = '/data/rdm-workspace/RDM-twohead-lr10-v1'
PID = 1677394


def write(name, value):
    p = ROOT/'control'/name
    tmp = p.with_name(p.name+'.partial')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); os.replace(tmp, p)


def main():
    authorization = dict(user_instruction='200iter after save, restart/resume; retain independent200/250/300/350/400 weights and every50 thereafter',
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        raw_messages=['200iter\u540e\u91cd\u65b0resume\u4e00\u4e0b\uff0c\u6bcf50iter\u7684\u6743\u91cd\u6700\u597d\u90fd\u4fdd\u5b58',
                      '200\uff0c250\uff0c300\uff0c350\uff0c400iter\u7684\u6743\u91cd'],
        target_pid=PID, target_source=EXPECTED, scope='Intentional stop of this launcher only after verified200 save; goal remains active and resume is authorized',
        G_Adam_limitation='Old FSDP checkpoint omits G Adam; disclosed that this authorized resume resets G Adam only')
    write('twohead-resume200-authorization.json', authorization)
    deadline = time.monotonic()+3600
    log = ROOT/'launch-logs'/f'{PREFIX}.log'
    marker = 'resume state (step 200)'
    while marker not in log.read_text(errors='replace'):
        assert time.monotonic() < deadline, 'Step200 save wait exceeded1h; no signal sent'
        exitp = ROOT/'control'/f'{PREFIX}.exit'
        assert not exitp.exists(), 'Training exited before authorized handoff'
        time.sleep(2)
    command = [str(ROOT/'envs/rdm/bin/python'),
        '/data/rdm-workspace/RDM-twohead-readonly-audit-v2/ops/h20-audit-twohead-checkpoint.py', '--step', '200']
    result = subprocess.run(command, text=True, capture_output=True, timeout=120,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'))
    assert result.returncode == 0, result.stderr[-1500:]+result.stdout[-1500:]
    audit = json.loads(result.stdout)
    assert audit['passed'] and audit['step'] == 200
    write('twohead-resume200-checkpoint-audit.json', audit)
    pidfile = ROOT/'control'/f'{PREFIX}.pid'
    assert int(pidfile.read_text()) == PID
    proc = pathlib.Path(f'/proc/{PID}')
    cmdline = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert EXPECTED+'/ops/h20-run-twohead-lr10-a.sh 2' in cmdline, 'Launcher identity changed; no signal sent'
    write('twohead-resume200-stop-intent.json', dict(authorization,
        save200_verified=True, signal='SIGTERM', issued_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    os.kill(PID, signal.SIGTERM)
    exitp = ROOT/'control'/f'{PREFIX}.exit'
    deadline = time.monotonic()+180
    while not exitp.exists():
        assert time.monotonic() < deadline, 'Await same launcher true exit; do not relaunch blindly'
        time.sleep(1)
    receipt = dict(authorization, passed=True, old_actual_exit=int(exitp.read_text()),
        classification='intentional-user-authorized-handoff-not-training-failure',
        stopped_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint_audit='twohead-resume200-checkpoint-audit.json')
    write('twohead-resume200-stop-verified.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    rc = 0
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        rc = 1
    finally:
        (ROOT/'control/twohead-resume200-guard.exit').write_text(str(rc)+'\n')
    raise SystemExit(rc)
