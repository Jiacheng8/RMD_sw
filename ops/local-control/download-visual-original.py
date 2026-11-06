#!/usr/bin/env python3
"""Binary visual readback over the existing owned ControlMaster, with SHA256 verification."""
import argparse, hashlib, json, os, pathlib, re, subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--remote', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sha256', required=True)
    a = ap.parse_args()
    allowed = '/data/rdm-workspace/rdm-sets/visualizations/twohead-lr10-A/'
    path = pathlib.PurePosixPath(a.remote)
    assert a.remote.startswith(allowed) and '..' not in path.parts and path.suffix in ('.png', '.jpg', '.json')
    assert re.fullmatch('[0-9a-f]{64}', a.sha256)
    state_file = pathlib.Path(os.environ.get('RDM_RELAY_STATE_FILE',
        '/Users/shaoshitong/Documents/Codex/2026-11-01/ben-d/work/rdm-relay-state.json'))
    socket = json.loads(state_file.read_text())['control_socket']
    target = pathlib.Path(a.out); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.part')
    code = 'import pathlib,sys,shutil\np=pathlib.Path('+repr(a.remote)+')\nwith p.open("rb") as f: shutil.copyfileobj(f,sys.stdout.buffer,1024*1024)\n'
    command = ['ssh', '-F', '/dev/null', '-T', '-S', socket, '-oControlMaster=no',
               '-oProxyCommand=/usr/bin/false', '-oProxyJump=none', '-oBatchMode=yes',
               'jump_h20-1', 'python3 -']
    with temporary.open('wb') as output:
        result = subprocess.run(command, input=code.encode(), stdout=output, stderr=subprocess.PIPE, timeout=180)
    assert result.returncode == 0, result.stderr.decode(errors='replace')[-1000:]
    digest = hashlib.sha256()
    with temporary.open('rb') as source:
        for chunk in iter(lambda: source.read(8*1024**2), b''):
            digest.update(chunk)
    assert digest.hexdigest() == a.sha256, 'Downloaded bytes failed SHA256; partial retained'
    os.replace(temporary, target)
    print(json.dumps(dict(passed=True, path=str(target), bytes=target.stat().st_size, sha256=a.sha256)), flush=True)


if __name__ == '__main__':
    main()
