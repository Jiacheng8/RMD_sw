"""Mac control entry: Python stdin over the existing owned ControlMaster only.

Backup of work/h20-recovery-ssh.py. Reads only the control socket from private
local state; never prints that state or creates a relay/direct-SSH fallback.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

state = Path(os.environ.get('RDM_RELAY_STATE_FILE',
    '/Users/shaoshitong/Documents/Codex/2026-11-01/ben-d/work/rdm-relay-state.json'))
socket = json.loads(state.read_text())['control_socket']
result = subprocess.run([
    'ssh', '-F', '/dev/null', '-T', '-S', socket, '-oControlMaster=no',
    '-oProxyCommand=/usr/bin/false', '-oProxyJump=none', '-oBatchMode=yes',
    'jump_h20-1', 'python3 -',
], input=sys.stdin.read(), text=True, capture_output=True, timeout=110)
print(result.stdout, end='')
print(result.stderr, end='', file=sys.stderr)
raise SystemExit(result.returncode)
