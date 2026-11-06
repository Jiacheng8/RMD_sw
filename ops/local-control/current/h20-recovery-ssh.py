import json,pathlib,subprocess,sys
state=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-11-01/ben-d/work/rdm-relay-state.json')
s=json.loads(state.read_text())
r=subprocess.run(['ssh','-F','/dev/null','-T','-S',s['control_socket'],'-oControlMaster=no','-oProxyCommand=/usr/bin/false','-oProxyJump=none','-oBatchMode=yes','jump_h20-1','python3 -'],input=sys.stdin.read(),text=True,capture_output=True,timeout=110)
print(r.stdout,end=''); print(r.stderr,end='',file=sys.stderr);sys.exit(r.returncode)
