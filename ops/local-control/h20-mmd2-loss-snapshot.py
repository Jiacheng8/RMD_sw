#!/usr/bin/env python3
"""Read existing MMD2 scalar records through the existing SSH helper. No GPU/model loads."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path('/data/rdm-workspace/rdm-sets')
CONTROL=ROOT/'control'
EXP=ROOT/'zimage/work_dirs/zimage-2step-1024-dino-qwenvl-mmd2-from0'
PREFIX='dino-qwenvl-mmd2-from0'
# This reader itself is supplied on stdin from a later, published code-only tree.
# Import its small contract module from the cold visualization deployment, never hot-edit SOURCE.
CONTRACT=Path('/data/rdm-workspace/RDM-dino-qwenvl-mmd2-v8/ops/h20-mmd2-loss-data.py')
spec=importlib.util.spec_from_file_location('mmd2_loss_contract',CONTRACT)
contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
command_path=CONTROL/(PREFIX+'-formal.command.json')
if not command_path.exists():
    print(json.dumps(dict(status='not_started',rows=[],experiment=EXP.name,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
    raise SystemExit(0)
command=json.loads(command_path.read_text().replace('/data/shaoshitong/', '/data/rdm-workspace/'));preflight=json.loads((CONTROL/(PREFIX+'-preflight.json')).read_text())
SOURCE=Path(command['source']).resolve()
assert SOURCE.is_relative_to('/data/rdm-workspace') and SOURCE.name.startswith('RDM-dino-qwenvl-mmd2-v')
SHA=command['source_commit']
assert command['source']==str(SOURCE) and command['source_commit']==SHA and command['CUDA_VISIBLE_DEVICES']=='4,5'
assert (SOURCE/'SOURCE_COMMIT').read_text().strip()==SHA
assert preflight['passed'] and preflight['source_commit']==SHA and preflight['init']=='official pretrained0'
assert preflight['MMD_encoders']==preflight['GAN_encoders']==contract.NAMES
assert not any('load_from' in x or 'resume_from' in x for x in command['command'])
path=EXP/'train_log.jsonl'
raw=path.read_bytes() if path.exists() else b''
complete=raw[:raw.rfind(b'\n')+1];ignored=len(raw)-len(complete)
rows=[json.loads(line) for line in complete.splitlines() if line.strip()]
if not rows:
    print(json.dumps(dict(status='initializing_no_complete_iteration',rows=[],source_commit=SHA,experiment=EXP.name,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),ignored_incomplete_log_bytes=ignored)))
    raise SystemExit(0)
contract.validate_rows(rows)
select=contract.keys()
actions=[{k:r['gan/d_lr_control'][k] for k in ['completed_step','effective_from_step','old_lr','new_lr','action','reason']}
    for r in rows if r['gan/d_lr_control']['changed']]
config=SOURCE/'z-image/configs/zimage_2step_1024_dino_qwenvl_mmd2_from0.yaml'
payload=dict(status='actual_completed_iterations',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_commit=SHA,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),experiment=EXP.name,
    init='official pretrained0',restart_boundaries=[],MMD_encoders=contract.NAMES,GAN_encoders=contract.NAMES,
    GAN_G_first_iteration=51,D_lr_actions=actions,rows=[{k:r[k] for k in select if k in r} for r in rows],
    ignored_incomplete_log_bytes=ignored,log_prefix_sha256=hashlib.sha256(complete).hexdigest(),
    actual_formal_exit=json.loads((CONTROL/(PREFIX+'-formal.exit.json')).read_text()) if (CONTROL/(PREFIX+'-formal.exit.json')).exists() else None)
contract.validate_payload(payload)
print(json.dumps(payload,allow_nan=False))
