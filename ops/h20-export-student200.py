#!/usr/bin/env python3
"""Export the actual saved200 FP32 model once, without critics/caches or G changes."""
import json, os, pathlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
torch.set_num_threads(1)
root = pathlib.Path('/data/rdm-workspace/rdm-sets')
folder = root/'zimage/work_dirs/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A'
proof = json.loads((root/'control/twohead-resume200-stop-verified.json').read_text())
assert proof['passed'] and proof['classification'].startswith('intentional-user-authorized')
state = torch.load(folder/'resume.pth', map_location='cpu', weights_only=False, mmap=True)
assert state['step'] == state['schedule']['step'] == 200
assert len(state['model']) == 521 and all(v.dtype == torch.float32 for v in state['model'].values())
student = {k:v for k,v in state.items() if k not in ['optimizer','critic','critics','cache','spatial_cache']}
path = folder/'step_0000200.pth'
assert not path.exists()
torch.save(student, str(path)+'.partial')
os.replace(str(path)+'.partial', path)
print(json.dumps(dict(passed=True, step=200, path=str(path), bytes=path.stat().st_size,
    dtype='float32', source='actual saved200 resume; no weight cast or modification')), flush=True)
