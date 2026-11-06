#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
export CUDA_VISIBLE_DEVICES='' TMPDIR="$ROOT/.cache/tmp" PIP_NO_CACHE_DIR=1 PYTHONNOUSERSITE=1
ENV="$ROOT/qib/env"
[[ ! -f "$ENV/.ready" ]] || exit 0
"$ROOT/envs/rdm/bin/python" -m venv --system-site-packages "$ENV"
"$ENV/bin/python" -m pip install --no-cache-dir 'ms-swift==4.6.0' 'transformers==5.17.0' \
  'torch==2.8.0' 'torchvision==0.23.0' 'flash-linear-attention==0.5.2' 'fla-core==0.5.2' \
  'qwen-vl-utils>=0.0.14' 'pandas>=2' 'openpyxl>=3.1' 'Pillow>=10' tqdm
"$ENV/bin/python" - <<'PY'
import importlib.metadata as m, json, pathlib
import swift, transformers, torch
import fla.ops.gated_delta_rule
assert torch.__version__.startswith('2.8.0') and transformers.__version__=='5.17.0'
assert m.version('ms-swift')=='4.6.0'
p=pathlib.Path('/data/rdm-workspace/rdm-sets/qib/env/.ready')
p.write_text(json.dumps(dict(passed=True,versions={n:m.version(n) for n in
    ['torch','torchvision','transformers','ms-swift','flash-linear-attention','fla-core']}),indent=2)+'\n')
print(p.read_text(),flush=True)
PY
