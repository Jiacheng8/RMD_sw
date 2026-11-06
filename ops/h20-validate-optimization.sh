#!/usr/bin/env bash
# Explicitly authorized isolated checks, not the formal A/B experiments.
set -euo pipefail
REPO=/data/rdm-workspace/RDM-qwen
ROOT=/data/rdm-workspace/rdm-sets
source "$REPO/ops/h20-data-env.sh"
exec >>"$ROOT/launch-logs/validate-optimization.log" 2>&1
on_exit() {
  rc=$?; trap - EXIT
  printf '%s\n' "$rc" > "$ROOT/control/validate-optimization.exit"
  date -u +%FT%TZ > "$ROOT/control/validate-optimization.finished"
  printf 'OPTIMIZATION_VALIDATION_EXIT=%s\n' "$rc"
  exit "$rc"
}
trap on_exit EXIT
printf '%s\n' "$$" > "$ROOT/control/validate-optimization.pid"
date -u +%FT%TZ > "$ROOT/control/validate-optimization.started"
deadline=$((SECONDS + 43200))
while [[ ! -f "$ROOT/control/prepare-qwen.exit" ]]; do
  (( SECONDS < deadline )) || { echo 'Data preparation wait timed out'; exit 20; }
  sleep 30
done
[[ "$(cat "$ROOT/control/prepare-qwen.exit")" == 0 ]] || {
  echo 'Data preparation failed; GPU tests were not started'; exit 21;
}
source "$ROOT/env.sh"
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
cd "$REPO"
conda run -n rdm --no-capture-output python ops/h20-assemble-qwen.py
conda run -n rdm --no-capture-output python -m pytest \
  z-image/tests/test_zimage_multistep.py z-image/tests/test_qwen_text_store.py -q
conda run -n rdm --no-capture-output python -m pip install 'kernels==0.12.0'
exec 9>/data/rdm-workspace/rdm-run.lock
flock -n 9 || { echo 'Another experiment holds the run lock'; exit 22; }
python3 - <<'PY'
import subprocess
out = subprocess.check_output(['nvidia-smi','-i','4,5','--query-gpu=index,memory.used,utilization.gpu',
                               '--format=csv,noheader,nounits'], text=True)
for row in out.splitlines():
    index, memory, utilization = map(int, row.split(','))
    assert memory <= 32 and utilization == 0, row
PY
conda run -n rdm --no-capture-output python ops/h20-fa3-check.py
for spec in 'native 1' '_flash_3_varlen_hub 1' '_flash_3_varlen_hub 2'; do
  read -r backend micro <<< "$spec"
  conda run -n rdm --no-capture-output torchrun --nproc_per_node=2 --master_port=29531 \
    ops/h20-benchmark.py --backend "$backend" --micro-batch "$micro" --iterations 3
done
conda run -n rdm --no-capture-output python -m pip freeze > "$ROOT/control/validated-environment.txt"
echo 'OPTIMIZATION_CHECKS_COMPLETE; formal A/B training has not been started'
