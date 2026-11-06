#!/usr/bin/env bash
# Authorized incident recovery; original source/config and historical logs stay intact.
set -euo pipefail
MODE=${1:---preflight}
[[ "$MODE" == --preflight || "$MODE" == --run ]] || exit 2
ROOT=/share-evpfs/shaoshitong/rdm-sets
REPO=/data/rdm-workspace/RDM
CONTROL=/data/rdm-workspace/rdm-control
ATTEMPT="$CONTROL/A-recovery-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$ATTEMPT"
exec >>"$ATTEMPT/controller.log" 2>&1
on_exit() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$ATTEMPT/exit"
  date -u +%FT%TZ > "$ATTEMPT/finished"
  printf 'RECOVERY_EXIT=%s\n' "$rc"
  exit "$rc"
}
trap on_exit EXIT
printf '%s\n' "$$" > "$ATTEMPT/pid"
date -u +%FT%TZ > "$ATTEMPT/started"
printf 'MODE=%s ATTEMPT=%s\n' "$MODE" "$ATTEMPT"
exec 9>"/data/rdm-workspace/rdm-run.lock"
flock -n 9 || { echo 'Existing experiment holds the run lock'; exit 3; }
test -d "$ROOT" || { echo "STORAGE_UNAVAILABLE: $ROOT is absent or inaccessible"; exit 20; }
# Reject a bare directory exposed on the full system disk after an unmount.
[[ "$(stat -c %d "$ROOT")" != "$(stat -c %d /)" ]] || {
  echo 'STORAGE_UNAVAILABLE: ROOT resolves to the system disk'; exit 20;
}
PIN=e1670e86448d43295e7170443fe4c6b339346812
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$PIN" ]]
[[ "$(git -C "$REPO" branch --show-current)" == zimage ]]
git -C "$REPO" diff --quiet
git -C "$REPO" diff --cached --quiet
CONFIG="$REPO/z-image/configs/zimage_2step_1024_dlr1e-4.yaml"
[[ "$(sha256sum "$CONFIG" | cut -d' ' -f1)" == f2083054d8bcdfc006d39068f25dafbbc899884ff39db7839db54863c1091ead ]]
RUN="$ROOT/zimage/work_dirs/zimage-2step-1024-dlr1e-4"
test -s "$RUN/resume.pth"
test -s "$ROOT/zimage/zimage-ref-v1/reference_store/metadata.json"
test -s "$ROOT/zimage/zimage-ref-v1/contexts/manifest.json"
test -x "$ROOT/envs/rdm/bin/python"
test -s "$ROOT/control/env.sh"
python3 - "$ROOT" "$CONFIG" "$ATTEMPT/config.yaml" <<'PY'
import os, re, sys, tempfile
root, original, target = sys.argv[1:]
free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
if free < 200 * 1024**3:
    raise SystemExit('Less than 200 GiB available for five student saves and atomic resume replacement')
with tempfile.NamedTemporaryFile(dir=root, prefix='.A-recovery-preflight-', delete=True) as f:
    f.write(b'fsync-check'); f.flush(); os.fsync(f.fileno())
text = open(original).read()
for key in ('save_freq', 'resume_every'):
    text, n = re.subn(r'^' + key + r':[^\n]*', key + ': 50', text, flags=re.M)
    assert n == 1, key
open(target, 'w').write(text)
print('Only config overrides: save_freq=50 resume_every=50; total steps remains 500')
PY
source "$ROOT/control/env.sh"
export PYTHONUNBUFFERED=1
"$ROOT/envs/rdm/bin/python" - "$RUN/resume.pth" "$ATTEMPT/resume-metadata.json" <<'PY'
import json, sys, torch
state = torch.load(sys.argv[1], map_location='cpu', weights_only=False, mmap=True)
step = int(state['step'])
assert 0 < step < 500, step
record = {'resume_from': sys.argv[1], 'step': step, 'target_step': 500,
          'save_freq': 50, 'resume_every': 50,
          'generator_optimizer_moments': 'not saved under original FSDP configuration'}
open(sys.argv[2], 'w').write(json.dumps(record, indent=2))
print(json.dumps(record))
PY
[[ "$MODE" == --run ]] || { echo 'PREFLIGHT_COMPLETE; no training started'; exit 0; }
python3 - <<'PY'
import subprocess
out = subprocess.check_output(['nvidia-smi','-i','4,5','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True)
for row in out.splitlines():
    index, memory, utilization = map(int, row.split(','))
    assert memory <= 32 and utilization == 0, row
print('Physical GPUs 4 and 5 idle')
PY
cd "$REPO"
printf 'COMMAND: bash z-image/scripts/ours_two-cards.sh --root %s --config %s --resume\n' "$ROOT" "$ATTEMPT/config.yaml"
bash z-image/scripts/ours_two-cards.sh --root "$ROOT" --config "$ATTEMPT/config.yaml" --resume
