#!/usr/bin/env bash
set -euo pipefail
ROOT=/share-evpfs/shaoshitong/rdm-sets
REPO=/data/rdm-workspace/RDM
SETTING="${1:?Expected setting B}"
[[ $# -eq 1 ]] || { printf '%s\n' 'Expected exactly one argument: B' >&2; exit 2; }
case "$SETTING" in
  B) CONFIG=z-image/configs/zimage_2step_1024_dlr1e-4_stepw0.25.yaml ;;
  *) printf '%s\n' 'Expected setting B' >&2; exit 2 ;;
esac
source "$ROOT/control/env.sh"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
exec 9>/data/rdm-workspace/rdm-run.lock
flock -n 9 || { printf '%s\n' 'Another experiment controller holds the run lock.' >&2; exit 3; }
test ! -e "$ROOT/control/$SETTING.exit" || {
  printf '%s\n' 'This setting has already exited; refusing an automatic restart.' >&2
  exit 4
}
if [[ "$SETTING" == B ]]; then
  test "$(cat "$ROOT/control/A.exit")" = 0
  test -f "$ROOT/control/A.reported"
  test -s "$ROOT/zimage/work_dirs/zimage-2step-1024-dlr1e-4/qwen_image_bench.md"
fi
# B reuses A's prepared assets; reject a changed scientific checkout before recording B.
cd "$REPO"
test "$(git rev-parse HEAD)" = e1670e86448d43295e7170443fe4c6b339346812
test "$(git symbolic-ref --short HEAD)" = zimage
git diff --exit-code
test -z "$(git status --porcelain --untracked-files=normal)"
test "$(sha256sum "$CONFIG" | cut -d ' ' -f 1)" = f6030b32bbd3988cd81013156e2e55be84a78360f20d8af2d2c418a5657fd2a5
test -s "$ROOT/env.sh"
test -f "$ROOT/zimage/zimage-ref-v1/.staged"
test -s "$ROOT/zimage/zimage-ref-v1/reference_store/metadata.json"
test -s "$ROOT/zimage/zimage-ref-v1/contexts/manifest.json"
test -f "$ROOT/qib/env/.ready"
test -x "$ROOT/qib/env/bin/python"
test -x "$ROOT/qib/env/bin/pip"
test -s "$ROOT/qib/bench/judge.py"
test -s "$ROOT/qib/judge/config.json"
on_exit() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$ROOT/control/$SETTING.exit"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$ROOT/control/$SETTING.finished"
  printf '\nEXPERIMENT_%s_EXIT_CODE=%s\n' "$SETTING" "$rc"
  exit "$rc"
}
trap on_exit EXIT
printf '%s\n' "$$" > "$ROOT/control/$SETTING.pid"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$ROOT/control/$SETTING.started"
GPU_STATE="$(nvidia-smi -i 4,5 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
python3 - "$GPU_STATE" <<'PY'
import sys
for row in sys.argv[1].splitlines():
    index, memory, utilization = [int(value.strip()) for value in row.split(',')]
    if memory > 32 or utilization != 0:
        raise SystemExit(f'GPU {index} is occupied ({memory} MiB, {utilization}%); stopping without using it.')
print('GPU 4 and GPU 5 are idle; using only these two GPUs.')
PY
printf 'EXPERIMENT=%s WANDB_MODE=%s CONFIG=%s\n' "$SETTING" "$WANDB_MODE" "$CONFIG"
if [[ ! -x "$ROOT/qib/env/bin/pip" ]]; then
  printf '%s\n' 'Preparing the specified Python 3.11 judge environment with pip.'
  mkdir -p "$ROOT/qib"
  if [[ -x "$ROOT/qib/env/bin/python" ]]; then
    CONDA_PKGS_DIRS="$ROOT/qib/.cache/conda-pkgs" conda install -y -p "$ROOT/qib/env" pip
  else
    CONDA_PKGS_DIRS="$ROOT/qib/.cache/conda-pkgs" conda create -y -p "$ROOT/qib/env" python=3.11 pip
  fi
  "$ROOT/qib/env/bin/python" -m pip --version
fi
printf 'COMMAND: CUDA_VISIBLE_DEVICES=4,5 bash z-image/scripts/ours_two-cards.sh --root %s --config %s --skip-env --skip-download --skip-reference --skip-contexts\n' "$ROOT" "$CONFIG"
bash z-image/scripts/ours_two-cards.sh --root "$ROOT" --config "$CONFIG" --skip-env --skip-download --skip-reference --skip-contexts
