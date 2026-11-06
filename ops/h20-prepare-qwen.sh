#!/usr/bin/env bash
# Data/environment preparation only: no experiment training and no evaluation.
set -euo pipefail
REPO=/data/rdm-workspace/RDM-qwen
ROOT=/data/rdm-workspace/rdm-sets
source "$REPO/ops/h20-data-env.sh"
mkdir -p "$ROOT/control" "$ROOT/launch-logs" "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME"
exec 8>"$ROOT/control/prepare-qwen.lock"
flock -n 8 || exit 3
exec >>"$ROOT/launch-logs/prepare-qwen.log" 2>&1
on_exit() {
  rc=$?; trap - EXIT
  printf '%s\n' "$rc" > "$ROOT/control/prepare-qwen.exit"
  date -u +%FT%TZ > "$ROOT/control/prepare-qwen.finished"
  printf 'PREPARATION_EXIT=%s\n' "$rc"
  exit "$rc"
}
trap on_exit EXIT
printf '%s\n' "$$" > "$ROOT/control/prepare-qwen.pid"
date -u +%FT%TZ > "$ROOT/control/prepare-qwen.started"
cd "$REPO"
git rev-parse HEAD
# Downloads do not need the training environment or GPUs.
/usr/bin/python3 ops/h20-qwen-prefetch.py >"$ROOT/launch-logs/reference-qwen.log" 2>&1 &
ref_pid=$!
/usr/bin/python3 scripts/fetch_prerequisites.py --root "$ROOT" --group zimage --group encoders \
  --encoders dinov3_l,siglip2,aimv2_huge >"$ROOT/launch-logs/models.log" 2>&1 &
models_pid=$!
prepare_env() {
  if [[ ! -x "$ROOT/miniconda/bin/conda" ]]; then
    installer="$TMPDIR/Miniconda3-latest-Linux-x86_64.sh"
    curl --fail --location --retry 5 --retry-delay 5 --connect-timeout 20 \
      --max-time 1800 --continue-at - --output "$installer" \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sha256sum "$installer" > "$ROOT/control/miniconda-installer.sha256"
    bash "$installer" -b -p "$ROOT/miniconda"
  fi
  bash scripts/setup_env.sh --name rdm --python 3.12
}
prepare_env >"$ROOT/launch-logs/environment.log" 2>&1 &
env_pid=$!
failed=0
for pid in "$ref_pid" "$models_pid" "$env_pid"; do
  wait "$pid" || failed=1
done
[[ "$failed" == 0 ]] || { echo 'Preparation failed; inspect reference-qwen.log/models.log/environment.log'; exit 10; }
export PYTHONPATH="$REPO/z-image:$REPO:${PYTHONPATH:-}"
conda run -n rdm --no-capture-output python ops/h20-assemble-qwen.py
# --no-train and --no-eval retain only original contexts/Qwen-store preparation.
bash z-image/scripts/ours_two-cards.sh --root "$ROOT" \
  --config z-image/configs/zimage_2step_1024_dlr1e-4_qwen.yaml \
  --skip-env --skip-download --skip-reference --no-train --no-eval
printf '%s\n' 'DATA_PREPARATION_COMPLETE; no experiment training started'
