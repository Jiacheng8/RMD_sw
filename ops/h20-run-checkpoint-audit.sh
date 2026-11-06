#!/usr/bin/env bash
# CPU-only audit runner. Detach this wrapper through the existing owned SSH helper.
set -euo pipefail
ROOT=/data/rdm-workspace/rdm-sets
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STEP="${1:?Usage: h20-run-checkpoint-audit.sh STEP [auditor options]}"
shift
case "$STEP" in 50|100|150|200|250|300|350|400|450|500) ;; *) exit 2 ;; esac
BASE="$ROOT/control/twohead-checkpoint-audits"
RUN="$BASE/$(date -u +%Y%m%dT%H%M%SZ)-$$-step$STEP"
mkdir -p "$RUN"
exec >>"$RUN/launcher.log" 2>&1
printf '%s\n' "$$" >"$RUN/pid"
CHILD=''
finish() {
  rc=$?; trap - EXIT
  printf '%s\n' "$rc" >"$RUN/exit"
  printf 'audit_exit=%s utc=%s\n' "$rc" "$(date -u +%FT%TZ)"
  exit "$rc"
}
on_signal() {
  trap '' TERM INT HUP
  if [[ -n "$CHILD" ]] && kill -0 "$CHILD" 2>/dev/null; then
    kill -TERM "$CHILD" 2>/dev/null || true
    set +e; wait "$CHILD"; exit "$?"
  fi
  exit 143
}
trap finish EXIT
trap on_signal TERM INT HUP
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TMPDIR="$ROOT/.cache/tmp"
export H20_CONTROL_PREFIX=twohead-lr10-resume200-500-v3
export H20_TRAIN_SOURCE=/data/rdm-workspace/RDM-twohead-resume200-v3
export H20_TRAIN_SHA=1e06bf49569b4ee7847a417bc1ff1445bb8ba51b
PYTHON="$ROOT/envs/rdm/bin/python"
"$PYTHON" - "$REPO" "$RUN" "$STEP" "$@" <<'PY'
import json, os, pathlib, sys, time
repo, run = map(pathlib.Path, sys.argv[1:3])
command = ['timeout','--signal=TERM','--kill-after=30s','3600',sys.executable,
           str(repo/'ops/h20-audit-twohead-checkpoint.py'),'--step',sys.argv[3],*sys.argv[4:]]
value = dict(command=command, wrapper_pid=os.getppid(), run_dir=str(run),
    created_epoch=time.time(), source_commit=(repo/'SOURCE_COMMIT').read_text().strip(),
    training_source_commit=os.environ['H20_TRAIN_SHA'], CPU_only=True,
    stdout=str(run/'stdout.log'), stderr=str(run/'stderr.log'), exit=str(run/'exit'))
(run/'command.json').write_text(json.dumps(value,indent=2)+'\n')
latest = run.parent/'latest.json'; temporary = latest.with_name(latest.name+'.partial-'+str(os.getpid()))
temporary.write_text(json.dumps(value,indent=2)+'\n'); os.replace(temporary,latest)
print(json.dumps(value),flush=True)
PY
# Keep expensive final scans sequential and independent from the training lock.
exec 9>"$BASE/run.lock"
flock -n 9 || { echo 'Another owned CPU checkpoint audit is active'; exit 3; }
timeout --signal=TERM --kill-after=30s 3600 "$PYTHON" \
  "$REPO/ops/h20-audit-twohead-checkpoint.py" --step "$STEP" "$@" \
  >"$RUN/stdout.log" 2>"$RUN/stderr.log" &
CHILD=$!
printf '%s\n' "$CHILD" >"$RUN/child.pid"
set +e
wait "$CHILD"
exit "$?"
