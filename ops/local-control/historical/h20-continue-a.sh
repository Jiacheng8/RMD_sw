#!/usr/bin/env bash
# One manually authorized continuation of the reported preparation failure.
set -euo pipefail
ROOT=/share-evpfs/shaoshitong/rdm-sets
REPO=/data/rdm-workspace/RDM
SOCKET=/data/rdm-workspace/rdm-tmux.sock
exec 8>"$ROOT/control/A-continuation.lock"
flock -n 8 || { echo 'Another A continuation owns the claim.'; exit 3; }
test ! -e "$ROOT/control/A-continuation.exit" || { echo 'A prior continuation already exited.'; exit 4; }
on_exit() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$ROOT/control/A-continuation.exit"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$ROOT/control/A-continuation.finished"
  printf 'A_CONTINUATION_EXIT=%s\n' "$rc"
  exit "$rc"
}
trap on_exit EXIT
printf '%s\n' "$$" > "$ROOT/control/A-continuation.pid"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$ROOT/control/A-continuation.started"
echo 'WAITING_FOR_VERIFIED_JUDGE_PREFETCH'
while [[ ! -e "$ROOT/control/judge-prefetch.exit" ]]; do
  if ! tmux -S "$SOCKET" has-session -t rdm-judge-prefetch 2>/dev/null; then
    sleep 2
    test -e "$ROOT/control/judge-prefetch.exit" || { echo 'Prefetch session disappeared without an exit record.'; exit 5; }
  fi
  sleep 15
done
test "$(cat "$ROOT/control/judge-prefetch.exit")" = 0
test -s "$ROOT/control/judge-prefetch.sha256"
test -s "$ROOT/qib/judge/model-00001-of-00002.safetensors"
test "$(git -C "$REPO" rev-parse HEAD)" = e1670e86448d43295e7170443fe4c6b339346812
test "$(git -C "$REPO" branch --show-current)" = zimage
test -z "$(git -C "$REPO" status --porcelain)"
printf '%s  %s\n' f2083054d8bcdfc006d39068f25dafbbc899884ff39db7839db54863c1091ead \
  "$REPO/z-image/configs/zimage_2step_1024_dlr1e-4.yaml" | sha256sum --check --status
test "$(cat "$ROOT/control/A.exit")" = 1
test "$(cat "$ROOT/control/A.finished")" = 2026-11-02T12:45:48Z
test ! -e "$ROOT/control/A.reported"
test ! -e "$ROOT/control/B.started"
test ! -e "$ROOT/zimage/work_dirs/zimage-2step-1024-dlr1e-4/train_log.jsonl"
test ! -e "$ROOT/zimage/work_dirs/zimage-2step-1024-dlr1e-4/resume.pth"
while true; do
  if ! gpu_state=$(nvidia-smi -i 4,5 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits); then
    echo 'GPU_QUERY_FAILED; refusing to treat a driver error as busy GPUs.'
    exit 6
  fi
  if printf '%s\n' "$gpu_state" | awk -F, '{if ($1+0>32 || $2+0!=0) busy=1; n++} END {exit (busy || n!=2)}'; then break; fi
  printf 'UTC=%s WAITING_FOR_GPU4_5\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  sleep 30
done
exec 7>/data/rdm-workspace/rdm-run.lock
flock -w 30 7
ARCHIVE="$ROOT/control/history/A-preparation-$(date -u '+%Y%m%dT%H%M%SZ')"
mkdir -p "$ARCHIVE"
for name in A.pid A.started A.exit A.finished; do
  if [[ -f "$ROOT/control/$name" ]]; then mv --no-clobber "$ROOT/control/$name" "$ARCHIVE/$name"; fi
done
if [[ -f "$ROOT/launch-logs/A.log" ]]; then mv --no-clobber "$ROOT/launch-logs/A.log" "$ARCHIVE/A.log"; fi
(cd "$ARCHIVE" && sha256sum A.* > SHA256SUMS)
printf '%s\n' "$ARCHIVE" > "$ROOT/control/A-prior-failure.archive"
printf 'UTC=%s STARTING_ORIGINAL_A_ENTRYPOINT ARCHIVE=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$ARCHIVE"
# Transport policy only: HTTP downloads inherit the same task-specific proxy.
export HF_HUB_DISABLE_XET=1
flock -u 7
exec 7>&-
bash "$ROOT/control/run-one.sh" A >> "$ROOT/launch-logs/A.log" 2>&1
