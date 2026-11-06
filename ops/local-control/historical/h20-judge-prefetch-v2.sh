#!/usr/bin/env bash
# Same pinned bytes and destination; retry budget applies to lack of progress.
set -euo pipefail
ROOT=/share-evpfs/shaoshitong/rdm-sets
source "$ROOT/control/env.sh"
exec 9>/data/rdm-workspace/rdm-run.lock
flock -n 9 || { echo 'An existing task holds the experiment lock.'; exit 3; }
test ! -e "$ROOT/control/judge-prefetch.exit" || { echo 'Prior prefetch exit exists; refusing automatic restart.'; exit 4; }
on_exit() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$ROOT/control/judge-prefetch.exit.tmp.$$"
  mv "$ROOT/control/judge-prefetch.exit.tmp.$$" "$ROOT/control/judge-prefetch.exit"
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$ROOT/control/judge-prefetch.finished"
  printf 'JUDGE_PREFETCH_EXIT=%s\n' "$rc"
  exit "$rc"
}
trap on_exit EXIT
printf '%s\n' "$$" > "$ROOT/control/judge-prefetch.pid"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$ROOT/control/judge-prefetch.started"
test "$(cat "$ROOT/control/A.exit")" = 1
test ! -e "$ROOT/control/B.started"
REV=1b77ff83564ac4e4e8140769eacff7ff73f35f3c
NAME=model-00001-of-00002.safetensors
SIZE=49825162976
SHA=cd93ba04b9dad1af628ecffdcdce16554621cae6ef6f00334c2baf28316aff3e
FINAL="$ROOT/qib/judge/$NAME"
PARTIAL="$FINAL.curl-part"
URL="https://huggingface.co/Qwen/Qwen-Image-Bench/resolve/$REV/$NAME?download=true"
mkdir -p "$ROOT/qib/judge"
verified=0
if [[ ! -f "$FINAL" ]]; then
  attempt=0
  no_progress=0
  deadline=$((SECONDS + 86400))
  while (( no_progress < 12 && SECONDS < deadline )); do
    attempt=$((attempt + 1))
    before=$(stat -c %s "$PARTIAL" 2>/dev/null || echo 0)
    printf 'UTC=%s DOWNLOAD_ATTEMPT=%s BYTES=%s EXPECTED=%s NO_PROGRESS_FAILURES=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$attempt" "$before" "$SIZE" "$no_progress"
    (( before <= SIZE )) || { echo 'Partial file exceeds expected size; preserving it.'; exit 5; }
    if (( before == SIZE )); then break; fi
    rc=0
    curl --proxy "$HTTP_PROXY" --location --fail --show-error --silent \
      --connect-timeout 20 --speed-time 60 --speed-limit 1024 \
      --max-time "$((deadline - SECONDS))" --continue-at - --output "$PARTIAL" \
      --write-out 'HTTP=%{http_code} TRANSFER_BYTES=%{size_download} TIME=%{time_total}\n' \
      "$URL" || rc=$?
    after=$(stat -c %s "$PARTIAL" 2>/dev/null || echo 0)
    printf 'DOWNLOAD_ATTEMPT_EXIT=%s TOTAL_BYTES=%s\n' "$rc" "$after"
    (( after >= before && after <= SIZE )) || { echo 'Unexpected file size; preserving received data.'; exit 6; }
    if (( after == SIZE )); then break; fi
    if (( after - before >= 1048576 )); then no_progress=0; else no_progress=$((no_progress + 1)); fi
    if (( no_progress < 12 )); then
      delay=$((no_progress * 5)); (( delay >= 2 )) || delay=2; (( delay <= 30 )) || delay=30
      sleep "$delay"
    fi
  done
  test "$(stat -c %s "$PARTIAL")" = "$SIZE"
  printf 'UTC=%s HASHING_STARTED=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$PARTIAL"
  printf '%s  %s\n' "$SHA" "$PARTIAL" | sha256sum --check --status
  test ! -e "$FINAL"
  mv --no-clobber "$PARTIAL" "$FINAL"
  verified=1
fi
test "$(stat -c %s "$FINAL")" = "$SIZE"
if (( ! verified )); then printf '%s  %s\n' "$SHA" "$FINAL" | sha256sum --check --status; fi
printf '%s  %s\n' 07959f893b7e5315989c9f8644a375afbf447cb15f46b23db91bd13893f25602 \
  "$ROOT/qib/judge/model-00002-of-00002.safetensors" | sha256sum --check --status
printf '%s  %s\n' "$SHA" "$NAME" \
  07959f893b7e5315989c9f8644a375afbf447cb15f46b23db91bd13893f25602 model-00002-of-00002.safetensors \
  > "$ROOT/control/judge-prefetch.sha256"
printf 'UTC=%s BOTH_JUDGE_SHARDS_SHA256_OK\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
