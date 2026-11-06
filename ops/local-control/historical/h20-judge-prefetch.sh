#!/usr/bin/env bash
# Transfer repair only. Project source and scientific settings are unchanged.
set -euo pipefail
ROOT=/share-evpfs/shaoshitong/rdm-sets
source "$ROOT/control/env.sh"
exec 9>/data/rdm-workspace/rdm-run.lock
flock -n 9 || { echo 'An existing task holds the experiment lock.'; exit 3; }
test ! -e "$ROOT/control/judge-prefetch.exit" || { echo 'Prior prefetch exit exists; refusing automatic restart.'; exit 4; }
on_exit() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$ROOT/control/judge-prefetch.exit"
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
  for attempt in $(seq 1 12); do
    bytes=$(stat -c %s "$PARTIAL" 2>/dev/null || echo 0)
    printf 'UTC=%s DOWNLOAD_ATTEMPT=%s BYTES=%s EXPECTED=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$attempt" "$bytes" "$SIZE"
    (( bytes <= SIZE )) || { echo 'Partial file exceeds expected size; preserving it.'; exit 5; }
    if (( bytes == SIZE )); then break; fi
    rc=0
    curl --proxy "$HTTP_PROXY" --location --fail --show-error --silent \
      --connect-timeout 20 --speed-time 60 --speed-limit 1024 \
      --continue-at - --output "$PARTIAL" \
      --write-out 'HTTP=%{http_code} TRANSFER_BYTES=%{size_download} TIME=%{time_total}\n' \
      "$URL" || rc=$?
    bytes=$(stat -c %s "$PARTIAL" 2>/dev/null || echo 0)
    printf 'DOWNLOAD_ATTEMPT_EXIT=%s TOTAL_BYTES=%s\n' "$rc" "$bytes"
    if (( bytes == SIZE )); then break; fi
    # Bounded retries within this transfer, preserving the received bytes.
    if (( attempt < 12 )); then sleep "$(( attempt < 6 ? attempt * 5 : 30 ))"; fi
  done
  test "$(stat -c %s "$PARTIAL")" = "$SIZE"
  printf 'UTC=%s HASHING_STARTED=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$PARTIAL"
  printf '%s  %s\n' "$SHA" "$PARTIAL" | sha256sum --check --status
  test ! -e "$FINAL"
  mv --no-clobber "$PARTIAL" "$FINAL"
  verified=1
fi
test "$(stat -c %s "$FINAL")" = "$SIZE"
if (( ! verified )); then
  printf '%s  %s\n' "$SHA" "$FINAL" | sha256sum --check --status
fi
printf '%s  %s\n' 07959f893b7e5315989c9f8644a375afbf447cb15f46b23db91bd13893f25602 \
  "$ROOT/qib/judge/model-00002-of-00002.safetensors" | sha256sum --check --status
printf '%s  %s\n' "$SHA" "$NAME" \
  07959f893b7e5315989c9f8644a375afbf447cb15f46b23db91bd13893f25602 model-00002-of-00002.safetensors \
  > "$ROOT/control/judge-prefetch.sha256"
printf 'UTC=%s BOTH_JUDGE_SHARDS_SHA256_OK\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
