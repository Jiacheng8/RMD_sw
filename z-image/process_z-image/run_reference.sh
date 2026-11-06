#!/usr/bin/env bash
# Full local pipeline. Activate the inference Python environment before running.
# GPU_IDS=0,1 bash z-image/process_z-image/run_reference.sh
# bash z-image/process_z-image/run_reference.sh --dry-run     # coverage check only; no writes/downloads/GPU work
# ROOT=/path/to/new-run GPU_IDS=0 bash z-image/process_z-image/run_reference.sh --cpu-offload
# PROMPT_ROOT is the data folder holding prompt_build/ (captions + images; not in git), and the
# default ROOT lives under it too, so the run's artifacts never land in the checkout.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_ROOT="${PROMPT_ROOT:-/data/ironman/jiacheng/RDM/z-image}"
cd "$PROMPT_ROOT"                       # the inputs below are relative to it
PYTHON="${PYTHON:-python}"
ROOT="${ROOT:-$PROMPT_ROOT/process/artifacts/reference_v1}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
ENCODERS="${ENCODERS:-dinov3_l,siglip2,aimv2_huge}"
RDM_REPO="${RDM_REPO:-$(cd "$HERE/../.." && pwd)}"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
GPU_IDS=0,1 bash z-image/process_z-image/run_reference.sh [render options]
  --dry-run          Check full prompt coverage; no output writes or GPU work.
  --cpu-offload      Example render option; see python z-image/process_z-image/render.py --help.
Environment: ROOT, PROMPT_ROOT, GPU_IDS, ENCODERS, RDM_REPO, PYTHON.
Default: 24 candidates -> top-4; dinov3_l,siglip2,aimv2_huge.
HELP
    exit 0
fi

# Explicit production outputs; add new production JSONLs here if captioning outputs change.
inputs=(
    prompt_build/pilot_claude.jsonl
    prompt_build/claude_extra.jsonl
    prompt_build/claude_s0001.jsonl
    prompt_build/outputs/luna_merged.jsonl   # luna_all.jsonl with its skips replaced by luna_retry prompts
    prompt_build/outputs/sonnet_b1.jsonl
    prompt_build/outputs/codex_luna_b1.jsonl
)
prepare=("$PYTHON" "$HERE/prepare.py" --out "$ROOT" --shards 0-259 --prompt-build "$PROMPT_ROOT/prompt_build" --inputs "${inputs[@]}")
if [[ "${1:-}" == --dry-run ]]; then
    [[ $# -eq 1 ]] || { echo '--dry-run takes no additional render options' >&2; exit 2; }
    exec "${prepare[@]}" --dry-run
fi

"${prepare[@]}"
exec "$PYTHON" "$HERE/launch.py" all --root "$ROOT" \
    --gpu-ids "$GPU_IDS" --encoders "$ENCODERS" --rdm-repo "$RDM_REPO" -- "$@"
