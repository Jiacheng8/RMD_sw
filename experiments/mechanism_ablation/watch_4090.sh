#!/usr/bin/env bash
# Watch the Hub and score each mechanism-ablation arm on this 6 x 4090 box as soon as its H100 machine has
# finished uploading -- no manual step between training and evaluation.
#
#   tmux new -d -s mechwatch 'bash experiments/mechanism_ablation/watch_4090.sh'
#   tail -f /data/thor/jiacheng/rdm-sets-eval/sw_lmmd/work_dirs/mechanism_ablation/logs/watch.log
#
# Every --poll seconds (default 600) it checks, for each run not scored yet:
#   1. datasets/jiachengcui888/evaluation-result/mechanism-ablation/<run>/train_side/UPLOAD_DONE.json exists
#      (the H100 script writes it LAST, after the checkpoint upload was verified), and
#   2. jiachengcui888/<run>/step_0003400.pth on the Hub has the marker's size and sha256.
# Then: bash eval_4090.sh --runs "<ready runs>" (download, score with the paper's pipeline, report, upload).
# No idle GPU at that moment, or a failed evaluation: it tries again at the next poll (at most --max-tries
# times per run). It exits once every run is scored.
#
# --runs LIST     default "window-nodiag global-plus-self-b1"
# --poll SECONDS  default 600
# --max-tries N   evaluation attempts per run (default 3)
# --once          one check (and evaluation if ready), then exit
# Other options are passed on to eval_4090.sh (e.g. --anchor, --gpus 0,1, --no-upload).
set -euo pipefail
export PYTHONNOUSERSITE=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/data/thor/jiacheng/rdm-sets-eval
RUNS="window-nodiag global-plus-self-b1"; POLL=600; MAX_TRIES=3; ONCE=0; PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)      RUNS="$2"; shift 2 ;;
    --poll)      POLL="$2"; shift 2 ;;
    --max-tries) MAX_TRIES="$2"; shift 2 ;;
    --once)      ONCE=1; shift ;;
    --root)      ROOT="$2"; PASS+=("$1" "$2"); shift 2 ;;
    --gpus|--eval-repo) PASS+=("$1" "$2"); shift 2 ;;
    --anchor|--no-upload) PASS+=("$1"); shift ;;
    -h|--help)   sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           echo "unknown option '$1' (see --help)" >&2; exit 2 ;;
  esac
done
WORK="$ROOT/sw_lmmd/work_dirs/mechanism_ablation"
mkdir -p "$WORK/logs"
exec > >(tee -a "$WORK/logs/watch.log") 2>&1
exec 8> "$WORK/.watch.lock"
flock -n 8 || { echo "another watch_4090.sh is already running" >&2; exit 1; }
source "$ROOT/env.sh"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

ready() {   # <run>: 0 when the marker is there and the Hub checkpoint matches it
  conda run -n "${CONDA_ENV:-rdm}" --no-capture-output python - "$1" <<'PY' 2>/dev/null
import json, sys
from huggingface_hub import HfApi, hf_hub_download
run = sys.argv[1]
try:
    m = json.load(open(hf_hub_download("jiachengcui888/evaluation-result",
                                       f"mechanism-ablation/{run}/train_side/UPLOAD_DONE.json",
                                       repo_type="dataset", force_download=True)))
    i = HfApi().get_paths_info(m["model_repo"], [m["checkpoint"]], repo_type="model")[0]
except Exception:
    sys.exit(1)
sys.exit(0 if (i.size == m["size"] and i.lfs and i.lfs.sha256 == m["sha256"]) else 1)
PY
}

declare -A TRIES=()
log "watching: $RUNS (poll ${POLL}s)"
while true; do
  todo=(); go=()
  for r in $RUNS; do
    [[ -s "$WORK/$r/eval_step_0003400/summary.json" ]] && continue
    todo+=("$r")
    if (( ${TRIES[$r]:-0} >= MAX_TRIES )); then continue; fi
    if ready "$r"; then go+=("$r"); fi
  done
  if (( ${#todo[@]} == 0 )); then log "every run is scored; done"; break; fi
  if (( ${#go[@]} )); then
    idle=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2 < 1000' | wc -l)
    if [[ " ${PASS[*]-} " != *" --gpus "* ]] && (( idle == 0 )); then
      log "upload complete: ${go[*]}, but no idle GPU right now; next poll"
    else
      for r in "${go[@]}"; do TRIES[$r]=$(( ${TRIES[$r]:-0} + 1 )); done
      log "upload complete: ${go[*]} -> evaluating"
      bash "$HERE/eval_4090.sh" --runs "${go[*]}" ${PASS[@]+"${PASS[@]}"} || log "eval_4090.sh exited with an error"
      scored=0
      for r in "${go[@]}"; do [[ -s "$WORK/$r/eval_step_0003400/summary.json" ]] && scored=1; done
      if (( scored )); then log "scored: $(for r in "${go[@]}"; do [[ -s "$WORK/$r/eval_step_0003400/summary.json" ]] && printf '%s ' "$r"; done)"; continue; fi
      log "evaluation failed for ${go[*]} (see $WORK/logs/eval_*.log); trying again at the next poll"
    fi
  fi
  gave_up=(); for r in "${todo[@]}"; do (( ${TRIES[$r]:-0} >= MAX_TRIES )) && gave_up+=("$r"); done
  if (( ${#gave_up[@]} == ${#todo[@]} )); then log "gave up after $MAX_TRIES attempts: ${gave_up[*]} (see $WORK/logs)"; exit 1; fi
  log "waiting for: ${todo[*]}"
  (( ONCE )) && break
  sleep "$POLL"
done
