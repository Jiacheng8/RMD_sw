#!/usr/bin/env bash
# Experiment 7, then experiment 6, on one fresh machine in one command -- TRAINING ONLY, no
# evaluation. Both are K=128 / B=32 + GAN on the COCO reference (no GenEval block), 2000 steps:
#   exp 7  configs/sw_lmmd_train_h100_2gpu_gan_grouped_g2_w15.yaml   2 seeds per prompt, sibling weight 1.5
#   exp 6  configs/sw_lmmd_train_h100_2gpu_gan_grouped_g2.yaml       2 seeds per prompt, plain force
#
# exp 7 goes through scripts/run_machine-new7.sh: conda env, downloads, the Qwen3 context, training.
# The GenEval scorer (stage 4) is skipped, since nothing is evaluated here. exp 6 then trains on the
# same setup.
#
#   git clone -b unbiased-siblings https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   echo hf_xxx > .hf_token && chmod 600 .hf_token        # jiachengcui888's Read token
#   tmux new -s train
#   bash scripts/run_machine-new7-then-6.sh --dry-run     # checks + plan only, changes nothing
#   bash scripts/run_machine-new7-then-6.sh               # the two runs (root /root/rdm-sets)
#
# After an interruption, run the same command again. A run with a resume.pth is resumed (a
# finished one trains nothing); a run that has not started yet starts. If exp 7 fails while
# training, exp 6 still runs and the summary at the end names what failed. If exp 7 fails before
# training starts (env, downloads, context), the script stops: exp 6 would fail the same way.
#
# --root DIR   data root (default /root/rdm-sets); needs ~500 GB free (two runs x 10 checkpoints).
# --gpus N     training GPUs (default: every visible card, rounded down to 1/2/4/8). 80 GB cards.
# Every other option goes to both runs (e.g. --micro-batch 2).
#
# Evaluate later on this machine (installs the GenEval scorer first), once per run:
#   bash scripts/new_machine.sh --root <root> --config <config> --no-train --eval \
#        --skip-env --skip-download --skip-preprocess
# or upload a run (a WRITE token) and evaluate it elsewhere:
#   bash scripts/upload_run_hf.sh --root <root> --run <root>/sw_lmmd/work_dirs/<exp_name>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

ROOT="/root/rdm-sets"
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)    ROOT="$2"; shift 2 ;;
    --resume)  shift ;;                       # automatic, per run (see above)
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         PASS+=("$1"); shift ;;
  esac
done

EXP7_CFG=configs/sw_lmmd_train_h100_2gpu_gan_grouped_g2_w15.yaml
EXP6_CFG=configs/sw_lmmd_train_h100_2gpu_gan_grouped_g2.yaml

run_dir() {                                   # <root>/sw_lmmd/work_dirs/<exp_name of the config>
  local exp
  exp="$(sed -n 's/^exp_name:[[:space:]]*//p' "$REPO/$1" | head -1)"
  [[ -n "$exp" ]] || { echo "no exp_name in $1" >&2; exit 1; }
  echo "$ROOT/sw_lmmd/work_dirs/$exp"
}

# train_one <label> <launcher> <config> [extra launcher options...]
train_one() {
  local label="$1" launcher="$2" dir mode=()
  dir="$(run_dir "$3")"
  shift 3
  if [[ -f "$dir/resume.pth" ]]; then
    mode=(--resume)                           # interrupted -> continue; finished -> trains nothing
    printf '\n\033[1m==> %s: %s has a resume.pth -- resuming it\033[0m\n' "$label" "$dir"
  elif [[ -s "$dir/train_log.jsonl" ]]; then
    printf '\n\033[1m==> %s: %s holds a run without resume.pth (it stopped before its first\n' "$label" "$dir"
    printf '    resume save) -- not touching it. To restart it: mv %s{,_failed}\033[0m\n' "$dir"
    return 1
  fi
  bash "$HERE/$launcher" --root "$ROOT" --no-eval "${mode[@]}" "$@" "${PASS[@]}"
}

FAILED=()
if ! train_one "exp 7" run_machine-new7.sh "$EXP7_CFG" --skip-geneval; then
  FAILED+=("exp 7")
  if [[ ! -s "$(run_dir "$EXP7_CFG")/train_log.jsonl" ]]; then
    echo "exp 7 failed before training started (setup) -- fix that and re-run; exp 6 not started" >&2
    exit 1
  fi
fi
# stages 1-4 are done by now; exp 6 only trains
train_one "exp 6" run_machine-new6.sh "$EXP6_CFG" \
  --skip-env --skip-download --skip-preprocess --skip-geneval || FAILED+=("exp 6")

printf '\n\033[1m==> summary\033[0m\n'
for pair in "exp 7:$EXP7_CFG" "exp 6:$EXP6_CFG"; do
  label="${pair%%:*}"; dir="$(run_dir "${pair#*:}")"
  n=0; [[ -d "$dir" ]] && n=$(find "$dir" -maxdepth 1 -name 'step_*.pth' | wc -l)
  status=ok; [[ " ${FAILED[*]-} " == *" $label "* ]] && status=FAILED
  printf '    %-6s %-7s %2d checkpoints in %s\n' "$label" "$status" "$n" "$dir"
done
(( ${#FAILED[@]} == 0 )) || { echo "failed: ${FAILED[*]} -- see the messages above, then re-run this command" >&2; exit 1; }
