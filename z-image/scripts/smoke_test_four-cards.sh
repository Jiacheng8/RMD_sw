#!/usr/bin/env bash
# Smoke test of the three one-command launchers on 4 GPUs (80 GB): each one end to end -- prepare,
# train 20 steps, save the model, render + judge 20 Qwen-Image-Bench prompts -- then a summary table.
#
#   git clone -b zimage https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   tmux new -s smoke
#   bash z-image/scripts/smoke_test_four-cards.sh --root /root/rdm-sets
#   bash z-image/scripts/smoke_test_four-cards.sh --root /root/rdm-sets --only dmd2      # one of them
#
#   1 ours   ours_four-cards.sh --steps 20 --eval-limit 20 (+ the teacher at 9 and 2 steps, 20 prompts)
#   2 rdm    rdm-global_four-cards.sh --steps 20 --rows 128 --eval-limit 20
#            (10240 rows per step -- the real setting -- is ~3.5 h per step on 4 x A100: 20 of them do
#            not fit a smoke test; --rdm-rows 10240 --rdm-steps 1 times one real step instead)
#   3 dmd2   dmd2_four-cards.sh --iterations 20 --eval-limit 20, batch 128 (DMD2's), real data from
#            2 tar shards (~2.8k renders) in a smoke-only dir; 20 iterations = 4 student updates
#
# Everything heavy is shared with the real runs under --root (env, weights, reference pack, contexts,
# judge) and done once; the first launcher pays for it (~1-2 h: downloads, the 70 GB context
# rebuild, the 55 GB judge). The smoke runs themselves go to <root>/zimage/smoke/ and their
# judgements are named *_limit20, so nothing here touches or enters a real run's tables. A smoke
# run's checkpoints are deleted once it has been judged (--keep-ckpts keeps them; ~25-75 GB each).
#
# Summary per launcher: pass/fail, wall time, median s/step (first step left out), peak GPU memory,
# the checkpoint it judged, and its 20-prompt Qwen-Image-Bench total. Logs: <root>/zimage/smoke/logs/.
#
# --root DIR        data root (default /root/rdm-sets)
# --only LIST       ours,rdm,dmd2 (default all three, in this order)
# --steps N         training steps / DMD2 iterations (default 20)
# --eval-limit N    prompts rendered + judged (default 20)
# --rdm-rows N      RDM's rows per step (default 128); --rdm-steps N its steps (default --steps)
# --keep-ckpts      keep the smoke runs' checkpoints
# --fresh           delete <root>/zimage/smoke first (re-run everything of the smoke runs)
set -uo pipefail
export PYTHONNOUSERSITE=1
ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="/root/rdm-sets"; ONLY="ours,rdm,dmd2"; STEPS=20; LIMIT=20; RDM_ROWS=128; RDM_STEPS=""
KEEP=0; FRESH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)       ROOT="$2"; shift 2 ;;
    --only)       ONLY="$2"; shift 2 ;;
    --steps)      STEPS="$2"; shift 2 ;;
    --eval-limit) LIMIT="$2"; shift 2 ;;
    --rdm-rows)   RDM_ROWS="$2"; shift 2 ;;
    --rdm-steps)  RDM_STEPS="$2"; shift 2 ;;
    --keep-ckpts) KEEP=1; shift ;;
    --fresh)      FRESH=1; shift ;;
    -h|--help) sed -n '2,/^set -uo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
ROOT="$(realpath -m "$ROOT")"
SMOKE="$ROOT/zimage/smoke"; LOGS="$SMOKE/logs"; WORK="$SMOKE/work_dirs"
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
(( FRESH )) && rm -rf "$SMOKE"
mkdir -p "$LOGS" "$WORK"
EXP=$(python3 -c "import re,sys; print(re.search(r'^exp_name:\s*(\S+)', open(sys.argv[1]).read(), re.M).group(1))" \
      "$ZDIR/configs/zimage_2step_1024_h100.yaml")
N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
say "smoke test: $ONLY on 4 GPUs ($N_GPU visible) | $STEPS steps, $LIMIT prompts judged | root $ROOT"
(( N_GPU >= 4 )) || { echo "needs 4 GPUs, $N_GPU visible" >&2; exit 1; }
[[ -z "${TMUX:-}${STY:-}" ]] && info "not inside tmux/screen: a dropped SSH session stops the test"

declare -A STATUS WALL SPS PEAK CKPT SCORE
run_one() {   # run_one <key> <run dir> <eval name> <command...>
  local key="$1" run="$2" name="$3"; shift 3
  local log="$LOGS/$key.log" mem="$LOGS/$key.mem" t0=$SECONDS
  say "$key -> $run (log: $log)"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 5000 > "$mem" 2>/dev/null &
  local smi=$!
  "$@" > "$log" 2>&1
  local rc=$?
  kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
  WALL[$key]="$(( (SECONDS - t0) / 60 )) min"
  PEAK[$key]="$(sort -n "$mem" 2>/dev/null | tail -1) MiB"
  SPS[$key]=$(python3 - "$run/train_log.jsonl" <<'PY' 2>/dev/null || echo "-"
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
s = [r["seconds"] for r in rows if "seconds" in r][1:] or [r["seconds"] for r in rows]
stu = [r["seconds"] for r in rows if r.get("student_step")]
extra = f" (student it {statistics.median(stu):.0f} s)" if stu else ""
print(f"{statistics.median(s):.1f} s x {len(rows)}{extra}")
PY
)
  CKPT[$key]=$(find "$run" -maxdepth 1 -name 'step_*.pth' 2>/dev/null | sort | tail -1)
  SCORE[$key]=$(python3 -c "import json,sys; print(round(json.load(open(sys.argv[1]))['total'], 2))" \
                "$ROOT/qib/runs/$name/input_bench_scores.json" 2>/dev/null || echo "-")
  if (( rc == 0 )) && [[ "${SCORE[$key]}" != "-" ]]; then
    STATUS[$key]="PASS"
    if (( ! KEEP )); then
      find "$run" -maxdepth 1 \( -name 'step_*.pth' -o -name 'resume.pth' \) -delete
      CKPT[$key]="${CKPT[$key]:+${CKPT[$key]##*/} (judged, deleted)}"
    fi
  else
    STATUS[$key]="FAIL (rc $rc)"
    info "FAILED -- last lines of $log:"
    tail -n 25 "$log" | sed 's/^/      /'
  fi
  info "${STATUS[$key]} | ${WALL[$key]} | s/step ${SPS[$key]} | peak ${PEAK[$key]} | total ${SCORE[$key]}"
}

IFS=, read -r -a WHICH <<< "$ONLY"
for key in "${WHICH[@]}"; do
  case "$key" in
    ours)
      run_one ours "$WORK/$EXP" "${EXP}_s${STEPS}_limit$LIMIT" \
        bash "$ZDIR/scripts/ours_four-cards.sh" --root "$ROOT" --steps "$STEPS" \
          --output-dir "$WORK" --eval-limit "$LIMIT" ;;
    rdm)
      rs="${RDM_STEPS:-$STEPS}"
      run_one rdm "$WORK/$EXP-rdm-global-N$RDM_ROWS" "rdm-global-N${RDM_ROWS}_s${rs}_limit$LIMIT" \
        bash "$ZDIR/scripts/rdm-global_four-cards.sh" --root "$ROOT" --rows "$RDM_ROWS" --steps "$rs" \
          --checkpoints 1 --output-dir "$WORK" --eval-limit "$LIMIT" ;;
    dmd2)
      run_one dmd2 "$WORK/$EXP-dmd2" "dmd2_it${STEPS}_limit$LIMIT" \
        bash "$ZDIR/scripts/dmd2_four-cards.sh" --root "$ROOT" --iterations "$STEPS" \
          --real-shards 2 --latents-dir "$SMOKE/dmd2_latents" --checkpoints 1 \
          --output-dir "$WORK" --eval-limit "$LIMIT" ;;
    *) echo "unknown launcher '$key' (ours, rdm, dmd2)" >&2; exit 2 ;;
  esac
done

say "summary"
printf '    %-6s %-14s %-9s %-34s %-12s %s\n' launcher status wall "s/step (steps)" "peak GPU" "QIB total ($LIMIT prompts)"
for key in "${WHICH[@]}"; do
  printf '    %-6s %-14s %-9s %-34s %-12s %s\n' "$key" "${STATUS[$key]}" "${WALL[$key]}" "${SPS[$key]}" \
    "${PEAK[$key]}" "${SCORE[$key]}"
done
info "checkpoints judged: $(for k in "${WHICH[@]}"; do printf '%s=%s ' "$k" "${CKPT[$k]:--}"; done)"
info "logs: $LOGS   (send these, and the table, if anything failed)"
info "the $LIMIT-prompt totals only show the chain works -- 20 prompts are too few to compare models"
for key in "${WHICH[@]}"; do [[ "${STATUS[$key]}" == PASS ]] || exit 1; done
exit 0
