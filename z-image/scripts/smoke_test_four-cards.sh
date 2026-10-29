#!/usr/bin/env bash
# Smoke test of the three one-command launchers on 4 GPUs (80 GB): each one end to end -- prepare,
# train 10 steps, save the model, render + judge 10 Qwen-Image-Bench prompts -- then a summary table.
#
#   git clone -b zimage https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   tmux new -s smoke
#   bash z-image/scripts/smoke_test_four-cards.sh --root /root/rdm-sets              # dmd2 only (for now)
#   bash z-image/scripts/smoke_test_four-cards.sh --root /root/rdm-sets --only ours,rdm,rdm-oom,dmd2   # all four
#
#   1 ours     ours_four-cards.sh --steps 10 --eval-limit 10 (+ the teacher at 9 and 2 steps, 10 prompts)
#   2 rdm      rdm-global_four-cards.sh --rows 128 --steps 10 --eval-limit 10: the code path end to end
#              (a real 10240-row step is ~3.5 h on 4 x A100 [estimate] -- too long for a smoke test)
#   3 rdm-oom  rdm-global_four-cards.sh --rows 10240 --oom-probe: does the real batch fit? Every
#              buffer of a 10240-row step at full size (2560 rows' features per GPU, the gathered
#              10240 x 10240 loss and its backward, an optimizer step), only 2 micro-batches really
#              rolled out -- a few minutes; PASS/FAIL + peak memory per GPU, nothing saved
#   4 dmd2     dmd2_four-cards.sh --iterations 10 --eval-limit 10, batch 128 (DMD2's), real data =
#              the prepared latents from --dmd2-latents-hf (~18 GB, into <root>/zimage/dmd2_latents,
#              where the real run reuses them); 10 iterations = 2 student updates
#
# Everything heavy is shared with the real runs under --root (env, weights, reference pack, contexts,
# judge) and done once; the first launcher pays for it (~1-2 h: downloads, the 70 GB context
# rebuild, the 55 GB judge). The smoke runs themselves go to <root>/zimage/smoke/ and their
# judgements are named *_limit10, so nothing here touches or enters a real run's tables. A smoke
# run's checkpoints are deleted once it has been judged (--keep-ckpts keeps them; ~25-75 GB each).
#
# Summary per launcher: pass/fail, wall time, median s/step (first step left out), peak GPU memory,
# the checkpoint it judged, and its 10-prompt Qwen-Image-Bench total. Logs: <root>/zimage/smoke/logs/.
#
# --root DIR        data root (default /root/rdm-sets)
# --only LIST       ours,rdm,rdm-oom,dmd2, in this order (default for now: dmd2 -- ours, rdm and
#                   rdm-oom were run on 4 x A100; DMD2 is being re-tested after its FSDP fixes)
# --steps N         training steps / DMD2 iterations (default 10)
# --eval-limit N    prompts rendered + judged (default 10)
# --rdm-rows N      RDM's rows per step in the smoke run (default 128)
# --probe-rows N    rows the OOM probe tests (default 10240)
# --dmd2-latents-hf REPO  DMD2's real latents (default jiachengcui888/zimage-dmd2-latents; '' encodes
#                   2 tar shards into a smoke-only dir instead)
# --keep-ckpts      keep the smoke runs' checkpoints
# W&B: the key is asked for once here (Enter = none; or export WANDB_API_KEY / `wandb login` first);
# the smoke runs go to sw-lmmd/zimage-distill-smoke (WANDB_PROJECT overrides), tagged smoke
# --fresh           delete <root>/zimage/smoke first (re-run everything of the smoke runs)
set -uo pipefail
export PYTHONNOUSERSITE=1
ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# ONLY="ours,rdm,rdm-oom,dmd2"     # all four -- back to this once DMD2 passes
ROOT="/root/rdm-sets"; ONLY="dmd2"; STEPS=10; LIMIT=10; RDM_ROWS=128; PROBE_ROWS=10240
DMD2_LAT_HF="jiachengcui888/zimage-dmd2-latents"
KEEP=0; FRESH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)       ROOT="$2"; shift 2 ;;
    --only)       ONLY="$2"; shift 2 ;;
    --steps)      STEPS="$2"; shift 2 ;;
    --eval-limit) LIMIT="$2"; shift 2 ;;
    --rdm-rows)   RDM_ROWS="$2"; shift 2 ;;
    --probe-rows) PROBE_ROWS="$2"; shift 2 ;;
    --dmd2-latents-hf) DMD2_LAT_HF="$2"; shift 2 ;;
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
source "$ZDIR/scripts/_wandb.sh"
wandb_ask
export WANDB_PROJECT="${WANDB_PROJECT:-zimage-distill-smoke}" WANDB_TAGS="${WANDB_TAGS:-smoke}"
wandb_on && info "wandb: on, ${WANDB_ENTITY:-sw-lmmd}/$WANDB_PROJECT" || info "wandb: off"
(( FRESH )) && rm -rf "$SMOKE"
mkdir -p "$LOGS" "$WORK"
EXP=$(python3 -c "import re,sys; print(re.search(r'^exp_name:\s*(\S+)', open(sys.argv[1]).read(), re.M).group(1))" \
      "$ZDIR/configs/zimage_2step_1024_h100.yaml")
N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
say "smoke test: $ONLY on 4 GPUs ($N_GPU visible) | $STEPS steps (RDM $RDM_ROWS rows; OOM probe $PROBE_ROWS), $LIMIT prompts judged | root $ROOT"
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

run_probe() {   # run_probe <key> <command...>: the OOM probe -- no training, no eval
  local key="$1"; shift
  local log="$LOGS/$key.log" mem="$LOGS/$key.mem" t0=$SECONDS
  say "$key (log: $log)"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 2000 > "$mem" 2>/dev/null &
  local smi=$!
  "$@" > "$log" 2>&1
  local rc=$?
  kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
  WALL[$key]="$(( (SECONDS - t0) / 60 )) min"
  PEAK[$key]="$(sort -n "$mem" 2>/dev/null | tail -1) MiB"
  SPS[$key]="-"; CKPT[$key]=""
  local passes; passes=$(grep -c "oom-probe: PASS" "$log")
  # the torch peak of the busiest GPU (nvidia-smi's also counts the CUDA context etc.)
  SCORE[$key]=$(grep -o "reserved [0-9.]* of [0-9.]* GiB" "$log" | sort -k2 -n | tail -1)
  SCORE[$key]="${SCORE[$key]:--}"
  if (( rc == 0 && passes == 4 )); then
    STATUS[$key]="PASS"
  else
    STATUS[$key]="FAIL (rc $rc)"
    grep -q "oom-probe: FAIL" "$log" && STATUS[$key]="OOM"
    info "${STATUS[$key]} -- last lines of $log:"
    tail -n 25 "$log" | sed 's/^/      /'
  fi
  info "${STATUS[$key]} | ${WALL[$key]} | peak ${PEAK[$key]} | torch ${SCORE[$key]}"
}

if [[ -n "$DMD2_LAT_HF" ]]; then DMD2_DATA=(--latents-hf "$DMD2_LAT_HF")
else DMD2_DATA=(--real-shards 2 --latents-dir "$SMOKE/dmd2_latents"); fi

IFS=, read -r -a WHICH <<< "$ONLY"
for key in "${WHICH[@]}"; do
  case "$key" in
    ours)
      run_one ours "$WORK/$EXP" "${EXP}_s${STEPS}_limit$LIMIT" \
        bash "$ZDIR/scripts/ours_four-cards.sh" --root "$ROOT" --steps "$STEPS" \
          --output-dir "$WORK" --eval-limit "$LIMIT" ;;
    rdm)
      run_one rdm "$WORK/$EXP-rdm-global-N$RDM_ROWS" "rdm-global-N${RDM_ROWS}_s${STEPS}_limit$LIMIT" \
        bash "$ZDIR/scripts/rdm-global_four-cards.sh" --root "$ROOT" --rows "$RDM_ROWS" --steps "$STEPS" \
          --checkpoints 1 --output-dir "$WORK" --eval-limit "$LIMIT" ;;
    rdm-oom)
      run_probe rdm-oom bash "$ZDIR/scripts/rdm-global_four-cards.sh" --root "$ROOT" \
        --rows "$PROBE_ROWS" --oom-probe --output-dir "$WORK" ;;
    dmd2)
      run_one dmd2 "$WORK/$EXP-dmd2" "dmd2_it${STEPS}_limit$LIMIT" \
        bash "$ZDIR/scripts/dmd2_four-cards.sh" --root "$ROOT" --iterations "$STEPS" \
          "${DMD2_DATA[@]}" --checkpoints 1 --output-dir "$WORK" --eval-limit "$LIMIT" ;;
    *) echo "unknown launcher '$key' (ours, rdm, rdm-oom, dmd2)" >&2; exit 2 ;;
  esac
done

say "summary"
printf '    %-7s %-14s %-9s %-34s %-12s %s\n' launcher status wall "s/step (steps)" "peak GPU" "QIB total ($LIMIT prompts)"
for key in "${WHICH[@]}"; do
  printf '    %-7s %-14s %-9s %-34s %-12s %s\n' "$key" "${STATUS[$key]}" "${WALL[$key]}" "${SPS[$key]}" \
    "${PEAK[$key]}" "${SCORE[$key]}"
done
info "checkpoints judged: $(for k in "${WHICH[@]}"; do printf '%s=%s ' "$k" "${CKPT[$k]:--}"; done)"
info "logs: $LOGS   (send these, and the table, if anything failed)"
info "rdm-oom's last column is the torch peak (reserved) of its busiest GPU, not a score"
info "the $LIMIT-prompt totals only show the chain works -- so few prompts are too few to compare models"
for key in "${WHICH[@]}"; do [[ "${STATUS[$key]}" == PASS ]] || exit 1; done
exit 0
