#!/usr/bin/env bash
# Evaluate every checkpoint of a training run, keeping all GPUs busy, then print one comparison table.
#
#   bash scripts/eval_run.sh <run dir> --root <root>                      # every step_*.pth, all GPUs
#   bash scripts/eval_run.sh <run dir> --root <root> --baselines          # + teacher (4 step) + released s180
#   bash scripts/eval_run.sh <run dir> --root <root> --only 2000,1800     # just these steps
#   bash scripts/eval_run.sh <run dir> --root <root> --gpus 0,1 --per-gpu 2   # 2 evaluations per 80 GB card
#   bash scripts/eval_run.sh <run dir> --root <root> --dry-run            # print the plan
#
# Each job is scripts/eval_checkpoint.sh on one GPU (GenEval + PickScore, ~22 min on a 4090), queued
# so every GPU slot always has work. Results stay in the run dir -- eval_step_NNNNNNN/,
# eval_teacher_4step/, eval_s180_release/ -- and the table goes to <run dir>/eval_summary.{md,json}.
# A failed job does not stop the others; it is listed at the end with its log. A re-run picks up
# where the last one stopped (eval_checkpoint.sh skips finished stages); Ctrl-C stops every job.
# Newest checkpoints go first, the baselines last.
#
# --root / --geneval-root / --env / --no-geneval / --force are passed on to eval_checkpoint.sh.
# --gpus defaults to $CUDA_VISIBLE_DEVICES, else every GPU nvidia-smi lists; --per-gpu 2 needs
# cards of 48 GB or more (one evaluation peaks at ~21 GB). --s180 PATH uses a local copy of the
# released checkpoint instead of fetching it from the Hub.
set -euo pipefail
export PYTHONNOUSERSITE=1   # packages in ~/.local/lib/pythonX.Y must never shadow the conda envs'

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ASSETS:-/data/thor/jiacheng/rdm-sets}"
CONDA_ENV="${CONDA_ENV:-rdm}"
RUN=""
GPUS=""
PER_GPU=1
ONLY=""
BASELINES=0
S180=""
DRY=0
PASS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)         ROOT="$2"; PASS+=(--root "$2"); shift 2 ;;
    --geneval-root) PASS+=(--geneval-root "$2"); shift 2 ;;
    --env)          CONDA_ENV="$2"; PASS+=(--env "$2"); shift 2 ;;
    --no-geneval)   PASS+=(--no-geneval); shift ;;
    --force)        PASS+=(--force); shift ;;
    --gpus)         GPUS="$2"; shift 2 ;;
    --per-gpu)      PER_GPU="$2"; shift 2 ;;
    --only)         ONLY="$2"; shift 2 ;;
    --baselines)    BASELINES=1; shift ;;
    --s180)         S180="$2"; shift 2 ;;
    --dry-run)      DRY=1; shift ;;
    -h|--help)      sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)             echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    *)              [[ -z "$RUN" ]] || { echo "two run dirs given: $RUN $1" >&2; exit 2; }
                    RUN="$1"; shift ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -n "$RUN" ]] || die "usage: bash scripts/eval_run.sh <run dir> --root <root> [options]  (see --help)"
[[ -d "$RUN" ]] || die "run dir not found: $RUN"
RUN="$(realpath "$RUN")"
[[ "$PER_GPU" =~ ^[1-9]$ ]] || die "--per-gpu must be 1-9"

# ---------------------------------------------------------------------------
# jobs: checkpoints newest first, then the baselines
# ---------------------------------------------------------------------------
NAMES=(); TARGETS=(); OUTS=()
mapfile -t CKPTS < <(find "$RUN" -maxdepth 1 -name 'step_*.pth' | sort -r)
if [[ -n "$ONLY" ]]; then
  KEEP=()
  for s in ${ONLY//,/ }; do
    [[ "$s" =~ ^[0-9]+$ ]] || die "--only takes step numbers, got '$s'"
    f="$RUN/$(printf 'step_%07d.pth' "$((10#$s))")"
    [[ -f "$f" ]] || die "no checkpoint for step $s in $RUN"
    KEEP+=("$f")
  done
  mapfile -t CKPTS < <(printf '%s\n' "${KEEP[@]}" | sort -ru)
fi
for f in "${CKPTS[@]}"; do
  n="$(basename "$f" .pth)"
  NAMES+=("$n"); TARGETS+=("$f"); OUTS+=("$RUN/eval_$n")
done
if [[ $BASELINES -eq 1 ]]; then
  NAMES+=("teacher_4step"); TARGETS+=("base"); OUTS+=("$RUN/eval_teacher_4step")
  NAMES+=("s180_release"); TARGETS+=("${S180:-<released s180, fetched from the Hub>}"); OUTS+=("$RUN/eval_s180_release")
fi
NJOBS=${#NAMES[@]}
[[ $NJOBS -gt 0 ]] || die "no step_*.pth in $RUN (and no --baselines)"

# ---------------------------------------------------------------------------
# GPU slots
# ---------------------------------------------------------------------------
if [[ -z "$GPUS" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then GPUS="$CUDA_VISIBLE_DEVICES"
  else GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)"; fi
fi
[[ -n "$GPUS" ]] || die "no GPU found: pass --gpus 0,1"
IFS=, read -r -a GPU_LIST <<< "$GPUS"
if [[ $PER_GPU -gt 1 ]]; then
  for g in "${GPU_LIST[@]}"; do
    MIB=$(nvidia-smi -i "$g" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9' || true)
    [[ -z "$MIB" || "$MIB" -ge 47000 ]] || die "--per-gpu $PER_GPU on GPU $g (${MIB} MiB): one evaluation peaks at ~21 GB, two need a 48 GB+ card"
  done
fi
SLOTS=()
for g in "${GPU_LIST[@]}"; do for ((k = 0; k < PER_GPU; k++)); do SLOTS+=("$g"); done; done

say "evaluate $NJOBS job(s) from $RUN on GPU(s) $GPUS (x$PER_GPU per GPU)"
for ((j = 0; j < NJOBS; j++)); do info "$(printf '%-16s %s' "${NAMES[j]}" "${TARGETS[j]}")"; done
info "results     $RUN/eval_*/summary.json -> $RUN/eval_summary.md"
if [[ $DRY -eq 1 ]]; then say "dry run: nothing was started"; exit 0; fi

# The released s180 (public repo) into the root's HF cache -- once, before any GPU is busy.
if [[ $BASELINES -eq 1 && -z "$S180" ]]; then
  say "fetching the released s180 checkpoint (15.5 GB, public)"
  [[ -f "$ROOT/env.sh" ]] || die "$ROOT/env.sh not found -- pass --root"
  S180=$( (source "$ROOT/env.sh" && conda run -n "$CONDA_ENV" --no-capture-output python -c \
            "from huggingface_hub import hf_hub_download; print(hf_hub_download('epfl-vita/flux2-klein-1step-rdm', 'flux2_klein_1step_rdm_geallcoco_s180.pth'))") \
          | sed '/^[[:space:]]*$/d' | tail -1) || die "could not fetch the released s180 checkpoint"
  [[ -f "$S180" ]] || die "released s180 checkpoint not found after the download: $S180"
  info "$S180"
  TARGETS[NJOBS - 1]="$S180"
fi

# ---------------------------------------------------------------------------
# the queue
# ---------------------------------------------------------------------------
LOGDIR="$RUN/eval_logs"
mkdir -p "$LOGDIR"
SLOT_PID=(); SLOT_JOB=(); SLOT_T0=()
for s in "${!SLOTS[@]}"; do SLOT_PID[s]=""; done
FAILED=()

stop_all() {        # every job runs in its own session: kill the whole group (conda run, python, ...)
  for s in "${!SLOTS[@]}"; do
    [[ -n "${SLOT_PID[s]}" ]] && { kill -TERM -- "-${SLOT_PID[s]}" 2>/dev/null || kill -TERM "${SLOT_PID[s]}" 2>/dev/null || true; }
  done
}
trap 'printf "\n"; info "interrupted: stopping the running evaluations"; stop_all; exit 130' INT TERM

launch() {          # launch <slot> <job>
  local s=$1 j=$2
  local log="$LOGDIR/${NAMES[j]}.log"
  CUDA_VISIBLE_DEVICES="${SLOTS[s]}" setsid -w bash "$REPO/scripts/eval_checkpoint.sh" "${TARGETS[j]}" \
      "${PASS[@]}" --out "${OUTS[j]}" > "$log" 2>&1 &
  SLOT_PID[s]=$!; SLOT_JOB[s]=$j; SLOT_T0[s]=$SECONDS
  info "$(date +%H:%M)  start  ${NAMES[j]} on GPU ${SLOTS[s]}  (log: $log)"
}
result() {          # result <out dir>: "GenEval 0.8238  PickScore 21.825" from its summary.json
  local f="$1/summary.json"
  [[ -f "$f" ]] || { echo "no summary.json"; return; }
  conda run -n "$CONDA_ENV" --no-capture-output python - "$f" 2>/dev/null <<'PY' | sed '/^$/d' || echo "summary.json unreadable"
import json, sys
s = json.load(open(sys.argv[1]))
g = s.get("geneval") or {}
ge = f"GenEval {g['overall']:.4f}" if "overall" in g else "GenEval -"
print(f"{ge}  PickScore {s['pickscore']:.3f}" if s.get("pickscore") is not None else ge)
PY
}

next=0
say "running"
while true; do
  busy=0
  for s in "${!SLOTS[@]}"; do
    pid="${SLOT_PID[s]}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then rc=0; else rc=$?; fi
      j=${SLOT_JOB[s]}
      mins=$(( (SECONDS - SLOT_T0[s]) / 60 ))
      if [[ $rc -eq 0 ]]; then
        info "$(date +%H:%M)  done   ${NAMES[j]} in ${mins} min: $(result "${OUTS[j]}")"
      else
        info "$(date +%H:%M)  FAILED ${NAMES[j]} (exit $rc) after ${mins} min -- see $LOGDIR/${NAMES[j]}.log"
        FAILED+=("${NAMES[j]}")
      fi
      SLOT_PID[s]=""
      pid=""
    fi
    if [[ -z "$pid" && $next -lt $NJOBS ]]; then
      launch "$s" "$next"
      next=$((next + 1))
      sleep 2           # stagger the starts: each one checks its GPU's free memory first
    fi
    [[ -n "${SLOT_PID[s]}" ]] && busy=1
  done
  [[ $busy -eq 1 ]] || break
  sleep 10
done
trap - INT TERM

# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
say "summary"
printf '%s\n' "${NAMES[@]}" > "$LOGDIR/.jobs"
conda run -n "$CONDA_ENV" --no-capture-output python - "$RUN" "$LOGDIR/.jobs" <<'PY'
import json, os, sys
run, jobs = sys.argv[1], [l.strip() for l in open(sys.argv[2]) if l.strip()]
tasks = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
short = ["single", "two", "count", "colors", "position", "attr"]
rows = []
for name in jobs:
    f = os.path.join(run, f"eval_{name}", "summary.json")
    s = json.load(open(f)) if os.path.exists(f) else None
    g = (s or {}).get("geneval") or {}
    rows.append({"name": name, "checkpoint": (s or {}).get("checkpoint"),
                 "steps": (s or {}).get("num_sampling_steps"),
                 "geneval": g.get("overall"), **{t: g.get(t) for t in tasks},
                 "pickscore": (s or {}).get("pickscore"), "summary": f if s else None})
ckpts = [r for r in rows if r["name"].startswith("step_") and r["geneval"] is not None]
best = max(ckpts, key=lambda r: r["geneval"])["name"] if ckpts else None
fmt = lambda v, p: "-" if v is None else f"{v:.{p}f}"
head = "| checkpoint | GenEval | " + " | ".join(short) + " | PickScore |"
lines = [head, "|" + "---|" * (len(short) + 3)]
for r in rows:
    mark = " **(best)**" if r["name"] == best else ""
    lines.append(f"| {r['name']}{mark} | {fmt(r['geneval'], 4)} | "
                 + " | ".join(fmt(None if r[t] is None else 100 * r[t], 1) for t in tasks)
                 + f" | {fmt(r['pickscore'], 3)} |")
md = "\n".join(lines) + "\n"
open(os.path.join(run, "eval_summary.md"), "w").write(
    "GenEval (overall = mean of the 6 task rates, tasks in %) and PickScore (Pick-a-Pic 499, ctx 232).\n"
    "Compare only numbers measured on the same machine.\n\n" + md)
json.dump(rows, open(os.path.join(run, "eval_summary.json"), "w"), indent=1)
print(md)
print(f"-> {os.path.join(run, 'eval_summary.md')}")
PY
if [[ ${#FAILED[@]} -gt 0 ]]; then
  die "${#FAILED[@]} job(s) failed: ${FAILED[*]} -- logs in $LOGDIR; re-run the same command to retry them"
fi
