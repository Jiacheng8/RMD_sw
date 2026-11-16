#!/usr/bin/env bash
# Continue six finished FLUX.2 klein-4B experiments from their resume.pth on Hugging Face, each for 2.8 more H100
# GPU-hours (1.4 h on 2 x H100), from an empty machine, in one command. Training only: no evaluation, no upload.
#
#   git clone -b main https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   (HF token for the gated FLUX.2 VAE + the private reference store: $HF_TOKEN, <repo>/.hf_token or ~/.hf_token)
#   tmux new -s resume
#   bash scripts/run_new-machine-resume.sh --root /root/rdm-sets --dry-run     # plan only
#   bash scripts/run_new-machine-resume.sh --root /root/rdm-sets
#
# Steps come from each run's own logs: steps = 5040 s (2.8 GPU-h / 2 GPUs) / wall seconds per step, saves and probes
# included. All six ran on 2 x H100, so no A100 -> H100 conversion is applied to these rates. The step counts are the
# same on any card; on A100 the same steps just take longer (the plan prints an estimate at about half H100 speed).
#
#   key      run                                   B    resume   + steps   to      s/step   code
#   k128b4   ablation-sw/k128-b4-gan               4    8359     5850      14209   0.861    366f718
#   k512     ablation-sw/k512-b32-gan              32   1413     1000      2413    5.086    366f718
#   nc32     nocache-k32b32                        32   1600     1400      3000    ~3.8     fd5ebef (no-cache)
#   nc128    nocache-k128b128-2h                   128  510      350       860     ~14.4    fd5ebef (no-cache)
#   global   global-ablation-nogroup               32   2000     1200      3200    ~4.2     958a38d (ref-ablation)
#   wogan    w128_s32_2000steps_wo-gan             32   2000     1200      3200    4.097    61c4c9b (main)
#
# nc32: the resume.pth on HF is the step-1600 state, although that run went on to step 2000.
#
# Each experiment trains with the code that produced it, in a detached worktree <root>/code/<commit>; this script
# itself, the environment and the data are shared. Runs land in <root>/sw_lmmd/work_dirs/resume/<name>/.
#
# Learning rate:
#   --lr restart  (default) the cosine decay is stretched to the new last step: the lr jumps up at the resume point
#                 (e.g. 2.83e-7 -> ~1.1e-6 for K128 at step 2000) and decays back to 2.83e-7 at the end
#   --lr floor    the original schedule stays: constant 2.83e-7 for the new steps (nc32 first finishes its decay)
#
# Run it again after any interruption: a finished experiment is skipped, a stopped one goes on from its own resume.pth.
# A failed experiment does not stop the next one; a summary comes at the end.
#
# --root DIR        data root (default /root/rdm-sets); ~660 GB free for all six
# --gpus N          4 (default, set up for 4 x A100 80 GB) or 2 cards, 80 GB each
# --only LIST       run only these keys, e.g. --only k512,nc32
# --lr MODE         restart | floor
# --skip-setup      skip the environment and download stage (the machine is already set up under --root)
# --allow-low-disk  go on although the checkpoints will not all fit (you prune them)
# --env NAME        conda env (default rdm)
# --dry-run         print the plan, change nothing
set -euo pipefail
export PYTHONNOUSERSITE=1
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "bash >= 4.4 required (this is $BASH_VERSION)" >&2; exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${HF_TOKEN:-}" && -s "$REPO/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$REPO/.hf_token")"; export HF_TOKEN
elif [[ -z "${HF_TOKEN:-}" && -s "$HOME/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.hf_token")"; export HF_TOKEN
fi

ROOT="/root/rdm-sets"; GPUS=4; ONLY="k128b4,k512,nc32,nc128,global,wogan"; LR_MODE="restart"
ENV_NAME="${CONDA_ENV:-rdm}"; SETUP=1; LOW_DISK=0; DRY=0
HF_DATASET="jiachengcui888/sw-rdm-reference-store"

C_BUDGET=366f718; C_NOCACHE=fd5ebef; C_REF=958a38d; C_MAIN=61c4c9b
#      key     name             HF repo                                    subfolder     code        module                     config                                                 B   start extra save  old lr horizon
EXPS=(
  "k128b4 |k128-b4-gan      |jiachengcui888/ablation-sw                 |k128-b4-gan  |$C_BUDGET  |rdm.sw_lmmd.launch         |configs/sw_lmmd_budget_k128_b4_gan.yaml               |4  |8359|5850|1500|8359"
  "k512   |k512-b32-gan     |jiachengcui888/ablation-sw                 |k512-b32-gan |$C_BUDGET  |rdm.sw_lmmd.launch         |configs/sw_lmmd_budget_k512_b32_gan.yaml              |32 |1413|1000|250 |1413"
  "nc32   |nocache-k32b32   |jiachengcui888/nocache-k32b32              |             |$C_NOCACHE |experiments.no_cache.train |experiments/no_cache/configs/nocache_k32_b32.yaml     |32 |1600|1400|400 |2000"
  "nc128  |nocache-k128b128 |jiachengcui888/nocache-k128b128-2h         |             |$C_NOCACHE |experiments.no_cache.train |experiments/no_cache/configs/nocache_k128_b128.yaml   |128|510 |350 |100 |510"
  "global |global-ungrouped |jiachengcui888/global-ablation-nogroup     |             |$C_REF     |experiments.ref_ablation.train|experiments/ref_ablation/configs/ref_global.yaml   |32 |2000|1200|400 |2000"
  "wogan  |wo-gan           |jiachengcui888/w128_s32_2000steps_wo-gan   |             |$C_MAIN    |rdm.sw_lmmd.launch         |configs/sw_lmmd_train_h100_2gpu.yaml                  |32 |2000|1200|400 |2000"
)
BRANCHES=(zimage no-cache ref-ablation main)     # where those commits live on origin

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)           ROOT="$2"; shift 2 ;;
    --gpus)           GPUS="$2"; shift 2 ;;
    --only)           ONLY="$2"; shift 2 ;;
    --lr)             LR_MODE="$2"; shift 2 ;;
    --env)            ENV_NAME="$2"; shift 2 ;;
    --skip-setup)     SETUP=0; shift ;;
    --allow-low-disk) LOW_DISK=1; shift ;;
    --dry-run)        DRY=1; shift ;;
    -h|--help)        sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; printf '%s' "${s%"${s##*[![:space:]]}"}"; }
field() { local IFS='|' a; read -r -a a <<< "$1"; trim "${a[$2]}"; }     # field <row> <index>

[[ "$GPUS" == 2 || "$GPUS" == 4 ]] || die "--gpus must be 2 or 4: B = 4, 32 and 128 must split evenly over the cards"
[[ "$LR_MODE" == restart || "$LR_MODE" == floor ]] || die "--lr must be restart or floor"
for x in ${ONLY//,/ }; do
  found=0; for r in "${EXPS[@]}"; do [[ "$(field "$r" 0)" == "$x" ]] && found=1; done
  (( found )) || die "--only: unknown experiment '$x'"
done
wanted() { [[ ",$ONLY," == *",$1,"* ]]; }

ROOT="$(realpath -m "$ROOT")"
SW="$ROOT/sw_lmmd"; STORE="$SW/reference_store"; OUT="$SW/work_dirs/resume"; NYS="$SW/ref_ablation/nystrom_m4096"
cd "$REPO"

if ! command -v conda >/dev/null 2>&1; then       # a non-interactive shell (an agent's, tmux -c)
  for c in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /root/miniconda3; do
    [[ -x "$c/bin/conda" ]] && { export PATH="$c/bin:$PATH"; break; }
  done
fi
command -v conda >/dev/null 2>&1 || die "conda not found: install Miniconda3 first"
crun() { conda run -n "$ENV_NAME" --no-capture-output "$@"; }

lr_at() {  # lr_at <step> <horizon>: rdm/sw_lmmd/config.py LRSchedule (base 2.83e-6, warm-up 100, floor 0.1)
  python3 -c "
import math, sys
s, h = int(sys.argv[1]), int(sys.argv[2])
p = min(1.0, (s - 100) / max(1, h - 100))
print(f'{2.83e-6 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))):.3g}')" "$1" "$2"
}

# ---- the plan --------------------------------------------------------------------------------------------------------
CARD="unknown"; SLOWER=1
if command -v nvidia-smi >/dev/null 2>&1; then
  CARD="$(nvidia-smi --query-gpu=name --format=csv,noheader | awk 'NR == 1')"     # not head: pipefail + SIGPIPE
  [[ "$CARD" == *A100* ]] && SLOWER=2
fi
say "resume plan: +2.8 H100 GPU-h per experiment | repo $(git rev-parse --short HEAD 2>/dev/null || echo '?') | $GPUS x $CARD"
info "root  $ROOT"
info "runs  $OUT/<name>"
info "lr    $LR_MODE"
N_CKPT=0; N_RUNS=0
for r in "${EXPS[@]}"; do
  key="$(field "$r" 0)"; wanted "$key" || continue
  start="$(field "$r" 8)"; extra="$(field "$r" 9)"; save="$(field "$r" 10)"; old="$(field "$r" 11)"; b="$(field "$r" 7)"
  target=$(( start + extra )); horizon=$old; [[ "$LR_MODE" == restart ]] && horizon=$target
  n=$(( target / save - start / save )); (( target % save )) && n=$(( n + 1 ))
  N_CKPT=$(( N_CKPT + n )); N_RUNS=$(( N_RUNS + 1 ))
  info "$(printf '%-7s B=%-4s step %5d -> %5d (+%d) | lr %s -> %s | %d checkpoints | code %s' "$key" "$b" "$start" "$target" "$extra" \
       "$(lr_at "$start" "$horizon")" "$(lr_at "$(( target - 1 ))" "$horizon")" "$n" "$(field "$r" 4)")"
done
WALL=$(python3 -c "print(f'{$N_RUNS * 2.8 / $GPUS * $SLOWER:.1f}')")
info "time  about $WALL h of training on $GPUS x $CARD$( (( SLOWER > 1 )) && echo ' (estimate: A100 taken as half H100 speed, not measured)')"
NEED_BASE=$(( 117 + N_RUNS * 24 + 24 + N_CKPT * 31 / 2 ))
info "disk  ~$NEED_BASE GB in all"
[[ $DRY -eq 1 || -n "${TMUX:-}${STY:-}" ]] || info "not inside tmux/screen: a dropped SSH session would stop this"

if (( ! DRY )); then
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: no GPUs"
  mkdir -p "$ROOT"
  mapfile -t ROWS < <(nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits)
  (( ${#ROWS[@]} >= GPUS )) || die "need $GPUS GPUs, found ${#ROWS[@]}"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=, read -r -a SEL <<< "$CUDA_VISIBLE_DEVICES"; else mapfile -t SEL < <(seq 0 $(( GPUS - 1 ))); fi
  (( ${#SEL[@]} >= GPUS )) || die "CUDA_VISIBLE_DEVICES has fewer than $GPUS GPUs"
  export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${SEL[*]:0:$GPUS}")"
  for g in "${SEL[@]:0:$GPUS}"; do
    IFS=', ' read -r _ total free <<< "$(printf '%s\n' "${ROWS[@]}" | awk -F', ' -v g="$g" '$1 == g')"
    info "GPU $g  $free / $total MiB free"
    (( total >= 79000 )) || die "GPU $g has $total MiB: these runs need 80 GB cards"
    (( free >= 70000 )) || die "GPU $g has only $free MiB free: stop what is running there, or choose idle cards with CUDA_VISIBLE_DEVICES"
  done
  FREE_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  HAVE_GB=$(( $(du -sk "$ROOT" 2>/dev/null | cut -f1 || echo 0) / 1048576 ))
  NEED_GB=$(( NEED_BASE - HAVE_GB )); (( NEED_GB < 50 )) && NEED_GB=50
  info "disk  $FREE_GB GB free, ~$NEED_GB GB still needed"
  (( FREE_GB >= NEED_GB || LOW_DISK )) || die "under ~$NEED_GB GB free on $ROOT: choose a bigger --root, use --only, or pass --allow-low-disk"
fi

# ---- environment, models, reference store, Qwen3 context (no GenEval env) --------------------------------------------
say "1 setup (environment, models, reference store, Qwen3 context)"
if (( SETUP )); then
  NM=(bash scripts/new_machine.sh --root "$ROOT" --gpus "$GPUS" --config configs/sw_lmmd_train_h100_2gpu_gan.yaml
      --env "$ENV_NAME" --no-train --skip-geneval --allow-low-disk)
  (( DRY )) && NM+=(--dry-run)
  # Training only: no GenEval scorer env, no PickScore weights (download_all.sh --minimal would fetch them), as
  # experiments/no_cache/run_nocache.sh does. new_machine.sh does the conda env and the context; the downloads are here.
  info "conda env"
  "${NM[@]}" --skip-download --skip-preprocess
  info "downloads for training only (three encoders, FLUX.2, flux2 source, reference store)"
  FETCH=(conda run -n "$ENV_NAME" --no-capture-output python -u "$REPO/scripts/fetch_prerequisites.py"
         --root "$ROOT" --group encoders --group flux --group flux2src --encoders dinov3_l,siglip2,aimv2_huge)
  STORE_DL=(env CONDA_ENV="$ENV_NAME" bash "$REPO/scripts/download_all.sh" --root "$ROOT" --env "$ENV_NAME" --sw-lmmd-store-only)
  (( DRY )) && { FETCH+=(--dry-run); STORE_DL+=(--dry-run); }
  "${FETCH[@]}" || die "downloads incomplete (see above); re-run"
  "${STORE_DL[@]}" || die "reference store download incomplete (see above); re-run"
  if (( ! DRY )); then
    # the model-revision check of new_machine.sh, minus the evaluation-only PickScore repos
    REV="$ROOT/.hf_revisions_train_only.json"
    python3 - "$REPO/assets/hf_revisions.json" "$REV" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
json.dump({k: v for k, v in d.items() if k not in ("yuvalkirstain/PickScore_v1",
          "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")}, open(sys.argv[2], "w"), indent=1)
PY
    ( source "$ROOT/env.sh" && crun python "$REPO/scripts/check_hf_revisions.py" --check "$REV" ) \
      || die "a training model on the Hub changed or did not download (see above)"
  fi
  info "Qwen3 context for the store"
  "${NM[@]}" --skip-env --skip-download
else
  info "skipped (--skip-setup)"
fi
if (( ! DRY )); then
  [[ -f "$ROOT/env.sh" ]] || die "$ROOT/env.sh missing: stage 1 did not finish"
  [[ -f "$STORE/qwen_context.npy" || -e "$STORE/qwen_context" ]] || die "the reference store has no Qwen3 context: stage 1 did not finish"
fi

# ---- the code of each experiment ---------------------------------------------------------------------------------------
say "2 code: one worktree per commit -> $ROOT/code/"
worktree() {  # worktree <commit>
  local c="$1" dir="$ROOT/code/$1"
  if [[ -f "$dir/.git" ]] && [[ "$(git -C "$dir" rev-parse --short=7 HEAD 2>/dev/null)" == "$c" ]]; then info "$c present"; return; fi
  if (( DRY )); then info "would add $dir"; return; fi
  git cat-file -e "$c^{commit}" 2>/dev/null || git fetch origin "${BRANCHES[@]}" || die "cannot fetch the branches holding $c"
  git cat-file -e "$c^{commit}" 2>/dev/null || die "commit $c not found on origin"
  mkdir -p "$ROOT/code"
  git worktree prune
  git worktree add --detach "$dir" "$c" >/dev/null || die "could not create the worktree $dir"
  info "$c -> $dir"
}
declare -A DONE_C
for r in "${EXPS[@]}"; do
  wanted "$(field "$r" 0)" || continue
  c="$(field "$r" 4)"
  [[ -n "${DONE_C[$c]:-}" ]] && continue
  DONE_C[$c]=1; worktree "$c"
done

# ---- the Nystrom bundles of the global ablation (the same ones it trained with) ----------------------------------------
landmarks_ok() { [[ -f "$NYS/SHA256SUMS" ]] && (cd "$NYS" && sha256sum --quiet -c SHA256SUMS) >/dev/null 2>&1; }
if wanted global; then
  say "3 Nystrom bundles -> $NYS"
  if (( DRY )); then info "would download $HF_DATASET:ref_ablation/nystrom_m4096 (0.5 GB)"
  elif landmarks_ok; then info "present, checksums OK"
  else
    mkdir -p "$NYS"
    crun python - "$HF_DATASET" "$SW" <<'PY' || die "could not download the Nystrom bundles: the token must read $HF_DATASET"
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type="dataset", local_dir=sys.argv[2], allow_patterns=["ref_ablation/nystrom_m4096/*"])
PY
    landmarks_ok || die "the Nystrom bundles in $NYS fail their checksums"
    info "downloaded, checksums OK"
  fi
fi

# ---- one experiment ----------------------------------------------------------------------------------------------------
fetch_state() {  # fetch_state <repo> <subfolder> <run dir>: resume.pth + train_log.jsonl from HF
  crun python - "$1" "$2" "$3" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download
repo, sub, out = sys.argv[1:]
tmp = os.path.join(out, ".dl")
for name in ("resume.pth", "train_log.jsonl"):
    if name == "train_log.jsonl" and os.path.exists(os.path.join(out, name)):
        continue
    p = hf_hub_download(repo, f"{sub}/{name}" if sub else name, local_dir=tmp)
    os.replace(p, os.path.join(out, name))
    print("   ", os.path.join(out, name), flush=True)
shutil.rmtree(tmp, ignore_errors=True)
PY
}
state_step() {  # state_step <resume.pth> <needs critic 0|1>: its step, after checking it can continue the run exactly
  crun python - "$1" "$2" <<'PY' | tail -1
import sys, torch
s = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
need = {"model", "step", "schedule", "optimizer", "cache"} | ({"critic"} if sys.argv[2] == "1" else set())
missing = need - set(s)
if missing:
    raise SystemExit(f"resume.pth has no {sorted(missing)}: it cannot continue the run exactly")
print(int(s["step"]))
PY
}
trim_log() {  # trim_log <train_log.jsonl> <step>: drop lines past the resume state, so the log has no step twice
  python3 - "$1" "$2" <<'PY'
import json, os, sys
path, at = sys.argv[1], int(sys.argv[2])
if os.path.exists(path):
    with open(path) as f:
        rows = [l for l in f if l.strip()]
    keep = [l for l in rows if json.loads(l).get("step", 0) <= at]
    if len(keep) != len(rows):
        with open(path + ".partial", "w") as f:
            f.writelines(keep)
        os.replace(path + ".partial", path)
        print(f"    train_log.jsonl: dropped {len(rows) - len(keep)} lines after step {at}")
PY
}

declare -A RESULT
run_one() {  # run_one <row>
  local r="$1" key name repo sub code module config b start extra save old
  key="$(field "$r" 0)"; name="$(field "$r" 1)"; repo="$(field "$r" 2)"; sub="$(field "$r" 3)"; code="$(field "$r" 4)"
  module="$(field "$r" 5)"; config="$(field "$r" 6)"; b="$(field "$r" 7)"; start="$(field "$r" 8)"; extra="$(field "$r" 9)"
  save="$(field "$r" 10)"; old="$(field "$r" 11)"
  local target=$(( start + extra )) dir="$OUT/$name" wt="$ROOT/code/$code" critic=1 micro=4 horizon=$old
  [[ "$key" == wogan ]] && critic=0
  [[ "$key" == k128b4 ]] && micro=$(( 4 / GPUS ))
  [[ "$LR_MODE" == restart ]] && horizon=$target

  say "$key: $name, step $start -> $target"
  if (( DRY )); then
    [[ -f "$dir/resume.pth" ]] && info "resume.pth present: kept" || info "would download resume.pth (23.5 GB) and train_log.jsonl from $repo${sub:+/$sub}"
  fi
  local at=$start
  if (( ! DRY )); then
    mkdir -p "$dir"
    if [[ -f "$dir/resume.pth" ]]; then info "resume.pth present: kept (a re-run goes on from it)"
    else fetch_state "$repo" "$sub" "$dir" || { RESULT[$key]="FAILED: download"; return; }
    fi
    at="$(state_step "$dir/resume.pth" "$critic")" || { RESULT[$key]="FAILED: resume.pth unreadable or incomplete"; return; }
    [[ "$at" =~ ^[0-9]+$ ]] || { RESULT[$key]="FAILED: no step in resume.pth (got '$at')"; return; }
    (( at >= start )) || { RESULT[$key]="FAILED: resume.pth is at step $at, before $start"; return; }
    info "resume.pth is at step $at"
    trim_log "$dir/train_log.jsonl" "$at"
    if (( at >= target )) && [[ -f "$dir/step_$(printf %07d "$target").pth" ]]; then RESULT[$key]="finished earlier"; info "finished earlier: skipped"; return; fi
  fi

  local sets=(--set "reference_root=$STORE" --set "output_dir=$OUT" --set "steps=$target" --set "save_freq=$save"
              --set "save_resume=true" --set "resume_every=$save" --set "memory.micro_batch=$micro"
              --set "lr_schedule.total_steps=$horizon"
              --set "resume_from=$dir/resume.pth" --set "load_from=$dir/resume.pth" --set "strict_load=true")
  [[ "$key" != global ]] && sets+=(--set "exp_name=$name")
  [[ "$code" == "$C_BUDGET" ]] && sets+=(--set "time_budget_gpu_hours=null")
  [[ "$key" == global ]] && sets+=(--set "ref_ablation.nystrom_dir=$NYS" --set "ref_ablation.pre_diagnostic=false")

  local cmd=(torchrun --nproc_per_node="$GPUS" --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29500}" -m "$module" "$config" "${sets[@]}")
  if (( DRY )); then info "in $wt: ${cmd[*]}"; RESULT[$key]="dry run"; return; fi

  local t0=$SECONDS
  if ( cd "$wt"; source "$ROOT/env.sh"; export PYTHONPATH="$wt:${PYTHONPATH:-}" ASSETS="$ROOT"
       export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
       crun python -c "import rdm, sys; f = rdm.__file__; print('    code', f); sys.exit(0 if f.startswith('$wt/') else 'rdm is not imported from $wt')"
       crun "${cmd[@]}" ) 2>&1 | tee -a "$dir/train.log"; then
    if [[ -f "$dir/step_$(printf %07d "$target").pth" ]]; then RESULT[$key]="ok, $(( (SECONDS - t0) / 60 )) min"
    else RESULT[$key]="FAILED: ended without step_$(printf %07d "$target").pth"; fi
  else
    RESULT[$key]="FAILED, see $dir/train.log"
  fi
}

say "4 training"
for r in "${EXPS[@]}"; do
  wanted "$(field "$r" 0)" || continue
  run_one "$r"
done

say "summary"
FAILED=0
for r in "${EXPS[@]}"; do
  key="$(field "$r" 0)"; wanted "$key" || continue
  info "$(printf '%-7s %s' "$key" "${RESULT[$key]:-not run}")"
  [[ "${RESULT[$key]:-}" == FAILED* ]] && FAILED=1
done
(( DRY )) && say "dry run: nothing was changed"
exit $FAILED
