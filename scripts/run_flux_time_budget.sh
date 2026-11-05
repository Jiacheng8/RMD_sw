#!/usr/bin/env bash
# Four FLUX.2 klein-4B experiments in one command, each on 4 GPU-hours (2 h on 2 x H100, 8 h in all). Training only:
# no evaluation, and no evaluation environment is installed.
#
#   git clone -b zimage https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   echo hf_xxx > .hf_token && chmod 600 .hf_token        # a Read token that can read the private reference store
#   tmux new -s train
#   bash scripts/run_flux_time_budget.sh --root /root/rdm-sets --dry-run
#   bash scripts/run_flux_time_budget.sh --root /root/rdm-sets
#
#   1 irdm      iRDM objective: every step 10240 fresh rows (K = B), exact repulsion, Nystrom attraction, no GAN
#   2 dmd2      DMD2, the one-step student against klein-4B's score
#   3 k512      SW-LMMD + GAN, K = 512, B = 32
#   4 b4        SW-LMMD + GAN, K = 128, B = 4
#
# Run it again after any interruption: finished experiments are skipped, a stopped one resumes from its last
# resume.pth with its time so far. A failed experiment does not stop the next one; a summary comes at the end.
# Each run saves about 4 checkpoints (15.5 GB each) -> <root>/sw_lmmd/work_dirs/<run>/. Disk: ~600 GB in all.
#
# --root DIR             data root (default /root/rdm-sets)
# --gpus N               GPUs per experiment: 2 (default) or 4, 80 GB cards. GPU time is fixed, so 4 takes half the wall time.
# --only LIST            run only these, e.g. --only irdm,b4
# --budget-gpu-hours H   GPU time per experiment (default 4)
# --skip-setup           skip the environment and download stage
# --env NAME             conda env (default rdm)
# --dry-run              print the plan, change nothing
set -euo pipefail
export PYTHONNOUSERSITE=1
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "bash >= 4.4 required (this is $BASH_VERSION)" >&2; exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="/root/rdm-sets"; GPUS=2; ONLY="irdm,dmd2,k512,b4"; BUDGET=4; SETUP=1; DRY=0; ENV_NAME="${CONDA_ENV:-rdm}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)             ROOT="$2"; shift 2 ;;
    --gpus)             GPUS="$2"; shift 2 ;;
    --only)             ONLY="$2"; shift 2 ;;
    --budget-gpu-hours) BUDGET="$2"; shift 2 ;;
    --skip-setup)       SETUP=0; shift ;;
    --env)              ENV_NAME="$2"; shift 2 ;;
    --dry-run)          DRY=1; shift ;;
    -h|--help)          sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                  echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
[[ "$GPUS" == 2 || "$GPUS" == 4 ]] || die "--gpus must be 2 or 4: B = 4 and B = 32 must split evenly over the GPUs"
for x in ${ONLY//,/ }; do case "$x" in irdm|dmd2|k512|b4) ;; *) die "--only: unknown experiment '$x'" ;; esac; done
wanted() { [[ ",$ONLY," == *",$1,"* ]]; }

ROOT="$(realpath -m "$ROOT")"
SW="$ROOT/sw_lmmd"; STORE="$SW/reference_store"; OUT="$SW/work_dirs"; NYS="$SW/ref_ablation/nystrom_m4096"
HF_DATASET="jiachengcui888/sw-rdm-reference-store"
if [[ -z "${HF_TOKEN:-}" && -s "$REPO/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$REPO/.hf_token")"; export HF_TOKEN
fi
cd "$REPO"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then   # every run uses exactly these cards
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $(( GPUS - 1 )))"
fi
say "plan: $ONLY | $BUDGET GPU-h each on $GPUS GPU(s) -> $OUT"

# ---- environment, models, reference store, Qwen3 context (no GenEval env) --------------------------------------------
if (( SETUP )); then
  NM=(bash scripts/new_machine.sh --root "$ROOT" --gpus "$GPUS" --config configs/sw_lmmd_budget_k512_b32_gan.yaml
      --no-train --skip-geneval --allow-low-disk)
  (( DRY )) && NM+=(--dry-run)
  say "setup: ${NM[*]}"
  "${NM[@]}"
fi

crun() { conda run -n "$ENV_NAME" --no-capture-output "$@"; }
[[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# ---- the Nystrom bundles of the iRDM run ------------------------------------------------------------------------------
landmarks_ok() { [[ -f "$NYS/SHA256SUMS" ]] && (cd "$NYS" && sha256sum --quiet -c SHA256SUMS) >/dev/null 2>&1; }
if wanted irdm; then
  say "Nystrom bundles -> $NYS"
  if (( DRY )); then
    info "would download $HF_DATASET:ref_ablation/nystrom_m4096 (3 x ~170 MB), or build them from the store on one GPU"
  elif landmarks_ok; then
    info "present, checksums OK"
  else
    mkdir -p "$NYS"
    if crun python - "$HF_DATASET" "$SW" <<'PY' && landmarks_ok; then
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type="dataset", local_dir=sys.argv[2], allow_patterns=["ref_ablation/nystrom_m4096/*"])
PY
      info "downloaded, checksums OK"
    else
      info "not downloadable with this token: building them from the store (~1 min)"
      rm -f "$NYS/SHA256SUMS"
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" crun python -m experiments.ref_ablation.build_landmarks \
        experiments/ref_ablation/configs/ref_global_n10240.yaml --set "reference_root=$STORE" --out "$NYS" \
        || die "the Nystrom bundles could not be built"
    fi
  fi
fi

# ---- one experiment ----------------------------------------------------------------------------------------------------
finished() { [[ -f "$1/time_budget.json" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['finished'] else 1)" "$1/time_budget.json"; }
declare -A RESULT
run_exp() {  # run_exp <key> <run dir> <command...>
  local key="$1" dir="$2"; shift 2
  say "$key -> $dir"
  if finished "$dir"; then RESULT[$key]="finished earlier"; info "finished earlier: skipped"; return; fi
  info "${*}"
  (( DRY )) && { RESULT[$key]="dry run"; return; }
  mkdir -p "$dir"
  local t0=$SECONDS
  if "$@" 2>&1 | tee -a "$dir/train.log"; then RESULT[$key]="ok, $(( (SECONDS - t0) / 60 )) min"; else RESULT[$key]="FAILED, see $dir/train.log"; fi
}
sw_args() {  # sw_args <run dir>: the overrides every SW-LMMD run gets
  SW_ARGS=(--set "time_budget_gpu_hours=$BUDGET" --set "reference_root=$STORE" --set "output_dir=$OUT")
  if [[ -f "$1/resume.pth" ]]; then
    SW_ARGS+=(--set "resume_from=$1/resume.pth" --set "load_from=$1/resume.pth" --set "strict_load=true")
  fi
}
torchrun_() { crun torchrun --nproc_per_node="$GPUS" --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29500}" "$@"; }

if wanted irdm; then
  DIR="$OUT/global-n10240-4gph"; sw_args "$DIR"
  run_exp irdm "$DIR" torchrun_ -m experiments.ref_ablation.train experiments/ref_ablation/configs/ref_global_n10240.yaml \
    "${SW_ARGS[@]}" --set "ref_ablation.nystrom_dir=$NYS"
fi
if wanted dmd2; then
  DIR="$OUT/flux-dmd2-4gph"
  run_exp dmd2 "$DIR" env CONDA_ENV="$ENV_NAME" bash scripts/train_dmd2_flux.sh --root "$ROOT" --exp-name flux-dmd2-4gph \
    --iterations 100000 --save-every 100000 --budget-gpu-hours "$BUDGET"
fi
if wanted k512; then
  DIR="$OUT/flux-k512-b32-gan-4gph"; sw_args "$DIR"
  run_exp k512 "$DIR" torchrun_ -m rdm.sw_lmmd.launch configs/sw_lmmd_budget_k512_b32_gan.yaml "${SW_ARGS[@]}"
fi
if wanted b4; then
  DIR="$OUT/flux-k128-b4-gan-4gph"; sw_args "$DIR"
  run_exp b4 "$DIR" torchrun_ -m rdm.sw_lmmd.launch configs/sw_lmmd_budget_k128_b4_gan.yaml "${SW_ARGS[@]}" \
    --set "memory.micro_batch=$(( 4 / GPUS ))"
fi

say "summary"
FAILED=0
for key in irdm dmd2 k512 b4; do
  wanted "$key" || continue
  info "$(printf '%-6s %s' "$key" "${RESULT[$key]:-not run}")"
  [[ "${RESULT[$key]:-}" == FAILED* ]] && FAILED=1
done
exit $FAILED
