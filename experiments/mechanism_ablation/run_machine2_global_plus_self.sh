#!/usr/bin/env bash
# Machine 2 of the mechanism ablation: run (a) "global-plus-self-b1" to 3,400 updates on ONE fresh rented 2 x H100
# box, evaluated there together with the anchor, then uploaded. Independent of machine 1.
#
#   HF_TOKEN=hf_... bash run_machine2_global_plus_self.sh          # optional: WORK_DIR=/path (default ./work)
#
# (copy this file to the box, e.g. curl -fsSLO https://raw.githubusercontent.com/Jiacheng8/RMD_sw/mechanism-ablation/experiments/mechanism_ablation/run_machine2_global_plus_self.sh;
#  run it inside tmux: ~7 h). HF_TOKEN: jiachengcui888's WRITE token, with access to the private dataset
#  jiachengcui888/sw-rdm-reference-store and the gated black-forest-labs/FLUX.2-dev (licence accepted).
#
# global-plus-self-b1 = the Table 2 Global arm (jiachengcui888/global-ablation-nogroup: Nystrom attraction to
# all references, same K / B / cache / critic) plus beta* (2/(|E| B)) sum_e sum_i [1 - kappa_e(u_i, v_i)]
# (own_term.beta = beta*). The critic's adaptive weight is balanced against the global loss PLUS beta* L_self
# (author's decision). Everything else as the main run; code path = the Global arm's plus the flag.
#
# Learning rate, as the Global arm reached its step 3400: phase 1 = 0 -> 2000, warm-up 100, cosine to 0.1x at
# 2000 (resume.pth at 2000); phase 2 = resume -> 3400 with the decay stretched to 3400 (the Global arm's
# continuation, scripts/run_new-machine-resume.sh --lr restart, ending at 3400). step_*.pth every 400.
#
# Stages (markers in WORK_DIR/done/; a rerun skips finished stages and continues training from resume.pth):
#   setup            preflight, Miniconda, repo @ pinned commit, env (= the main run's versions), FLUX.2
#                    klein-4B / encoders / COCO store (md5-checked) / Qwen3 ctx, GenEval scorer, PickScore model
#   landmarks        the Global arm's published Nystrom bundles (SHA256-checked; never rebuilt)
#   step0            main run steps 200, 1000, 2000 (step_0002000.pth), 3400: |g_own|/|g_total| and the two
#                    cosines, 8 consecutive windows each, per encoder and concatenated     (~1 h)
#   calibrate        beta*: 50-step runs of Global + beta L_self from the base model until the share is within
#                    20 % of the main run's at step 200; round 1 at beta_0 = 1/(K-1); at most 4 rounds, else
#                    STOP -> WORK_DIR/beta_star.json (read by both training phases)
#   train_phase1     0 -> 2000 with own_term.beta = beta*                         (~2 h)
#   train_phase2     2000 -> 3400, decay horizon 3400                             (~1.4 h)
#   evaluate         step 3400 (GPU 0) + the anchor = main run step 3400 (GPU 1), as machine 1
#   report, upload   as machine 1, plus the Step 0 table and the calibration log
RUN_NAME="global-plus-self-b1"
CONFIG_NAME="global_plus_self"
PHASE2_HORIZON=3400
NEED_GB=380; NEED_AFTER_SETUP_GB=250
NEED_LINES=("setup ~140 (models 26, store 8.5 + 8.5 untar, Qwen3 ctx 61, GenEval env + detector 12, conda env 12, caches)"
            "training ~187 (9 checkpoints x 15.5 + resume.pth 23.5 x 2 while it is replaced)"
            "main-run checkpoints 15.5 at a time (step 0; step 3400 kept as the anchor) + 15.5, Nystrom 0.5,"
            "evaluation ~3, headroom ~15")
set -euo pipefail
export PYTHONNOUSERSITE=1
GIT_URL="https://github.com/Jiacheng8/RMD_sw.git"
GIT_BRANCH="mechanism-ablation"
PINNED_COMMIT="054d575e700b75a6c5ba49a495b56890133cafa0"     # the training code: attraction.skip_diagonal / own_term.beta, tested
OWNER="jiachengcui888"

WORK="$(realpath -m "${WORK_DIR:-./work}")"
LOGS="$WORK/logs"; DONE="$WORK/done"; DATA="$WORK/data"; REPO="$WORK/RDM"
mkdir -p "$LOGS" "$DONE"
exec > >(tee -a "$LOGS/run.log") 2>&1
exec 9> "$WORK/.lock"
flock -n 9 || { echo "ERROR: another run is already using $WORK" >&2; exit 1; }
printf '\n\033[1m==> %s: %s, %s\033[0m\n' "$(basename "$0")" "$RUN_NAME" "$(date -Is)"
fail() { printf '\033[31m    PREFLIGHT FAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# ---- preflight: nothing over 1 GB is downloaded before this passes ---------------------------------
echo "    work dir    $WORK"
[[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN is not set (HF_TOKEN=hf_... bash $(basename "$0"))"
export HF_TOKEN
for t in git curl nvidia-smi flock df; do command -v "$t" >/dev/null 2>&1 || fail "'$t' is not installed"; done
NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader ${CUDA_VISIBLE_DEVICES:+--id=$CUDA_VISIBLE_DEVICES})"
N_GPU=$(grep -c . <<< "$NAMES" || true)
echo "    GPUs        $N_GPU: $(paste -sd'|' <<< "$NAMES")"
(( N_GPU == 2 )) || fail "exactly 2 GPUs needed (the main run's count), found $N_GPU"
grep -qv "H100" <<< "$NAMES" && fail "both GPUs must be H100"
echo "    driver      $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -1)"
command -v nvcc >/dev/null 2>&1 && echo "    nvcc        $(nvcc --version | grep -o 'release [0-9.]*')"
mkdir -p "$WORK"
FREE_GB=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
echo "    disk        $FREE_GB GB free under $WORK; needs ~$NEED_GB GB:"
printf '                %s\n' "${NEED_LINES[@]}"
if [[ ! -f "$DONE/setup" ]]; then
  (( FREE_GB >= NEED_GB )) || fail "not enough disk: $FREE_GB GB free, ~$NEED_GB GB needed"
else
  (( FREE_GB >= NEED_AFTER_SETUP_GB )) || fail "not enough disk to go on: $FREE_GB GB free, ~$NEED_AFTER_SETUP_GB GB needed after setup"
fi
hf_api() { curl -sI -o /dev/null -w '%{http_code}' -H @- "$1" <<< "Authorization: Bearer $HF_TOKEN"; }
WHO="$(curl -sf -H @- https://huggingface.co/api/whoami-v2 <<< "Authorization: Bearer $HF_TOKEN")" \
  || fail "HF_TOKEN is rejected by huggingface.co"
grep -Eq "\"name\": *\"$OWNER\"" <<< "$WHO" || fail "HF_TOKEN does not belong to $OWNER (or an org it is in): it must upload to $OWNER"
grep -Eq '"role": *"read"' <<< "$WHO" && fail "HF_TOKEN is a READ token; the uploads need write access"
echo "    HF token    $OWNER, $(grep -Eo '"role": *"[a-zA-Z]*"' <<< "$WHO" | head -1)"
[[ "$(hf_api https://huggingface.co/api/datasets/$OWNER/sw-rdm-reference-store)" == 200 ]] \
  || fail "HF_TOKEN cannot read the private dataset $OWNER/sw-rdm-reference-store"
code="$(hf_api https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/main/ae.safetensors)"
[[ "$code" =~ ^(200|302|307)$ ]] || fail "HF_TOKEN cannot read the gated black-forest-labs/FLUX.2-dev (HTTP $code): accept its licence on huggingface.co"
echo "    preflight   OK"

# ---- conda (Miniconda in ~/miniconda3 if none) and the training repo at the pinned commit ---------
for d in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
  [[ -x "$d/bin/conda" ]] && ! command -v conda >/dev/null 2>&1 && export PATH="$d/bin:$PATH"
done
if ! command -v conda >/dev/null 2>&1; then
  echo "==> installing Miniconda3 -> $HOME/miniconda3"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$WORK/miniconda.sh"
  bash "$WORK/miniconda.sh" -b -p "$HOME/miniconda3"
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
if [[ ! -d "$REPO/.git" ]]; then
  git clone --branch "$GIT_BRANCH" "$GIT_URL" "$REPO"
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$PINNED_COMMIT" ]]; then
  [[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] || fail "$REPO has local changes"
  git -C "$REPO" fetch -q origin "$GIT_BRANCH" || true
  git -C "$REPO" checkout -q --detach "$PINNED_COMMIT" \
    || fail "commit $PINNED_COMMIT is not on $GIT_URL ($GIT_BRANCH): push it first"
fi
echo "    repo        $REPO @ $(git -C "$REPO" log -1 --format='%h %s')"
source "$REPO/experiments/mechanism_ablation/machine_lib.sh"
init_env

do_step0() {
  local s json ckpt
  mkdir -p "$WORK/step0"
  for s in 200 1000 2000 3400; do
    json="$WORK/step0/step0_share_step$(printf '%04d' "$s").json"
    [[ -s "$json" ]] && { info "step $s: done before"; continue; }
    ckpt="$(dl_main "$s")"
    [[ -s "$ckpt" ]] || die "download of $MAIN_REPO:$(step_file "$s") failed"
    conda run -n "$ENV_NAME" --no-capture-output torchrun --nproc_per_node="$GPUS" --master_addr=127.0.0.1 \
      --master_port=29500 -m experiments.mechanism_ablation.probe \
      "$REPO/experiments/mechanism_ablation/configs/main_probe.yaml" --ckpt "$ckpt" --out "$json" \
      --set "reference_root=$STORE" 2>&1 | tee "$LOGS/step0_$s.log" || die "Step 0 at step $s failed"
    [[ -s "$json" ]] || die "Step 0 at step $s wrote no $json"
    (( s == 3400 )) || rm -f "$ckpt"          # this script's own download; step 3400 stays for the anchor
  done
  (cd "$REPO" && pyrun -m experiments.mechanism_ablation.probe --table "$WORK/step0/step0_share_step*.json") \
    | tee "$WORK/step0/step0_table.md"
}

do_calibrate() {
  local cal="$WORK/calibration"
  if [[ -s "$WORK/beta_star.json" && -s "$cal/calibration.json" ]] && grep -q '"status": "accepted"' "$cal/calibration.json"; then
    info "beta* was calibrated before ($WORK/beta_star.json): kept, never recalibrated"; return 0
  fi
  if [[ -d "$cal/runs" ]]; then
    warn "an interrupted calibration in $cal/runs: moved to $cal/runs.interrupted-$(date +%s)"
    mv "$cal/runs" "$cal/runs.interrupted-$(date +%s)"
  fi
  if [[ -s "$cal/calibration.json" ]] && grep -q '"status": "failed"' "$cal/calibration.json"; then
    die "the calibration failed before ($cal/calibration.json): STOP and report to the author"
  fi
  mkdir -p "$cal"
  (cd "$REPO" && env ASSETS="$DATA" CONDA_ENV="$ENV_NAME" GPUS="$GPUS" REFERENCE_ROOT="$STORE" NYSTROM_DIR="$NYS" \
     conda run -n "$ENV_NAME" --no-capture-output python -m experiments.mechanism_ablation.calibrate \
       --target "$WORK/step0/step0_share_step0200.json" --work "$cal/runs" --out "$cal/calibration.json" \
       --gpus "$GPUS" --window 128) 2>&1 | tee "$LOGS/calibrate.log" \
    || die "the calibration stopped ($cal/calibration.json): report to the author before training"
  pyrun - "$cal/calibration.json" "$WORK/beta_star.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
b = c.get("beta_star")
if not (isinstance(b, float) and b > 0):
    sys.exit(f"no beta* in {sys.argv[1]}")
json.dump({"beta_star": b, "target_share": c["target"]["share"], "rounds": len(c["rounds"]),
           "calibration": sys.argv[1]}, open(sys.argv[2], "w"), indent=2)
print(f"    beta* = {b!r} -> {sys.argv[2]}")
PY
}

beta_star() {      # the calibrated beta*, from WORK_DIR/beta_star.json only
  pyrun -c "import json; b=json.load(open('$WORK/beta_star.json'))['beta_star']; assert b > 0; print(repr(b))" | tail -1
}

stage setup          do_setup
stage landmarks      do_landmarks
stage step0          do_step0
stage calibrate      do_calibrate
BETA="$(beta_star)"; [[ -n "$BETA" ]] || die "no beta* in $WORK/beta_star.json"
info "own_term.beta = beta* = $BETA (from $WORK/beta_star.json)"
stage train_phase1   do_train_phase1 --set "own_term.beta=$BETA"
stage train_phase2   do_train_phase2 --set "own_term.beta=$BETA"
stage evaluate       do_evaluate
stage report         do_report --step0 "$WORK/step0/step0_share_step*.json" --calibration "$WORK/calibration/calibration.json"
do_upload_m2() {
  mkdir -p "$RESULT/step0" "$RESULT/calibration"
  cp -a "$WORK/step0"/*.json "$WORK/step0/step0_table.md" "$RESULT/step0/"
  cp -a "$WORK/calibration/calibration.json" "$WORK/beta_star.json" "$RESULT/calibration/"
  for d in "$WORK/calibration/runs"/calib-r*; do
    [[ -d "$d" ]] || continue
    mkdir -p "$RESULT/calibration/$(basename "$d")"
    cp -a "$d/train_log.jsonl" "$RESULT/calibration/$(basename "$d")/" 2>/dev/null || true
  done
  do_upload
}
stage upload         do_upload_m2
say "done"
grep -h '^==> SUMMARY' "$LOGS/run.log" | tail -1
