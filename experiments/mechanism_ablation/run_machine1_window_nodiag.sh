#!/usr/bin/env bash
# Machine 1 of the mechanism ablation: run (b) "window-nodiag" to 3,400 updates on ONE fresh rented 2 x H100
# box, evaluated there together with the anchor, then uploaded. Independent of machine 2.
#
#   HF_TOKEN=hf_... bash run_machine1_window_nodiag.sh          # optional: WORK_DIR=/path (default ./work)
#
# (copy this file to the box, e.g. curl -fsSLO https://raw.githubusercontent.com/Jiacheng8/RMD_sw/mechanism-ablation/experiments/mechanism_ablation/run_machine1_window_nodiag.sh;
#  run it inside tmux: ~6 h). HF_TOKEN: jiachengcui888's WRITE token, with access to the private dataset
#  jiachengcui888/sw-rdm-reference-store and the gated black-forest-labs/FLUX.2-dev (licence accepted).
#
# window-nodiag = the main run (jiachengcui888/w128_s32_2000steps_w-gan) with the cross diagonal k(X_i, Y_i)
# dropped from the window attraction and the rest rescaled by K/(K-1) (attraction.skip_diagonal: true);
# repulsion, critic (sees the paired window), K=128, B=32, sliding cache, encoders, bandwidths, store, row
# order 3407 (hash 6559fefb9a872952, checked), fixed per-row noise, seed 0, 2 GPUs: unchanged. The code path is
# the Table 2 Global arm's (experiments/ref_ablation) plus the flag.
#
# Learning rate, as the main run reached its step 3400: phase 1 = 0 -> 2000 with warm-up 100 and cosine to
# 0.1x at 2000 (resume.pth at 2000); phase 2 = resume -> 3400 with the decay stretched to step 4000 (the
# main run's continuation 2000 -> 4000, scripts/continue_k128_gan.sh --lr restart). step_*.pth every 400.
#
# Stages (markers in WORK_DIR/done/; a rerun skips finished stages and continues training from resume.pth):
#   setup            preflight, Miniconda, repo @ pinned commit, env (torch 2.8.0+cu126 + requirements-lock.txt
#                    = the main run's), FLUX.2 klein-4B / encoders / COCO store (md5-checked) / Qwen3 ctx,
#                    GenEval scorer + detector, PickScore model
#   train_phase1     0 -> 2000                                                     (~2 h)
#   train_phase2     2000 -> 3400, decay horizon 4000                              (~1.4 h)
#   evaluate         step 3400 of window-nodiag (GPU 0) + the anchor = main run step 3400 (GPU 1):
#                    GenEval 553 x 4, shared noise, official detector; PickScore 499 Pick-a-Pic, 1 image each
#   report           WORK_DIR/out/window-nodiag/{scores,report,run_info}.json|md, train_log.jsonl
#   upload           -> datasets/jiachengcui888/evaluation-result/mechanism-ablation/window-nodiag/ and the final
#                       checkpoint -> model repo jiachengcui888/window-nodiag
RUN_NAME="window-nodiag"
CONFIG_NAME="window_nodiag"
PHASE2_HORIZON=4000
NEED_GB=360; NEED_AFTER_SETUP_GB=230
NEED_LINES=("setup ~140 (models 26, store 8.5 + 8.5 untar, Qwen3 ctx 61, GenEval env + detector 12, conda env 12, caches)"
            "training ~187 (9 checkpoints x 15.5 + resume.pth 23.5 x 2 while it is replaced)"
            "anchor checkpoint 15.5, evaluation ~3, headroom ~15")
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

stage setup          do_setup
stage train_phase1   do_train_phase1
stage train_phase2   do_train_phase2
stage evaluate       do_evaluate
stage report         do_report
stage upload         do_upload
say "done"
grep -h '^==> SUMMARY' "$LOGS/run.log" | tail -1
