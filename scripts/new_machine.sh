#!/usr/bin/env bash
# New machine -> SW-LMMD training in one command: environment, downloads, context rebuild, the
# GenEval scorer environment (for scripts/eval_checkpoint.sh), train.
#
#   git clone <repo> RDM && cd RDM
#   bash scripts/new_machine.sh --root /data/<you>/rdm-sets              # 2 x H100, the 2000-step run
#   bash scripts/new_machine.sh --root <dir> --gate                      # 20-step memory/speed check
#   bash scripts/new_machine.sh --root <dir> --gpus 4 --config configs/sw_lmmd_train_4x4090.yaml
#   bash scripts/new_machine.sh --root <dir> --skip-env --skip-download  # resume from preprocessing
#   bash scripts/new_machine.sh --root <dir> --dry-run                   # print the plan, change nothing
#   bash scripts/new_machine.sh --root <dir> --resume                    # continue a stopped run (stage 5 only)
#
# Stages -- all idempotent; training refuses a run dir that already holds a run, unless --resume:
#   1 env         setup_env.sh                   conda env (torch 2.8.0+cu126, bitsandbytes)
#   2 download    download_all.sh --minimal      encoders + FLUX.2 (klein-4B, AE, Qwen3-4B) + flux2
#                                                source + our prebuilt reference store (rclone)
#   3 preprocess  preprocess_all-new-machine.sh  rebuild the store's 61 GB Qwen3 context (~10 min, 1 GPU)
#   4 geneval     setup_geneval.sh               the GenEval scorer env in <root>/geneval (5-10 min;
#                                                mmcv is source-built on H100); a failure here only
#                                                warns -- training still starts
#   5 train       train_sw_lmmd.sh               SW-LMMD, configs/sw_lmmd_train_h100_2gpu.yaml
#
# A config with `reference_extension: <name>` (e.g. ..._gan_grouped_geneval.yaml: COCO + the GenEval
# block) trains on the store PLUS that block. Stage 2 then also downloads the block from its private
# HF dataset (EXT_HF, default jiachengcui888/sw-rdm-geneval-block) into <root>/sw_lmmd/<name> and
# reads it back with the store; preflight checks the token can read it, and training refuses to
# start without it.
# After training, evaluate a checkpoint with
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_checkpoint.sh <run dir>/step_NNNNNNN.pth --root <root>
#
# Everything lands under --root: env.sh, hf/ (weights), geneval/ (scorer env + detector),
# sw_lmmd/{reference_store, qwen3_ctx_coco.npy, work_dirs/, logs/}, .cache/ (pip, conda). Budget
# ~315 GB: weights ~26, store 8.5 (+8.5 while untarring), context 61, geneval ~12, caches ~5,
# checkpoints 15.5 GB each (10 over the H100 run), resume.pth 23 (x2 while it is replaced).
# The rdm conda env (~12 GB) goes to conda's own envs dir.
#
# Manual prerequisites, once per machine:
#   - conda (miniconda/anaconda) on PATH, and this repo cloned with the current scripts;
#   - the FLUX.2 VAE (black-forest-labs/FLUX.2-dev) is a GATED repo: accept its licence on
#     huggingface.co (once per HF account), then give this machine a token before stage 2 --
#     `conda run -n rdm hf auth login --token hf_...` after stage 1, or `export HF_TOKEN=hf_...`.
#     Use a classic "Read" token; a fine-grained one needs "read access to public gated repos"
#     AND "read access to contents of all repos under your personal namespace": the same token
#     fetches the reference store from the private dataset jiachengcui888/sw-rdm-reference-store;
#   - optional: rclone with a Google Drive remote "gdrive" (account that owns SW-RDM/), the slow
#     fallback for the store if the HF dataset is not readable.
#
# Options: --root DIR (required)  --gpus N (2)  --config YAML  --gate  --steps N  --micro-batch N
#          --output-dir DIR  --env NAME (rdm)  --hf-cache DIR  --full-download
#          --skip-env  --skip-download  --skip-preprocess  --skip-geneval  --no-train  --dry-run
#          --resume  (continue the run in --output-dir from its resume.pth; skips stages 1-4)
#          --eval  (after training: GenEval + PickScore on every step_*.pth over all visible GPUs,
#                   scripts/eval_run.sh -> <run dir>/eval_summary.md; with --no-train it evaluates
#                   the existing run)  --eval-baselines  (also the 4-step teacher and released s180)
#          --allow-low-disk  (train even if the checkpoints will not all fit; you prune them)
#          --allow-hf-drift  (continue although a model file on the Hub changed since assets/hf_revisions.json)
set -euo pipefail
export PYTHONNOUSERSITE=1   # packages in ~/.local/lib/pythonX.Y must never shadow the conda envs'
# Empty "${arr[@]}" expansions under set -u (all over these scripts) need bash >= 4.4.
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "bash >= 4.4 required (this is $BASH_VERSION)" >&2; exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# This machine's Hugging Face token: write it into <repo>/.hf_token (git-ignored, never committed)
# instead of exporting it in every shell. An exported HF_TOKEN still wins; every stage inherits it.
if [[ -z "${HF_TOKEN:-}" && -s "$REPO/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$REPO/.hf_token")"; export HF_TOKEN
fi
ROOT=""
GPUS=2
CONFIG="configs/sw_lmmd_train_h100_2gpu.yaml"
ENV_NAME="rdm"
HF_CACHE=""
OUTPUT_DIR=""
STEPS=""
MICRO_BATCH=""
GATE=0
MINIMAL=1
DO_ENV=1; DO_DOWNLOAD=1; DO_PREPROCESS=1; DO_GENEVAL=1; DO_TRAIN=1
RESUME=0
ALLOW_LOW_DISK=0
ALLOW_HF_DRIFT=0
DO_EVAL=0
EVAL_BASELINES=0
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)            ROOT="$2"; shift 2 ;;
    --gpus)            GPUS="$2"; shift 2 ;;
    --config)          CONFIG="$2"; shift 2 ;;
    --env)             ENV_NAME="$2"; shift 2 ;;
    --hf-cache)        HF_CACHE="$2"; shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
    --steps)           STEPS="$2"; shift 2 ;;
    --micro-batch)     MICRO_BATCH="$2"; shift 2 ;;
    --gate)            GATE=1; shift ;;
    --full-download)   MINIMAL=0; shift ;;
    --skip-env)        DO_ENV=0; shift ;;
    --skip-download)   DO_DOWNLOAD=0; shift ;;
    --skip-preprocess) DO_PREPROCESS=0; shift ;;
    --skip-geneval)    DO_GENEVAL=0; shift ;;
    --no-train)        DO_TRAIN=0; shift ;;
    --resume)          RESUME=1; shift ;;
    --allow-low-disk)  ALLOW_LOW_DISK=1; shift ;;
    --allow-hf-drift)  ALLOW_HF_DRIFT=1; shift ;;
    --eval)            DO_EVAL=1; shift ;;
    --eval-baselines)  DO_EVAL=1; EVAL_BASELINES=1; shift ;;
    --dry-run)         DRY=1; shift ;;
    -h|--help)         sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[[ -n "$ROOT" ]] || { echo "usage: bash scripts/new_machine.sh --root <data-dir> [options]  (see --help)" >&2; exit 2; }
# --resume continues a run whose stages 1-4 completed before it started: go straight to training.
[[ $RESUME -eq 1 ]] && { DO_ENV=0; DO_DOWNLOAD=0; DO_PREPROCESS=0; DO_GENEVAL=0; }
ROOT="$(realpath -m "$ROOT")"
cd "$REPO"
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 1; }

SW_DIR="$ROOT/sw_lmmd"
STORE="$SW_DIR/reference_store"
if [[ $GATE -eq 1 ]]; then
  STEPS="${STEPS:-20}"
  OUTPUT_DIR="${OUTPUT_DIR:-$SW_DIR/work_dirs_gate}"   # never mixes with the real run's logs
fi
OUTPUT_DIR="${OUTPUT_DIR:-$SW_DIR/work_dirs}"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    \033[33mWARNING:\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
# `reference_extension:` of the config, following its `extends:` chain like rdm.train.launch.load_config
# (child keys win). Plain YAML scalars only -- enough for this key, and it works before stage 1.
cfg_get() {   # cfg_get <config> <key>
  local f="$1" key="$2" v parent
  [[ -f "$f" ]] || return 0
  v=$(grep -E "^$key:" "$f" | head -1 | sed -E "s/^$key:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^['\"]//; s/['\"]$//")
  if [[ -n "$v" ]]; then echo "$v"; return 0; fi
  parent=$(grep -E '^extends:' "$f" | head -1 | sed -E "s/^extends:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^['\"]//; s/['\"]$//")
  [[ -n "$parent" ]] && cfg_get "$(dirname "$f")/$parent" "$key"
  return 0
}
EXT_NAME="$(cfg_get "$CONFIG" reference_extension)"
EXT_DIR=""
EXT_HF="${EXT_HF-jiachengcui888/sw-rdm-geneval-block}"
if [[ -n "$EXT_NAME" ]]; then
  [[ "$EXT_NAME" == /* ]] && EXT_DIR="$EXT_NAME" || EXT_DIR="$SW_DIR/$EXT_NAME"   # next to the store
fi
# nvidia-smi can fail transiently (e.g. while another process initialises a GPU): retry first.
smi() { local i out; for i in 1 2 3; do out=$(nvidia-smi "$@" 2>/dev/null) && { printf '%s\n' "$out"; return 0; }; sleep 3; done; return 1; }

if [[ $DRY -eq 0 ]]; then
  mkdir -p "$SW_DIR/logs"
  LOG="$SW_DIR/logs/new_machine_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$LOG") 2>&1
fi

# ---------------------------------------------------------------------------
# preflight: everything that would otherwise fail hours in
# ---------------------------------------------------------------------------
say "preflight"
info "repo        $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"
info "root        $ROOT"
info "stages      env=$DO_ENV download=$DO_DOWNLOAD preprocess=$DO_PREPROCESS geneval=$DO_GENEVAL train=$DO_TRAIN"
info "training    $CONFIG on $GPUS GPU(s)${STEPS:+, $STEPS steps} -> $OUTPUT_DIR"
[[ $DRY -eq 1 ]] || info "log         $LOG"
command -v conda >/dev/null 2>&1 || die "conda not found on PATH"
# conda keeps the ~12 GB rdm env in its own envs dir -- on rented machines often a small root disk.
CONDA_BASE="$(conda info --base 2>/dev/null)" || die "conda info --base failed -- is conda initialised in this shell?"
ENV_FREE_GB=$(df -BG --output=avail "$CONDA_BASE" 2>/dev/null | tail -1 | tr -dc '0-9' || true)
info "conda       $CONDA_BASE (${ENV_FREE_GB:-?} GB free there; the rdm env needs ~12 GB)"
if [[ $DO_ENV -eq 1 && -n "${ENV_FREE_GB:-}" && "$ENV_FREE_GB" -lt 15 ]] \
   && ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  die "only ${ENV_FREE_GB} GB free where conda keeps its envs ($CONDA_BASE); the rdm env needs ~12 GB.
  Install Miniconda on the data volume instead, e.g.
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $ROOT/miniconda3 && source $ROOT/miniconda3/bin/activate
  and re-run this script from that shell."
fi
# Download caches go under --root as well, not into ~/.cache on the root disk.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$ROOT/.cache/conda-pkgs}"
[[ -z "${TMUX:-}${STY:-}" && $DRY -eq 0 && $DO_TRAIN -eq 1 ]] && \
  warn "not inside tmux/screen -- training runs for hours; a dropped SSH session kills it"

d="$ROOT"; while [[ ! -d "$d" ]]; do d="$(dirname "$d")"; done
FREE_GB=$(df -BG --output=avail "$d" | tail -1 | tr -dc '0-9')
info "free space  ${FREE_GB} GB on $(df --output=target "$d" | tail -1)"
[[ "$FREE_GB" -lt 315 ]] && warn "under ~315 GB free: weights + store + 61 GB context + geneval + 10 x 15.5 GB checkpoints + resume.pth"

if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then N_GPU=0; else N_GPU=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES"); fi
elif command -v nvidia-smi >/dev/null 2>&1; then
  N_GPU=$(smi -L | wc -l) || die "nvidia-smi failed (driver/library mismatch? reboot or reinstall the driver)"
else
  N_GPU=0
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER=$(smi --query-gpu=driver_version --format=csv,noheader | head -1) \
    || die "nvidia-smi failed (driver/library mismatch? reboot or reinstall the driver)"
  info "driver      $DRIVER"
  [[ "${DRIVER%%.*}" -ge 525 ]] || die "NVIDIA driver $DRIVER is too old: torch 2.8 (CUDA 12.6) needs >= 525"
  info "gpus        $N_GPU visible: $(smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
  MIN_MIB=$(smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1) || MIN_MIB=0
  if [[ "$CONFIG" == *h100* && "${MIN_MIB:-0}" -lt 79000 ]]; then
    warn "the H100 config needs ~66 GB per card (it is not sharded); on 24 GB cards use"
    warn "  --gpus 4 --config configs/sw_lmmd_train_4x4090.yaml"
  fi
fi
SHM_MB=$(df -BM --output=size /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9' || true)
if [[ $DO_TRAIN -eq 1 && -n "${SHM_MB:-}" && "$SHM_MB" -lt 1024 ]]; then
  warn "/dev/shm is only ${SHM_MB} MB (a container default?) -- NCCL can fail on it once training"
  warn "  starts; restart the container with --shm-size=16g (or --ipc=host) if it does"
fi
# Hugging Face: which account the token stage 2 will use belongs to, and whether it can download the
# gated FLUX.2-dev VAE and read the private reference-store dataset -- found out now, not mid-download.
# (The auth header comes from a file descriptor, so the token never shows in ps or in the log.)
SW_STORE_HF="${SW_STORE_HF-jiachengcui888/sw-rdm-reference-store}"
TOKEN="${HF_TOKEN:-}"
TOKFILE="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
[[ -z "$TOKEN" && -f "$TOKFILE" ]] && TOKEN="$(cat "$TOKFILE")"
hf_http() {   # hf_http <url> [head]: the HTTP status of a request made with the token
  curl -s -o /dev/null -w '%{http_code}' ${2:+-I} -H @<(printf 'Authorization: Bearer %s\n' "$TOKEN") "$1" \
    2>/dev/null || true
}
if [[ $DO_DOWNLOAD -eq 1 ]]; then
  if [[ -z "$TOKEN" ]]; then
    warn "no Hugging Face token yet: export HF_TOKEN=hf_... before stage 2 (FLUX.2-dev and the store need it)"
  else
    WHO=$(curl -s -H @<(printf 'Authorization: Bearer %s\n' "$TOKEN") https://huggingface.co/api/whoami-v2 \
            2>/dev/null | grep -o '"name":"[^"]*"' | head -1 | cut -d'"' -f4 || true)
    info "HF token    account ${WHO:-<unknown: token invalid?>}"
    if ! ls "${HF_CACHE:-$ROOT/hf/hub}"/models--black-forest-labs--FLUX.2-dev/snapshots/*/ae.safetensors >/dev/null 2>&1; then
      CODE=$(hf_http https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/main/ae.safetensors head)
      case "$CODE" in
        2??|3??) info "FLUX.2-dev  gated VAE downloadable with this token" ;;
        *) MSG="this token (account ${WHO:-?}) cannot download the gated FLUX.2-dev VAE (HTTP ${CODE:-?}).
  Log in to huggingface.co as ${WHO:-that account}, accept the licence at
  https://huggingface.co/black-forest-labs/FLUX.2-dev, and if the token is fine-grained, give it
  \"read access to contents of all public gated repos you can access\" (or use a classic Read token)."
           if [[ $DRY -eq 1 ]]; then warn "$MSG"; else die "$MSG"; fi ;;
      esac
    fi
  fi
fi
if [[ $DO_DOWNLOAD -eq 1 && ! -f "$STORE/metadata.json" && -z "${SW_STORE_TAR:-}" ]]; then
  # Stage 2 fetches the reference store from the private HF dataset, else from Google Drive with
  # rclone (a fallback: ~2 h against ~2 min). Check now that one of them works.
  RCLONE_SRC="${SW_STORE_RCLONE:-gdrive:SW-RDM}"
  HF_STATE="off (SW_STORE_HF empty)"
  if [[ -n "$SW_STORE_HF" ]]; then
    if [[ -z "$TOKEN" ]]; then
      HF_STATE="no token yet"
    else
      CODE=$(hf_http "https://huggingface.co/api/datasets/$SW_STORE_HF")
      if [[ "$CODE" == 200 ]]; then HF_STATE="ok"
      else HF_STATE="not readable by this token's account ${WHO:-?} (HTTP ${CODE:-?})"; fi
    fi
  fi
  RC_STATE="ok"
  if ! command -v rclone >/dev/null 2>&1; then
    RC_STATE="rclone not installed"
  elif ! rclone listremotes 2>/dev/null | grep -qx "${RCLONE_SRC%%:*}:"; then
    RC_STATE="no rclone remote '${RCLONE_SRC%%:*}:'"
  elif ! rclone lsf "$RCLONE_SRC/" 2>/dev/null | grep -qx "reference_store_noctx.tar"; then
    RC_STATE="$RCLONE_SRC/reference_store_noctx.tar not found"
  fi
  if [[ "$HF_STATE" == ok ]]; then
    info "store       HF dataset $SW_STORE_HF (readable); rclone fallback: $RC_STATE"
  elif [[ "$RC_STATE" == ok ]]; then
    warn "store: HF dataset $SW_STORE_HF: $HF_STATE -- stage 2 falls back to rclone $RCLONE_SRC (Drive: ~2 h)"
  elif [[ $DRY -eq 1 || "$HF_STATE" == "no token yet" ]]; then
    warn "store: HF dataset $SW_STORE_HF: $HF_STATE; rclone: $RC_STATE -- stage 2 needs one of them"
  else
    die "cannot fetch the reference store. HF dataset $SW_STORE_HF: $HF_STATE; rclone: $RC_STATE.
  Give this machine a token that can read $SW_STORE_HF (export HF_TOKEN=hf_...; a classic \"Read\"
  token reads your own private repos), or set up the rclone fallback (rclone config, remote gdrive)."
  fi
fi
if [[ -n "$EXT_DIR" ]]; then
  info "reference   store + extension '$EXT_NAME' -> $EXT_DIR"
  if [[ -f "$EXT_DIR/metadata.json" ]]; then
    info "extension   present"
  elif [[ $DO_DOWNLOAD -eq 1 && "$EXT_NAME" != /* ]]; then
    CODE=$([[ -n "$TOKEN" ]] && hf_http "https://huggingface.co/api/datasets/$EXT_HF" || echo "no token")
    if [[ "$CODE" == 200 ]]; then
      info "extension   HF dataset $EXT_HF (readable) -> stage 2 downloads $EXT_NAME/"
    else
      MSG="the config trains on the extension '$EXT_NAME', but HF dataset $EXT_HF is not readable
  with this token (account ${WHO:-?}: ${CODE:-?}). Use a token of the account that owns it (it is
  private), or set EXT_HF=<user>/<dataset>, or copy the block to $EXT_DIR yourself."
      if [[ $DRY -eq 1 || -z "$TOKEN" ]]; then warn "$MSG"; else die "$MSG"; fi
    fi
  elif [[ $DO_TRAIN -eq 1 && $DRY -eq 0 ]]; then
    die "the config trains on the extension '$EXT_NAME', but $EXT_DIR does not exist and downloads are
  skipped. Fetch it:  hf download $EXT_HF --repo-type dataset --include \"$EXT_NAME/*\" --local-dir $SW_DIR"
  fi
fi
if [[ $DRY -eq 0 ]]; then
  [[ $DO_PREPROCESS -eq 1 && "$N_GPU" -lt 1 ]] && die "stage 3 needs 1 GPU; none visible"
  [[ $DO_TRAIN -eq 1 && "$N_GPU" -lt "$GPUS" ]] && die "training asks for $GPUS GPUs but $N_GPU are visible"
fi

# ---------------------------------------------------------------------------
# 1  environment
# ---------------------------------------------------------------------------
if [[ $DO_ENV -eq 1 ]]; then
  say "stage 1/5: environment ($ENV_NAME)"
  bash "$REPO/scripts/setup_env.sh" --name "$ENV_NAME" $([[ $DRY -eq 1 ]] && echo --dry-run)
fi

# ---------------------------------------------------------------------------
# 2  downloads
# ---------------------------------------------------------------------------
if [[ $DO_DOWNLOAD -eq 1 ]]; then
  say "stage 2/5: downloads -> $ROOT"
  # The gated FLUX.2-dev VAE fails ~40 GB into the downloads without a token -- check first,
  # unless it is already in the cache.
  HUB="${HF_CACHE:-$ROOT/hf/hub}"
  if ! ls "$HUB"/models--black-forest-labs--FLUX.2-dev/snapshots/*/ae.safetensors >/dev/null 2>&1; then
    if conda run -n "$ENV_NAME" python -c \
         "import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)" 2>/dev/null; then
      info "Hugging Face token found"
    elif [[ $DRY -eq 1 ]]; then
      warn "no Hugging Face token yet -- needed for the gated FLUX.2-dev VAE"
    else
      die "no Hugging Face token. The FLUX.2 VAE is gated:
  1) accept the licence at https://huggingface.co/black-forest-labs/FLUX.2-dev (once per account)
  2) create a \"Read\" token at https://huggingface.co/settings/tokens, then either
       conda run -n $ENV_NAME hf auth login --token hf_...     (saved for this machine)
     or
       export HF_TOKEN=hf_...                                  (this shell only)
  then re-run with --skip-env"
    fi
  fi
  DL=(--root "$ROOT" --env "$ENV_NAME")
  [[ $MINIMAL -eq 1 ]] && DL+=(--minimal)
  [[ -n "$HF_CACHE" ]] && DL+=(--hf-cache "$HF_CACHE")
  [[ $DRY -eq 1 ]] && DL+=(--dry-run)
  CONDA_ENV="$ENV_NAME" bash "$REPO/scripts/download_all.sh" "${DL[@]}" \
    || die "downloads incomplete (see above); re-run with --skip-env to resume"
  # The Hub serves each repo's newest commit: make sure the model files are the ones the reference
  # results were produced with (scripts/check_hf_revisions.py).
  if [[ $DRY -eq 0 ]]; then
    if ! ( source "$ROOT/env.sh" && conda run -n "$ENV_NAME" --no-capture-output \
             python "$REPO/scripts/check_hf_revisions.py" --check "$REPO/assets/hf_revisions.json" ); then
      if [[ $ALLOW_HF_DRIFT -eq 1 ]]; then
        warn "model files differ from assets/hf_revisions.json (see above); continuing (--allow-hf-drift)"
      else
        die "a model on the Hugging Face Hub changed since the reference results were made (CHANGED above),
  or did not download (MISSING). Training on it would not reproduce them. Re-run with
  --skip-env --allow-hf-drift only if you accept that."
      fi
    fi
  fi
  # The reference extension (e.g. the GenEval block): its own private HF dataset, next to the store.
  if [[ -n "$EXT_DIR" && "$EXT_NAME" != /* ]]; then
    if [[ $DRY -eq 1 ]]; then
      info "would fetch $EXT_NAME/ from HF dataset $EXT_HF into $SW_DIR (skipped if present)"
    else
      say "stage 2/5: reference extension '$EXT_NAME' <- HF dataset $EXT_HF"
      # snapshot_download skips what is already complete and verifies every LFS file's sha256
      conda run -n "$ENV_NAME" --no-capture-output python - "$EXT_HF" "$EXT_NAME" "$SW_DIR" <<'PY' \
        || die "could not download the extension from $EXT_HF (see above); re-run with --skip-env"
import sys
from huggingface_hub import snapshot_download
repo, name, dest = sys.argv[1:]
snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{name}/*"], local_dir=dest)
PY
      [[ -f "$EXT_DIR/metadata.json" ]] || die "$EXT_HF holds no $EXT_NAME/metadata.json"
      # read it back WITH the store, for the config's encoders (the context comes in stage 3)
      ( source "$ROOT/env.sh" && conda run -n "$ENV_NAME" --no-capture-output python - "$CONFIG" "$STORE" "$EXT_DIR" <<'PY' ) \
        || die "the extension $EXT_DIR does not read back with the store (see above)"
import sys
import numpy as np
from rdm.sw_lmmd import ReferenceFeatureStore
from rdm.train.launch import load_config
cfg, store, ext = sys.argv[1:]
names = list(load_config(cfg).encoders)
s = ReferenceFeatureStore(store, names, extension=ext, require_context=False)
rows = np.arange(s.extension["first_row"], s.num_rows, 997)
for n in names:
    assert np.isfinite(s.reference_joint_features(n, rows).numpy()).all(), n
print(f"    ok  {ext}: {s.extension['rows']:,} rows behind the store's {s.extension['first_row']:,} "
      f"({100 * s.extension['rows'] / s.num_rows:.1f}% of {s.num_rows:,}), encoders {', '.join(names)}")
PY
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 3  rebuild the reference store's Qwen3 context
# ---------------------------------------------------------------------------
PRE=(env ASSETS="$ROOT" RDM_REPO="$REPO" CONDA_ENV="$ENV_NAME" bash "$REPO/scripts/preprocess_all-new-machine.sh")
if [[ $DO_PREPROCESS -eq 1 ]]; then
  say "stage 3/5: Qwen3 context for the reference store"
  if [[ $DRY -eq 1 ]]; then
    info "would run: ${PRE[*]}"
  else
    [[ -f "$ROOT/env.sh" ]] || die "$ROOT/env.sh missing -- stage 2 has not completed"
    "${PRE[@]}"
  fi
fi

# ---------------------------------------------------------------------------
# 4  GenEval scorer environment (scripts/eval_checkpoint.sh finds it at <root>/geneval)
# ---------------------------------------------------------------------------
# Installed now because its self-test needs a GPU, and training then holds them for days. The env
# goes under --root (not conda's envs dir, which is often a small home quota). It is not needed
# for training, so a failure warns instead of stopping the run.
GENEVAL=(bash "$REPO/scripts/setup_geneval.sh" --root "$ROOT/geneval" --prefix "$ROOT/geneval/env")
if [[ $DO_GENEVAL -eq 1 ]]; then
  say "stage 4/5: GenEval scorer environment -> $ROOT/geneval"
  if [[ $DRY -eq 1 ]]; then
    "${GENEVAL[@]}" --dry-run || warn "setup_geneval.sh --dry-run failed"
  elif ! "${GENEVAL[@]}"; then
    warn "the GenEval setup failed (see above); training starts anyway. Retry it any time with"
    warn "  ${GENEVAL[*]}"
  fi
fi

# ---------------------------------------------------------------------------
# 5  training
# ---------------------------------------------------------------------------
TRAIN_ENV=(ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" GPUS="$GPUS" REFERENCE_ROOT="$STORE" OUTPUT_DIR="$OUTPUT_DIR")
[[ -n "$STEPS" ]] && TRAIN_ENV+=(STEPS="$STEPS")
[[ -n "$MICRO_BATCH" ]] && TRAIN_ENV+=(MICRO_BATCH="$MICRO_BATCH")
TRAIN_CMD=(bash "$REPO/scripts/train_sw_lmmd.sh" "$CONFIG")
if [[ $DO_TRAIN -eq 1 ]]; then
  say "stage 5/5: training"
  if [[ $DRY -eq 1 ]]; then
    info "would run: env ${TRAIN_ENV[*]}$([[ $RESUME -eq 1 ]] && echo ' RESUME_FROM=<run dir>/resume.pth') ${TRAIN_CMD[*]}"
  else
    [[ -f "$STORE/qwen_context.npy" ]] || die "$STORE has no qwen_context.npy -- stage 3 has not completed"
    if [[ -n "$EXT_DIR" ]]; then
      [[ -f "$EXT_DIR/metadata.json" ]] || die "the config trains on the extension '$EXT_NAME' but $EXT_DIR is missing.
  Fetch it:  hf download $EXT_HF --repo-type dataset --include \"$EXT_NAME/*\" --local-dir $SW_DIR"
      info "reference   $STORE + $EXT_DIR"
    fi
    CFGVALS=$(conda run -n "$ENV_NAME" --no-capture-output python -c \
      "from rdm.train.launch import load_config; c = load_config('$CONFIG'); print(c.exp_name, c.steps, c.save_freq, int(bool(getattr(c, 'save_resume', False))))" \
      | sed '/^[[:space:]]*$/d' | tail -1) || die "could not load $CONFIG (see the error above)"
    read -r EXP CFG_STEPS SAVE_FREQ SAVE_RESUME <<< "$CFGVALS"
    [[ -n "$EXP" && -n "${SAVE_RESUME:-}" ]] || die "could not read exp_name/steps/save_freq from $CONFIG"
    RUN_DIR="$OUTPUT_DIR/$EXP"
    # Never two trainings at once: they would write the same run dir and checkpoints. Only a python
    # process running the module counts (torchrun, its workers, conda run around them) -- not an
    # editor, a grep or a shell that merely mentions it.
    TRAIN_PAT='^[^ ]*python[0-9.]* .*-m rdm[.]sw_lmmd[.]launch'
    if pgrep -u "$(id -u)" -f "$TRAIN_PAT" >/dev/null 2>&1; then
      die "a training process is already running:
$(pgrep -u "$(id -u)" -af "$TRAIN_PAT" | cut -c1-160)
  stop it (kill <pid>) or let it finish before starting another"
    fi
    # Every checkpoint still to come must fit, or the run dies at a save days in.
    N_LEFT=$(( (${STEPS:-$CFG_STEPS} + SAVE_FREQ - 1) / SAVE_FREQ ))
    N_HAVE=0                                   # (find on a missing dir fails, and pipefail would end the script)
    [[ -d "$RUN_DIR" ]] && N_HAVE=$(find "$RUN_DIR" -maxdepth 1 -name 'step_*.pth' | wc -l)
    N_LEFT=$(( N_LEFT > N_HAVE ? N_LEFT - N_HAVE : 0 ))
    NEED_GB=$(( N_LEFT * 31 / 2 + SAVE_RESUME * 47 + 10 ))     # 15.5 GB each; resume.pth x2 while replaced
    d="$OUTPUT_DIR"; while [[ ! -d "$d" ]]; do d="$(dirname "$d")"; done
    OUT_FREE_GB=$(df -BG --output=avail "$d" | tail -1 | tr -dc '0-9')
    info "disk        $OUT_FREE_GB GB free for $N_LEFT more checkpoints x 15.5 GB$( (( SAVE_RESUME )) && echo ' + resume.pth') -> need ~$NEED_GB GB"
    if [[ "$OUT_FREE_GB" -lt "$NEED_GB" ]]; then
      if [[ $ALLOW_LOW_DISK -eq 1 ]]; then
        warn "not enough disk for every checkpoint (--allow-low-disk): delete old step_*.pth as the run goes"
      else
        die "$OUT_FREE_GB GB free under $OUTPUT_DIR but the run will write ~$NEED_GB GB of checkpoints;
  it would die at a save. Free space, pass --output-dir on a bigger volume, or --allow-low-disk
  (then delete old step_*.pth yourself while it trains)"
      fi
    fi
    if [[ $RESUME -eq 1 ]]; then
      [[ -f "$RUN_DIR/resume.pth" ]] || die "--resume: $RUN_DIR/resume.pth does not exist (it is written every
  resume_every steps by configs with save_resume: true; check --output-dir / --gate match the run)"
      TRAIN_ENV+=(RESUME_FROM="$RUN_DIR/resume.pth")
    elif [[ -s "$RUN_DIR/train_log.jsonl" ]]; then
      # Starting over would restart at step 0, append to the old log and overwrite its checkpoints.
      die "$RUN_DIR already holds a run. To continue it, re-run with --resume. To start over,
  move it away (mv $RUN_DIR{,_old}) or pass --output-dir"
    fi
    env "${TRAIN_ENV[@]}" "${TRAIN_CMD[@]}"
    say "done"
    info "logs + checkpoints: $RUN_DIR"
    info "evaluate one: CUDA_VISIBLE_DEVICES=0 bash scripts/eval_checkpoint.sh $RUN_DIR/step_NNNNNNN.pth --root $ROOT"
  fi
fi

# ---------------------------------------------------------------------------
# 6  evaluation (--eval): every checkpoint of the run, all visible GPUs, one table
# ---------------------------------------------------------------------------
if [[ $DO_EVAL -eq 1 ]]; then
  say "stage 6/6: evaluation (GenEval + PickScore)"
  if [[ -z "${RUN_DIR:-}" ]]; then                 # --no-train: evaluate the run already on disk
    if [[ $DRY -eq 1 ]]; then
      RUN_DIR="$OUTPUT_DIR/<exp_name of $CONFIG>"
    else
      EXP=$(conda run -n "$ENV_NAME" --no-capture-output python -c \
        "from rdm.train.launch import load_config; print(load_config('$CONFIG').exp_name)" \
        | sed '/^[[:space:]]*$/d' | tail -1) || die "could not load $CONFIG"
      RUN_DIR="$OUTPUT_DIR/$EXP"
    fi
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then EVAL_GPUS="$CUDA_VISIBLE_DEVICES"
  else EVAL_GPUS=$(smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -); fi
  # one evaluation peaks at ~21 GB: two per card from 48 GB up (eval_run.sh --per-gpu)
  EVAL_PER_GPU=1; [[ "${MIN_MIB:-0}" -ge 48000 ]] && EVAL_PER_GPU=2
  EVAL=(bash "$REPO/scripts/eval_run.sh" "$RUN_DIR" --root "$ROOT" --env "$ENV_NAME"
        --gpus "$EVAL_GPUS" --per-gpu "$EVAL_PER_GPU")
  [[ $EVAL_BASELINES -eq 1 ]] && EVAL+=(--baselines)
  EVAL_GROOT="${GENEVAL_ROOT:-$ROOT/geneval}"       # stage 4 installs it at <root>/geneval
  if [[ -f "$EVAL_GROOT/geneval_env.sh" ]]; then
    EVAL+=(--geneval-root "$EVAL_GROOT")
  else
    warn "no GenEval scorer env at $EVAL_GROOT (stage 4 skipped or failed): PickScore only"
    EVAL+=(--no-geneval)
  fi
  if [[ $DRY -eq 1 ]]; then
    info "would run: env ASSETS=$ROOT CONDA_ENV=$ENV_NAME ${EVAL[*]}"
  else
    N_CKPT=0
    [[ -d "$RUN_DIR" ]] && N_CKPT=$(find "$RUN_DIR" -maxdepth 1 -name 'step_*.pth' | wc -l)
    if [[ $N_CKPT -eq 0 && $EVAL_BASELINES -eq 0 ]]; then
      warn "no step_*.pth in $RUN_DIR to evaluate$([[ $GATE -eq 1 ]] && echo ' (a --gate run saves none)')"
    else
      info "$N_CKPT checkpoint(s)$([[ $EVAL_BASELINES -eq 1 ]] && echo ' + teacher + s180') on GPU(s) $EVAL_GPUS, $EVAL_PER_GPU per card"
      # an evaluation failure must not hide that training finished: warn, the command retries it
      if env ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" "${EVAL[@]}"; then
        info "table: $RUN_DIR/eval_summary.md"
      else
        warn "some evaluations failed (see above); re-run to retry only those:"
        warn "  bash scripts/new_machine.sh --root $ROOT --config $CONFIG --skip-env --skip-download --skip-preprocess --skip-geneval --output-dir $OUTPUT_DIR --no-train --eval$([[ $EVAL_BASELINES -eq 1 ]] && echo -baselines)"
      fi
    fi
  fi
fi

if [[ $DRY -eq 1 ]]; then
  say "dry run: nothing was changed"
elif [[ $DO_TRAIN -eq 0 && $DO_EVAL -eq 0 ]]; then
  say "ready -- to train:"
  info "env ${TRAIN_ENV[*]} ${TRAIN_CMD[*]}"
fi
