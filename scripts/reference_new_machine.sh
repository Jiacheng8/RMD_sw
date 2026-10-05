#!/usr/bin/env bash
# New machine -> the GenEval reference block in one command: environment, downloads, the GenEval
# scorer, then render + score + select + extract (scripts/build_geneval_block.sh). The block-making
# counterpart of scripts/new_machine.sh; it needs neither COCO nor the COCO reference store.
#
#   git clone -b geneval-block https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   echo hf_xxx > .hf_token && chmod 600 .hf_token
#   bash scripts/reference_new_machine.sh --root /data/<you>/rdm-sets --dry-run     # plan + checks only
#   bash scripts/reference_new_machine.sh --root /data/<you>/rdm-sets --limit 4 --seeds 8   # ~15 min smoke test
#   bash scripts/reference_new_machine.sh --root /data/<you>/rdm-sets                # the real block
#   bash scripts/reference_new_machine.sh --root <dir> --upload <hf-user>/sw-geneval-block  # + push it to HF
#
# Stages -- all idempotent; a re-run skips finished work (renders and scores resume per image/shard):
#   1 env        setup_env.sh                     conda env "rdm" (torch 2.8.0+cu126)
#   2 download   fetch_prerequisites.py           FLUX.2 klein-4B + AE + Qwen3-4B, the flux2 source,
#                                                 the block's encoders, PickScore, the released tau(c)
#                                                 table; then the model revisions are checked against
#                                                 assets/hf_revisions.json
#   3 geneval    setup_geneval.sh                 the official scorer env in <root>/geneval (5-10 min;
#                                                 mmcv is source-built on H100) + its self-test
#   4 build      build_geneval_block.sh           Qwen3 ctx of the 553 prompts; SEEDS teacher renders
#                                                 per prompt, one shard per GPU; the scorer per shard;
#                                                 keep the correct renders (whole groups of 4, <= 96 per
#                                                 prompt); encoder features -> <root>/sw_lmmd/geneval_block
#   5 upload     (only with --upload)             the block dir to a PRIVATE HF dataset
#
# Time (553 x 128 = 70,784 renders): ~3.6 img/s per RTX 4090 for the 4-step render, so ~6.5 h on one
# 4090, ~1.6 h on four; H100s are faster. Scoring adds roughly a third of that, features a few min per
# encoder. Disk ~90 GB under --root: weights ~25, scorer env ~12, renders ~0.45 MB each (~32 GB at
# SEEDS=128), the block ~1-2 GB. After the block is built the renders can be deleted.
#
# Manual prerequisites, once per machine (as for new_machine.sh):
#   - conda on PATH (on a small root disk install Miniconda under --root);
#   - an HF "Read" token whose account accepted the FLUX.2-dev licence (the VAE is gated), in
#     <repo>/.hf_token or HF_TOKEN; --upload needs a token with WRITE access instead.
#
# Then on the TRAINING machine the block must sit next to the store, <root>/sw_lmmd/geneval_block:
#   hf download <hf-user>/sw-geneval-block --repo-type dataset --local-dir <root>/sw_lmmd
# and train with configs/sw_lmmd_train_h100_2gpu_gan_grouped_geneval.yaml.
#
# Options: --root DIR (required)  --gpus N (all visible)  --seeds N (128)  --max-per-prompt N (96)
#          --encoders LIST (dinov3_l,siglip2,aimv2_huge)  --limit N (first N prompts: smoke test,
#          written to *_smoke dirs)  --upload HF_DATASET  --env NAME (rdm)  --hf-cache DIR
#          --geneval-root DIR (<root>/geneval; an existing scorer env with --skip-geneval)
#          --skip-env  --skip-download  --skip-geneval  --no-build  --allow-hf-drift  --dry-run
set -euo pipefail
export PYTHONNOUSERSITE=1   # packages in ~/.local/lib/pythonX.Y must never shadow the conda envs'
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "bash >= 4.4 required (this is $BASH_VERSION)" >&2; exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${HF_TOKEN:-}" && -s "$REPO/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$REPO/.hf_token")"; export HF_TOKEN
fi
ROOT=""
GPUS=""
SEEDS=128
MAX_PER_PROMPT=96
ENCODERS="dinov3_l,siglip2,aimv2_huge"
LIMIT=0
UPLOAD=""
GENEVAL_ROOT=""
ENV_NAME="rdm"
HF_CACHE=""
DO_ENV=1; DO_DOWNLOAD=1; DO_GENEVAL=1; DO_BUILD=1
ALLOW_HF_DRIFT=0
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)            ROOT="$2"; shift 2 ;;
    --gpus)            GPUS="$2"; shift 2 ;;
    --seeds)           SEEDS="$2"; shift 2 ;;
    --max-per-prompt)  MAX_PER_PROMPT="$2"; shift 2 ;;
    --encoders)        ENCODERS="$2"; shift 2 ;;
    --limit)           LIMIT="$2"; shift 2 ;;
    --upload)          UPLOAD="$2"; shift 2 ;;
    --geneval-root)    GENEVAL_ROOT="$2"; shift 2 ;;
    --env)             ENV_NAME="$2"; shift 2 ;;
    --hf-cache)        HF_CACHE="$2"; shift 2 ;;
    --skip-env)        DO_ENV=0; shift ;;
    --skip-download)   DO_DOWNLOAD=0; shift ;;
    --skip-geneval)    DO_GENEVAL=0; shift ;;
    --no-build)        DO_BUILD=0; shift ;;
    --allow-hf-drift)  ALLOW_HF_DRIFT=1; shift ;;
    --dry-run)         DRY=1; shift ;;
    -h|--help)         sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[[ -n "$ROOT" ]] || { echo "usage: bash scripts/reference_new_machine.sh --root <data-dir> [options]  (see --help)" >&2; exit 2; }
(( MAX_PER_PROMPT % 4 == 0 )) || { echo "--max-per-prompt must be a multiple of 4 (the window group size)" >&2; exit 2; }
ROOT="$(realpath -m "$ROOT")"
cd "$REPO"

SW_DIR="$ROOT/sw_lmmd"
SUFFIX=""; [[ "$LIMIT" -gt 0 ]] && SUFFIX="_smoke"        # a smoke test never mixes with the real block
RENDER_DIR="$SW_DIR/geneval_renders$SUFFIX"
BLOCK_DIR="$SW_DIR/geneval_block$SUFFIX"
GENEVAL_ROOT="$(realpath -m "${GENEVAL_ROOT:-$ROOT/geneval}")"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    \033[33mWARNING:\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
smi() { local i out; for i in 1 2 3; do out=$(nvidia-smi "$@" 2>/dev/null) && { printf '%s\n' "$out"; return 0; }; sleep 3; done; return 1; }

if [[ $DRY -eq 0 ]]; then
  mkdir -p "$SW_DIR/logs"
  LOG="$SW_DIR/logs/reference_new_machine_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$LOG") 2>&1
fi

# ---------------------------------------------------------------------------
# preflight: everything that would otherwise fail hours in
# ---------------------------------------------------------------------------
say "preflight"
info "repo        $REPO ($(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')@$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"
info "root        $ROOT"
info "stages      env=$DO_ENV download=$DO_DOWNLOAD geneval=$DO_GENEVAL build=$DO_BUILD upload=${UPLOAD:-no}"
info "block       553 prompts$([[ $LIMIT -gt 0 ]] && echo " (first $LIMIT only: smoke test)") x $SEEDS seeds, <= $MAX_PER_PROMPT kept, encoders $ENCODERS -> $BLOCK_DIR"
[[ $DRY -eq 1 ]] || info "log         $LOG"
[[ -f scripts/build_geneval_block.sh ]] || die "this checkout has no scripts/build_geneval_block.sh -- check out the geneval-block branch"
command -v conda >/dev/null 2>&1 || die "conda not found on PATH"
CONDA_BASE="$(conda info --base 2>/dev/null)" || die "conda info --base failed -- is conda initialised in this shell?"
ENV_FREE_GB=$(df -BG --output=avail "$CONDA_BASE" 2>/dev/null | tail -1 | tr -dc '0-9' || true)
info "conda       $CONDA_BASE (${ENV_FREE_GB:-?} GB free there; the rdm env needs ~12 GB)"
if [[ $DO_ENV -eq 1 && -n "${ENV_FREE_GB:-}" && "$ENV_FREE_GB" -lt 15 ]] \
   && ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  die "only ${ENV_FREE_GB} GB free where conda keeps its envs ($CONDA_BASE); the rdm env needs ~12 GB.
  Install Miniconda on the data volume instead, e.g.
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $ROOT/miniconda3 && source $ROOT/miniconda3/bin/activate"
fi
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$ROOT/.cache/conda-pkgs}"
[[ -z "${TMUX:-}${STY:-}" && $DRY -eq 0 && $DO_BUILD -eq 1 ]] && \
  warn "not inside tmux/screen -- the render runs for hours; a dropped SSH session kills it"

d="$ROOT"; while [[ ! -d "$d" ]]; do d="$(dirname "$d")"; done
FREE_GB=$(df -BG --output=avail "$d" | tail -1 | tr -dc '0-9')
N_PROMPTS=553; [[ $LIMIT -gt 0 && $LIMIT -lt 553 ]] && N_PROMPTS=$LIMIT
NEED_GB=$(( 45 + N_PROMPTS * SEEDS * 45 / 100000 + 3 ))     # weights + scorer env + 0.45 MB/render + block
info "free space  ${FREE_GB} GB on $(df --output=target "$d" | tail -1) (need ~${NEED_GB} GB)"
[[ "$FREE_GB" -lt "$NEED_GB" ]] && { if [[ $DRY -eq 1 ]]; then warn "not enough free space"; else die "need ~${NEED_GB} GB under $ROOT, have $FREE_GB"; fi; }

# GPUs: the physical ids the shards run on (CUDA_VISIBLE_DEVICES is honoured), first --gpus of them.
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  IFS=',' read -r -a ALL_IDS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t ALL_IDS < <(smi --query-gpu=index --format=csv,noheader | tr -d ' ') \
    || die "nvidia-smi failed (driver/library mismatch? reboot or reinstall the driver)"
else
  ALL_IDS=()
fi
N_GPU=${#ALL_IDS[@]}
[[ -z "$GPUS" ]] && GPUS=$N_GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER=$(smi --query-gpu=driver_version --format=csv,noheader | head -1) || die "nvidia-smi failed"
  info "driver      $DRIVER"
  [[ "${DRIVER%%.*}" -ge 525 ]] || die "NVIDIA driver $DRIVER is too old: torch 2.8 (CUDA 12.6) needs >= 525"
  info "gpus        $N_GPU visible: $(smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
  MIN_MIB=$(smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1) || MIN_MIB=0
  [[ "${MIN_MIB:-0}" -lt 16000 ]] && warn "the 4-step render needs ~11 GB per GPU; the smallest card has ${MIN_MIB} MiB"
fi
if [[ $DRY -eq 0 && $DO_BUILD -eq 1 ]]; then
  [[ "$N_GPU" -ge 1 ]] || die "no GPU visible"
  [[ "$GPUS" -le "$N_GPU" ]] || die "--gpus $GPUS but only $N_GPU visible"
fi
GPU_IDS=$(IFS=,; echo "${ALL_IDS[*]:0:$GPUS}")
info "render on   GPU(s) ${GPU_IDS:-none} ($GPUS shard(s))"

# Hugging Face: the token's account, the gated FLUX.2-dev VAE, and (for --upload) write access.
TOKEN="${HF_TOKEN:-}"
TOKFILE="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
[[ -z "$TOKEN" && -f "$TOKFILE" ]] && TOKEN="$(cat "$TOKFILE")"
hf_http() { curl -s -o /dev/null -w '%{http_code}' ${2:+-I} -H @<(printf 'Authorization: Bearer %s\n' "$TOKEN") "$1" 2>/dev/null || true; }
if [[ $DO_DOWNLOAD -eq 1 || -n "$UPLOAD" ]]; then
  if [[ -z "$TOKEN" ]]; then
    MSG="no Hugging Face token: write one to $REPO/.hf_token (or export HF_TOKEN=hf_...); the FLUX.2-dev VAE is gated"
    if [[ $DRY -eq 1 ]]; then warn "$MSG"; else die "$MSG"; fi
  else
    WHOAMI=$(curl -s -H @<(printf 'Authorization: Bearer %s\n' "$TOKEN") https://huggingface.co/api/whoami-v2 2>/dev/null || true)
    WHO=$(grep -o '"name":"[^"]*"' <<< "$WHOAMI" | head -1 | cut -d'"' -f4 || true)
    info "HF token    account ${WHO:-<unknown: token invalid?>}"
    if [[ $DO_DOWNLOAD -eq 1 ]] && ! ls "${HF_CACHE:-$ROOT/hf/hub}"/models--black-forest-labs--FLUX.2-dev/snapshots/*/ae.safetensors >/dev/null 2>&1; then
      CODE=$(hf_http https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/main/ae.safetensors head)
      case "$CODE" in
        2??|3??) info "FLUX.2-dev  gated VAE downloadable with this token" ;;
        *) MSG="this token (account ${WHO:-?}) cannot download the gated FLUX.2-dev VAE (HTTP ${CODE:-?}).
  Accept the licence at https://huggingface.co/black-forest-labs/FLUX.2-dev as ${WHO:-that account}
  (a fine-grained token also needs \"read access to contents of all public gated repos\")."
           if [[ $DRY -eq 1 ]]; then warn "$MSG"; else die "$MSG"; fi ;;
      esac
    fi
    if [[ -n "$UPLOAD" ]]; then
      ROLE=$(grep -o '"role":"[^"]*"' <<< "$WHOAMI" | head -1 | cut -d'"' -f4 || true)
      info "upload      -> HF dataset $UPLOAD (private); token role: ${ROLE:-?}"
      [[ "$ROLE" == read ]] && { MSG="--upload needs a token with write access; this one is read-only"
        if [[ $DRY -eq 1 ]]; then warn "$MSG"; else die "$MSG"; fi; }
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 1  environment
# ---------------------------------------------------------------------------
if [[ $DO_ENV -eq 1 ]]; then
  say "stage 1/5: environment ($ENV_NAME)"
  bash "$REPO/scripts/setup_env.sh" --name "$ENV_NAME" $([[ $DRY -eq 1 ]] && echo --dry-run)
fi

# ---------------------------------------------------------------------------
# 2  downloads: no COCO, no reference store -- only what rendering, scoring and the features need
# ---------------------------------------------------------------------------
FETCH=(conda run -n "$ENV_NAME" --no-capture-output python -u "$REPO/scripts/fetch_prerequisites.py"
       --root "$ROOT" --group encoders --group flux --group pickscore --group flux2src --group assets
       --encoders "$ENCODERS")
[[ -n "$HF_CACHE" ]] && FETCH+=(--hf-cache "$HF_CACHE")
if [[ $DO_DOWNLOAD -eq 1 ]]; then
  say "stage 2/5: downloads -> $ROOT"
  if [[ $DRY -eq 1 ]]; then
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then "${FETCH[@]}" --dry-run || true
    else info "would run: ${FETCH[*]}"; fi
  else
    "${FETCH[@]}" || die "downloads incomplete (see above); re-run with --skip-env to resume"
    if ! ( source "$ROOT/env.sh" && conda run -n "$ENV_NAME" --no-capture-output \
             python "$REPO/scripts/check_hf_revisions.py" --check "$REPO/assets/hf_revisions.json" ); then
      if [[ $ALLOW_HF_DRIFT -eq 1 ]]; then
        warn "model files differ from assets/hf_revisions.json (see above); continuing (--allow-hf-drift)"
      else
        die "a model on the Hugging Face Hub changed since the reference results were made (CHANGED above),
  or did not download (MISSING). A block rendered with it would not match the COCO store. Re-run
  with --skip-env --allow-hf-drift only if you accept that."
      fi
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 3  GenEval scorer environment -- required here: it decides which renders enter the block
# ---------------------------------------------------------------------------
GENEVAL=(bash "$REPO/scripts/setup_geneval.sh" --root "$GENEVAL_ROOT" --prefix "$GENEVAL_ROOT/env")
if [[ $DO_GENEVAL -eq 1 ]]; then
  say "stage 3/5: GenEval scorer environment -> $GENEVAL_ROOT"
  if [[ $DRY -eq 1 ]]; then
    "${GENEVAL[@]}" --dry-run || warn "setup_geneval.sh --dry-run failed"
  else
    "${GENEVAL[@]}" || die "the GenEval setup failed (see above); fix it and re-run with --skip-env --skip-download"
  fi
fi

# ---------------------------------------------------------------------------
# 4  build the block
# ---------------------------------------------------------------------------
BUILD_ENV=(ASSETS="$ROOT" RDM_REPO="$REPO" CONDA_ENV="$ENV_NAME" GPU_IDS="$GPU_IDS" GENEVAL_SEEDS="$SEEDS"
           MAX_PER_PROMPT="$MAX_PER_PROMPT" LIMIT="$LIMIT" BLOCK_ENCODERS="$ENCODERS"
           RENDER_DIR="$RENDER_DIR" BLOCK_DIR="$BLOCK_DIR")
BUILD=(bash "$REPO/scripts/build_geneval_block.sh" --geneval-root "$GENEVAL_ROOT")
if [[ $DO_BUILD -eq 1 ]]; then
  say "stage 4/5: render, score, select, extract -> $BLOCK_DIR"
  if [[ $DRY -eq 1 ]]; then
    info "would run: env ${BUILD_ENV[*]} ${BUILD[*]}"
  else
    [[ -f "$ROOT/env.sh" ]] || die "$ROOT/env.sh missing -- stage 2 has not completed"
    [[ -f "$GENEVAL_ROOT/geneval_env.sh" ]] || die "$GENEVAL_ROOT/geneval_env.sh missing -- stage 3 has not completed"
    env "${BUILD_ENV[@]}" "${BUILD[@]}"
    info "block: $(du -sh "$BLOCK_DIR" | cut -f1) at $BLOCK_DIR"
    conda run -n "$ENV_NAME" --no-capture-output python -c "
import json; m = json.load(open('$BLOCK_DIR/metadata.json'))
print(f\"    {m['num_rows']:,} rows over {m['prompts_covered']}/{m['num_prompts']} prompts; per task {m['rows_per_task']}\")
print(f\"    vs the 331,132-row COCO store: {m['num_rows'] / (m['num_rows'] + 331132):.1%} of the combined reference (iRDM: 17.8%)\")"
  fi
fi

# ---------------------------------------------------------------------------
# 5  upload (optional): the block only -- the renders stay here
# ---------------------------------------------------------------------------
if [[ -n "$UPLOAD" ]]; then
  say "stage 5/5: upload $BLOCK_DIR -> HF dataset $UPLOAD (private)"
  UP=(conda run -n "$ENV_NAME" --no-capture-output hf upload "$UPLOAD" "$BLOCK_DIR" "$(basename "$BLOCK_DIR")"
      --repo-type dataset --private --commit-message "GenEval reference block ($SEEDS seeds, <= $MAX_PER_PROMPT kept)")
  if [[ $DRY -eq 1 ]]; then
    info "would run: ${UP[*]}"
  else
    [[ -f "$BLOCK_DIR/metadata.json" ]] || die "no block at $BLOCK_DIR to upload"
    "${UP[@]}" || die "upload failed (see above); re-run with --skip-env --skip-download --skip-geneval"
  fi
fi

if [[ $DRY -eq 1 ]]; then
  say "dry run: nothing was changed"
else
  say "done"
  info "block       $BLOCK_DIR"
  info "on the training machine it must sit at <root>/sw_lmmd/$(basename "$BLOCK_DIR"):"
  if [[ -n "$UPLOAD" ]]; then
    info "  hf download $UPLOAD --repo-type dataset --local-dir <root>/sw_lmmd"
  else
    info "  rsync -a $BLOCK_DIR <training-host>:<root>/sw_lmmd/"
  fi
  info "then train with configs/sw_lmmd_train_h100_2gpu_gan_grouped_geneval.yaml"
  info "the renders ($(du -sh "$RENDER_DIR" 2>/dev/null | cut -f1)) are no longer needed: rm -rf $RENDER_DIR"
fi
