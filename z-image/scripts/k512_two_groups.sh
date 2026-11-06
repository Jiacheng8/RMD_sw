#!/usr/bin/env bash
# The current two-group protocol (README "Current H20 protocol") on ANY machine, in one command:
#   Group1  MMD only    z-image/configs/zimage_2step_1024_dino_qwenvl_k512_b64_mmd_only_from0.yaml
#   Group2  MMD + GAN   z-image/configs/zimage_2step_1024_dino_qwenvl_k512_b64_fixed_from0.yaml
# Both start from the original pretrained Z-Image-Turbo, 500 updates, K=512, B=64, MMD encoders
# DINOv3 + Qwen3-VL-4B. Everything they need is rebuilt from public Hugging Face repos.
#
#   git clone -b zimage-exp https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   tmux new -s k512
#   bash z-image/scripts/k512_two_groups.sh --root /data/<you>/rdm-sets --dry-run          # plan only
#   bash z-image/scripts/k512_two_groups.sh --root /data/<you>/rdm-sets --gate             # prepare + 3 updates per group
#   bash z-image/scripts/k512_two_groups.sh --root /data/<you>/rdm-sets                    # Group1, then Group2
#   bash z-image/scripts/k512_two_groups.sh --root /data/<you>/rdm-sets mmd-gan            # one group only
#   bash z-image/scripts/k512_two_groups.sh --root /data/<you>/rdm-sets mmd-gan --resume   # continue after a stop
#
# Stages (each finished stage is detected and skipped on a re-run):
#   1 env        scripts/setup_env.sh                  conda env rdm (+ aria2 for the teacher tars)
#   2 download   scripts/fetch_prerequisites.py         Z-Image-Turbo (31 GB) + dinov3_l (1.2 GB) -> <root>/hf
#                Qwen/Qwen3-VL-4B-Instruct@ebb281ec      (~9 GB) -> <root>/models/qwen3vl-4b
#   3 reference  jiachengcui888/zimage-ref-v1           base store + prompts (~1.6 GB) -> <root>/zimage/zimage-ref-v1
#   4 contexts   zimage_distill.contexts                 Qwen3 contexts on every visible GPU (~70 GB)
#   5 qwen text  jiachengcui888/zimage-ref-qwen-text    Qwen3-4B text features -> reference_store_qwen
#   6 images     ops/h20-patchgan-prepare-images.py     the 87 teacher tars of jiachengcui888/zimage-turbo-reference
#                                                        (173 GB, kept: the GAN reads real PNGs from them) + index
#   7 qwen-vl    ops/h20-build-dino-qwenvl-reference.py Qwen3-VL-4B image features of all 125,808 renders on
#                                                        every visible GPU -> reference_store_dino_qwenvl
#   8 train      z-image/scripts/train.sh                Group1 then Group2, fresh from pretrained, separate dirs
#   9 eval       z-image/eval_qib/run_eval.sh            Qwen-Image-Bench on step 500 of each group, plus the teacher
#                                                        Z-Image-Turbo at 9 and 2 steps (once per root); judge ~65 GB
#
# Options
#   [all|mmd-only|mmd-gan]  which group(s) to train (default all = mmd-only, then mmd-gan)
#   --root DIR         data root (required). Budget ~950 GB: weights 41, tars 162, contexts 70,
#                      per group 10 checkpoints x 24.6 + resume.pth ~26 (x2 while replaced), judge 65
#   --gpus N           training GPUs (default 4: the reference machine is 4 x A100 80GB; must divide
#                      64 / micro-batch). Stages 4, 7 and 9 use every visible card.
#   --micro-batch N    rows per GPU per forward (default 2, as on H20). If --gate runs out of memory, use 1:
#                      the global batch stays 64 (grad_accum = 64 / (gpus * micro)), the objective is the same
#   --attention NAME   transformer attention backend. Default: the config's FA3 on Hopper (H100/H20),
#                      `native` (PyTorch SDPA) on anything else (A100 has no FA3). Any diffusers backend name.
#   --gate             stages 1-7, then 3 updates per selected group into <root>/zimage/work_dirs_gate
#                      (no checkpoints; drift probe at update 2; Group2 with GAN-G from update 1);
#                      prints each card's peak memory and GATE PASS. Do this first on new hardware.
#   --resume           continue the selected group from its resume.pth (skips stages 1-7)
#   --no-eval          skip stage 9 (it is on by default)
#   --no-baselines     stage 9 judges only the two students, not the teacher at 9 / 2 steps
#   --eval-only        stage 9 alone, on runs that already finished (skips stages 1-8)
#   --ref-budget-gib G GPU memory cap per card for stage 7 (default 20; H20 used 3.5 to share cards)
#   --skip-env --skip-download --skip-prepare (stages 3-7) --allow-low-disk --dry-run
#   --no-wandb         run without Weights & Biases (otherwise a working `wandb login` is required)
set -euo pipefail
export PYTHONNOUSERSITE=1
if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
  echo "bash >= 4.4 required (this is $BASH_VERSION)" >&2; exit 1
fi

ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ZDIR/.." && pwd)"
if [[ -z "${HF_TOKEN:-}" && -s "$REPO/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$REPO/.hf_token")"; export HF_TOKEN
fi

ROOT=""; GROUP="all"; GPUS=4; MICRO=2; ATTN=""; ENV_NAME="rdm"; REF_BUDGET=20
GATE=0; RESUME=0; EVAL=1; EVAL_ONLY=0; BASELINES=1; DRY=0; ALLOW_LOW_DISK=0; DO_ENV=1; DO_DOWNLOAD=1; DO_PREPARE=1
QWEN3VL_REPO="Qwen/Qwen3-VL-4B-Instruct"; QWEN3VL_REV="ebb281ec70b05090aa6165b016eac8ec08e71b17"
REF_HF="jiachengcui888/zimage-ref-v1"; REF_NAME="zimage-ref-v1"; QWEN_TEXT_HF="jiachengcui888/zimage-ref-qwen-text"
while [[ $# -gt 0 ]]; do
  case "$1" in
    all|mmd-only|mmd-gan) GROUP="$1"; shift ;;
    --root)           ROOT="$2"; shift 2 ;;
    --gpus)           GPUS="$2"; shift 2 ;;
    --micro-batch)    MICRO="$2"; shift 2 ;;
    --attention)      ATTN="$2"; shift 2 ;;
    --env)            ENV_NAME="$2"; shift 2 ;;
    --ref-budget-gib) REF_BUDGET="$2"; shift 2 ;;
    --gate)           GATE=1; shift ;;
    --resume)         RESUME=1; shift ;;
    --eval)           EVAL=1; shift ;;   # the default; kept for old command lines
    --no-eval)        EVAL=0; shift ;;
    --no-baselines)   BASELINES=0; shift ;;
    --eval-only)      EVAL=1; EVAL_ONLY=1; shift ;;
    --skip-env)       DO_ENV=0; shift ;;
    --skip-download)  DO_DOWNLOAD=0; shift ;;
    --skip-prepare)   DO_PREPARE=0; shift ;;
    --allow-low-disk) ALLOW_LOW_DISK=1; shift ;;
    --dry-run)        DRY=1; shift ;;
    --no-wandb)       export WANDB_MODE=disabled; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[[ -n "$ROOT" ]] || { echo "--root DIR is required (see --help)" >&2; exit 2; }
[[ $RESUME -eq 1 && "$GROUP" == all ]] && { echo "--resume continues ONE group: pass mmd-only or mmd-gan" >&2; exit 2; }
[[ $RESUME -eq 1 && $GATE -eq 1 ]] && { echo "--resume and --gate are exclusive" >&2; exit 2; }
[[ $RESUME -eq 1 ]] && { DO_ENV=0; DO_DOWNLOAD=0; DO_PREPARE=0; }
[[ $EVAL_ONLY -eq 1 ]] && { DO_ENV=0; DO_DOWNLOAD=0; DO_PREPARE=0; }
[[ $EVAL_ONLY -eq 1 && $GATE -eq 1 ]] && { echo "--eval-only and --gate are exclusive" >&2; exit 2; }
[[ $GATE -eq 1 ]] && EVAL=0

ROOT="$(realpath -m "$ROOT")"
ZROOT="$ROOT/zimage"; PACK="$ZROOT/$REF_NAME"
BASE_STORE="$PACK/reference_store"; QWEN_STORE="$PACK/reference_store_qwen"
IMAGES="$PACK/reference_images_patchgan"; STORE="$PACK/reference_store_dino_qwenvl"
QWEN3VL_DIR="$ROOT/models/qwen3vl-4b"
OUTPUT_DIR="$ZROOT/work_dirs"; [[ $GATE -eq 1 ]] && OUTPUT_DIR="$ZROOT/work_dirs_gate"
LOG_DIR="$ZROOT/logs"
case "$GROUP" in all) GROUPS_RUN=(mmd-only mmd-gan) ;; *) GROUPS_RUN=("$GROUP") ;; esac
config_of() { [[ "$1" == mmd-only ]] && echo "$ZDIR/configs/zimage_2step_1024_dino_qwenvl_k512_b64_mmd_only_from0.yaml" \
                                     || echo "$ZDIR/configs/zimage_2step_1024_dino_qwenvl_k512_b64_fixed_from0.yaml"; }
exp_of() { sed -n 's/^exp_name:[[:space:]]*\([^[:space:]#]*\).*/\1/p' "$(config_of "$1")"; }

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    \033[33mWARNING:\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
smi()  { local i out; for i in 1 2 3; do out=$(nvidia-smi "$@" 2>/dev/null) && { printf '%s\n' "$out"; return 0; }; sleep 3; done; return 1; }
crun() { conda run -n "$ENV_NAME" --no-capture-output "$@"; }
penv() { [[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"; export PYTHONPATH="$ZDIR:$REPO:${PYTHONPATH:-}"; }
require_idle() {   # no foreign job is ever touched: GPU work waits for the user to free the cards
  local state; state=$(smi "${SEL[@]}" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  python3 - "$state" <<'PY' || die "visible GPUs are busy; free them or choose others with CUDA_VISIBLE_DEVICES"
import sys
busy = [r for r in sys.argv[1].splitlines() if int(r.split(',')[1]) > 1024 or int(r.split(',')[2]) > 5]
for r in busy: print(f"    GPU {r.split(',')[0].strip()} busy: {r.split(',')[1].strip()} MiB, {r.split(',')[2].strip()}%")
raise SystemExit(1 if busy else 0)
PY
}
source "$ZDIR/scripts/_wandb.sh"

# ---------------------------------------------------------------------------
say "preflight"
info "repo        $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"
info "root        $ROOT"
info "groups      ${GROUPS_RUN[*]}$([[ $GATE -eq 1 ]] && echo ' (gate: 3 updates each)')$([[ $RESUME -eq 1 ]] && echo ' (resume)')"
info "eval        $( (( EVAL )) && echo "Qwen-Image-Bench on step 500$( (( BASELINES )) && echo ' + teacher 9/2 steps')" || echo off)"
command -v conda >/dev/null 2>&1 || die "conda not found on PATH"
command -v nvidia-smi >/dev/null 2>&1 || [[ $DRY -eq 1 ]] || die "nvidia-smi not found: no GPUs"
wandb_require "$DRY"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.cache/pip}" CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$ROOT/.cache/conda-pkgs}"
[[ -z "${TMUX:-}${STY:-}" && $DRY -eq 0 ]] && warn "not inside tmux/screen -- this runs for many hours; a dropped SSH session kills it"

if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then mapfile -t GPU_IDS < <(smi --query-gpu=index --format=csv,noheader | tr -d ' ')
else GPU_IDS=(); fi
N_GPU=${#GPU_IDS[@]}
SEL=(-i "$(IFS=,; echo "${GPU_IDS[*]}")")
(( GPUS >= 1 && 64 % (GPUS * MICRO) == 0 )) || die "--gpus $GPUS x --micro-batch $MICRO must divide the global batch 64"
info "batching    $GPUS GPUs x micro $MICRO x accum $(( 64 / (GPUS * MICRO) )) = 64 rows per update"
if (( N_GPU > 0 )); then
  info "gpus        $(smi "${SEL[@]}" --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
  MIN_MIB=$(smi "${SEL[@]}" --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
  (( MIN_MIB >= 79000 )) || warn "cards under 80 GB: the 6 B student will most likely run out of memory"
  CAP=$(smi "${SEL[@]}" --query-gpu=compute_cap --format=csv,noheader | sort -n | head -1 | tr -d ' ')
else CAP="?"; fi
if [[ -z "$ATTN" ]]; then
  if [[ "$CAP" =~ ^(9|1[0-9])\. ]]; then ATTN="config"; else ATTN="native"; fi
fi
info "attention   $([[ $ATTN == config ]] && echo "the configs' FA3 (_flash_3_varlen_hub; compute capability $CAP)" || echo "$ATTN (compute capability $CAP; FA3 needs Hopper)")"
(( DRY == 0 && EVAL_ONLY == 0 && N_GPU < GPUS )) && die "training needs $GPUS GPUs (--gpus, default 4) but only $N_GPU are visible"

d="$ROOT"; while [[ ! -d "$d" ]]; do d="$(dirname "$d")"; done
FREE_GB=$(df -BG --output=avail "$d" | tail -1 | tr -dc '0-9')
NEED_GB=$(( 41 + 162 + 72 + ${#GROUPS_RUN[@]} * 300 + EVAL * 65 ))
[[ $GATE -eq 1 ]] && NEED_GB=$(( 41 + 162 + 72 + 10 ))
[[ $RESUME -eq 1 ]] && NEED_GB=300
[[ $EVAL_ONLY -eq 1 ]] && NEED_GB=$(( 65 + 5 ))
info "disk        $FREE_GB GB free on $(df --output=target "$d" | tail -1); this run needs up to ~$NEED_GB GB more (already-present parts excluded only approximately)"
if (( FREE_GB < NEED_GB )); then
  if [[ $ALLOW_LOW_DISK -eq 1 || $DRY -eq 1 ]]; then warn "less free disk than the estimate"
  else die "under ~$NEED_GB GB free under $ROOT; free space, choose a bigger --root, or pass --allow-low-disk"; fi
fi
git -C "$REPO" rev-parse HEAD > "$REPO/SOURCE_COMMIT" 2>/dev/null || [[ -s "$REPO/SOURCE_COMMIT" ]] || echo unknown > "$REPO/SOURCE_COMMIT"

# ---------------------------------------------------------------------------
if [[ $DO_ENV -eq 1 ]]; then
  say "stage 1/9: environment ($ENV_NAME)"
  bash "$REPO/scripts/setup_env.sh" --name "$ENV_NAME" $([[ $DRY -eq 1 ]] && echo --dry-run)
  if [[ $DRY -eq 0 ]] && ! crun bash -c 'command -v aria2c' >/dev/null 2>&1 && ! command -v aria2c >/dev/null 2>&1; then
    info "installing aria2 into $ENV_NAME (teacher tar downloads)"
    conda install -y -n "$ENV_NAME" -c conda-forge aria2 >/dev/null || die "could not install aria2 (apt install aria2 also works)"
  fi
fi

# ---------------------------------------------------------------------------
if [[ $DO_DOWNLOAD -eq 1 ]]; then
  say "stage 2/9: downloads"
  DL=(python "$REPO/scripts/fetch_prerequisites.py" --root "$ROOT" --group zimage --group encoders --encoders dinov3_l)
  [[ $DRY -eq 1 ]] && DL+=(--dry-run)
  crun "${DL[@]}" || [[ $DRY -eq 1 ]] || die "downloads incomplete (see above); re-run with --skip-env to resume"
  if [[ -f "$QWEN3VL_DIR/.complete" ]]; then info "Qwen3-VL-4B present: $QWEN3VL_DIR"
  elif [[ $DRY -eq 1 ]]; then info "would download $QWEN3VL_REPO@$QWEN3VL_REV (*.json, *.safetensors) -> $QWEN3VL_DIR"
  else
    crun python - "$QWEN3VL_REPO" "$QWEN3VL_REV" "$QWEN3VL_DIR" <<'PY' || die "Qwen3-VL-4B download failed (re-run resumes)"
import sys
from huggingface_hub import snapshot_download
repo, rev, dest = sys.argv[1:]
snapshot_download(repo, revision=rev, local_dir=dest, allow_patterns=["*.json", "*.safetensors"])
PY
    touch "$QWEN3VL_DIR/.complete"; info "ok: $(du -sh "$QWEN3VL_DIR" | cut -f1)"
  fi
fi
export RDM_QWEN3VL_SNAPSHOT="$QWEN3VL_DIR" RDM_QWEN3VL_SPATIAL_SNAPSHOT="$QWEN3VL_DIR"

# ---------------------------------------------------------------------------
if [[ $DO_PREPARE -eq 1 ]]; then
  say "stage 3/9: reference pack $REF_HF -> $PACK"
  if [[ -f "$PACK/.staged" ]]; then info "present"
  elif [[ $DRY -eq 1 ]]; then info "would download $REF_HF:$REF_NAME/*"
  else
    mkdir -p "$ZROOT"
    crun python - "$REF_HF" "$REF_NAME" "$ZROOT" <<'PY' || die "could not download the pack (see above)"
import sys
from huggingface_hub import snapshot_download
repo, name, dest = sys.argv[1:]
snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{name}/*"], local_dir=dest)
PY
    [[ -f "$BASE_STORE/metadata.json" ]] || die "$REF_HF holds no $REF_NAME/reference_store/metadata.json"
    touch "$PACK/.staged"; info "ok: $(du -sh "$PACK" | cut -f1)"
  fi

  say "stage 4/9: Qwen3 contexts on $N_GPU GPU(s) -> $PACK/contexts"
  if [[ -f "$PACK/contexts/manifest.json" ]]; then info "present"
  elif [[ $DRY -eq 1 ]]; then info "would run zimage_distill.contexts build on GPUs ${GPU_IDS[*]:-?}, then finalize"
  else
    require_idle; mkdir -p "$LOG_DIR"; pids=()
    for ((r=0; r<N_GPU; r++)); do
      ( penv; CUDA_VISIBLE_DEVICES="${GPU_IDS[$r]}" crun python -m zimage_distill.contexts build \
          --pack "$PACK" --rank "$r" --world "$N_GPU" ) > "$LOG_DIR/contexts_rank$r.log" 2>&1 &
      pids+=($!); info "rank $r -> GPU ${GPU_IDS[$r]} (log: $LOG_DIR/contexts_rank$r.log)"
    done
    fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || { warn "rank $i failed -- see $LOG_DIR/contexts_rank$i.log"; fail=1; }; done
    [[ $fail -eq 0 ]] || die "context rebuild incomplete; re-run with --skip-env --skip-download (finished ranks are kept)"
    ( penv; crun python -m zimage_distill.contexts finalize --pack "$PACK" --world "$N_GPU" ) || die "could not merge the context shards"
  fi

  say "stage 5/9: Qwen3-4B text features $QWEN_TEXT_HF -> $QWEN_STORE"
  if [[ -f "$QWEN_STORE/metadata.json" ]]; then info "present"
  elif [[ $DRY -eq 1 ]]; then info "would assemble from $QWEN_TEXT_HF"
  else
    ( penv; crun python -m zimage_distill.qwen_text_store --source "$BASE_STORE" --out "$QWEN_STORE" --from-hub "$QWEN_TEXT_HF" ) \
      || die "could not assemble the Qwen3 text store (see above)"
  fi

  say "stage 6/9: teacher renders (87 tars, 173 GB, kept) -> $IMAGES"
  if [[ -f "$IMAGES/manifest.json" ]]; then info "present"
  elif [[ $DRY -eq 1 ]]; then info "would run ops/h20-patchgan-prepare-images.py --root $ROOT --endpoint ${HF_ENDPOINT:-https://huggingface.co}"
  else
    mkdir -p "$LOG_DIR"; info "log: $LOG_DIR/teacher_images.log (one JSON line per tar; restartable)"
    ( penv; crun python "$REPO/ops/h20-patchgan-prepare-images.py" --root "$ROOT" \
        --endpoint "${HF_ENDPOINT:-https://huggingface.co}" ) 2>&1 | tee -a "$LOG_DIR/teacher_images.log" \
      | grep --line-buffered -E '"event": "(disk_preflight|teacher_tar_index_committed|real_images_complete|already_complete|failed)"' \
      | sed -u 's/^/    /' || true
    [[ -f "$IMAGES/manifest.json" ]] || die "teacher images incomplete -- see $LOG_DIR/teacher_images.log; re-run to resume"
  fi

  say "stage 7/9: Qwen3-VL-4B image features on $N_GPU GPU(s) -> $STORE"
  if [[ -f "$STORE/reference-verified.json" ]]; then info "present"
  elif [[ $DRY -eq 1 ]]; then info "would run ops/h20-build-dino-qwenvl-reference.py on GPUs ${GPU_IDS[*]:-?} (cap ${REF_BUDGET} GiB each)"
  else
    require_idle
    info "per-rank logs: $PACK/reference_store_dino_qwenvl.partial/rank*.log (restartable)"
    ( penv; RDM_ROOT="$ROOT" RDM_REF_GPUS="$(IFS=,; echo "${GPU_IDS[*]}")" RDM_REF_GPU_BUDGET_GIB="$REF_BUDGET" \
        crun python "$REPO/ops/h20-build-dino-qwenvl-reference.py" ) | tail -1 | sed 's/^/    /' \
      || die "Qwen3-VL features failed -- see the rank logs above; re-run to resume"
    [[ -f "$STORE/reference-verified.json" ]] || die "Qwen3-VL store not verified -- see the rank logs"
  fi
fi

# ---------------------------------------------------------------------------
QIB=(env ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" bash "$ZDIR/eval_qib/run_eval.sh" --root "$ROOT/qib")
if [[ $EVAL -eq 1 ]]; then
  # before training: a judge that cannot be installed should not surface only after ~1 day of training
  say "stage 9/9 (setup): Qwen-Image-Bench judge env + Q-Judger weights -> $ROOT/qib"
  if [[ $EVAL_ONLY -eq 1 ]]; then
    for G in "${GROUPS_RUN[@]}"; do   # before the ~65 GB judge install, not after it
      [[ $DRY -eq 1 || -f "$OUTPUT_DIR/$(exp_of "$G")/step_0000500.pth" ]] || die "no $OUTPUT_DIR/$(exp_of "$G")/step_0000500.pth: train $G first"
    done
  fi
  if [[ $DRY -eq 1 ]]; then info "would run: ${QIB[*]} --setup"
  else "${QIB[@]}" --setup || die "the judge setup failed (see above); re-run to retry, or pass --no-eval to train without it"; fi
fi

# ---------------------------------------------------------------------------
if [[ $EVAL_ONLY -eq 0 ]]; then
say "stage 8/9: training ${GROUPS_RUN[*]}"
[[ $DRY -eq 1 || -f "$STORE/reference-verified.json" ]] || die "$STORE is not built; run without --skip-prepare"
for G in "${GROUPS_RUN[@]}"; do
  CONFIG="$(config_of "$G")"; EXP="$(exp_of "$G")"; RUN_DIR="$OUTPUT_DIR/$EXP"
  ARGS=("$CONFIG" --set "spatial_gan.qwen_snapshot=$QWEN3VL_DIR" --set "spatial_gan.reference_images=$IMAGES"
        --set "memory.micro_batch=$MICRO")
  [[ "$ATTN" != config ]] && ARGS+=(--set "zimage.attention_backend=$ATTN")
  # the gate exercises every memory peak of the real run in 3 updates: the drift probe, and for
  # Group2 the generator's GAN term from update 1 (the real run starts it at 51)
  [[ $GATE -eq 1 ]] && ARGS+=(--set "steps=3" --set "probe_every=2")
  [[ $GATE -eq 1 && $G == mmd-gan ]] && ARGS+=(--set "gan.g_start_step=1")
  TENV=(ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" GPUS="$GPUS" REFERENCE_ROOT="$STORE" OUTPUT_DIR="$OUTPUT_DIR"
        MASTER_PORT="${MASTER_PORT:-29501}")
  say "  group $G: $(basename "$CONFIG") -> $RUN_DIR"
  if [[ $RESUME -eq 1 ]]; then
    [[ -f "$RUN_DIR/resume.pth" ]] || die "--resume: $RUN_DIR/resume.pth does not exist"
    TENV+=(RESUME_FROM="$RUN_DIR/resume.pth")
  elif [[ -s "$RUN_DIR/train_log.jsonl" ]]; then
    if [[ $GATE -eq 1 ]]; then mv "$RUN_DIR" "$RUN_DIR.old.$(date +%s)"; info "moved the previous gate run aside"
    else die "$RUN_DIR already holds a run: continue it with '$G --resume', or move it away (mv $RUN_DIR{,_old})"; fi
  fi
  if [[ $DRY -eq 1 ]]; then info "would run: env ${TENV[*]} bash $ZDIR/scripts/train.sh ${ARGS[*]}"; continue; fi
  require_idle
  pgrep -u "$(id -u)" -f 'zimage_distill[.]launch' >/dev/null 2>&1 && die "a Z-Image training process is already running"
  PEAK="$LOG_DIR/peak_$G.txt"; mkdir -p "$LOG_DIR"; : > "$PEAK"
  ( while sleep 5; do smi "${SEL[@]}" --query-gpu=index,memory.used --format=csv,noheader,nounits >> "$PEAK" || true; done ) &
  MON=$!
  (( GATE == 0 )) && wandb_start "$RUN_DIR" "k512-$G" "$GPUS" 500 "$CONFIG"
  set +e; env "${TENV[@]}" bash "$ZDIR/scripts/train.sh" "${ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/train_$G$([[ $GATE -eq 1 ]] && echo _gate).log"
  RC=${PIPESTATUS[0]}; set -e
  kill "$MON" 2>/dev/null || true
  info "peak memory per GPU (MiB): $(sort -t, -k1,1n -k2,2n "$PEAK" | awk -F, '{m[$1]=$2>m[$1]?$2:m[$1]} END{for(g in m) printf "GPU%s=%s ", g, m[g]}')"
  if (( RC != 0 )); then
    grep -qiE "out of memory|OutOfMemoryError" "$LOG_DIR/train_$G"*.log 2>/dev/null && (( MICRO > 1 )) && \
      warn "CUDA out of memory: re-run with --micro-batch 1 (global batch stays 64)"
    die "group $G failed (exit $RC); full log: $LOG_DIR/train_$G$([[ $GATE -eq 1 ]] && echo _gate).log"
  fi
  if [[ $GATE -eq 1 ]]; then
    N=$(wc -l < "$RUN_DIR/train_log.jsonl" 2>/dev/null || echo 0)
    (( N >= 3 )) || die "gate $G: expected 3 logged updates in $RUN_DIR/train_log.jsonl, found $N"
    info "GATE PASS $G: 3 updates, last row: $(tail -1 "$RUN_DIR/train_log.jsonl" | cut -c1-300)"
  else
    [[ -f "$RUN_DIR/step_0000500.pth" ]] || die "group $G ended without step_0000500.pth"
    info "DONE $G: checkpoints $(ls "$RUN_DIR" | grep -c '^step_') in $RUN_DIR"
  fi
done
fi

# ---------------------------------------------------------------------------
if [[ $EVAL -eq 1 ]]; then
  say "stage 9/9: Qwen-Image-Bench: step 500 of ${GROUPS_RUN[*]}$( (( BASELINES )) && echo ', teacher at 9 and 2 steps')"
  ALL="$(IFS=,; echo "${GPU_IDS[*]}")"
  EVQ=("${QIB[@]}" --gen-gpus "$ALL" --judge-gpus "$ALL")
  if [[ $DRY -eq 1 ]]; then
    for G in "${GROUPS_RUN[@]}"; do info "would run: ${EVQ[*]} --name $(exp_of "$G")_s500 --ckpt $OUTPUT_DIR/$(exp_of "$G")/step_0000500.pth"; done
    (( BASELINES )) && info "would run: ${EVQ[*]} --name turbo9 --teacher-steps 9  (and turbo2 --teacher-steps 2)"
  else
    NAMES=()
    for G in "${GROUPS_RUN[@]}"; do
      EXP="$(exp_of "$G")"; n="${EXP}_s500"
      "${EVQ[@]}" --name "$n" --ckpt "$OUTPUT_DIR/$EXP/step_0000500.pth" \
        || die "Qwen-Image-Bench failed on $G (re-run with --eval-only: finished parts are kept)"
      NAMES+=("$n")
    done
    if (( BASELINES )); then          # finished parts are skipped, so the teacher is judged once per root
      for t in 9 2; do
        "${EVQ[@]}" --name "turbo$t" --teacher-steps "$t" || die "Qwen-Image-Bench failed on the teacher at $t steps"
        NAMES+=("turbo$t")
      done
    fi
    "${QIB[@]}" --report "${NAMES[@]}" >/dev/null || die "could not build the report"
    sed -n '/^| model/,/^$/p' "$ROOT/qib/runs/report.md" | sed 's/^/    /'
    info "table: $ROOT/qib/runs/report.md"
  fi
fi
[[ $DRY -eq 1 ]] && say "dry run: nothing was changed"
say "all requested stages finished"
exit 0
