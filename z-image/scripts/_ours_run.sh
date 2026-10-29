#!/usr/bin/env bash
# Z-Image-Turbo -> 2-step student on a fresh machine, in one command: conda env, downloads (Z-Image-
# Turbo + the 3 encoders), the Qwen-Image-Bench judge env, the reference pack from a Hugging Face
# dataset, the Qwen3 contexts rebuilt on every GPU, training (z-image/configs/zimage_2step_1024_h100.yaml),
# then Qwen-Image-Bench on the last checkpoint and on the teacher (z-image/eval_qib).
#
# This is the shared body of the launchers ours_two-cards.sh, ours_four-cards.sh and
# ours_eight-cards.sh, which fix --gpus and the reference pack; run one of those:
#
#   git clone -b zimage https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   tmux new -s train
#   bash z-image/scripts/ours_two-cards.sh --root /data/<you>/rdm-sets --dry-run   # plan only
#   bash z-image/scripts/ours_two-cards.sh --root /data/<you>/rdm-sets --gate      # 20 steps: memory/speed
#   bash z-image/scripts/ours_two-cards.sh --root /data/<you>/rdm-sets             # the run
#   bash z-image/scripts/ours_two-cards.sh --root /data/<you>/rdm-sets --resume    # continue it
#
# Stages (all idempotent: a finished stage is skipped on a re-run):
#   1 env        scripts/setup_env.sh                 conda env rdm (torch 2.8, diffusers 0.40, bnb)
#   2 download   scripts/fetch_prerequisites.py       Z-Image-Turbo (31 GB) + dinov3_l, siglip2,
#                                                     aimv2_huge (5.6 GB) -> <root>/hf; writes <root>/env.sh
#   3 eval env   eval_qib/run_eval.sh --setup         the judge's own env (ms-swift) + Qwen-Image-Bench
#                                                     + Q-Judger weights (54.7 GB) -> <root>/qib
#                                                     (no GenEval, no PickScore)
#   4 reference  the pack <ref-name>/ from the HF dataset (pack_reference.sh made it), ~1.6 GB
#                -> <root>/zimage/<ref-name>/, sha256-checked by the Hub client, read back
#   5 contexts   zimage_distill.contexts on every visible GPU: each prompt's Qwen3 context,
#                length-bucketed (~70 GB, minutes on 2 x H100) -> <root>/zimage/<ref-name>/contexts/
#   6 train      z-image/scripts/train.sh             FSDP over --gpus cards -> <root>/zimage/work_dirs
#   7 eval       eval_qib/run_eval.sh                 Qwen-Image-Bench (1000 prompts, 1 image each) on
#                                                     the last checkpoint, and once on the teacher at
#                                                     9 and 2 steps; one judge copy per 80 GB card in
#                                                     parallel (~40 min per model on 2 x H100)
#                                                     -> <root>/qib/runs/, table <run dir>/qwen_image_bench.md
#
# --root DIR       data root (default /root/rdm-sets); ~330 GB: weights 37, judge env + weights 65,
#                  contexts ~70, 4 checkpoints x 24.6 GB, resume.pth ~26 GB (x2 while it is replaced),
#                  ~1.5 GB of images per evaluated model
# --ref-hf REPO    the HF dataset holding the pack (needed until stage 3 has run once)
# --ref-name NAME  the pack's folder in it (default zimage-ref-v1)
# --gpus N         training GPUs, 2 / 4 / 8 (set by the launcher; default: every visible card,
#                  rounded down to 1/2/4/8). 80 GB cards. Stages 5 and 7 use every visible card.
#                  Choose cards with CUDA_VISIBLE_DEVICES.
# --micro-batch N  rows per rank per forward (default 1, the config's); 2 only if --gate shows room
# --config YAML    (default z-image/configs/zimage_2step_1024_h100.yaml)
# --gate           20 steps into <root>/zimage/work_dirs_gate: memory and s/step before the real run
# --steps N        override the config's steps
# --resume         continue the run from its resume.pth (skips stages 1, 2, 4, 5)
# --eval-ckpts W   which checkpoints stage 7 judges: last (default) | all (every step_*.pth)
# --no-baselines   do not judge the teacher (turbo9, turbo2; each is judged once per --root anyway)
# --eval-only      stage 7 alone, on the run already in --output-dir (implies --skip-*)
# --no-eval        skip stages 3 and 7
# --eval-limit N   judge the first N prompts only (a smoke test; runs are named *_limitN and kept
#                  out of the comparison tables)
# --output-dir DIR --env NAME (rdm) --hf-cache DIR (adopt an existing HF hub cache)
# --skip-env --skip-download --skip-reference --skip-contexts --no-train --allow-low-disk --dry-run
# A --gate run saves no checkpoint: stage 7 is skipped for it.
# W&B (on when a key is there): z-image/scripts/_wandb.sh; WANDB_MODE=disabled turns it off
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
ROOT="/root/rdm-sets"; REF_HF=""; REF_NAME="zimage-ref-v1"; GPUS=""
CONFIG="$ZDIR/configs/zimage_2step_1024_h100.yaml"; ENV_NAME="rdm"; HF_CACHE=""; OUTPUT_DIR=""
STEPS=""; GATE=0; RESUME=0; DRY=0; ALLOW_LOW_DISK=0; MICRO=""
DO_ENV=1; DO_DOWNLOAD=1; DO_REFERENCE=1; DO_CONTEXTS=1; DO_TRAIN=1; DO_EVAL=1
EVAL_CKPTS="last"; BASELINES=1; EVAL_ONLY=0; EVAL_LIMIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)           ROOT="$2"; shift 2 ;;
    --ref-hf)         REF_HF="$2"; shift 2 ;;
    --ref-name)       REF_NAME="$2"; shift 2 ;;
    --gpus)           GPUS="$2"; shift 2 ;;
    --config)         CONFIG="$(realpath "$2")"; shift 2 ;;
    --env)            ENV_NAME="$2"; shift 2 ;;
    --hf-cache)       HF_CACHE="$2"; shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
    --steps)          STEPS="$2"; shift 2 ;;
    --micro-batch)    MICRO="$2"; shift 2 ;;
    --gate)           GATE=1; shift ;;
    --resume)         RESUME=1; shift ;;
    --skip-env)       DO_ENV=0; shift ;;
    --skip-download)  DO_DOWNLOAD=0; shift ;;
    --skip-reference) DO_REFERENCE=0; shift ;;
    --skip-contexts)  DO_CONTEXTS=0; shift ;;
    --no-train)       DO_TRAIN=0; shift ;;
    --no-eval)        DO_EVAL=0; shift ;;
    --eval-ckpts)     EVAL_CKPTS="$2"; shift 2 ;;
    --no-baselines)   BASELINES=0; shift ;;
    --eval-only)      EVAL_ONLY=1; shift ;;
    --eval-limit)     EVAL_LIMIT="$2"; shift 2 ;;
    --allow-low-disk) ALLOW_LOW_DISK=1; shift ;;
    --dry-run)        DRY=1; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[[ $RESUME -eq 1 ]] && { DO_ENV=0; DO_DOWNLOAD=0; DO_REFERENCE=0; DO_CONTEXTS=0; }
[[ $EVAL_ONLY -eq 1 ]] && { DO_ENV=0; DO_DOWNLOAD=0; DO_REFERENCE=0; DO_CONTEXTS=0; DO_TRAIN=0; DO_EVAL=1; }
[[ "$EVAL_CKPTS" == last || "$EVAL_CKPTS" == all ]] || { echo "--eval-ckpts last|all" >&2; exit 2; }
ROOT="$(realpath -m "$ROOT")"
ZROOT="$ROOT/zimage"
PACK="$ZROOT/$REF_NAME"
STORE="$PACK/reference_store"
if [[ $GATE -eq 1 ]]; then STEPS="${STEPS:-20}"; OUTPUT_DIR="${OUTPUT_DIR:-$ZROOT/work_dirs_gate}"; fi
OUTPUT_DIR="${OUTPUT_DIR:-$ZROOT/work_dirs}"
LOG_DIR="$ZROOT/logs"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    \033[33mWARNING:\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
smi()  { local i out; for i in 1 2 3; do out=$(nvidia-smi "$@" 2>/dev/null) && { printf '%s\n' "$out"; return 0; }; sleep 3; done; return 1; }
crun() { conda run -n "$ENV_NAME" --no-capture-output "$@"; }
source "$ZDIR/scripts/_wandb.sh"

# ---------------------------------------------------------------------------
say "preflight"
(( DRY == 0 && (DO_TRAIN || DO_EVAL) )) && wandb_ask
info "repo        $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"
info "root        $ROOT"
[[ $GATE -eq 1 ]] && DO_EVAL=0                     # a gate run saves no checkpoint to judge
info "stages      env=$DO_ENV download=$DO_DOWNLOAD eval-env=$DO_EVAL reference=$DO_REFERENCE contexts=$DO_CONTEXTS train=$DO_TRAIN eval=$DO_EVAL"
[[ -f "$CONFIG" ]] || die "config not found: $CONFIG"
command -v conda >/dev/null 2>&1 || die "conda not found on PATH"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$ROOT/.cache/conda-pkgs}"
[[ -z "${TMUX:-}${STY:-}" && $DRY -eq 0 && $DO_TRAIN -eq 1 ]] && \
  warn "not inside tmux/screen -- training runs for hours; a dropped SSH session kills it"

# GPUs: the visible ones (CUDA_VISIBLE_DEVICES or all), their ids for the per-GPU context workers
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_IDS < <(smi --query-gpu=index --format=csv,noheader | tr -d ' ')
else
  GPU_IDS=()
fi
N_GPU=${#GPU_IDS[@]}
if [[ -z "$GPUS" ]]; then
  GPUS=1
  while (( GPUS * 2 <= N_GPU && GPUS * 2 <= 8 )); do GPUS=$(( GPUS * 2 )); done
fi
(( 32 % GPUS == 0 && 128 % GPUS == 0 )) || die "--gpus $GPUS does not divide B = 32 and K = 128 (use 1/2/4/8)"
[[ -z "$MICRO" ]] || (( (32 / GPUS) % MICRO == 0 )) || die "--micro-batch $MICRO does not divide the $(( 32 / GPUS )) rows per GPU"
if command -v nvidia-smi >/dev/null 2>&1 && (( N_GPU > 0 )); then
  SEL=(-i "$(IFS=,; echo "${GPU_IDS[*]}")")
  info "gpus        $N_GPU visible: $(smi "${SEL[@]}" --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
  MIN_MIB=$(smi "${SEL[@]}" --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1) || MIN_MIB=0
  if [[ $DO_TRAIN -eq 1 ]] && (( ${MIN_MIB:-0} < 79000 || GPUS < 2 )); then
    warn "the 6 B student with fp32 master weights needs 80 GB cards, 2 at least (FSDP); training"
    warn "  on $GPUS x $(( ${MIN_MIB:-0} / 1024 )) GB will most likely run out of memory"
  fi
fi
info "training    $(basename "$CONFIG") on $GPUS GPU(s)${STEPS:+, $STEPS steps} -> $OUTPUT_DIR"
if [[ $DRY -eq 0 ]]; then
  [[ $DO_CONTEXTS -eq 1 && $N_GPU -lt 1 ]] && die "stage 4 needs a GPU; none visible"
  [[ $DO_TRAIN -eq 1 && $N_GPU -lt $GPUS ]] && die "training asks for $GPUS GPUs but $N_GPU are visible"
fi

d="$ROOT"; while [[ ! -d "$d" ]]; do d="$(dirname "$d")"; done
FREE_GB=$(df -BG --output=avail "$d" | tail -1 | tr -dc '0-9')
info "free space  ${FREE_GB} GB on $(df --output=target "$d" | tail -1)"
(( FREE_GB < 330 )) && warn "under ~330 GB free: weights 37 + judge 65 + contexts ~70 + 4 checkpoints x 24.6 + resume.pth"

# the pack: present, or readable from the HF dataset with this machine's token
if [[ $DO_REFERENCE -eq 1 && ! -f "$PACK/.staged" ]]; then
  [[ -n "$REF_HF" ]] || die "no reference pack at $PACK: pass --ref-hf <owner>/<dataset> (made by
  z-image/scripts/pack_reference.sh), or copy the pack there yourself"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    CODE=$(curl -s -o /dev/null -w '%{http_code}' -H @<(printf 'Authorization: Bearer %s\n' "$HF_TOKEN") \
             "https://huggingface.co/api/datasets/$REF_HF" 2>/dev/null || true)
    [[ "$CODE" == 200 ]] && info "reference   HF dataset $REF_HF readable -> $PACK" || {
      MSG="HF dataset $REF_HF is not readable with this token (HTTP ${CODE:-?}): a private dataset needs a
  token of the account that owns it (echo hf_... > $REPO/.hf_token)"
      [[ $DRY -eq 1 ]] && warn "$MSG" || die "$MSG"; }
  else
    info "reference   HF dataset $REF_HF (no token: works only if it is public) -> $PACK"
  fi
fi

# ---------------------------------------------------------------------------
if [[ $DO_ENV -eq 1 ]]; then
  say "stage 1/7: environment ($ENV_NAME)"
  bash "$REPO/scripts/setup_env.sh" --name "$ENV_NAME" $([[ $DRY -eq 1 ]] && echo --dry-run)
fi

# ---------------------------------------------------------------------------
if [[ $DO_DOWNLOAD -eq 1 ]]; then
  say "stage 2/7: downloads -> $ROOT/hf"
  ENCODERS=$(python3 -c "import re,sys; t=open(sys.argv[1]).read(); print(re.search(r'^encoders:\s*\[(.*)\]', t, re.M).group(1).replace(' ', ''))" "$CONFIG")
  DL=(python "$REPO/scripts/fetch_prerequisites.py" --root "$ROOT" --group zimage --group encoders
      --encoders "$ENCODERS")
  [[ -n "$HF_CACHE" ]] && DL+=(--hf-cache "$HF_CACHE")
  [[ $DRY -eq 1 ]] && DL+=(--dry-run)
  crun "${DL[@]}" || [[ $DRY -eq 1 ]] || die "downloads incomplete (see above); re-run with --skip-env to resume"
fi

# ---------------------------------------------------------------------------
QIB="$ROOT/qib"
EVAL=(env ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" bash "$ZDIR/eval_qib/run_eval.sh" --root "$QIB")
if [[ $DO_EVAL -eq 1 ]]; then
  # before training: a judge that cannot be installed should not surface only after it. Quick when
  # it is already there (the env has a .ready marker, the downloads are checked, not repeated).
  say "stage 3/7: Qwen-Image-Bench judge env + weights -> $QIB"
  if [[ $DRY -eq 1 ]]; then
    info "would run: ${EVAL[*]} --setup"
  else
    "${EVAL[@]}" --setup || die "the judge setup failed (see above); re-run with --skip-env --skip-download
  to retry it, or --no-eval to train without it"
  fi
fi

# ---------------------------------------------------------------------------
if [[ $DO_REFERENCE -eq 1 ]]; then
  say "stage 4/7: reference pack $REF_NAME <- $REF_HF"
  if [[ -f "$PACK/.staged" ]]; then
    info "present: $PACK"
  elif [[ $DRY -eq 1 ]]; then
    info "would download $REF_HF:$REF_NAME/* -> $ZROOT"
  else
    mkdir -p "$ZROOT"
    crun python - "$REF_HF" "$REF_NAME" "$ZROOT" <<'PY' || die "could not download the pack (see above)"
import sys
from huggingface_hub import snapshot_download
repo, name, dest = sys.argv[1:]
snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{name}/*"], local_dir=dest)
PY
    [[ -f "$STORE/metadata.json" ]] || die "$REF_HF holds no $REF_NAME/reference_store/metadata.json"
    touch "$PACK/.staged"
    info "ok: $(du -sh "$PACK" | cut -f1)"
  fi
fi

# ---------------------------------------------------------------------------
if [[ $DO_CONTEXTS -eq 1 ]]; then
  say "stage 5/7: Qwen3 contexts on $N_GPU GPU(s) -> $PACK/contexts"
  if [[ -f "$PACK/contexts/manifest.json" ]]; then
    info "present"
  elif [[ $DRY -eq 1 ]]; then
    info "would run zimage_distill.contexts build on GPUs ${GPU_IDS[*]:-?}, then finalize"
  else
    [[ -f "$ROOT/env.sh" ]] || die "$ROOT/env.sh missing -- stage 2 has not completed"
    mkdir -p "$LOG_DIR"
    pids=()
    for ((r=0; r<N_GPU; r++)); do
      ( source "$ROOT/env.sh"; export PYTHONPATH="$ZDIR:$REPO:${PYTHONPATH:-}"
        CUDA_VISIBLE_DEVICES="${GPU_IDS[$r]}" crun python -m zimage_distill.contexts build \
          --pack "$PACK" --rank "$r" --world "$N_GPU" ) > "$LOG_DIR/contexts_rank$r.log" 2>&1 &
      pids+=($!)
      info "rank $r -> GPU ${GPU_IDS[$r]} (log: $LOG_DIR/contexts_rank$r.log)"
    done
    fail=0
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || { warn "rank $i failed -- see $LOG_DIR/contexts_rank$i.log"; fail=1; }
    done
    [[ $fail -eq 0 ]] || die "context rebuild incomplete; re-run with --skip-env --skip-download (finished ranks are kept)"
    ( source "$ROOT/env.sh"; export PYTHONPATH="$ZDIR:$REPO:${PYTHONPATH:-}"
      crun python -m zimage_distill.contexts finalize --pack "$PACK" --world "$N_GPU" ) \
      || die "could not merge the context shards (see above)"
    grep -h "done:" "$LOG_DIR"/contexts_rank*.log | sed 's/^/    /' || true
  fi
fi

# ---------------------------------------------------------------------------
TRAIN_ENV=(ASSETS="$ROOT" CONDA_ENV="$ENV_NAME" GPUS="$GPUS" REFERENCE_ROOT="$STORE" OUTPUT_DIR="$OUTPUT_DIR")
TRAIN_ARGS=("$CONFIG")
[[ -n "$STEPS" ]] && TRAIN_ARGS+=(--set "steps=$STEPS")
[[ -n "$MICRO" ]] && TRAIN_ARGS+=(--set "memory.micro_batch=$MICRO")
read -r EXP CFG_STEPS SAVE_FREQ <<< "$(python3 - "$CONFIG" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
get = lambda k: re.search(rf"^{k}:\s*(\S+)", t, re.M).group(1)
print(get("exp_name"), get("steps"), get("save_freq"))
PY
)"
RUN_DIR="$OUTPUT_DIR/$EXP"
(( DRY == 0 && (DO_TRAIN || DO_EVAL) )) && wandb_start "$RUN_DIR" ours "$GPUS" "${STEPS:-$CFG_STEPS}" "$CONFIG"
if [[ $DO_TRAIN -eq 1 ]]; then
  say "stage 6/7: training"
  if [[ $DRY -eq 1 ]]; then
    info "would run: env ${TRAIN_ENV[*]}$([[ $RESUME -eq 1 ]] && echo " RESUME_FROM=$RUN_DIR/resume.pth") bash $ZDIR/scripts/train.sh ${TRAIN_ARGS[*]}"
  else
    [[ -f "$PACK/contexts/manifest.json" ]] || die "$PACK/contexts is missing -- stage 4 has not completed"
    pgrep -u "$(id -u)" -f '^[^ ]*python[0-9.]* .*-m zimage_distill[.]launch' >/dev/null 2>&1 && \
      die "a Z-Image training process is already running (pgrep -af zimage_distill.launch)"
    N_LEFT=$(( (${STEPS:-$CFG_STEPS} + SAVE_FREQ - 1) / SAVE_FREQ ))
    N_HAVE=0; [[ -d "$RUN_DIR" ]] && N_HAVE=$(find "$RUN_DIR" -maxdepth 1 -name 'step_*.pth' | wc -l)
    N_LEFT=$(( N_LEFT > N_HAVE ? N_LEFT - N_HAVE : 0 ))
    NEED_GB=$(( N_LEFT * 25 + 55 ))                # 24.6 GB each + resume.pth twice while replaced
    d="$OUTPUT_DIR"; while [[ ! -d "$d" ]]; do d="$(dirname "$d")"; done
    OUT_FREE_GB=$(df -BG --output=avail "$d" | tail -1 | tr -dc '0-9')
    info "disk        $OUT_FREE_GB GB free for $N_LEFT more checkpoints + resume.pth -> need ~$NEED_GB GB"
    if (( OUT_FREE_GB < NEED_GB )); then
      [[ $ALLOW_LOW_DISK -eq 1 ]] && warn "not enough disk for every checkpoint (--allow-low-disk)" || \
        die "$OUT_FREE_GB GB free under $OUTPUT_DIR, the run writes ~$NEED_GB GB; free space, use
  --output-dir on a bigger volume, or --allow-low-disk (and delete old step_*.pth as it trains)"
    fi
    if [[ $RESUME -eq 1 ]]; then
      [[ -f "$RUN_DIR/resume.pth" ]] || die "--resume: $RUN_DIR/resume.pth does not exist"
      TRAIN_ENV+=(RESUME_FROM="$RUN_DIR/resume.pth")
    elif [[ -s "$RUN_DIR/train_log.jsonl" ]]; then
      die "$RUN_DIR already holds a run. Continue it with --resume, or move it away (mv $RUN_DIR{,_old})"
    fi
    env "${TRAIN_ENV[@]}" bash "$ZDIR/scripts/train.sh" "${TRAIN_ARGS[@]}"
    say "done"
    info "logs + checkpoints: $RUN_DIR"
  fi
fi

# ---------------------------------------------------------------------------
if [[ $DO_EVAL -eq 1 ]]; then
  say "stage 7/7: Qwen-Image-Bench on GPUs ${GPU_IDS[*]:-?}"
  ALL_GPUS="$(IFS=,; echo "${GPU_IDS[*]}")"
  EV=("${EVAL[@]}" --gen-gpus "$ALL_GPUS" --judge-gpus "$ALL_GPUS")
  SUF=""; (( EVAL_LIMIT )) && { EV+=(--limit "$EVAL_LIMIT"); SUF="_limit$EVAL_LIMIT"; }
  if [[ $DRY -eq 1 ]]; then
    info "would judge: $EVAL_CKPTS checkpoint(s) of $RUN_DIR$( (( BASELINES )) && echo ', turbo9, turbo2')"
    info "  e.g. ${EV[*]} --name ${EXP}_sNNNNNNN --ckpt $RUN_DIR/step_NNNNNNN.pth"
  else
    mapfile -t CKPTS < <(find "$RUN_DIR" -maxdepth 1 -name 'step_*.pth' 2>/dev/null | sort)
    (( ${#CKPTS[@]} )) || die "no step_*.pth in $RUN_DIR to evaluate"
    [[ "$EVAL_CKPTS" == last ]] && CKPTS=("${CKPTS[-1]}")
    NAMES=()
    if (( BASELINES )); then                       # the teacher: judged once per --root
      for t in 9 2; do
        wandb_eval "turbo$t$SUF" "$EVAL_LIMIT"
        "${EV[@]}" --name "turbo$t$SUF" --teacher-steps "$t" || die "Qwen-Image-Bench failed on the teacher ($t steps)"
        NAMES+=("turbo$t$SUF")
      done
    fi
    for c in "${CKPTS[@]}"; do
      step=$(basename "$c" .pth); step=${step#step_}
      n="${EXP}_s$((10#$step))$SUF"
      wandb_eval "$n" "$EVAL_LIMIT"
      "${EV[@]}" --name "$n" --ckpt "$c" || die "Qwen-Image-Bench failed on $c (re-run with --eval-only: finished parts are kept)"
      NAMES+=("$n")
    done
    "${EVAL[@]}" --report "${NAMES[@]}" >/dev/null
    cp "$QIB/runs/report.md" "$RUN_DIR/qwen_image_bench.md"
    sed -n '/^| model/,/^$/p' "$RUN_DIR/qwen_image_bench.md" | sed 's/^/    /'
    info "table: $RUN_DIR/qwen_image_bench.md (runs: $QIB/runs/)"
  fi
fi
[[ $DRY -eq 1 ]] && say "dry run: nothing was changed"
exit 0
