#!/usr/bin/env bash
# Evaluate one FLUX checkpoint end to end: GenEval (the primary metric) + PickScore-pa.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_checkpoint.sh <ckpt.pth> --geneval-root <dir>
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_checkpoint.sh base --geneval-root <dir>        # klein-4B teacher, 4 steps
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_checkpoint.sh base --steps 1 --geneval-root <dir>   # untrained 1-step base
#   bash scripts/eval_checkpoint.sh <ckpt.pth> --root /data/<you>/rdm-sets --geneval-root <dir>  # another machine
#
#   1 render   reproduce.py eval-flux in the rdm env: the 553 GenEval prompts x 4 canonical seeds
#              at ctx_len 48 (the training geometry) -> <out>/geneval/, and the 499 Pick-a-Pic
#              prompts at ctx_len 232 (the paper's PickScore-pa protocol) -> PickScore
#   2 geneval  scripts/score_geneval.sh in the geneval env (scripts/setup_geneval.sh)
#   3 seed diversity  scripts/seed_diversity.py on the same 4-per-prompt GenEval images: how
#              different one prompt's images are (pixel; 1-cos of DreamSim and DINOv3), mode collapse
#   -> <out>/summary.json   {geneval: {overall, 6 tasks}, pickscore, seed_diversity, checkpoint, steps}
#
# --root       the data root new_machine.sh / download_all.sh used (holds env.sh); default $ASSETS
#              or /data/thor/jiacheng/rdm-sets
# --geneval-root  the --root given to setup_geneval.sh; default $GENEVAL_ROOT, else <root>/geneval
# --out        default <ckpt dir>/eval_<ckpt name> (a checkpoint in the HF cache, and `base`, go to
#              <root>/sw_lmmd/work_dirs/eval/ instead)
# --steps N    sampling steps: 1 for a checkpoint, 4 for `base` (= the klein-4B teacher)
# --env NAME   the rdm conda env (default $CONDA_ENV or rdm)
# --no-geneval render + PickScore only        --force  redo stages already done
# --dry-run    print the plan and the eval config, run nothing
#
# A re-run skips the stages whose outputs exist, and refuses an --out that holds a different
# evaluation. One idle 24 GB card is enough (peak 21.2 GB). On a 4090 a 1-step checkpoint takes
# 22 min (render 10 + GenEval 11), the 4-step teacher ~30 -- run it inside tmux.
set -euo pipefail
export PYTHONNOUSERSITE=1   # packages in ~/.local/lib/pythonX.Y must never shadow the conda envs'

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ASSETS:-/data/thor/jiacheng/rdm-sets}"
CONDA_ENV="${CONDA_ENV:-rdm}"
GROOT="${GENEVAL_ROOT:-}"
CKPT=""
OUT=""
STEPS=""
GENEVAL=1
FORCE=0
DRY=0
PICKSCORE_CTX_LEN=232
NEED_MIB=22000              # peak 21191 MiB in nvidia-smi on a 4090 (torch 19.3 GiB) + margin

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)         ROOT="$2"; shift 2 ;;
    --geneval-root) GROOT="$2"; shift 2 ;;
    --out)          OUT="$2"; shift 2 ;;
    --steps)        STEPS="$2"; shift 2 ;;
    --env)          CONDA_ENV="$2"; shift 2 ;;
    --no-geneval)   GENEVAL=0; shift ;;
    --force)        FORCE=1; shift ;;
    --dry-run)      DRY=1; shift ;;
    -h|--help)      sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)             echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    *)              [[ -z "$CKPT" ]] || { echo "two checkpoints given: $CKPT $1" >&2; exit 2; }
                    CKPT="$1"; shift ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    \033[33mWARNING:\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
CRUN=(conda run -n "$CONDA_ENV" --no-capture-output)

# ---------------------------------------------------------------------------
# What to evaluate and where
# ---------------------------------------------------------------------------
[[ -n "$CKPT" ]] || die "usage: bash scripts/eval_checkpoint.sh <ckpt.pth | base> --geneval-root <dir> [options]  (see --help)"
ROOT="$(realpath -m "$ROOT")"
[[ -f "$ROOT/env.sh" ]] || die "$ROOT/env.sh not found -- pass --root <the data root download_all.sh / new_machine.sh used>"
if [[ "$CKPT" == "base" ]]; then
  STEPS="${STEPS:-4}"
  LOAD_FROM=""
  LABEL="klein-4B base"
  OUT="${OUT:-$ROOT/sw_lmmd/work_dirs/eval/klein4b_base_${STEPS}step}"
else
  STEPS="${STEPS:-1}"
  [[ -f "$CKPT" ]] || die "checkpoint not found: $CKPT"
  CKPT="$(realpath -s "$CKPT")"           # absolute, but keep symlinks (HF snapshot names)
  LOAD_FROM="$CKPT"
  LABEL="$CKPT"
  NAME="$(basename "$CKPT" .pth)"
  [[ "$STEPS" == 1 ]] || NAME+="_${STEPS}step"
  if [[ "$CKPT" == */snapshots/* ]]; then  # do not write renders into the HF cache
    OUT="${OUT:-$ROOT/sw_lmmd/work_dirs/eval/$NAME}"
  else
    OUT="${OUT:-$(dirname "$CKPT")/eval_$NAME}"
  fi
fi
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || die "--steps must be a positive integer, got '$STEPS'"
OUT="$(realpath -m "$OUT")"
[[ "$OUT$LOAD_FROM" != *"'"* ]] || die "paths with a single quote are not supported: $OUT $LOAD_FROM"
if [[ $GENEVAL -eq 1 ]]; then
  if [[ -z "$GROOT" && -f "$ROOT/geneval/geneval_env.sh" ]]; then GROOT="$ROOT/geneval"; fi
  [[ -n "$GROOT" ]] || die "no GenEval install: pass --geneval-root <the --root given to setup_geneval.sh> (or set GENEVAL_ROOT), or --no-geneval"
  GROOT="$(realpath -m "$GROOT")"
  [[ -f "$GROOT/geneval_env.sh" ]] || die "$GROOT/geneval_env.sh not found -- run: bash scripts/setup_geneval.sh --root $GROOT"
fi
N_GENEVAL=$(( $(grep -c . "$REPO/assets/geneval_prompts.jsonl") * 4 ))

say "evaluate $LABEL (${STEPS}-step)"
info "out           $OUT"
info "data root     $ROOT  (conda env $CONDA_ENV)"
info "geneval       ${GROOT:-skipped (--no-geneval)}"

# ---------------------------------------------------------------------------
# Preflight: everything that would otherwise fail only after the ~12 min render
# ---------------------------------------------------------------------------
say "preflight"
DEV="${CUDA_VISIBLE_DEVICES-0}"
DEV="${DEV%%,*}"
[[ -n "$DEV" ]] || die "CUDA_VISIBLE_DEVICES is empty: no GPU to render on"
if FREE=$(nvidia-smi -i "$DEV" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null); then
  FREE="${FREE//[!0-9]/}"
  info "gpu           $DEV (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}): ${FREE} MiB free, ~${NEED_MIB} MiB needed"
  if [[ $FREE -lt $NEED_MIB ]]; then
    msg="GPU $DEV has only $FREE MiB free -- pick an idle card with CUDA_VISIBLE_DEVICES=<i>:"$'\n'"$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader)"
    if [[ $DRY -eq 1 ]]; then warn "$msg"; else die "$msg"; fi
  fi
else
  warn "nvidia-smi cannot query GPU '$DEV'; not checking free memory"
fi
if [[ $GENEVAL -eq 1 ]]; then
  ( source "$GROOT/geneval_env.sh"
    [[ -f "$GENEVAL_SRC/evaluation/evaluate_images.py" && -f "$GENEVAL_MODELS/mask2former.pth" ]] \
      || { echo "    incomplete GenEval install in $GROOT -- re-run scripts/setup_geneval.sh --root $GROOT" >&2; exit 1; }
    conda run "${GENEVAL_ENV_ARGS[@]}" python -c "import mmdet, mmcv, open_clip" \
      || { echo "    the geneval conda env (${GENEVAL_ENV_ARGS[*]}) cannot import mmdet/mmcv/open_clip" >&2; exit 1; }
  ) || die "GenEval install check failed"
  info "geneval env   ok"
fi

# One config per run, extending the canonical eval config (relative paths in it resolve
# against the repo, which is where reproduce.py runs). load_from '' = the klein-4B base.
CONFIG="$(cat <<EOF
# Written by scripts/eval_checkpoint.sh
extends: '$REPO/configs/eval_flux.yaml'
exp_name: eval
load_from: '$LOAD_FROM'
strict_load: true                 # a checkpoint that does not load cleanly stops the run
output_dir: '$OUT'
num_sampling_steps: $STEPS
pickscore_ctx_len: $PICKSCORE_CTX_LEN            # GenEval stays at flux_ctx_len 48
geneval_repo: null                # scored below by scripts/score_geneval.sh in the geneval env
autocast_cache: false             # no bf16 weight copy: identical images, fits a 24 GB card
vae_decode_batch: 4               # PickScore renders 16 at a time; one fp32 decode of 16 needs ~8 GB
EOF
)"

# A previous evaluation in $OUT: reuse it only if it evaluated exactly this.
PRIOR=none
if [[ -f "$OUT/flux_eval.json" ]]; then
  PRIOR=$("${CRUN[@]}" python - "$OUT/flux_eval.json" "$LOAD_FROM" "$STEPS" "$PICKSCORE_CTX_LEN" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
want = {"load_from": sys.argv[2], "num_sampling_steps": int(sys.argv[3]), "pickscore_ctx_len": int(sys.argv[4])}
diff = [f"{k}: {j.get(k)!r} there, {v!r} here" for k, v in want.items() if j.get(k) != v]
print("; ".join(diff) if diff else "same")
PY
)
  if [[ "$PRIOR" != same && $FORCE -eq 0 ]]; then
    die "$OUT already holds a different evaluation ($PRIOR) -- pass another --out, or --force to replace it"
  fi
fi

if [[ $DRY -eq 1 ]]; then
  say "dry run: would write $OUT/eval_config.yaml"
  printf '%s\n' "$CONFIG" | sed 's/^/    /'
  info "then: python reproduce.py eval-flux --config $OUT/eval_config.yaml   ($CONDA_ENV env, in $REPO)"
  [[ $GENEVAL -eq 0 ]] || info "then: bash scripts/score_geneval.sh $OUT/geneval --root $GROOT"
  info "previous evaluation in --out: $PRIOR"
  exit 0
fi

# ---------------------------------------------------------------------------
say "1/3 render + PickScore"
GJSONL="$OUT/geneval_results.jsonl"
n_images() { find "$OUT/geneval" -mindepth 3 -maxdepth 3 -path '*/samples/*.png' 2>/dev/null | wc -l; }
if [[ $FORCE -eq 0 && "$PRIOR" == same && $(n_images) -eq $N_GENEVAL ]]; then
  info "done before ($OUT/flux_eval.json, $N_GENEVAL GenEval images); skipping"
  RENDERED=0
else
  mkdir -p "$OUT"
  rm -f "$OUT/flux_eval.json" "$GJSONL" "$GJSONL.summary.json" "$OUT/summary.json" \
        "$OUT/seed_diversity.json"                                                   # stale results
  printf '%s\n' "$CONFIG" > "$OUT/eval_config.yaml"
  t0=$SECONDS
  # expandable_segments: less fragmentation next to a 15.5 GB fp32 model; numerics unchanged.
  if ! ( cd "$REPO" && source "$ROOT/env.sh" \
         && export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
         && "${CRUN[@]}" python reproduce.py eval-flux --config "$OUT/eval_config.yaml" ) 2>&1 \
       | tee "$OUT/eval_flux.log"; then
    die "reproduce.py eval-flux failed -- see $OUT/eval_flux.log"
  fi
  [[ -f "$OUT/flux_eval.json" ]] || die "reproduce.py wrote no $OUT/flux_eval.json -- see $OUT/eval_flux.log"
  n=$(n_images)
  [[ $n -eq $N_GENEVAL ]] || die "expected $N_GENEVAL GenEval images under $OUT/geneval, found $n"
  info "rendered in $(( (SECONDS - t0) / 60 )) min"
  RENDERED=1
fi

# ---------------------------------------------------------------------------
say "2/3 GenEval"
if [[ $GENEVAL -eq 0 ]]; then
  info "skipped (--no-geneval)"
elif [[ $FORCE -eq 0 && $RENDERED -eq 0 && -f "$GJSONL.summary.json" ]]; then
  info "done before ($GJSONL.summary.json); skipping"
else
  t0=$SECONDS
  if ! bash "$REPO/scripts/score_geneval.sh" "$OUT/geneval" --root "$GROOT" --out "$GJSONL" 2>&1 \
       | tee "$OUT/geneval_score.log"; then
    die "GenEval scoring failed -- see $OUT/geneval_score.log"
  fi
  info "scored in $(( (SECONDS - t0) / 60 )) min"
fi

# ---------------------------------------------------------------------------
say "3/3 seed diversity (the 4 images of each GenEval prompt)"
DIV="$OUT/seed_diversity.json"
if [[ $FORCE -eq 0 && $RENDERED -eq 0 && -f "$DIV" ]]; then
  info "done before ($DIV); skipping"
else
  # never fails the evaluation: a missing encoder is skipped inside, anything else only warns
  ( cd "$REPO" && source "$ROOT/env.sh" \
      && "${CRUN[@]}" python scripts/seed_diversity.py "$OUT/geneval" --out "$DIV" ) 2>&1 \
    | grep -v "it/s]" | tee "$OUT/seed_diversity.log" \
    || warn "seed diversity failed -- see $OUT/seed_diversity.log"
fi

# ---------------------------------------------------------------------------
say "summary"
"${CRUN[@]}" python - "$OUT" "$LABEL" <<'PY'
import json, os, sys
out, label = sys.argv[1], sys.argv[2]
ev = json.load(open(os.path.join(out, "flux_eval.json")))
gpath = os.path.join(out, "geneval_results.jsonl.summary.json")
gen = json.load(open(gpath)) if os.path.exists(gpath) else None
dpath = os.path.join(out, "seed_diversity.json")
div = json.load(open(dpath)) if os.path.exists(dpath) else None
summary = {"checkpoint": label, "num_sampling_steps": ev["num_sampling_steps"],
           "geneval": gen, "pickscore": ev.get("pickscore"), "seed_diversity": div,
           "pickscore_ctx_len": ev["pickscore_ctx_len"], "geneval_ctx_len": ev["flux_ctx_len"],
           "out_dir": out}
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
print(f"    checkpoint   {label} ({summary['num_sampling_steps']}-step)")
if gen:
    tasks = "  ".join(f"{k} {100 * v:.1f}" for k, v in gen.items() if k not in ("overall", "images"))
    print(f"    GenEval      {gen['overall']:.4f}   ({tasks})")
else:
    print("    GenEval      -")
print(f"    PickScore    {summary['pickscore']:.3f}   (Pick-a-Pic 499, ctx_len {summary['pickscore_ctx_len']})")
if div:
    parts = "  ".join(f"{k} {div[k]:.4f}" for k in ("pixel", "dreamsim", "dinov3_l") if k in div)
    print(f"    diversity    {parts}   (teacher 4-step: pixel 0.1905, dreamsim 0.2314, dinov3_l 0.2674)")
print(f"    -> {os.path.join(out, 'summary.json')}")
PY
