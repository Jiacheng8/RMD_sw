# Shared stages of the two mechanism-ablation machine scripts (sourced, not run):
#   experiments/mechanism_ablation/run_machine1_window_nodiag.sh
#   experiments/mechanism_ablation/run_machine2_global_plus_self.sh
# The caller sets WORK, REPO (the clone at the pinned commit), DATA, LOGS, DONE, RUN_NAME, CONFIG_NAME,
# PHASE2_HORIZON and sources this file after its preflight and clone. Everything fixed for the paper
# comparison lives here, not on the command line.

# ---- fixed settings ---------------------------------------------------------------------------------
OWNER="jiachengcui888"
MAIN_REPO="$OWNER/w128_s32_2000steps_w-gan"          # the main run (Table 1(a) / Table 2 "Local")
STORE_DATASET="$OWNER/sw-rdm-reference-store"
RESULT_DATASET="$OWNER/evaluation-result"
RESULT_PATH="mechanism-ablation"
ENV_NAME="rdm"
GPUS=2
PHASE1_STEPS=2000          # cosine to 2000 -> resume.pth, as both Table 2 runs
FINAL_STEP=3400            # the 6.8 H100 GPU-h budget of Table 2, in updates
SAVE_FREQ=400              # (in the configs) step_*.pth and resume.pth every 400 steps
STORE="$DATA/sw_lmmd/reference_store"
NYS="$DATA/sw_lmmd/ref_ablation/nystrom_m4096"
OUT="$DATA/sw_lmmd/work_dirs"
RUN_DIR="$OUT/$RUN_NAME"
CKPT_MAIN="$WORK/ckpt_main"
RESULT="$WORK/out/$RUN_NAME"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m    WARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m    ERROR: %s\033[0m\n' "$*" >&2; exit 1; }
pyrun() { conda run -n "$ENV_NAME" --no-capture-output python "$@"; }
step_file() { printf 'step_%07d.pth' "$1"; }

# stage <name> <function> [args]: skipped when $DONE/<name> exists; wall-clock -> $WORK/stages.jsonl
stage() {
  local name="$1"; shift
  if [[ -f "$DONE/$name" ]]; then say "$name: done before ($DONE/$name), skipped"; return 0; fi
  say "$name"
  local t0; t0=$(date +%s)
  "$@"
  printf '{"stage": "%s", "start": %s, "end": %s, "seconds": %s}\n' "$name" "$t0" "$(date +%s)" \
    "$(( $(date +%s) - t0 ))" >> "$WORK/stages.jsonl"
  date -Is > "$DONE/$name"
}

init_env() {       # every (re)start: conda on PATH, the data root's env.sh once setup wrote it
  [[ -f "$DATA/env.sh" ]] && source "$DATA/env.sh"
  mkdir -p "$CKPT_MAIN" "$RESULT"
}

# ---- setup ------------------------------------------------------------------------------------------
do_setup() {
  # conda env (torch 2.8.0+cu126, bitsandbytes 0.50.2, requirements-lock.txt: the main run's versions),
  # FLUX.2 [klein] 4B + AE + Qwen3 at the snapshots pinned in assets/hf_revisions.json (the main run's),
  # the encoders, the COCO reference store (tar md5 2165876e5cc25b2694b4f553f7a7a107, checked by
  # download_all.sh) and its 61 GB Qwen3 context. No GenEval / PickScore here: the checkpoints are
  # scored on the 6 x 4090 paper evaluation box (experiments/mechanism_ablation/eval_4090.sh).
  bash "$REPO/scripts/new_machine.sh" --root "$DATA" --config "$REPO/experiments/mechanism_ablation/configs/$CONFIG_NAME.yaml" \
    --gpus "$GPUS" --env "$ENV_NAME" --no-train --skip-geneval
  source "$DATA/env.sh"
  [[ -f "$STORE/metadata.json" ]] || die "reference store missing after setup: $STORE"
  pyrun - "$STORE" <<'PY'
import json, os, sys, torch
m = json.load(open(os.path.join(sys.argv[1], "metadata.json")))
print(f"    store: {m.get('num_rows', '?')} rows, encoders {list(m.get('encoders', m.get('encoder_names', [])))}")
print(f"    torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
import diffusers, timm, transformers, bitsandbytes
print(f"    diffusers {diffusers.__version__}, timm {timm.__version__}, transformers {transformers.__version__}, "
      f"bitsandbytes {bitsandbytes.__version__}")
PY
}

do_landmarks() {   # the Global arm's published Nystrom bundles; never rebuilt (k-means is not bit-reproducible)
  ok() { [[ -f "$NYS/SHA256SUMS" ]] && (cd "$NYS" && sha256sum --quiet -c SHA256SUMS) >/dev/null 2>&1; }
  if ok; then info "present, checksums OK"; return 0; fi
  mkdir -p "$NYS"
  pyrun - "$STORE_DATASET" "$DATA/sw_lmmd" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type="dataset", local_dir=sys.argv[2],
                  allow_patterns=["ref_ablation/nystrom_m4096/*"])
PY
  ok || die "the Nystrom bundles from $STORE_DATASET fail their SHA256SUMS"
  info "downloaded, checksums OK"
}

dl_main() {        # <step> -> prints the local path of the main run's step_<step>.pth
  local f; f="$(step_file "$1")"
  if [[ -s "$CKPT_MAIN/$f" ]]; then echo "$CKPT_MAIN/$f"; return 0; fi
  pyrun - "$MAIN_REPO" "$f" "$CKPT_MAIN" <<'PY' | tail -1
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3]))
PY
}

# ---- training: phase 1 (0 -> 2000, cosine over 2000), phase 2 (resume -> 3400, horizon PHASE2_HORIZON) --
_train() {         # <phase> <steps> [--set ...]: resumes from resume.pth when the run has one
  local phase="$1" steps="$2"; shift 2
  local resume=""
  if [[ -f "$RUN_DIR/resume.pth" ]]; then
    resume="$RUN_DIR/resume.pth"
  elif [[ -e "$RUN_DIR" ]]; then
    # stopped before its first resume.pth (step 400): nothing to continue from; keep it aside, start over
    local aside="$RUN_DIR.stopped-$(date +%s)"
    warn "$RUN_DIR has no resume.pth yet: moved to $aside, phase 1 starts over"
    mv "$RUN_DIR" "$aside"
  fi
  [[ "$phase" == 2 && -z "$resume" ]] && die "phase 2 needs $RUN_DIR/resume.pth (the step-$PHASE1_STEPS state)"
  if [[ -n "$resume" ]]; then
    local at; at=$(pyrun -c "import torch; print(torch.load('$resume', map_location='cpu', mmap=True, weights_only=False)['step'])" | tail -1)
    info "continuing from $resume (step $at)"
    [[ "$phase" == 1 ]] || (( at >= PHASE1_STEPS )) || die "resume.pth is at step $at, before $PHASE1_STEPS"
  fi
  env ASSETS="$DATA" CONDA_ENV="$ENV_NAME" GPUS="$GPUS" REFERENCE_ROOT="$STORE" OUTPUT_DIR="$OUT" \
      NYSTROM_DIR="$NYS" RESUME_FROM="$resume" \
    bash "$REPO/experiments/mechanism_ablation/run_arm.sh" "$CONFIG_NAME" --gpus "$GPUS" \
      --set "steps=$steps" "$@" 2>&1 | tee -a "$LOGS/train_$RUN_NAME.log" \
    || die "training ($RUN_NAME, phase $phase) failed: rerun the same command to continue from resume.pth"
  [[ -f "$RUN_DIR/$(step_file "$steps")" ]] || die "phase $phase ended without $(step_file "$steps")"
}

do_train_phase1() { _train 1 "$PHASE1_STEPS" "$@"; }
do_train_phase2() { _train 2 "$FINAL_STEP" --set "lr_schedule.total_steps=$PHASE2_HORIZON" "$@"; }

# ---- report + upload --------------------------------------------------------------------------------
do_report() {      # [extra report.py args]
  mkdir -p "$RESULT"
  (cd "$REPO" && pyrun -m experiments.mechanism_ablation.report train --run-name "$RUN_NAME" --run-dir "$RUN_DIR" \
     --work "$WORK" --out "$RESULT" "$@")
  cp -a "$RUN_DIR/train_log.jsonl" "$RESULT/train_log.jsonl"
  cp -a "$RUN_DIR"/pre_diagnostic_step*.json "$RESULT/" 2>/dev/null || true
}

do_upload() {
  local final; final="$RUN_DIR/$(step_file "$FINAL_STEP")"
  cp -a "$LOGS/run.log" "$RESULT/run.log"
  info "training-side files -> datasets/$RESULT_DATASET/$RESULT_PATH/$RUN_NAME/train_side/"
  pyrun - "$RESULT_DATASET" "$RESULT_PATH/$RUN_NAME/train_side" "$RESULT" <<'PY'
import sys
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id=sys.argv[1], repo_type="dataset", path_in_repo=sys.argv[2], folder_path=sys.argv[3],
                      commit_message=f"mechanism ablation: {sys.argv[2]}")
print(f"    https://huggingface.co/datasets/{sys.argv[1]}/tree/main/{sys.argv[2]}")
PY
  info "final checkpoint -> $OWNER/$RUN_NAME (public model repo, step $FINAL_STEP only)"
  local stage_dir="$WORK/hf_upload/$RUN_NAME"
  rm -rf "$stage_dir"; mkdir -p "$stage_dir"
  ln "$final" "$stage_dir/$(step_file "$FINAL_STEP")" 2>/dev/null || cp -a "$final" "$stage_dir/"
  cp -a "$RESULT"/{train_log.jsonl,run_info.json,report.md} "$stage_dir/"
  pyrun - "$OWNER/$RUN_NAME" "$stage_dir" <<'PY'
import os, sys
from huggingface_hub import HfApi
api, repo, folder = HfApi(), sys.argv[1], sys.argv[2]
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
api.upload_large_folder(repo_id=repo, repo_type="model", folder_path=folder)
remote = {f.path: f.size for f in api.list_repo_tree(repo, repo_type="model", recursive=True)
          if getattr(f, "size", None) is not None}
bad = [fn for fn in os.listdir(folder) if os.path.isfile(os.path.join(folder, fn))
       and remote.get(fn) != os.path.getsize(os.path.join(folder, fn))]
if bad:
    sys.exit(f"    size mismatch on the Hub: {bad}")
print(f"    https://huggingface.co/{repo} (sizes verified)")
PY
  # LAST: the completion marker the 4090 watcher (watch_4090.sh) waits for; written only after the checkpoint
  # is on the Hub with the local size AND sha256
  info "completion marker -> datasets/$RESULT_DATASET/$RESULT_PATH/$RUN_NAME/train_side/UPLOAD_DONE.json"
  pyrun - "$OWNER/$RUN_NAME" "$(step_file "$FINAL_STEP")" "$final" "$RESULT_DATASET" \
      "$RESULT_PATH/$RUN_NAME/train_side/UPLOAD_DONE.json" "$(git -C "$REPO" rev-parse HEAD)" <<'PY'
import hashlib, io, json, os, sys, time
from huggingface_hub import HfApi
repo, fname, local, ds, marker, commit = sys.argv[1:7]
h = hashlib.sha256()
with open(local, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
sha, size = h.hexdigest(), os.path.getsize(local)
api = HfApi()
info = api.get_paths_info(repo, [fname], repo_type="model")[0]
remote_sha = getattr(getattr(info, "lfs", None), "sha256", None)
if info.size != size or remote_sha != sha:
    sys.exit(f"    {repo}/{fname} on the Hub does not match the local file (size {info.size} vs {size}, "
             f"sha256 {remote_sha} vs {sha}): no marker written; rerun to upload again")
body = {"run": repo.split("/")[-1], "model_repo": repo, "checkpoint": fname, "size": size, "sha256": sha,
        "training_commit": commit, "uploaded": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
api.upload_file(path_or_fileobj=io.BytesIO(json.dumps(body, indent=2).encode()), path_in_repo=marker,
                repo_id=ds, repo_type="dataset", commit_message=f"{body['run']}: upload complete")
print(f"    marker written (sha256 {sha[:12]}..., {size / 1e9:.1f} GB)")
PY
}
