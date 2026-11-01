#!/usr/bin/env bash
# Upload the three Z-Image experiments (ours, DMD2, RDM-global) to Hugging Face, so they outlive the
# machine they ran on. Run it on that machine.
#
#   bash z-image/scripts/upload_results_hf.sh --dry-run               # the plan only
#   bash z-image/scripts/upload_results_hf.sh                         # upload to <account>/z-image
#   bash z-image/scripts/upload_results_hf.sh --last-only             # 1 ckpt each
#
# Two destinations:
#
# 1. checkpoints + training logs -> ONE public model repo (default z-image under the token's account;
#    --repo NAME or OWNER/NAME changes it), a folder per experiment:
#
#      ours/        step_*.pth, train_log.jsonl, qwen_image_bench.md, the run dir's other small files
#      dmd2/        the same (step_*.pth = the student only)
#      rdm-global/  the same, plus plan.json
#
#    resume.pth (26 GB+, and DMD2's also carries the fake score and the GAN head) only with
#    --with-resume.
#
# 2. the Qwen-Image-Bench evaluations of each experiment's root (<root>/qib/runs: scores, verdicts per
#    image, report, renders) -> the public evaluation dataset, through upload_eval_hf.sh --all (default
#    jiachengcui888/evaluation-result; --eval-repo changes it, --no-eval skips this part).
#
# The experiment roots (the --root each launcher ran with; run dir = <root>/zimage/work_dirs/<name>):
#
#   --base DIR        default /plm-shared/zhangjunming/Workspace/ZImage-Experiments-20261030
#   --ours-root DIR   default <base>/rdm-sets                    run zimage-2step-1024
#   --dmd2-root DIR   default <base>/experiments/dmd2/assets       run zimage-2step-1024-dmd2
#   --rdm-root DIR    default <base>/experiments/rdm-global/assets run zimage-2step-1024-rdm-global-N10240
#
# --only LIST     a comma list of ours,dmd2,rdm-global (default all three)
# --last-only     the last step_*.pth of each experiment instead of every one (~25 GB each)
# --with-resume   also resume.pth
# --no-eval       checkpoints and logs only
# --workers N     hf upload-large-folder workers (default 16)
# --env NAME      conda env with huggingface_hub and the hf CLI (default rdm)
# --dry-run       list what would go up, upload nothing
#
# Needs a HF Write token: $HF_TOKEN, else asked for once (hidden; never stored) and handed to
# upload_eval_hf.sh. Files are staged as hard links (symlinks across disks) in <base>/hf_upload/<repo
# name>/, which also holds the upload's resume state: if it is interrupted, run the same command again.
# Ends by comparing every remote file size with the local one.
set -euo pipefail
export PYTHONNOUSERSITE=1
ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ZDIR/.." && pwd)"
BASE="/plm-shared/zhangjunming/Workspace/ZImage-Experiments-20261030"
OURS_ROOT=""; DMD2_ROOT=""; RDM_ROOT=""; REPO_ID="z-image"; EVAL_REPO="jiachengcui888/evaluation-result"
ONLY="ours,dmd2,rdm-global"; LAST_ONLY=0; WITH_RESUME=0; DO_EVAL=1; WORKERS=16; ENV_NAME="rdm"; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)        BASE="$2"; shift 2 ;;
    --ours-root)   OURS_ROOT="$2"; shift 2 ;;
    --dmd2-root)   DMD2_ROOT="$2"; shift 2 ;;
    --rdm-root)    RDM_ROOT="$2"; shift 2 ;;
    --repo)        REPO_ID="$2"; shift 2 ;;
    --eval-repo)   EVAL_REPO="$2"; shift 2 ;;
    --only)        ONLY="$2"; shift 2 ;;
    --last-only)   LAST_ONLY=1; shift ;;
    --with-resume) WITH_RESUME=1; shift ;;
    --no-eval)     DO_EVAL=0; shift ;;
    --workers)     WORKERS="$2"; shift 2 ;;
    --env)         ENV_NAME="$2"; shift 2 ;;
    --dry-run)     DRY=1; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
BASE="$(realpath -m "$BASE")"
OURS_ROOT="$(realpath -m "${OURS_ROOT:-$BASE/rdm-sets}")"
DMD2_ROOT="$(realpath -m "${DMD2_ROOT:-$BASE/experiments/dmd2/assets}")"
RDM_ROOT="$(realpath -m "${RDM_ROOT:-$BASE/experiments/rdm-global/assets}")"
declare -A ROOT_OF=([ours]="$OURS_ROOT" [dmd2]="$DMD2_ROOT" [rdm-global]="$RDM_ROOT")
declare -A RUN_OF=([ours]="$OURS_ROOT/zimage/work_dirs/zimage-2step-1024"
                   [dmd2]="$DMD2_ROOT/zimage/work_dirs/zimage-2step-1024-dmd2"
                   [rdm-global]="$RDM_ROOT/zimage/work_dirs/zimage-2step-1024-rdm-global-N10240")
IFS=, read -r -a KINDS <<< "$ONLY"
for k in "${KINDS[@]}"; do
  [[ -n "${RUN_OF[$k]+x}" ]] || die "--only: unknown experiment '$k' (ours, dmd2, rdm-global)"
  [[ -d "${RUN_OF[$k]}" ]] || die "$k: no run dir ${RUN_OF[$k]} (wrong --base / --$k-root?)"
done
crun() { ( export PYTHONPATH="$ZDIR:$REPO:${PYTHONPATH:-}"
           cd "$REPO" && conda run -n "$ENV_NAME" --no-capture-output "$@" ); }

# --- token: a Write token; a bare repo name goes under its account ---------------------------------
if [[ -z "${HF_TOKEN:-}" && $DRY -eq 1 ]]; then
  WHO="<token account>"; ROLE="-"          # a dry run needs no token
else
if [[ -z "${HF_TOKEN:-}" ]]; then
  [[ -t 0 ]] || die "no terminal to ask for the token: export HF_TOKEN=<write token> first"
  read -rs -p "HF write token: " HF_TOKEN; echo; export HF_TOKEN   # lives only in this process
fi
WHO_ROLE=$(crun python - <<'PY'
import os
from huggingface_hub import whoami
w = whoami(token=os.environ["HF_TOKEN"])
print(w["name"], w.get("auth", {}).get("accessToken", {}).get("role", "?"))
PY
) || die "token rejected by huggingface.co (whoami failed)"
read -r WHO ROLE <<< "$WHO_ROLE"
info "token account: $WHO (role: $ROLE)"
[[ "$ROLE" == read && $DRY -eq 0 ]] && die "this is a Read token; uploading needs a Write token"
fi
[[ "${REPO_ID#*/}" =~ ^[A-Za-z0-9._-]{1,96}$ ]] || \
  die "repo name '$REPO_ID': use letters, digits, '-', '_' or '.' (at most 96)"
[[ "$REPO_ID" == */* ]] || REPO_ID="$WHO/$REPO_ID"

# --- stage: <base>/hf_upload/<repo name>/<experiment>/ ---------------------------------------------
STAGE="$BASE/hf_upload/${REPO_ID#*/}"
link() {   # link <src> <dst>: hard link when possible, else a symlink to the real file
  local real; real="$(readlink -f "$1")"
  rm -f "$2"; ln "$real" "$2" 2>/dev/null || ln -s "$real" "$2"
}
say "checkpoints + logs -> https://huggingface.co/$REPO_ID (PUBLIC)"
for k in "${KINDS[@]}"; do
  run="${RUN_OF[$k]}"; s="$STAGE/$k"
  mkdir -p "$s"
  find "$s" -maxdepth 1 \( -type f -o -type l \) -delete   # restage from scratch; keeps .cache/
  ckpts=("$run"/step_*.pth)
  [[ -e "${ckpts[0]}" ]] || die "$k: no step_*.pth in $run"
  (( LAST_ONLY )) && ckpts=("${ckpts[-1]}")
  for f in "${ckpts[@]}"; do link "$f" "$s/$(basename "$f")"; done
  (( WITH_RESUME )) && [[ -e "$run/resume.pth" ]] && link "$run/resume.pth" "$s/resume.pth"
  # the run dir's small files: train_log.jsonl, qwen_image_bench.md, plan.json, configs, logs
  find "$run" -maxdepth 1 -type f -not -name '*.pth' -not -name '*.partial' -size -1048576k \
    -exec cp -p {} "$s/" \;
  info "$(printf '%-11s %s ckpt(s) %s| %s files, %s  <- %s' "$k" "${#ckpts[@]}" \
    "$( (( WITH_RESUME )) && [[ -e "$s/resume.pth" ]] && echo '+ resume.pth ')" \
    "$(find -L "$s" -maxdepth 1 -type f | wc -l)" "$(du -shL --exclude=.cache "$s" | cut -f1)" "$run")"
  (cd "$s" && find -L . -maxdepth 1 -type f -printf '      %P\n' | sort)
done
info "total $(du -shL --exclude=.cache "$STAGE" | cut -f1); the repo is public: anyone can download it"

if (( DO_EVAL )); then
  say "Qwen-Image-Bench evaluations -> dataset $EVAL_REPO (upload_eval_hf.sh --all, per root)"
  for k in "${KINDS[@]}"; do info "$k: ${ROOT_OF[$k]}/qib/runs"; done
fi
if (( DRY )); then
  if (( DO_EVAL )); then
    for k in "${KINDS[@]}"; do
      bash "$ZDIR/scripts/upload_eval_hf.sh" --root "${ROOT_OF[$k]}" --repo "$EVAL_REPO" \
        --env "$ENV_NAME" --all --dry-run || info "$k: no finished evaluation to upload"
    done
  fi
  say "dry run: nothing uploaded"; exit 0
fi

# --- upload the checkpoints (resumable) and verify ------------------------------------------------
crun python - "$REPO_ID" <<'PY' || die "could not create $REPO_ID"
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(sys.argv[1], repo_type="model", private=False, exist_ok=True)
api.update_repo_settings(sys.argv[1], private=False, repo_type="model")   # an existing private repo
print(f"    repo (public): https://huggingface.co/{sys.argv[1]}")
PY
crun hf upload-large-folder "$REPO_ID" "$STAGE" --repo-type model --num-workers "$WORKERS" \
  --exclude ".cache/**" || die "upload interrupted; run the same command again to resume"
crun python - "$REPO_ID" "$STAGE" <<'PY' || die "the hub does not match the staged files (rerun to resume)"
import os, sys
from huggingface_hub import HfApi
repo, stage = sys.argv[1], sys.argv[2]
remote = {f.path: f.size for f in HfApi(token=os.environ["HF_TOKEN"]).list_repo_tree(
    repo, repo_type="model", recursive=True) if getattr(f, "size", None) is not None}
bad = 0
for dp, dns, fns in os.walk(stage, followlinks=True):
    dns[:] = [d for d in dns if d != ".cache"]
    for fn in fns:
        p = os.path.join(dp, fn); rel = os.path.relpath(p, stage)
        if remote.get(rel) != os.path.getsize(p):
            bad += 1; print(f"    MISMATCH {rel}: local {os.path.getsize(p)}, hub {remote.get(rel)}")
print(f"    verify: {len(remote)} files on the hub, {sum(remote.values())/1e9:.1f} GB; mismatches: {bad}")
sys.exit(1 if bad else 0)
PY

# --- the evaluations ------------------------------------------------------------------------------
if (( DO_EVAL )); then
  for k in "${KINDS[@]}"; do
    say "$k: evaluations under ${ROOT_OF[$k]}/qib/runs"
    bash "$ZDIR/scripts/upload_eval_hf.sh" --root "${ROOT_OF[$k]}" --repo "$EVAL_REPO" \
      --env "$ENV_NAME" --all || die "$k: evaluation upload failed (run the same command again)"
  done
fi
say "done: https://huggingface.co/$REPO_ID$( (( DO_EVAL )) && echo " and https://huggingface.co/datasets/$EVAL_REPO")"
