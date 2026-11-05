#!/usr/bin/env bash
# Upload the four experiments of scripts/run_flux_time_budget.sh to ONE PUBLIC Hugging Face model repo, a folder
# per experiment. Run it on the machine that trained them.
#
#   bash scripts/upload_flux_time_budget_hf.sh --root /root/rdm-sets --dry-run     # the plan only
#   bash scripts/upload_flux_time_budget_hf.sh --root /root/rdm-sets               # -> <account>/ablation-sw
#
#   irdm/  dmd2/  k512-b32-gan/  k128-b4-gan/     step_*.pth, resume.pth, train_log.jsonl, time_budget.json, logs
#
# Up to ~90 GB per experiment, ~350 GB in all. Run it again after an interruption: the upload resumes. It ends by
# comparing every remote file size with the local one. An experiment without a run dir is reported and skipped.
#
# --root DIR      data root of the training (default /root/rdm-sets)
# --repo NAME     NAME or OWNER/NAME (default ablation-sw, under the token's account)
# --no-resume     leave out resume.pth
# --workers N     hf upload-large-folder workers (default 16)
# --env NAME      conda env with huggingface_hub and the hf CLI (default rdm)
# --dry-run       list what would go up, upload nothing
#
# Needs a HF Write token: $HF_TOKEN, else asked for once (hidden, never stored). The repo is created public, and an
# existing private one is switched to public. Files are staged as hard links (symlinks across disks) in
# <root>/hf_upload/<repo name>/, which also holds the upload's resume state.
set -euo pipefail
export PYTHONNOUSERSITE=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="/root/rdm-sets"; REPO_ID="ablation-sw"; WITH_RESUME=1; WORKERS=16; ENV_NAME="${CONDA_ENV:-rdm}"; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)      ROOT="$2"; shift 2 ;;
    --repo)      REPO_ID="$2"; shift 2 ;;
    --no-resume) WITH_RESUME=0; shift ;;
    --workers)   WORKERS="$2"; shift 2 ;;
    --env)       ENV_NAME="$2"; shift 2 ;;
    --dry-run)   DRY=1; shift ;;
    -h|--help)   sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
ROOT="$(realpath -m "$ROOT")"
OUT="$ROOT/sw_lmmd/work_dirs"
KINDS=(irdm dmd2 k512-b32-gan k128-b4-gan)
declare -A RUN_OF=([irdm]="$OUT/global-n10240-4gph" [dmd2]="$OUT/flux-dmd2-4gph"
                   [k512-b32-gan]="$OUT/flux-k512-b32-gan-4gph" [k128-b4-gan]="$OUT/flux-k128-b4-gan-4gph")
crun() { ( cd "$REPO" && conda run -n "$ENV_NAME" --no-capture-output "$@" ); }

# --- token: a Write token; a bare repo name goes under its account -------------------------------------------------
if [[ -z "${HF_TOKEN:-}" && $DRY -eq 1 ]]; then
  WHO="<token account>"; ROLE="-"           # a dry run needs no token
else
  if [[ -z "${HF_TOKEN:-}" ]]; then
    [[ -t 0 ]] || die "no terminal to ask for the token: export HF_TOKEN=<write token> first"
    read -rs -p "HF write token: " HF_TOKEN; echo; export HF_TOKEN    # lives only in this process
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
[[ "${REPO_ID#*/}" =~ ^[A-Za-z0-9._-]{1,96}$ ]] || die "repo name '$REPO_ID': use letters, digits, '-', '_' or '.'"
[[ "$REPO_ID" == */* ]] || REPO_ID="$WHO/$REPO_ID"

# --- stage ---------------------------------------------------------------------------------------------------------
STAGE="$ROOT/hf_upload/${REPO_ID#*/}"
link() {   # link <src> <dst>: hard link when possible, else a symlink to the real file
  local real; real="$(readlink -f "$1")"
  rm -f "$2"; ln "$real" "$2" 2>/dev/null || ln -s "$real" "$2"
}
say "-> https://huggingface.co/$REPO_ID (PUBLIC: anyone can download it)"
FOUND=0
shopt -s nullglob
for k in "${KINDS[@]}"; do
  run="${RUN_OF[$k]}"; s="$STAGE/$k"
  if [[ ! -d "$run" ]]; then info "$k: NO RUN DIR $run, skipped"; continue; fi
  ckpts=("$run"/step_*.pth)
  if (( ${#ckpts[@]} == 0 )); then info "$k: NO step_*.pth in $run, skipped"; continue; fi
  mkdir -p "$s"
  find "$s" -maxdepth 1 \( -type f -o -type l \) -delete     # restage from scratch; keeps .cache/
  for f in "${ckpts[@]}"; do link "$f" "$s/$(basename "$f")"; done
  (( WITH_RESUME )) && [[ -e "$run/resume.pth" ]] && link "$run/resume.pth" "$s/resume.pth"
  find "$run" -maxdepth 1 -type f -not -name '*.pth' -not -name '*.partial' -size -1048576k -exec cp -p {} "$s/" \;
  FOUND=$(( FOUND + 1 ))
  info "$(printf '%-13s %s ckpt(s)%s, %s  <- %s' "$k" "${#ckpts[@]}" \
    "$( [[ -e "$s/resume.pth" ]] && echo ' + resume.pth' )" "$(du -shL --exclude=.cache "$s" | cut -f1)" "$run")"
  (cd "$s" && find -L . -maxdepth 1 -type f -printf '      %P\n' | sort)
done
(( FOUND )) || die "nothing to upload under $OUT (wrong --root?)"
info "total $(du -shL --exclude=.cache "$STAGE" | cut -f1)"
(( DRY )) && { say "dry run: nothing uploaded"; exit 0; }

# --- upload (resumable) and verify ----------------------------------------------------------------------------------
crun python - "$REPO_ID" <<'PY' || die "could not create $REPO_ID"
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(sys.argv[1], repo_type="model", private=False, exist_ok=True)
api.update_repo_settings(sys.argv[1], private=False, repo_type="model")   # an existing private repo
PY
crun hf upload-large-folder "$REPO_ID" "$STAGE" --repo-type model --num-workers "$WORKERS" --exclude ".cache/**" \
  || die "upload interrupted; run the same command again to resume"
crun python - "$REPO_ID" "$STAGE" <<'PY' || die "the hub does not match the staged files (run the same command again)"
import os, sys
from huggingface_hub import HfApi
repo, stage = sys.argv[1], sys.argv[2]
remote = {f.path: f.size for f in HfApi(token=os.environ["HF_TOKEN"]).list_repo_tree(repo, repo_type="model", recursive=True)
          if getattr(f, "size", None) is not None}
bad = n = 0
for dp, dns, fns in os.walk(stage, followlinks=True):
    dns[:] = [d for d in dns if d != ".cache"]
    for fn in fns:
        p = os.path.join(dp, fn); rel = os.path.relpath(p, stage); n += 1
        if remote.get(rel) != os.path.getsize(p):
            bad += 1; print(f"    MISMATCH {rel}: local {os.path.getsize(p)}, hub {remote.get(rel)}")
print(f"    {n} files checked, {bad} mismatches")
sys.exit(1 if bad else 0)
PY
say "done: https://huggingface.co/$REPO_ID"
