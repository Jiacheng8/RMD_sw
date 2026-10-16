#!/usr/bin/env bash
# Upload everything a training run produced to a PUBLIC Hugging Face model repo, so it survives
# the rented machine: every step_*.pth (symlinked ones followed), resume.pth, train_log.jsonl,
# eval_summary*, eval_logs/, every finished eval_*/ dir (scores, logs, and the GenEval images
# packed as one uncompressed eval_*/geneval.tar each -- tens of thousands of small PNGs upload and
# download far better as one file), and <root>/sw_lmmd/logs/.
#
#   bash scripts/upload_run_hf.sh --root /root/RMD_data                      # defaults below
#   bash scripts/upload_run_hf.sh --root /root/RMD_data --repo my-run-name   # name the repo yourself
#   bash scripts/upload_run_hf.sh --root /root/RMD_data --dry-run            # show the plan only
#   bash scripts/upload_run_hf.sh --root /root/RMD_data --watch 600          # keep uploading
#
# It asks for the HF write token itself (hidden input) unless $HF_TOKEN is already set.
#
# Options: --run DIR (default <root>/sw_lmmd/work_dirs/sw-lmmd-flux-h100-2gpu), --repo NAME or
# OWNER/NAME (a bare NAME goes under the token's account; default <token's account>/<run dir name>),
# --no-resume (skip the 23 GB resume.pth), --workers N, --dry-run, --watch SECONDS.
#
# The repo is PUBLIC: created public, and switched to public if it already exists as private.
#
# --watch SECONDS: a pass every SECONDS while the run goes on (start it in its own tmux window next
# to training): each pass stages and uploads only what is new -- every finished checkpoint, the
# growing train_log.jsonl and logs, each eval_*/ dir once its summary.json exists. resume.pth (23 GB,
# rewritten every few hundred steps) is held back until a final pass, which runs once
# <run>/eval_summary.md appears (the one-command launchers write it when the evaluation ends) and
# then stops; a failed pass (network) is retried at the next one. A run trained with --no-eval
# writes no eval_summary.md: stop with Ctrl-C and run the plain command once more.
#
# Safe while training is still running: files are snapshotted into <root>/hf_upload/<run name>/ as
# hard links (the trainer replaces checkpoints and resume.pth with os.replace, so a link keeps the
# version that was there at staging time); *.partial files and eval dirs without summary.json are
# skipped. Uses `hf upload-large-folder`, which resumes where it stopped: if it is interrupted, or to
# add checkpoints/evals that appeared later, just run the same command again. Ends by comparing
# every remote file size with the local one.
#
# The token is only read from $HF_TOKEN (or prompted for); it never goes on a command line or into a
# file. <repo>/.hf_token is deliberately NOT used: on these machines it holds a Read token.
set -euo pipefail
export PYTHONNOUSERSITE=1
command -v conda >/dev/null 2>&1 || PATH="$PATH:/root/miniconda3/bin"

ROOT=""; RUN=""; REPO_ID=""; WITH_RESUME=1; WORKERS=16; DRY=0; WATCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --run) RUN="$2"; shift 2 ;;
    --repo) REPO_ID="$2"; shift 2 ;;
    --no-resume) WITH_RESUME=0; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --watch) WATCH="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$ROOT" ]] || { echo "need --root <data root> (e.g. /root/RMD_data)" >&2; exit 2; }
[[ "$WATCH" =~ ^[0-9]+$ ]] || { echo "--watch takes a number of seconds, got '$WATCH'" >&2; exit 2; }
RUN="${RUN:-$ROOT/sw_lmmd/work_dirs/sw-lmmd-flux-h100-2gpu}"
[[ -d "$RUN" ]] || { echo "no run dir $RUN" >&2; exit 1; }
NAME="$(basename "$RUN")"
STAGE="$ROOT/hf_upload/$NAME"
LOGS="$ROOT/sw_lmmd/logs"
py() { conda run -n rdm --no-capture-output python "$@"; }
hf() { conda run -n rdm --no-capture-output hf "$@"; }
info() { echo "[upload] $*"; }

# --- token: must be a Write token; print the account it belongs to --------------------------------
TOKEN_FROM_ENV=1
if [[ -z "${HF_TOKEN:-}" ]]; then
  [[ -t 0 ]] || { echo "no terminal to ask for the token: export HF_TOKEN=<write token> first" >&2; exit 1; }
  read -rs -p "HF write token: " HF_TOKEN; echo; export HF_TOKEN   # lives only in this process
  TOKEN_FROM_ENV=0
fi
WHO_ROLE=$(py - <<'EOF'
import os
from huggingface_hub import whoami
w = whoami(token=os.environ["HF_TOKEN"])
tok = w.get("auth", {}).get("accessToken", {})
print(w["name"], tok.get("role", "?"))
EOF
) || { echo "token rejected by huggingface.co (whoami failed)" >&2; exit 1; }
read -r WHO ROLE <<< "$WHO_ROLE"
info "token account: $WHO (role: $ROLE)"
if [[ "$ROLE" == "read" ]]; then
  echo "this is a Read token; uploading needs a Write (or fine-grained write) token" >&2
  [[ $DRY -eq 1 ]] || exit 1
fi
if [[ -z "$REPO_ID" ]]; then REPO_ID="$WHO/$NAME"            # default: the run dir's name
elif [[ "$REPO_ID" != */* ]]; then REPO_ID="$WHO/$REPO_ID"; fi  # a bare name: the token's account
[[ "$REPO_ID" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]{1,96}$ ]] || {
  echo "repo name '$REPO_ID': use letters, digits, '-', '_' or '.' (at most 96)" >&2; exit 2; }
info "target repo: https://huggingface.co/$REPO_ID (PUBLIC: created public, or switched to public)"

# --- stage: hard links (same disk) or symlinks (other disk) into $STAGE ---------------------------
link() {   # link <src> <dst>: snapshot src at dst; hard link when possible, else a symlink to the real file
  local src dst real
  src="$1"; dst="$2"; real="$(readlink -f "$src")"
  mkdir -p "$(dirname "$dst")"
  rm -f "$dst"
  ln "$real" "$dst" 2>/dev/null || ln -s "$real" "$dst"
}
one_pass() {   # one_pass <with_resume 1|0>: stage what exists now, upload what is new, verify
local with_resume="$1"
mkdir -p "$STAGE"
info "staging into $STAGE"

for f in "$RUN"/step_*.pth; do
  [[ -e "$f" ]] && link "$f" "$STAGE/$(basename "$f")"
done
if [[ $with_resume -eq 1 && -e "$RUN/resume.pth" ]]; then
  link "$RUN/resume.pth" "$STAGE/resume.pth"
else
  rm -f "$STAGE/resume.pth"       # never upload an older staged copy
fi
for f in "$RUN"/train_log.jsonl "$RUN"/eval_summary*; do
  [[ -f "$f" ]] && cp -p "$f" "$STAGE/"   # small and still appended to: copy, not link
done
[[ -d "$RUN/eval_logs" ]] && { mkdir -p "$STAGE/eval_logs"; cp -p "$RUN"/eval_logs/* "$STAGE/eval_logs/" 2>/dev/null || true; }
[[ -d "$LOGS" ]] && { mkdir -p "$STAGE/logs"; cp -p "$LOGS"/* "$STAGE/logs/" 2>/dev/null || true; }

for d in "$RUN"/eval_*/; do
  d="${d%/}"; e="$(basename "$d")"
  [[ "$e" == eval_logs ]] && continue
  if [[ ! -f "$d/summary.json" ]]; then info "skip $e (no summary.json yet: unfinished)"; continue; fi
  mkdir -p "$STAGE/$e"
  find "$d" -maxdepth 1 -type f -exec cp -p {} "$STAGE/$e/" \;
  if [[ -d "$d/geneval" ]]; then
    tarf="$STAGE/$e/geneval.tar"
    if [[ -f "$tarf" && -z "$(find "$d/geneval" -newer "$tarf" -print -quit)" ]]; then
      :   # images unchanged since the last tar
    elif [[ $DRY -eq 1 ]]; then
      info "would pack $e/geneval ($(du -sh "$d/geneval" | cut -f1)) -> $e/geneval.tar"
    else
      info "packing $e/geneval -> geneval.tar"
      tar -cf "$tarf.partial" -C "$d" geneval && mv "$tarf.partial" "$tarf"
    fi
  fi
done
rm -f "$STAGE"/*.partial "$STAGE"/*/*.partial

info "plan ($(find -L "$STAGE" -type f -not -path '*/.cache/*' | wc -l) files, $(du -shL --exclude=.cache "$STAGE" | cut -f1)):"
(cd "$STAGE" && find -L . -maxdepth 1 -type f -name '*.pth' -printf '  %P  %s bytes\n' | sort)
(cd "$STAGE" && find . -mindepth 1 -maxdepth 1 -type d -not -name .cache -printf '  %P/\n' | sort)
info "the repo is public: anyone can see and download these files"
[[ $DRY -eq 1 ]] && return 0

# --- upload (resumable) ----------------------------------------------------------------------------
info "uploading with $WORKERS workers (interrupt-safe: rerun the same command to continue)"
hf upload-large-folder "$REPO_ID" "$STAGE" --repo-type model --num-workers "$WORKERS" --exclude ".cache/**"

# --- verify: every local file is on the Hub with the same size --------------------------------------
py - "$REPO_ID" "$STAGE" <<'EOF'
import os, sys
from huggingface_hub import HfApi
repo, stage = sys.argv[1], sys.argv[2]
remote = {f.path: f.size for f in HfApi(token=os.environ["HF_TOKEN"]).list_repo_tree(
    repo, repo_type="model", recursive=True) if hasattr(f, "size") and f.size is not None}
bad = 0
for dp, dns, fns in os.walk(stage, followlinks=True):
    dns[:] = [d for d in dns if d != ".cache"]
    for fn in fns:
        p = os.path.join(dp, fn); rel = os.path.relpath(p, stage); size = os.path.getsize(p)
        if remote.get(rel) != size:
            bad += 1; print(f"[verify] MISMATCH {rel}: local {size}, hub {remote.get(rel)}")
total = sum(remote.values())
print(f"[verify] {len(remote)} files on the hub, {total/1e9:.1f} GB; mismatches: {bad}")
sys.exit(1 if bad else 0)
EOF
}

# --- run: once, or a pass every --watch seconds ----------------------------------------------------
if [[ $DRY -eq 1 ]]; then
  one_pass "$WITH_RESUME"
  info "dry run: nothing uploaded"
  exit 0
fi
py - "$REPO_ID" <<'EOF'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
url = api.create_repo(sys.argv[1], repo_type="model", private=False, exist_ok=True)
api.update_repo_settings(sys.argv[1], private=False, repo_type="model")   # an existing private repo
print("[upload] repo (public):", url)
EOF
if [[ $WATCH -eq 0 ]]; then
  one_pass "$WITH_RESUME"
else
  info "watch: a pass every ${WATCH}s; resume.pth only in the final pass, which runs once" \
       "$RUN/eval_summary.md exists (Ctrl-C to stop earlier)"
  while true; do
    if [[ -f "$RUN/eval_summary.md" ]]; then
      info "eval_summary.md is there: final pass"
      if one_pass "$WITH_RESUME"; then break; fi
      info "final pass incomplete; retrying in ${WATCH}s"
    elif ! one_pass 0; then
      info "pass incomplete (network?); retrying in ${WATCH}s"
    fi
    info "next pass at $(date -d "+${WATCH} seconds" +%H:%M:%S) (Ctrl-C to stop)"
    sleep "$WATCH"
  done
fi
info "done: https://huggingface.co/$REPO_ID  (download elsewhere: hf download $REPO_ID --local-dir <dir>)"
[[ $TOKEN_FROM_ENV -eq 1 ]] && info "remember: unset HF_TOKEN"
info "delete the write token on huggingface.co when you no longer need it"
