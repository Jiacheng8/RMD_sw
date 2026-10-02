#!/usr/bin/env bash
# Upload everything a training run produced to a private Hugging Face model repo, so it survives
# the rented machine: every step_*.pth (symlinked ones followed), resume.pth, train_log.jsonl,
# eval_summary*, eval_logs/, every finished eval_*/ dir (scores, logs, and the GenEval images
# packed as one uncompressed eval_*/geneval.tar each -- tens of thousands of small PNGs upload and
# download far better as one file), and <root>/sw_lmmd/logs/.
#
#   read -rs -p "HF write token: " HF_TOKEN && export HF_TOKEN
#   bash scripts/upload_run_hf.sh --root /root/RMD_data               # defaults below
#   bash scripts/upload_run_hf.sh --root /root/RMD_data --dry-run     # show the plan, upload nothing
#   unset HF_TOKEN
#
# Options: --run DIR (default <root>/sw_lmmd/work_dirs/sw-lmmd-flux-h100-2gpu), --repo ID (default
# <token's account>/<run name>), --no-resume (skip the 23 GB resume.pth), --workers N, --dry-run.
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

ROOT=""; RUN=""; REPO_ID=""; WITH_RESUME=1; WORKERS=16; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --run) RUN="$2"; shift 2 ;;
    --repo) REPO_ID="$2"; shift 2 ;;
    --no-resume) WITH_RESUME=0; shift ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$ROOT" ]] || { echo "need --root <data root> (e.g. /root/RMD_data)" >&2; exit 2; }
RUN="${RUN:-$ROOT/sw_lmmd/work_dirs/sw-lmmd-flux-h100-2gpu}"
[[ -d "$RUN" ]] || { echo "no run dir $RUN" >&2; exit 1; }
NAME="$(basename "$RUN")"
STAGE="$ROOT/hf_upload/$NAME"
LOGS="$ROOT/sw_lmmd/logs"
py() { conda run -n rdm --no-capture-output python "$@"; }
hf() { conda run -n rdm --no-capture-output hf "$@"; }
info() { echo "[upload] $*"; }

# --- token: must be a Write token; print the account it belongs to --------------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
  [[ -t 0 ]] || { echo "export HF_TOKEN=<write token> first" >&2; exit 1; }
  read -rs -p "HF write token: " HF_TOKEN; echo; export HF_TOKEN
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
REPO_ID="${REPO_ID:-$WHO/$NAME}"
info "target repo: $REPO_ID (created private if new; an existing repo keeps its visibility)"

# --- stage: hard links (same disk) or symlinks (other disk) into $STAGE ---------------------------
link() {   # link <src> <dst>: snapshot src at dst; hard link when possible, else a symlink to the real file
  local src dst real
  src="$1"; dst="$2"; real="$(readlink -f "$src")"
  mkdir -p "$(dirname "$dst")"
  rm -f "$dst"
  ln "$real" "$dst" 2>/dev/null || ln -s "$real" "$dst"
}
mkdir -p "$STAGE"
info "staging into $STAGE"

for f in "$RUN"/step_*.pth; do
  [[ -e "$f" ]] && link "$f" "$STAGE/$(basename "$f")"
done
if [[ $WITH_RESUME -eq 1 && -e "$RUN/resume.pth" ]]; then link "$RUN/resume.pth" "$STAGE/resume.pth"; fi
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
info "HF private storage counts against $WHO's quota -- check https://huggingface.co/settings/billing if unsure"
[[ $DRY -eq 1 ]] && { info "dry run: nothing uploaded"; exit 0; }

# --- upload (resumable) ----------------------------------------------------------------------------
py - "$REPO_ID" <<'EOF'
import os, sys
from huggingface_hub import create_repo
print("[upload] repo:", create_repo(sys.argv[1], repo_type="model", private=True, exist_ok=True,
                                    token=os.environ["HF_TOKEN"]))
EOF
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
info "done: https://huggingface.co/$REPO_ID  (download elsewhere: hf download $REPO_ID --local-dir <dir>)"
info "remember: unset HF_TOKEN, and delete the write token on huggingface.co when you no longer need it"
