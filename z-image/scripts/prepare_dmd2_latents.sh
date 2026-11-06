#!/usr/bin/env bash
# The DMD2 baseline's real data, made once on any GPU machine and uploaded to a public HF dataset, so
# the training machine downloads ~17 GB instead of encoding (dmd2_*-cards.sh --latents-hf <repo>).
#
#   git clone -b zimage https://github.com/Jiacheng8/RMD_sw.git RDM && cd RDM
#   tmux new -s latents
#   bash z-image/scripts/prepare_dmd2_latents.sh --root /root/rdm-sets --repo <owner>/<dataset>
#
# Stages (each skipped when already done; re-run the same command after an interruption):
#   1 prepare  _ours_run.sh --skip-contexts --no-train --no-eval: the conda env, Z-Image-Turbo (its
#              VAE and Qwen3 text encoder are what is used here), the reference pack (its prompt table:
#              every render's prompt is checked against it)
#   2 encode   zimage_distill.dmd2_latents on every visible GPU: --shards of the 87 tar shards of
#              jiachengcui888/zimage-turbo-reference (default 24 -> ~34k renders, ~48 GB downloaded, each
#              tar deleted once encoded) -> VAE latents, + the empty-prompt context (the CFG negative)
#   3 upload   <root>/zimage/dmd2_latents/{latents/, negative_context.pt} -> the HF dataset, public
#              (hf upload-large-folder: an interrupted upload resumes on a re-run)
#
# Any GPU with >= 24 GB works (VAE encoding, batch 8). On 4 GPUs: ~15 min setup + downloads, ~30 min
# encoding, ~10-20 min upload [estimates; 1 shard took ~5 min on a 4090]. Disk: ~100 GB.
# The upload needs a HF Write token: $HF_TOKEN, else asked for (hidden; never stored).
#
# --root DIR      data root (default /root/rdm-sets)
# --repo REPO     the HF dataset to upload to (required unless --no-upload), e.g. <you>/zimage-dmd2-latents
# --shards K      tar shards to encode (default 24, what dmd2_*-cards.sh uses)
# --no-upload     encode only
# --env NAME      conda env (default rdm)
set -euo pipefail
export PYTHONNOUSERSITE=1
ZDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ZDIR/.." && pwd)"
ROOT="/root/rdm-sets"; REPO_ID=""; SHARDS=24; UPLOAD=1; ENV_NAME="rdm"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)      ROOT="$2"; shift 2 ;;
    --repo)      REPO_ID="$2"; shift 2 ;;
    --shards)    SHARDS="$2"; shift 2 ;;
    --no-upload) UPLOAD=0; shift ;;
    --env)       ENV_NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
(( UPLOAD == 0 )) || [[ "$REPO_ID" == */* ]] || die "--repo <owner>/<dataset> (or --no-upload)"
ROOT="$(realpath -m "$ROOT")"
PACK="$ROOT/zimage/zimage-ref-v1"; LAT="$ROOT/zimage/dmd2_latents"
crun() { ( [[ -f "$ROOT/env.sh" ]] && source "$ROOT/env.sh"; export PYTHONPATH="$ZDIR:$REPO:${PYTHONPATH:-}"
           cd "$REPO" && conda run -n "$ENV_NAME" --no-capture-output "$@" ); }
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' '); fi
(( ${#GPU_IDS[@]} > 0 )) || die "no GPU visible"
say "DMD2 real latents: $SHARDS tar shards on GPUs ${GPU_IDS[*]} -> $LAT${REPO_ID:+ -> $REPO_ID}"
[[ -z "${TMUX:-}${STY:-}" ]] && info "not inside tmux/screen: a dropped SSH session stops it"

# ask for the token now, not after an hour of encoding
if (( UPLOAD )) && [[ -z "${HF_TOKEN:-}" ]]; then
  [[ -t 0 ]] || die "no terminal to ask for the token: export HF_TOKEN=<write token> first"
  read -rs -p "HF write token (for $REPO_ID): " HF_TOKEN; echo; export HF_TOKEN
fi

# 1 ------------------------------------------------------------------------------------------------
say "1/3 env, Z-Image-Turbo, the reference pack"
bash "$ZDIR/scripts/_ours_run.sh" --root "$ROOT" --env "$ENV_NAME" --ref-hf jiachengcui888/zimage-ref-v1 \
  --skip-contexts --no-train --no-eval --allow-low-disk | grep -v "^\s*$" || die "preparation failed (see above)"
[[ -f "$PACK/prompts.jsonl" ]] || die "$PACK/prompts.jsonl missing after preparation"

# 2 ------------------------------------------------------------------------------------------------
say "2/3 encode on ${#GPU_IDS[@]} GPU(s)"
if [[ -f "$LAT/latents/manifest.json" ]]; then info "present: $LAT/latents"
else
  mkdir -p "$LAT/logs"; pids=()
  for r in "${!GPU_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$r]}" crun python -m zimage_distill.dmd2_latents encode --out "$LAT" \
      --shards "$SHARDS" --rank "$r" --world "${#GPU_IDS[@]}" --pack "$PACK" > "$LAT/logs/encode_$r.log" 2>&1 &
    pids+=($!)
  done
  info "logs: $LAT/logs/encode_*.log   (tail -f $LAT/logs/encode_0.log)"
  fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || { info "rank $i failed: $LAT/logs/encode_$i.log"; fail=1; }; done
  [[ $fail -eq 0 ]] || die "encoding incomplete; re-run the same command (finished ranks are kept)"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" crun python -m zimage_distill.dmd2_latents finalize --out "$LAT" \
    --world "${#GPU_IDS[@]}" || die "could not finalize the latents"
fi
crun python -c "
from zimage_distill.dmd2_latents import Latents
l = Latents('$LAT/latents'); x = l.get(l.available[:2]).float()
assert x.isfinite().all() and x.std() > 0.1, 'latents look wrong'
print(f'    read back: {l.available.size} latents, shape {tuple(x.shape[1:])}, std {x.std():.2f}')" \
  || die "the latents do not read back"

# 3 ------------------------------------------------------------------------------------------------
if (( UPLOAD )); then
  say "3/3 upload -> HF dataset $REPO_ID (public)"
  crun python - "$REPO_ID" <<'PY' || die "could not create $REPO_ID"
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
who = api.whoami()
if who.get("auth", {}).get("accessToken", {}).get("role") == "read":
    sys.exit("this is a Read token; uploading needs a Write token")
api.create_repo(sys.argv[1], repo_type="dataset", private=False, exist_ok=True)
print(f"    account {who['name']}: dataset {sys.argv[1]} ready")
PY
  # upload-large-folder mirrors a folder: hard-link what the training machine needs into one
  STAGE="$LAT/.upload"
  if [[ ! -f "$STAGE/negative_context.pt" ]]; then     # kept on a re-run: it holds the upload's resume state
    mkdir -p "$STAGE"
    cp -al "$LAT/latents" "$STAGE/latents"; ln "$LAT/negative_context.pt" "$STAGE/negative_context.pt"
  fi
  crun hf upload-large-folder "$REPO_ID" "$STAGE" --repo-type dataset --exclude ".cache/**" \
    || die "upload interrupted; re-run the same command to resume"
  say "done: $REPO_ID"
  info "training machine: bash z-image/scripts/dmd2_eight-cards.sh --root <root> --latents-hf $REPO_ID"
else
  say "done (not uploaded): $LAT"
fi
