#!/usr/bin/env bash
# Create the conda environment for RDM / SW-LMMD and install requirements.txt into it.
#
# Python 3.12 by default: RDM itself needs >=3.10, but the native flux2 package caps at
# <3.13 (flux2/pyproject.toml), so 3.12 is the newest usable interpreter. The machine's
# base conda env is 3.13 and CANNOT run the FLUX path.
#
# torch is installed FIRST, pinned, from the PyTorch CUDA index -- requirements.txt only
# says "torch>=2.4", which would pull whatever default-CUDA wheel PyPI serves. flux2 pins
# torch==2.8.0 / torchvision==0.23.0, so that is the default here. The torch/torchvision
# lines are then filtered out of requirements.txt so the bulk install cannot silently
# replace that pinned build with a different CUDA variant.
#
# SW-LMMD training also needs bitsandbytes (the 8-bit AdamW both configs/sw_lmmd_train_*.yaml
# use), pinned to the version verified against that torch; --no-sw-lmmd-deps skips it.
#
# Every pip install after torch goes through requirements-lock.txt: the exact versions of the
# env the results were produced with, so a new machine does not pick up a newer transformers /
# timm / diffusers that day. --no-lock installs the newest versions instead.
#
#   bash scripts/setup_env.sh                          # rdm / py3.12 / torch 2.8.0+cu126
#   bash scripts/setup_env.sh --dry-run                # print the plan, change nothing
#   bash scripts/setup_env.sh --name rdm312 --python 3.11
#   bash scripts/setup_env.sh --cuda cu128             # newer CUDA wheels
#   bash scripts/setup_env.sh --force                  # delete and rebuild the env
#
# Afterwards:
#   conda activate rdm
#   source <assets-root>/env.sh        # written by scripts/fetch_prerequisites.py
set -euo pipefail
export PYTHONNOUSERSITE=1   # packages in ~/.local/lib/pythonX.Y must never shadow the conda envs'

ENV_NAME="rdm"
PY_VER="3.12"
TORCH_VER="2.8.0"
TVISION_VER="0.23.0"
CUDA_TAG="cu126"
BNB_VER="0.50.2"
FORCE=0
DRY_RUN=0
EDITABLE=1
FLUX2_DEPS=1
DEV_DEPS=1
SW_LMMD_DEPS=1
SKIP_TORCH=0
USE_LOCK=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="$REPO_ROOT/requirements.txt"
LOCK_FILE="$REPO_ROOT/requirements-lock.txt"

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)        ENV_NAME="$2"; shift 2 ;;
    --python)      PY_VER="$2"; shift 2 ;;
    --torch)       TORCH_VER="$2"; shift 2 ;;
    --torchvision) TVISION_VER="$2"; shift 2 ;;
    --cuda)        CUDA_TAG="$2"; shift 2 ;;
    --requirements) REQ_FILE="$2"; shift 2 ;;
    --force)       FORCE=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --no-editable) EDITABLE=0; shift ;;
    --no-flux2-deps) FLUX2_DEPS=0; shift ;;
    --no-dev-deps) DEV_DEPS=0; shift ;;
    --no-sw-lmmd-deps) SW_LMMD_DEPS=0; shift ;;
    --skip-torch)  SKIP_TORCH=1; shift ;;
    --no-lock)     USE_LOCK=0; shift ;;
    -h|--help)     usage ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
run()  {
  info "\$ $*"
  if [[ $DRY_RUN -eq 0 ]]; then "$@"; fi
}

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
say "Preflight"
command -v conda >/dev/null 2>&1 || { echo "conda not found on PATH" >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
info "conda            $(conda --version) at $CONDA_BASE"
info "repo             $REPO_ROOT"
[[ -f "$REQ_FILE" ]] || { echo "requirements file not found: $REQ_FILE" >&2; exit 1; }
info "requirements     $REQ_FILE"
PIPC=()
if [[ $USE_LOCK -eq 1 ]]; then
  [[ -f "$LOCK_FILE" ]] || { echo "version lock not found: $LOCK_FILE (or pass --no-lock)" >&2; exit 1; }
  PIPC=(-c "$LOCK_FILE")
  info "version lock     $LOCK_FILE ($(grep -c "^[A-Za-z0-9_.-]*==" "$LOCK_FILE") pins)"
else
  info "version lock     OFF (--no-lock): newest versions"
fi

# flux2's <3.13 cap is the binding constraint; refuse to build an env that cannot run it.
PY_MAJOR="${PY_VER%%.*}"; PY_MINOR="${PY_VER#*.}"; PY_MINOR="${PY_MINOR%%.*}"
if [[ "$PY_MAJOR" -ne 3 || "$PY_MINOR" -lt 10 || "$PY_MINOR" -ge 13 ]]; then
  echo "python $PY_VER is outside the supported range: RDM needs >=3.10 and flux2 caps at <3.13." >&2
  echo "use --python 3.12 (recommended), 3.11, or 3.10." >&2
  exit 1
fi
info "python           $PY_VER"

if [[ $SKIP_TORCH -eq 0 ]]; then
  info "torch            $TORCH_VER + torchvision $TVISION_VER ($CUDA_TAG)"
fi
if [[ $SW_LMMD_DEPS -eq 1 ]]; then
  info "bitsandbytes     $BNB_VER (SW-LMMD 8-bit AdamW)"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  info "driver           $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  info "gpus             $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | tr '\n' ' ')"
else
  info "driver           nvidia-smi not found (CPU-only install?)"
fi

# A full CUDA torch env is ~10-12 GB and lands on whatever filesystem holds conda's envs
# dir -- which is often the small root volume, not the data disk the assets go to.
ENVS_DIR="$CONDA_BASE/envs"
FREE_GB=$(df -BG --output=avail "$CONDA_BASE" 2>/dev/null | tail -1 | tr -dc '0-9')
info "envs dir         $ENVS_DIR (${FREE_GB:-?} GB free)"
if [[ -n "${FREE_GB:-}" && "$FREE_GB" -lt 20 ]]; then
  info "WARNING: under 20 GB free for a ~10-12 GB env. Consider pointing the pip cache at a"
  info "         bigger volume:  export PIP_CACHE_DIR=/data/hulk/jiacheng/.pip-cache"
fi

ENV_EXISTS=0
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then ENV_EXISTS=1; fi

# ---------------------------------------------------------------------------
# create the environment
# ---------------------------------------------------------------------------
say "Environment '$ENV_NAME'"
if [[ $ENV_EXISTS -eq 1 && $FORCE -eq 1 ]]; then
  info "exists; --force -> removing"
  run conda env remove -n "$ENV_NAME" -y
  ENV_EXISTS=0
fi
if [[ $ENV_EXISTS -eq 1 ]]; then
  info "exists; reusing (pass --force to rebuild from scratch)"
else
  # conda-forge only: a fresh Miniconda (2025+) refuses the Anaconda default channels in a
  # non-interactive create until their Terms of Service are accepted (CondaToSNonInteractiveError).
  run conda create -n "$ENV_NAME" "python=$PY_VER" -y --override-channels -c conda-forge
fi

# `conda run` avoids needing `conda activate` inside a non-interactive shell.
CRUN=(conda run -n "$ENV_NAME" --no-capture-output)

# ---------------------------------------------------------------------------
# torch first, pinned, from the CUDA index
# ---------------------------------------------------------------------------
if [[ $SKIP_TORCH -eq 1 ]]; then
  say "torch: skipped (--skip-torch)"
else
  say "torch $TORCH_VER ($CUDA_TAG)"
  if [[ "$CUDA_TAG" == "cpu" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  else
    TORCH_INDEX="https://download.pytorch.org/whl/$CUDA_TAG"
  fi
  run "${CRUN[@]}" python -m pip install --upgrade pip "${PIPC[@]}"
  run "${CRUN[@]}" python -m pip install \
      "torch==$TORCH_VER" "torchvision==$TVISION_VER" --index-url "$TORCH_INDEX"
fi

# ---------------------------------------------------------------------------
# requirements.txt, minus torch/torchvision (already pinned above)
# ---------------------------------------------------------------------------
say "requirements.txt"
FILTERED="$(mktemp -t rdm-requirements-XXXXXX.txt)"
trap 'rm -f "$FILTERED"' EXIT
# drop comments, blank lines, and any torch/torchvision requirement
sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$REQ_FILE" \
  | grep -vEi '^(torch|torchvision)([[:space:]]*[<>=!~]|$)' > "$FILTERED"
info "installing $(wc -l < "$FILTERED") requirements (torch/torchvision filtered out):"
sed 's/^/      /' "$FILTERED"
run "${CRUN[@]}" python -m pip install -r "$FILTERED" "${PIPC[@]}"

# ---------------------------------------------------------------------------
# native flux2 runtime deps not covered by requirements.txt
# ---------------------------------------------------------------------------
if [[ $DEV_DEPS -eq 1 ]]; then
  say "dev extras"
  info "pytest lives in pyproject's [dev] extra, not requirements.txt -- the verification"
  info "step and the 'pytest tests -q' below need it."
  run "${CRUN[@]}" python -m pip install pytest "${PIPC[@]}"
fi

if [[ $FLUX2_DEPS -eq 1 ]]; then
  say "flux2 runtime deps"
  info "flux2 is used via FLUX2_SRC (path injection), NOT pip-installed: its package name is"
  info "'flux' and it hard-pins torch/transformers/safetensors, which would fight this env."
  info "Only its missing runtime imports are installed here."
  run "${CRUN[@]}" python -m pip install fire accelerate "${PIPC[@]}"
fi

if [[ $SW_LMMD_DEPS -eq 1 ]]; then
  say "SW-LMMD extras"
  info "bitsandbytes holds the AdamW moments in 8 bits (memory.optimizer: adamw8bit): 31 GB of"
  info "fp32 moments -> 7.8 GB, what lets fp32 master weights fit an 80 GB H100 and, sharded,"
  info "a 24 GB 4090. Pinned: $BNB_VER is the version verified with torch $TORCH_VER, and its"
  info "torch>=2.4,<3 requirement leaves the pinned CUDA build above untouched."
  run "${CRUN[@]}" python -m pip install "bitsandbytes==$BNB_VER" "${PIPC[@]}"
fi

# ---------------------------------------------------------------------------
# the repo itself
# ---------------------------------------------------------------------------
if [[ $EDITABLE -eq 1 ]]; then
  say "rdm (editable)"
  run "${CRUN[@]}" python -m pip install -e "$REPO_ROOT" --no-deps
  info "--no-deps: requirements.txt above is the source of truth for dependencies"
fi

# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
say "Verification"
if [[ $DRY_RUN -eq 1 ]]; then
  info "--dry-run: nothing was executed; no environment was created or modified."
else
  SW_LMMD_DEPS=$SW_LMMD_DEPS "${CRUN[@]}" python - <<'PY'
import importlib, os, sys
print(f"    python           {sys.version.split()[0]}")
ok = True
try:
    import torch
    print(f"    torch            {torch.__version__}  cuda={torch.version.cuda}  "
          f"devices={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"    gpu0             {torch.cuda.get_device_name(0)} "
              f"(sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})")
    else:
        print("    gpu              CUDA NOT available -- check the driver / --cuda tag")
except Exception as e:
    ok = False
    print(f"    torch            FAILED: {type(e).__name__}: {e}")

for mod in ("numpy", "scipy", "timm", "open_clip", "transformers", "diffusers",
            "safetensors", "einops", "huggingface_hub", "yaml"):
    try:
        m = importlib.import_module(mod)
        print(f"    {mod:16s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        ok = False
        print(f"    {mod:16s} FAILED: {type(e).__name__}")

if os.environ.get("SW_LMMD_DEPS") == "1":             # required by the SW-LMMD configs
    try:
        import bitsandbytes as bnb
        bnb.optim.AdamW8bit                            # the optimizer adamw8bit resolves to
        print(f"    {'bitsandbytes':16s} {bnb.__version__}")
    except Exception as e:
        ok = False
        print(f"    {'bitsandbytes':16s} FAILED: {type(e).__name__}: {e}")

if ok and torch.cuda.is_available():                   # the libraries, not just the import:
    try:                                               # a foreign LD_LIBRARY_PATH cuDNN/cuBLAS
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)   # fails only here
        float((x @ x).sum())
        conv = torch.nn.Conv2d(3, 8, 3).cuda()
        float(conv(torch.randn(1, 3, 32, 32, device="cuda")).sum())
        print(f"    {'cuda smoke':16s} bf16 matmul + conv ok (cudnn {torch.backends.cudnn.version()})")
        if os.environ.get("SW_LMMD_DEPS") == "1":
            import bitsandbytes as bnb
            p = torch.nn.Parameter(torch.randn(4096, device="cuda"))     # 4096: 8-bit moments
            opt = bnb.optim.AdamW8bit([p], lr=1e-3)
            p.grad = torch.randn_like(p)
            opt.step()
            assert opt.state[p]["state1"].dtype == torch.uint8
            print(f"    {'bnb smoke':16s} 8-bit AdamW step ok")
    except Exception as e:
        ok = False
        print(f"    {'cuda smoke':16s} FAILED: {type(e).__name__}: {e}")

for mod in ("dreamsim", "wandb"):                      # optional extras
    try:
        importlib.import_module(mod)
        print(f"    {mod:16s} ok (optional)")
    except Exception:
        print(f"    {mod:16s} absent (optional)")

try:
    import rdm
    print("    rdm              importable")
except Exception as e:
    ok = False
    print(f"    rdm              FAILED: {type(e).__name__}: {e}")

sys.exit(0 if ok else 1)
PY
  VERIFY_RC=$?
  if [[ ${VERIFY_RC:-0} -ne 0 ]]; then
    echo
    echo "Some imports failed -- see above. Re-run with --force to rebuild cleanly." >&2
  fi
fi

# ---------------------------------------------------------------------------
say "Next"
cat <<EOF
    conda activate $ENV_NAME

    # 1. assets (weights, COCO, flux2 source) -- separate script, separate disk budget
    #    (scripts/new_machine.sh runs this and the steps below for you)
    bash scripts/download_all.sh --root <assets-root> --dry-run

    # 2. point the caches + FLUX2_SRC at that root, then check flux2 resolves
    source <assets-root>/env.sh
    python -c "import flux2; print('flux2 ok')"

    # 3. repo test suite (no downloaded weights needed)
    pytest tests -q
EOF
