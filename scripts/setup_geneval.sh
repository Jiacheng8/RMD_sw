#!/usr/bin/env bash
# GenEval scorer environment: the official djghosh13/geneval scorer, built the canonical way
# (docs/geneval_protocol.md) -- mmdet 3.3.0 Mask2Former Swin-S + the open_clip ViT-L-14 colour
# classifier -- in a conda env of its own. The rdm env renders the images; this env only scores.
#
#   bash scripts/setup_geneval.sh --root /data/<you>/geneval               # env "geneval"
#   bash scripts/setup_geneval.sh --root <dir> --prefix <dir>/env          # env at a path (off a full /)
#   bash scripts/setup_geneval.sh --root <dir> --dry-run                   # print the plan, change nothing
#   bash scripts/setup_geneval.sh --root <dir> --build-mmcv                # force the mmcv source build
#   GENEVAL_CUDA_ARCH_LIST="8.9;9.0" bash scripts/setup_geneval.sh ...     # build for these archs
#   bash scripts/setup_geneval.sh --root <dir> --force                     # rebuild the env from scratch
# then score a render dir (<dir>/<idx:05d>/{metadata.jsonl, samples/NNNN.png}):
#   bash scripts/score_geneval.sh <render_dir> --root <dir>
#
# Steps -- idempotent, a re-run skips what is already done:
#   1 conda env, python 3.10: torch 2.1.2+cu121, mmcv 2.1.0, mmdet 3.3.0, open_clip, clip-benchmark,
#     all pinned through one constraints file so no later install can swap torch or numpy.
#   2 mmcv. OpenMMLab's prebuilt wheel carries kernels for sm_50..sm_86 and NO PTX: it runs on
#     Ampere/Ada (A100, 3090, 4090 -- same major arch 8) and fails with "no kernel image" on Hopper
#     (H100, sm_90). For compute capability >= 9 it is built from source for the local GPUs, with a
#     conda CUDA 12.1 nvcc (and a conda gcc 11 when the system gcc is newer than 12, e.g. Ubuntu
#     24.04). Either way the CUDA op is then checked against its pure-PyTorch reference on this GPU.
#   3 the scorer: djghosh13/geneval at a pinned commit; evaluation/evaluate_images.py is ported to
#     the mmdet 3.x result API (pred_instances) -- upstream unpacks the mmdet 2.x tuple.
#   4 the detector: Mask2Former Swin-S COCO, the mmdet v3.0 checkpoint (..._001756-c9d0c4f2.pth),
#     sha256-checked, as <root>/models/mask2former.pth.
#   5 self-test: the full scorer on a COCO photo with known content (two cats, two remotes).
#
# Every pip install after torch goes through requirements-geneval-lock.txt -- the exact versions the
# reference scores in run.md were produced with; --no-lock installs the newest compatible ones.
#
# Disk: ~7 GB env (+~3 GB CUDA toolkit for a source build) + 0.3 GB detector + 1.7 GB CLIP.
# Needs an NVIDIA GPU (the scorer asserts CUDA) and internet access.
set -euo pipefail
export PYTHONNOUSERSITE=1   # packages in ~/.local/lib/pythonX.Y must never shadow the conda envs'

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=""
ENV_NAME="geneval"
PREFIX=""
BUILD_MMCV=0
FORCE=0
DRY=0
SELFTEST=1
USE_LOCK=1
LOCK_FILE="$REPO/requirements-geneval-lock.txt"

GENEVAL_GIT="https://github.com/djghosh13/geneval.git"
GENEVAL_COMMIT="af4902f24d3ca90ebbb446dd9891a59e0f82725f"
DET_URL="https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth"
DET_SHA256_PREFIX="c9d0c4f2"            # OpenMMLab names checkpoints after their sha256 prefix
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html"
CUDA_CHANNEL="nvidia/label/cuda-12.1.1"
SELFTEST_IMAGE="http://images.cocodataset.org/val2017/000000039769.jpg"   # two cats, two remotes

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)          ROOT="$2"; shift 2 ;;
    --name)          ENV_NAME="$2"; shift 2 ;;
    --prefix)        PREFIX="$2"; shift 2 ;;
    --build-mmcv)    BUILD_MMCV=1; shift ;;
    --force)         FORCE=1; shift ;;
    --dry-run)       DRY=1; shift ;;
    --skip-selftest) SELFTEST=0; shift ;;
    --no-lock)       USE_LOCK=0; shift ;;
    -h|--help)       usage ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[[ -n "$ROOT" ]] || { echo "usage: bash scripts/setup_geneval.sh --root <dir> [options]  (see --help)" >&2; exit 2; }
ROOT="$(realpath -m "$ROOT")"
[[ -n "$PREFIX" ]] && PREFIX="$(realpath -m "$PREFIX")"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    \033[33mWARNING:\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
run()  { info "\$ $*"; if [[ $DRY -eq 0 ]]; then "$@"; fi; }

if [[ -n "$PREFIX" ]]; then ENV_ARGS=(-p "$PREFIX"); ENV_DESC="$PREFIX"
else                        ENV_ARGS=(-n "$ENV_NAME"); ENV_DESC="$ENV_NAME"; fi
CRUN=(conda run "${ENV_ARGS[@]}" --no-capture-output)
env_exists() {
  if [[ -n "$PREFIX" ]]; then [[ -x "$PREFIX/bin/python" ]]
  else conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; fi
}

# ---------------------------------------------------------------------------
say "preflight"
command -v conda >/dev/null 2>&1 || die "conda not found on PATH"
command -v git   >/dev/null 2>&1 || die "git not found on PATH"
command -v curl  >/dev/null 2>&1 || die "curl not found on PATH"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found -- the GenEval scorer needs an NVIDIA GPU"
mapfile -t CAPS < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -u)
[[ ${#CAPS[@]} -gt 0 ]] || die "no GPU visible to nvidia-smi"
info "root        $ROOT"
info "env         $ENV_DESC"
info "gpus        $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | tr -s ' ' | tr '\n' ';') compute capability ${CAPS[*]}"
NEED_SOURCE=$BUILD_MMCV
for c in "${CAPS[@]}"; do [[ "${c%%.*}" -ge 9 ]] && NEED_SOURCE=1; done
# The archs a source build compiles for: this machine's GPUs, unless overridden -- e.g.
# GENEVAL_CUDA_ARCH_LIST="8.9;9.0" for one env that serves both a 4090 and an H100.
ARCH_LIST="${GENEVAL_CUDA_ARCH_LIST:-$(IFS=';'; echo "${CAPS[*]}")}"
if [[ $NEED_SOURCE -eq 1 ]]; then
  info "mmcv        source build for TORCH_CUDA_ARCH_LIST=$ARCH_LIST (the prebuilt wheel stops at sm_86)"
else
  info "mmcv        OpenMMLab prebuilt wheel (covers compute capability ${CAPS[*]})"
fi
d="$ROOT"; while [[ ! -d "$d" ]]; do d="$(dirname "$d")"; done
info "free space  $(df -BG --output=avail "$d" | tail -1 | tr -dc '0-9') GB at the root"

# ---------------------------------------------------------------------------
say "1/5 conda env ($ENV_DESC)"
if env_exists && [[ $FORCE -eq 1 ]]; then
  run conda env remove "${ENV_ARGS[@]}" -y
fi
if env_exists; then
  info "exists; reusing (--force rebuilds it)"
else
  # conda-forge only: a fresh Miniconda (2025+) refuses the Anaconda default channels in a
  # non-interactive create until their Terms of Service are accepted (CondaToSNonInteractiveError).
  run conda create "${ENV_ARGS[@]}" -y -q --override-channels -c conda-forge python=3.10
fi

# One constraints file for every pip call below: nothing may replace torch, numpy or mmcv.
CONSTRAINTS="$ROOT/.geneval_constraints.txt"
if [[ $DRY -eq 0 ]]; then
  mkdir -p "$ROOT"
  cat > "$CONSTRAINTS" <<'EOF'
torch==2.1.2
torchvision==0.16.2
numpy==1.26.4
mmcv==2.1.0
mmengine==0.10.4
mmdet==3.3.0
EOF
fi
PIP=("${CRUN[@]}" python -m pip install -c "$CONSTRAINTS")
# After torch (which comes from the PyTorch index alone), everything also goes through the lock.
PIPL=("${PIP[@]}")
if [[ $USE_LOCK -eq 1 ]]; then
  [[ -f "$LOCK_FILE" ]] || { echo "version lock not found: $LOCK_FILE (or pass --no-lock)" >&2; exit 1; }
  PIPL+=(-c "$LOCK_FILE")
fi
run "${CRUN[@]}" python -m pip install --upgrade "pip<25" "setuptools<70" wheel
run "${PIP[@]}" torch==2.1.2 torchvision==0.16.2 --index-url "$TORCH_INDEX"
run "${PIPL[@]}" numpy==1.26.4 mmengine==0.10.4 ninja psutil
# clip-benchmark 1.6.1, not upstream's 1.4.0: 1.4.0 pins torch<2. The two functions GenEval calls
# (zero_shot_classifier with a template LIST, run_classification) are the same code in both.
run "${PIPL[@]}" open-clip-torch==3.3.0 clip-benchmark==1.6.1 transformers==4.40.2 \
    pandas==2.2.2 pillow

# ---------------------------------------------------------------------------
say "2/5 mmcv 2.1.0"
MMCV_OPTEST=$(cat <<'PY'
import sys, torch, mmcv
from mmcv.ops.multi_scale_deform_attn import (MultiScaleDeformableAttnFunction,
                                              multi_scale_deformable_attn_pytorch)
cap = torch.cuda.get_device_capability()
torch.manual_seed(0)
shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
start = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
S = int(sum(int(h) * int(w) for h, w in shapes))
value = torch.rand(1, S, 2, 4).cuda() * 0.01
loc = torch.rand(1, 3, 2, 2, 2, 2).cuda()
w = torch.rand(1, 3, 2, 2, 2).cuda() + 1e-5
w /= w.sum(-1, keepdim=True).sum(-2, keepdim=True)
ref = multi_scale_deformable_attn_pytorch(value, shapes, loc, w)
out = MultiScaleDeformableAttnFunction.apply(value, shapes, start, loc, w, 2)
torch.cuda.synchronize()
err = (out - ref).abs().max().item()
print(f"    mmcv {mmcv.__version__} CUDA op on sm_{cap[0]}{cap[1]}: max |cuda - reference| = {err:.2e}")
sys.exit(0 if err < 1e-4 else 1)
PY
)
mmcv_ok() { "${CRUN[@]}" python -c "$MMCV_OPTEST" >/dev/null 2>&1; }
if [[ $DRY -eq 0 ]] && [[ $FORCE -eq 0 && $BUILD_MMCV -eq 0 ]] && mmcv_ok; then
  info "installed and its CUDA op runs on this GPU; skipping"
elif [[ $NEED_SOURCE -eq 0 ]]; then
  run "${PIPL[@]}" mmcv==2.1.0 -f "$MMCV_INDEX"
else
  # Source build: nvcc must be CUDA 12 like torch's cu121 build, and nvcc 12.1 accepts gcc <= 12.
  run conda install "${ENV_ARGS[@]}" -y -q --override-channels -c "$CUDA_CHANNEL" -c conda-forge \
      cuda-nvcc cuda-libraries-dev cuda-cudart-dev cuda-cccl
  GCC_MAJOR="$(gcc -dumpversion 2>/dev/null | cut -d. -f1 || true)"
  BUILD_ENV=()
  if [[ -z "$GCC_MAJOR" || "$GCC_MAJOR" -gt 12 || -n "${GENEVAL_CONDA_GCC:-}" ]]; then
    if [[ -n "${GENEVAL_CONDA_GCC:-}" ]]; then
      info "GENEVAL_CONDA_GCC set -> conda gcc/g++ 11"
    else
      info "system gcc '${GCC_MAJOR:-none}' is not usable with nvcc 12.1 (needs <= 12) -> conda gcc/g++ 11"
    fi
    run conda install "${ENV_ARGS[@]}" -y -q --override-channels -c conda-forge "gcc_linux-64=11" "gxx_linux-64=11"
    USE_CONDA_GCC=1
  else
    info "system gcc $GCC_MAJOR"
    USE_CONDA_GCC=0
  fi
  if [[ $DRY -eq 0 ]]; then
    EP="$("${CRUN[@]}" python -c 'import sys; print(sys.prefix)' | tail -1)"
    BUILD_ENV=(CUDA_HOME="$EP" PATH="$EP/bin:$PATH" TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
               MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS="$(( $(nproc) > 32 ? 32 : $(nproc) ))")
    if [[ $USE_CONDA_GCC -eq 1 ]]; then
      BUILD_ENV+=(CC="$EP/bin/x86_64-conda-linux-gnu-gcc" CXX="$EP/bin/x86_64-conda-linux-gnu-g++")
    fi
    info "building mmcv 2.1.0 from source for $ARCH_LIST (10-30 min)"
    env "${BUILD_ENV[@]}" "${PIPL[@]}" --no-build-isolation --no-binary mmcv --no-cache-dir mmcv==2.1.0
  else
    info "\$ ${PIPL[*]} --no-build-isolation --no-binary mmcv mmcv==2.1.0   (TORCH_CUDA_ARCH_LIST=$ARCH_LIST)"
  fi
fi
run "${PIPL[@]}" mmdet==3.3.0
if [[ $DRY -eq 0 ]]; then
  "${CRUN[@]}" python -c "$MMCV_OPTEST" \
    || die "mmcv's CUDA op does not run on this GPU (see above). Re-run with --build-mmcv."
fi

# ---------------------------------------------------------------------------
say "3/5 the GenEval scorer (djghosh13/geneval @ ${GENEVAL_COMMIT:0:7})"
SRC="$ROOT/geneval"
if [[ -d "$SRC/.git" ]]; then
  info "clone present"
else
  run git clone -q "$GENEVAL_GIT" "$SRC"
fi
if [[ $DRY -eq 0 ]]; then
  if [[ "$(git -C "$SRC" rev-parse HEAD)" != "$GENEVAL_COMMIT" ]]; then
    git -C "$SRC" diff --quiet -- evaluation/evaluate_images.py \
      || git -C "$SRC" checkout -q -- evaluation/evaluate_images.py    # drop our port before switching
    git -C "$SRC" fetch -q origin "$GENEVAL_COMMIT" 2>/dev/null || git -C "$SRC" fetch -q origin
    git -C "$SRC" checkout -q "$GENEVAL_COMMIT"
  fi
  info "at $(git -C "$SRC" rev-parse --short HEAD)"
  # mmdet 3.x returns a DetDataSample, not mmdet 2.x's (per-class boxes, per-class masks) tuple.
  # Convert it into exactly that shape, so every line of upstream's scoring logic stays as is.
  "${CRUN[@]}" python - "$SRC/evaluation/evaluate_images.py" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()
MARK = "# rdm: mmdet-3 port"
if MARK in src:
    print("    evaluate_images.py already ported"); sys.exit(0)
old = '''def evaluate_image(filepath, metadata):
    result = inference_detector(object_detector, filepath)
    bbox = result[0] if isinstance(result, tuple) else result
    segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
'''
new = f'''def _per_class(result, num_classes):
    {MARK}: DetDataSample.pred_instances -> the mmdet-2.x per-class
    # [x1, y1, x2, y2, score] arrays and per-class mask stacks that the code below indexes.
    inst = result.pred_instances
    boxes = np.concatenate([inst.bboxes.cpu().numpy(), inst.scores.cpu().numpy()[:, None]], axis=1)
    labels = inst.labels.cpu().numpy()
    masks = inst.masks.cpu().numpy() if "masks" in inst else None
    bbox = [boxes[labels == c] for c in range(num_classes)]
    segm = None if masks is None else [masks[labels == c] for c in range(num_classes)]
    return bbox, segm


def evaluate_image(filepath, metadata):
    result = inference_detector(object_detector, filepath)
    if hasattr(result, "pred_instances"):
        bbox, segm = _per_class(result, len(classnames))
    else:
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
'''
if src.count(old) != 1:
    sys.exit("evaluate_images.py does not match the pinned upstream -- refusing to patch")
open(path, "w").write(src.replace(old, new))
print("    evaluate_images.py ported to the mmdet 3.x result API")
PY
else
  info "would pin $GENEVAL_COMMIT and port evaluation/evaluate_images.py to mmdet 3.x"
fi

# ---------------------------------------------------------------------------
say "4/5 detector: Mask2Former Swin-S (COCO), mmdet v3.0 checkpoint"
DET="$ROOT/models/mask2former.pth"
det_ok() { [[ -f "$DET" ]] && sha256sum "$DET" | cut -c1-8 | grep -qx "$DET_SHA256_PREFIX"; }
if [[ $DRY -eq 1 ]]; then
  info "would download ${DET_URL##*/} -> $DET"
elif det_ok; then
  info "present, sha256 ${DET_SHA256_PREFIX}... ok"
else
  mkdir -p "$ROOT/models"
  curl -fL --retry 3 -o "$DET.part" "$DET_URL"
  mv "$DET.part" "$DET"
  det_ok || { rm -f "$DET"; die "detector checksum mismatch (expected sha256 ${DET_SHA256_PREFIX}...)"; }
  info "downloaded, sha256 ${DET_SHA256_PREFIX}... ok"
fi

# Record where everything is, for scripts/score_geneval.sh.
if [[ $DRY -eq 0 ]]; then
  MMDET_CFG="$("${CRUN[@]}" python -c 'import os, mmdet; print(os.path.join(os.path.dirname(mmdet.__file__), ".mim/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"))' | tail -1)"
  [[ -f "$MMDET_CFG" ]] || die "mmdet's builtin Mask2Former config is missing: $MMDET_CFG"
  cat > "$ROOT/geneval_env.sh" <<EOF
# Written by scripts/setup_geneval.sh -- sourced by scripts/score_geneval.sh.
GENEVAL_ENV_ARGS=(${ENV_ARGS[*]})
GENEVAL_SRC="$SRC"
GENEVAL_MODELS="$ROOT/models"
GENEVAL_MMDET_CONFIG="$MMDET_CFG"
GENEVAL_HF_HOME="$ROOT/hf"
EOF
  info "wrote $ROOT/geneval_env.sh"
fi

# ---------------------------------------------------------------------------
say "5/5 self-test"
if [[ $DRY -eq 1 || $SELFTEST -eq 0 ]]; then
  info "skipped"
else
  T="$ROOT/selftest"
  rm -rf "$T"; mkdir -p "$T/img"
  curl -fsL --retry 3 -o "$T/cats.jpg" "$SELFTEST_IMAGE" || die "could not fetch the self-test image"
  # expected: [correct?, prompt spec]. Class names are GenEval's (object_names.txt: "tv remote",
  # not COCO's "remote"). The colour row exercises the CLIP / mask path without asserting a colour.
  "${CRUN[@]}" python - "$T" "$SRC" <<'PY'
import json, os, sys
from PIL import Image
T, SRC = sys.argv[1], sys.argv[2]
# The scorer maps detector label i to object_names.txt line i: both must be COCO's 80 classes in
# order, differing only in GenEval's three renames.
from mmdet.datasets import CocoDataset
coco = list(CocoDataset.METAINFO["classes"])
names = [l.strip() for l in open(os.path.join(SRC, "evaluation", "object_names.txt"))]
renamed = {i for i, (a, b) in enumerate(zip(coco, names)) if a != b}
if len(coco) != len(names) or renamed != {64, 65, 66}:
    sys.exit(f"detector classes and object_names.txt are not aligned (differ at {sorted(renamed)})")
print("    detector labels align with object_names.txt (80 COCO classes)")
cases = [
    (True,  {"tag": "counting", "include": [{"class": "cat", "count": 2}], "exclude": [{"class": "cat", "count": 3}], "prompt": "a photo of two cats"}),
    (True,  {"tag": "two_object", "include": [{"class": "cat", "count": 1}, {"class": "tv remote", "count": 1}], "prompt": "a photo of a cat and a tv remote"}),
    (True,  {"tag": "position", "include": [{"class": "tv remote", "count": 1}, {"class": "cat", "count": 1, "position": ["right of", 0]}], "prompt": "a photo of a cat right of a tv remote"}),
    (False, {"tag": "single_object", "include": [{"class": "dog", "count": 1}], "prompt": "a photo of a dog"}),
    (False, {"tag": "counting", "include": [{"class": "cat", "count": 4}], "exclude": [{"class": "cat", "count": 5}], "prompt": "a photo of four cats"}),
    (False, {"tag": "position", "include": [{"class": "tv remote", "count": 1}, {"class": "cat", "count": 1, "position": ["left of", 0]}], "prompt": "a photo of a cat left of a tv remote"}),
    (None,  {"tag": "colors", "include": [{"class": "couch", "count": 1, "color": "pink"}], "prompt": "a photo of a pink couch"}),
]
img = Image.open(os.path.join(T, "cats.jpg")).convert("RGB")
for i, (_, meta) in enumerate(cases):
    d = os.path.join(T, "img", f"{i:05d}", "samples")
    os.makedirs(d, exist_ok=True)
    img.save(os.path.join(d, "0000.png"))
    json.dump(meta, open(os.path.join(T, "img", f"{i:05d}", "metadata.jsonl"), "w"))
json.dump([c[0] for c in cases], open(os.path.join(T, "expected.json"), "w"))
PY
  bash "$REPO/scripts/score_geneval.sh" "$T/img" --root "$ROOT" --out "$T/results.jsonl" >"$T/score.log" 2>&1 \
    || { tail -30 "$T/score.log"; die "self-test: the scorer failed (log: $T/score.log)"; }
  "${CRUN[@]}" python - "$T" <<'PY' || die "self-test: unexpected scorer output (see above)"
import json, os, sys
T = sys.argv[1]
expected = json.load(open(os.path.join(T, "expected.json")))
rows = {}
for line in open(os.path.join(T, "results.jsonl")):
    r = json.loads(line)
    rows[int(r["filename"].split(os.sep)[-3])] = r
bad = 0
for i, want in enumerate(expected):
    r = rows.get(i)
    if r is None:
        print(f"    FAIL case {i}: no result row"); bad += 1; continue
    got = bool(r["correct"])
    ok = want is None or got == want
    bad += not ok
    print(f"    {'ok  ' if ok else 'FAIL'} {r['prompt']!r:40s} correct={got}" + ("" if want is not None else "  (not asserted)")
          + ("" if ok else f"  expected {want}; reason: {r['reason']}"))
sys.exit(1 if bad else 0)
PY
  info "self-test passed"
fi

say "done"
cat <<EOF
    score a render dir:  bash scripts/score_geneval.sh <render_dir> --root $ROOT
    render dir layout:   <dir>/<idx:05d>/{metadata.jsonl, samples/NNNN.png}  (rdm: reproduce.py eval-flux)
EOF
