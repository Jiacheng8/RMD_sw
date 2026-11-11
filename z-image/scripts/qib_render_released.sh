#!/usr/bin/env bash
# Qwen-Image-Bench images of two released Z-Image-Turbo distillations, one command on a fresh machine:
# environment, code, weights, 2 x 1000 renders, checks, and (optional) upload. No judging.
#
#   twinflow-zimage-turbo-exp_2nfe_1024   inclusionAI/TwinFlow-Z-Image-Turbo (exp), 2 NFE, aux_time_embed
#   didr-zimage_1nfe_1024                 Junyi-W/DIDR zimage_transformer, 1 NFE (diffusers ZImagePipeline)
#
# Same protocol as z-image/eval_qib/generate.py (our zimage-2step-1024_s500, turbo9, dmd2, ...):
# prompt_cn of the 1000 prompts, 1 image each, 1024 px, bf16, seed 0, latent of prompt ID =
# torch.randn fp32 of Generator('cpu').manual_seed(ID), re-noise of step k = manual_seed(1000k*1000003+ID),
# prompts encoded first (512 tokens), images/{ID:04d}.png, input.jsonl (ID, prompt, image_path).
# Everything (env, versions, weights, revisions) is pinned to the run uploaded on 2026-11-10, whose
# images it reproduces bit for bit on RTX 4090.
#
#   bash qib_render_released.sh --root /data/qib_released --dry-run    # checks and the plan only
#   bash qib_render_released.sh --root /data/qib_released --setup-only # env + downloads, no GPU
#   bash qib_render_released.sh --root /data/qib_released              # everything
#   bash qib_render_released.sh --root /data/qib_released --limit 8    # smoke test, runs <name>_limit8
#   bash qib_render_released.sh --root /data/qib_released --upload --suffix _rerun
#
# Run inside tmux. Run again after an interruption: finished stages and images are kept.
#
# --root DIR      everything goes here (default ./qib_released): conda env, HF cache, code, runs;
#                 needs ~60 GB free
# --gpus IDS      cards to use, one process each, e.g. 0,1,2,3 (default all visible); >= 24 GB each
# --models LIST   twinflow,didr (default both)
# --limit N       first N prompts only; outputs are named <name>_limitN and are not for scoring
# --upload        upload the finished runs to --repo as qwen-image-bench/<name><suffix>; never
#                 overwrites: an existing folder of that name stops the upload
# --suffix S      appended to the uploaded folder names (both original names already exist on the Hub)
# --repo REPO     HF dataset (default jiachengcui888/evaluation-result)
# --dry-run       checks and the plan, change nothing
# --setup-only    environment, code, weights and prompts only; needs no GPU (e.g. on a login node
#                 with internet). Then run without it on a GPU node: rendering runs offline.
#
# Needs: NVIDIA driver >= 525 (CUDA 12.6 wheels), git or curl, internet. conda is installed under
# --root if missing. Behind a mirror: export HF_ENDPOINT=https://hf-mirror.com (and PIP_INDEX_URL)
# first. --upload needs a HF write token: $HF_TOKEN, else asked once (hidden, never stored).
# Log: <root>/qib_render_released.log
set -euo pipefail

ROOT="$PWD/qib_released"; GPUS=""; MODELS="twinflow,didr"; LIMIT=0; UPLOAD=0; SUFFIX=""
REPO="jiachengcui888/evaluation-result"; DRY=0; SETUP_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)    ROOT="$2"; shift 2 ;;
    --gpus)    GPUS="$2"; shift 2 ;;
    --models)  MODELS="$2"; shift 2 ;;
    --limit)   LIMIT="$2"; shift 2 ;;
    --upload)  UPLOAD=1; shift ;;
    --suffix)  SUFFIX="$2"; shift 2 ;;
    --repo)    REPO="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --setup-only) SETUP_ONLY=1; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- pins (the 2026-11-10 run) --------------------------------------------------------------------
PY_VER=3.12
TORCH="torch==2.8.0"; TORCH_INDEX="https://download.pytorch.org/whl/cu126"
PIPS=(diffusers==0.40.0 transformers==5.17.0 huggingface_hub==1.33.0 hf_xet==1.6.0 safetensors==0.8.0
      accelerate==1.15.0 tokenizers==0.23.2 numpy==2.5.2 pillow==12.3.0 pyyaml==6.0.3 jinja2==3.1.6)
TWINFLOW_DIFFUSERS="diffusers==0.36.0"   # TwinFlow's ZImage wrapper passes the constructor args by
                                         # position; 0.40 inserted siglip_feat_dim, so it needs 0.36
TWINFLOW_GIT="https://github.com/inclusionAI/TwinFlow"; TWINFLOW_COMMIT="a109a71b19634d9c5d5c45c3587d36b19f014917"

ROOT="$(realpath -m "$ROOT")"
ENV="$ROOT/env"; OVERLAY="$ROOT/diffusers036"; CODE="$ROOT/code"; RUNS="$ROOT/runs"
export HF_HOME="$ROOT/hf" PIP_CACHE_DIR="$ROOT/.cache/pip" PYTHONNOUSERSITE=1
IFS=, read -r -a MODEL_LIST <<< "$MODELS"
for m in "${MODEL_LIST[@]}"; do [[ "$m" == twinflow || "$m" == didr ]] || die "--models: twinflow and/or didr, got $m"; done
name_of() { local n; [[ "$1" == twinflow ]] && n=twinflow-zimage-turbo-exp_2nfe_1024 || n=didr-zimage_1nfe_1024
            (( LIMIT )) && n="${n}_limit$LIMIT"; echo "$n"; }

# ---- checks -----------------------------------------------------------------------------------------
if (( SETUP_ONLY )); then GG=(none); else
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: no NVIDIA driver? (--setup-only needs none)"
[[ -n "$GPUS" ]] || GPUS="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | paste -sd, -)}"
IFS=, read -r -a GG <<< "$GPUS"
(( ${#GG[@]} )) || die "no GPUs"
for g in "${GG[@]}"; do
  mem=$(nvidia-smi -i "$g" --query-gpu=memory.total --format=csv,noheader,nounits | tr -dc 0-9)
  (( mem >= 23000 )) || die "GPU $g has $mem MiB; each process needs a 24 GB card"
done
fi
mkdir -p "$ROOT"
NEED=60; [[ -f "$ENV/.ready" ]] && NEED=$(( NEED - 10 )); [[ -f "$HF_HOME/.ready" ]] && NEED=$(( NEED - 42 ))
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
info "root $ROOT | GPUs ${GG[*]} | models ${MODEL_LIST[*]} | disk ~$NEED GB needed, $FREE GB free"
(( FREE >= NEED )) || die "not enough space under $ROOT: choose another --root"
if (( UPLOAD )); then
  (( LIMIT == 0 )) || die "--upload with --limit: smoke-test runs are not uploaded"
  [[ -n "$SUFFIX" ]] || info "no --suffix: the upload stops if $REPO already has these names (it does for the 2026-11-10 run)"
fi
if (( DRY )); then
  say "dry run: nothing changed"
  [[ -f "$ENV/.ready" ]] && info "env ready: $ENV" || info "would create env $ENV (python $PY_VER, $TORCH cu126, ${PIPS[*]}) + $TWINFLOW_DIFFUSERS overlay"
  info "would fetch TwinFlow @ ${TWINFLOW_COMMIT:0:7}, weights and prompts into $HF_HOME"
  (( SETUP_ONLY )) && exit 0
  for m in "${MODEL_LIST[@]}"; do info "would render $RUNS/$(name_of "$m") on GPUs ${GG[*]}"; done
  (( UPLOAD )) && for m in "${MODEL_LIST[@]}"; do info "would upload to $REPO: qwen-image-bench/$(name_of "$m")$SUFFIX"; done
  exit 0
fi
if (( UPLOAD && ! SETUP_ONLY )) && [[ -z "${HF_TOKEN:-}" ]]; then
  [[ -t 0 ]] || die "no terminal to ask for the token: export HF_TOKEN=<write token> first"
  read -rs -p "HF write token (for $REPO): " HF_TOKEN; echo; export HF_TOKEN   # lives only in this process
fi
exec > >(tee -a "$ROOT/qib_render_released.log") 2>&1
say "start $(date '+%F %T') on $(hostname)"

# ---- 1. environment -----------------------------------------------------------------------------------
CONDA=""
if command -v conda >/dev/null 2>&1; then CONDA="$(command -v conda)"
elif [[ -x "$ROOT/miniforge/bin/conda" ]]; then CONDA="$ROOT/miniforge/bin/conda"
else
  say "conda not found: Miniforge -> $ROOT/miniforge"
  inst="$ROOT/.cache/Miniforge3.sh"; mkdir -p "$ROOT/.cache"
  curl -fsSL -o "$inst" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" \
    || die "could not download Miniforge"
  bash "$inst" -b -p "$ROOT/miniforge" >/dev/null; CONDA="$ROOT/miniforge/bin/conda"
fi
PY="$ENV/bin/python"
if [[ ! -f "$ENV/.ready" ]]; then
  say "environment -> $ENV"
  [[ -x "$PY" ]] || "$CONDA" create -y -q -p "$ENV" -c conda-forge --override-channels "python=$PY_VER" pip >/dev/null \
    || die "conda create failed"
  PIP=(env -u PYTHONPATH "$PY" -m pip install -q)
  "${PIP[@]}" "$TORCH" --index-url "$TORCH_INDEX" || die "torch install failed"
  "${PIP[@]}" "${PIPS[@]}" || die "pip install failed"
  rm -rf "$OVERLAY"; "${PIP[@]}" --no-deps --target "$OVERLAY" "$TWINFLOW_DIFFUSERS" || die "diffusers 0.36 overlay failed"
  touch "$ENV/.ready"
fi
env -u PYTHONPATH SETUP_ONLY=$SETUP_ONLY "$PY" -c "import os, torch, diffusers, transformers; assert os.environ['SETUP_ONLY'] == '1' or torch.cuda.is_available(), 'CUDA not usable (driver too old?)'; print('    ok  torch', torch.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__, '|', torch.cuda.device_count(), 'GPUs')" \
  || die "the environment does not work (see above); delete $ENV/.ready to rebuild it"
PYTHONPATH="$OVERLAY" "$PY" -c "import diffusers; assert diffusers.__version__ == '0.36.0', diffusers.__version__" \
  || die "the diffusers 0.36 overlay in $OVERLAY is broken; delete $ENV/.ready to rebuild"

# ---- 2. code and weights ------------------------------------------------------------------------------
TW="$CODE/TwinFlow"
if [[ "$(git -C "$TW" rev-parse HEAD 2>/dev/null)" != "$TWINFLOW_COMMIT" && ! -f "$TW/.commit_$TWINFLOW_COMMIT" ]]; then
  say "TwinFlow @ ${TWINFLOW_COMMIT:0:7} -> $TW"
  rm -rf "$TW"; mkdir -p "$CODE"
  if git clone -q "$TWINFLOW_GIT" "$TW" 2>/dev/null && git -C "$TW" checkout -q "$TWINFLOW_COMMIT"; then :
  else
    info "git failed; trying the GitHub tarball"
    rm -rf "$TW"; mkdir -p "$TW"
    curl -fsSL "$TWINFLOW_GIT/archive/$TWINFLOW_COMMIT.tar.gz" | tar -xz -C "$TW" --strip-components=1 \
      || die "could not fetch TwinFlow"
    touch "$TW/.commit_$TWINFLOW_COMMIT"
  fi
fi
[[ -f "$TW/unified_sampler.py" && -f "$TW/diffusers_patch/z_image/modeling_z_image.py" ]] || die "TwinFlow checkout incomplete: rm -rf $TW and re-run"

if [[ ! -f "$HF_HOME/.ready" ]]; then
  say "weights and prompts -> $HF_HOME (~42 GB)"
  env -u PYTHONPATH "$PY" - "$ROOT/bench" <<'PY' || die "download incomplete; re-run to resume"
import sys
from huggingface_hub import snapshot_download as get
bench = sys.argv[1]
get("Qwen/Qwen-Image-Bench", repo_type="dataset", revision="d2493deb153b020cf169c7e3f57d15e4dd697038",
    allow_patterns=["metadata/*", "qwen_image_bench_hf_v0518.jsonl"], local_dir=bench)
# Turbo: the pipeline parts DIDR runs with (its own transformer replaces Turbo's)
get("Tongyi-MAI/Z-Image-Turbo", revision="f332072aa78be7aecdf3ee76d5c247082da564a6",
    allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "vae/*", "transformer/config.json"])
get("inclusionAI/TwinFlow-Z-Image-Turbo", revision="bbb3e268784ed4349a49ca74a22fff1c29fb0646",
    allow_patterns=["TwinFlow-Z-Image-Turbo-exp/*"], ignore_patterns=["*/assets/*"])
get("Junyi-W/DIDR", revision="ee75d9eca3f5c4e3b4968e85c74b3b73b4689cd0", allow_patterns=["zimage_transformer/*"])
print("    ok")
PY
  touch "$HF_HOME/.ready"
fi

# ---- 3. the renderer (written fresh each run) -----------------------------------------------------------
REN="$CODE/render_qib.py"
cat > "$REN" <<'PYEOF'
"""Render the 1000 Qwen-Image-Bench prompts with TwinFlow-Z-Image-Turbo-exp (2 NFE) or DIDR (1 NFE),
on the protocol of z-image/eval_qib/generate.py. Written by qib_render_released.sh.

    render_qib.py --model twinflow|didr --out RUN --bench DIR --twinflow-code DIR --rank R --world W [--limit N]
    render_qib.py --model twinflow|didr --out RUN --bench DIR --twinflow-code DIR --finalize [--limit N]
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import torch

PROMPTS_FILE = "qwen_image_bench_hf_v0518.jsonl"
BENCH_REVISION = "d2493deb153b020cf169c7e3f57d15e4dd697038"
TWINFLOW_COMMIT = "a109a71b19634d9c5d5c45c3587d36b19f014917"
TWINFLOW_REPO, TWINFLOW_REV = "inclusionAI/TwinFlow-Z-Image-Turbo", "bbb3e268784ed4349a49ca74a22fff1c29fb0646"
TWINFLOW_SUB = "TwinFlow-Z-Image-Turbo-exp"
DIDR_REPO, DIDR_REV = "Junyi-W/DIDR", "ee75d9eca3f5c4e3b4968e85c74b3b73b4689cd0"
TURBO_REPO, TURBO_REV = "Tongyi-MAI/Z-Image-Turbo", "f332072aa78be7aecdf3ee76d5c247082da564a6"
MAX_SEQ = 512
TWINFLOW_SAMPLER = dict(sampling_steps=2, stochast_ratio=1.0, extrapol_ratio=0.0, sampling_order=1,
                        time_dist_ctrl=[1.0, 1.0, 1.0], rfba_gap_steps=[0.001, 0.6])


def bench_prompts(bench, lang="zh"):
    key = "prompt_cn" if lang == "zh" else "prompt_en"
    out = []
    with open(Path(bench) / PROMPTS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append((int(r["ID"]), r[key]))
    out.sort()
    if len(out) != 1000 or len({i for i, _ in out}) != 1000:
        raise SystemExit(f"expected 1000 distinct IDs, got {len(out)}")
    return out


def noise_seed(i, seed):
    return int(seed) * 1_000_003 + int(i)


def noise_for(i, seed, shape, device):
    g = torch.Generator(device="cpu").manual_seed(noise_seed(i, seed))
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device)


def write_image(images, i, rgb01):
    from PIL import Image
    arr = (rgb01.detach().clamp(0, 1) * 255.0).round().to("cpu", torch.uint8).permute(1, 2, 0).numpy()
    tmp = images / f".{i:04d}.tmp.png"
    Image.fromarray(arr).save(str(tmp))
    os.replace(tmp, images / f"{i:04d}.png")


def snapshot(repo, rev):
    p = os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo.replace("/", "--"), "snapshots", rev)
    if not os.path.isdir(p):
        raise SystemExit(f"{p} missing")
    return p


def progress(a, n, total, t0):
    if n % 25 == 0 or n == total:
        print(f"[render {a.rank}/{a.world}] {n}/{total} {(time.time() - t0) / n:.2f} s/image", flush=True)


def render_twinflow(a, mine, images):
    sys.path[:0] = [a.twinflow_code, a.twinflow_code + "/src"]
    from functools import partial
    from diffusers_patch.z_image.modeling_z_image import ZImage
    from safetensors import safe_open
    from unified_sampler import UnifiedSampler

    root = os.path.join(snapshot(TWINFLOW_REPO, TWINFLOW_REV), TWINFLOW_SUB)
    model = ZImage(root, aux_time_embed=True, imgs_dtype=torch.bfloat16, text_dtype=torch.bfloat16,
                   max_sequence_length=MAX_SEQ, device="cuda")
    tr = model.model.transformer
    idx = json.load(open(os.path.join(root, "transformer/diffusion_pytorch_model.safetensors.index.json")))
    ck = set(idx["weight_map"])
    sd = tr.state_dict()
    missing, unexpected = sorted(ck - set(sd)), sorted(set(sd) - ck)
    mism = []
    for fn in sorted(set(idx["weight_map"].values())):
        with safe_open(os.path.join(root, "transformer", fn), framework="pt") as f:
            for k in f.keys():
                if k in sd and not torch.equal(f.get_tensor(k).float(), sd[k].float()):
                    mism.append(k)
    keys = {"checkpoint_keys": len(ck), "model_keys": len(sd), "missing_in_model": missing,
            "unexpected_in_model": unexpected, "value_mismatches": mism,
            "t_embedder_2_keys": sorted(k for k in ck if "t_embedder_2" in k),
            "t_embedder_2_equals_t_embedder": all(
                torch.equal(p.float(), q.float()) for p, q in
                zip(tr.t_embedder_2.parameters(), tr.t_embedder.parameters()))}
    print("[keys]", json.dumps({k: v for k, v in keys.items() if k != "t_embedder_2_keys"}), flush=True)
    if missing or unexpected or mism:
        raise SystemExit("checkpoint does not match the model")

    caps, t0 = {}, time.time()
    for i, p in mine:
        emb, mask, _, _ = model.encode_prompt([p], None, do_cfg=False)
        caps[i] = (emb.cpu(), mask.cpu())
    print(f"[encode] {len(mine)} prompts {time.time() - t0:.0f}s", flush=True)
    model.model.text_encoder = None
    torch.cuda.empty_cache()
    model.transformer.to("cuda")
    model.eval()

    sampler = partial(UnifiedSampler().sampling_loop, **TWINFLOW_SAMPLER)
    trace = []

    def model_fn(x_t, t, c=None, tt=None, **kw):
        if len(trace) < 2:
            trace.append({"t_in": t.flatten()[0].item(), "tt_in": tt.flatten()[0].item(), "x_dtype": str(x_t.dtype)})
        return model.transformer(x_t, t, c=c, tt=tt, **kw)

    shape = (16, a.img_size // 8, a.img_size // 8)
    t0 = time.time()
    with torch.no_grad():
        for n, (i, _) in enumerate(mine):
            noise = noise_for(i, a.seed, shape, "cuda").unsqueeze(0).to(torch.bfloat16)
            # UnifiedSampler draws its re-noise from the global CPU generator: with one latent per
            # call this is noise_for(i, seed + 1000), the protocol's step-1 re-noise (checked in main)
            torch.manual_seed(noise_seed(i, a.seed + 1000))
            emb, mask = caps[i]
            traj = sampler(noise, model_fn, c=[emb.to("cuda"), mask.to("cuda")])
            px = model.latents_to_pixels(traj[-1].cuda())
            write_image(images, i, (px.float() * 0.5 + 0.5).clamp(0, 1)[0])
            progress(a, n + 1, len(mine), t0)
    return {"keys": keys, "model_inputs": trace, "sec_per_image": (time.time() - t0) / max(len(mine), 1)}


def render_didr(a, mine, images):
    from diffusers import ZImagePipeline, ZImageTransformer2DModel
    from diffusers.utils.torch_utils import randn_tensor

    tr, info = ZImageTransformer2DModel.from_pretrained(
        os.path.join(snapshot(DIDR_REPO, DIDR_REV), "zimage_transformer"), torch_dtype=torch.bfloat16,
        output_loading_info=True)
    keys = {k: list(v) for k, v in info.items() if k in ("missing_keys", "unexpected_keys", "mismatched_keys")}
    print("[keys]", json.dumps(keys), flush=True)
    if any(keys.values()):
        raise SystemExit("checkpoint does not match the model")
    pipe = ZImagePipeline.from_pretrained(snapshot(TURBO_REPO, TURBO_REV), transformer=None, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to("cuda")
    pipe.vae.to("cuda")
    caps, t0 = {}, time.time()
    with torch.no_grad():
        for i, p in mine:
            pos, _ = pipe.encode_prompt(prompt=p, device="cuda", do_classifier_free_guidance=False,
                                        max_sequence_length=MAX_SEQ)
            caps[i] = pos[0].cpu()
    print(f"[encode] {len(mine)} prompts {time.time() - t0:.0f}s", flush=True)
    pipe.text_encoder = None
    torch.cuda.empty_cache()
    pipe.transformer = tr.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    def gen(i):
        return torch.Generator(device="cpu").manual_seed(noise_seed(i, a.seed))

    shape = (1, 16, a.img_size // 8, a.img_size // 8)
    if not torch.equal(randn_tensor(shape, generator=gen(1), device="cuda", dtype=torch.float32)[0],
                       noise_for(1, a.seed, shape[1:], "cuda")):
        raise SystemExit("the pipeline's generator noise differs from the protocol's noise_for")
    sig, vae, t0 = None, pipe.vae, time.time()
    with torch.no_grad():
        for n, (i, _) in enumerate(mine):
            z = pipe(prompt=None, prompt_embeds=[caps[i].to("cuda")], height=a.img_size, width=a.img_size,
                     num_inference_steps=1, guidance_scale=0.0, generator=gen(i),
                     output_type="latent", return_dict=False)[0]
            if sig is None:
                sig = [float(s) for s in pipe.scheduler.sigmas]
            # the protocol's decode: cast to the VAE dtype, unscale, decode, then [0, 1] in fp32
            z = z.to(vae.dtype) / vae.config.scaling_factor + vae.config.shift_factor
            px = vae.decode(z, return_dict=False)[0]
            write_image(images, i, (px.float() * 0.5 + 0.5).clamp(0, 1)[0])
            progress(a, n + 1, len(mine), t0)
    return {"keys": keys, "scheduler_sigmas": sig, "sec_per_image": (time.time() - t0) / max(len(mine), 1)}


def finalize(a, out, prompts):
    from PIL import Image
    import numpy as np
    images = out / "images"
    names = sorted(p.name for p in images.iterdir())
    want = [f"{i:04d}.png" for i, _ in prompts]
    missing, extra = sorted(set(want) - set(names)), sorted(set(names) - set(want))
    qc, bad = [], []
    for name in want:
        if name in missing:
            continue
        try:
            with Image.open(images / name) as im:
                im.load()
                size, mode = im.size, im.mode
                arr = np.asarray(im.convert("RGB")).astype(np.float32)
        except Exception as e:
            bad.append((name, f"decode: {e}"))
            continue
        tv = float((np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean()) / 2)
        qc.append({"mean": float(arr.mean()), "std": float(arr.std()), "tv": tv})
        if size != (a.img_size, a.img_size) or mode != "RGB":
            bad.append((name, f"size {size} mode {mode}"))
        if arr.max() < 8 or arr.std() < 2:
            bad.append((name, f"blank: mean {arr.mean():.1f} std {arr.std():.1f}"))
        if tv > 45:
            bad.append((name, f"noise-like: mean neighbour difference {tv:.1f}"))
    report = {"png_count": len(names), "expected": len(want), "missing": missing, "extra": extra, "problems": bad}
    if qc:
        tvs = np.array([q["tv"] for q in qc])
        report.update(tv_p50=float(np.median(tvs)), tv_max=float(tvs.max()),
                      mean_min=min(q["mean"] for q in qc), std_min=min(q["std"] for q in qc))
    print("[qc]", json.dumps(report, ensure_ascii=False), flush=True)
    if missing or extra or bad:
        (out / "qc_failed.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(f"checks failed (see {out / 'qc_failed.json'}); nothing finalized")

    with open(out / "input.jsonl", "w", encoding="utf-8") as f:
        for i, p in prompts:
            f.write(json.dumps({"ID": i, "prompt": p, "image_path": str((images / f"{i:04d}.png").resolve())},
                               ensure_ascii=False) + "\n")
    picks = list(dict.fromkeys(prompts[round(k * (len(prompts) - 1) / 15)][0] for k in range(16)))
    tile = 384
    sheet = Image.new("RGB", (4 * tile, 4 * tile), "white")
    for n, i in enumerate(picks):
        with Image.open(images / f"{i:04d}.png") as im:
            sheet.paste(im.convert("RGB").resize((tile, tile), Image.LANCZOS), ((n % 4) * tile, (n // 4) * tile))
    sheet.save(out / "contact_sheet_16.png")

    ranks = [json.load(open(p)) for p in sorted(glob.glob(str(out / "logs/rank_*.json")))]
    r0 = (ranks[0] if ranks else {}).get("extra", {})
    is_tw = a.model == "twinflow"
    sampling = (TWINFLOW_SAMPLER if is_tw else
                {"num_inference_steps": 1, "guidance_scale": 0.0,
                 "scheduler": "FlowMatchEulerDiscreteScheduler of the Turbo snapshot", "sigmas": r0.get("scheduler_sigmas")})
    (out / "generation.json").write_text(json.dumps(
        {"ckpt": None, "teacher_steps": None, "sampling_args": sampling, "lang": a.lang, "seed": a.seed,
         "img_size": a.img_size, "images_per_prompt": 1}, indent=2))
    import diffusers
    info = {
        "model": a.model,
        "weights": ({"repo": TWINFLOW_REPO, "revision": TWINFLOW_REV, "subfolder": TWINFLOW_SUB} if is_tw else
                    {"repo": DIDR_REPO, "revision": DIDR_REV, "subfolder": "zimage_transformer",
                     "pipeline_repo": TURBO_REPO, "pipeline_revision": TURBO_REV}),
        "code": ({"repo": "github.com/inclusionAI/TwinFlow", "commit": TWINFLOW_COMMIT,
                  "entry": "diffusers_patch/z_image/modeling_z_image.py:ZImage + unified_sampler.py:UnifiedSampler"}
                 if is_tw else {"entry": "diffusers.ZImagePipeline"}),
        "sampling": sampling,
        "guidance": {"cfg_scale": 0.0} if is_tw else {"guidance_scale": 0.0},
        "aux_time_embed": True if is_tw else None,
        "seed_rule": {"seed": a.seed,
                      "initial_latent": "torch.randn((16,128,128), fp32, Generator('cpu').manual_seed(seed*1000003+ID))",
                      "renoise_step_k": "same with manual_seed((seed+1000*k)*1000003+ID)" if is_tw else "n/a (1 NFE)"},
        "prompt": {"dataset": "Qwen/Qwen-Image-Bench", "revision": BENCH_REVISION, "file": PROMPTS_FILE,
                   "rows": 1000, "language": "zh (prompt_cn)" if a.lang == "zh" else "en (prompt_en)",
                   "images_per_prompt": 1, "max_sequence_length": MAX_SEQ, "limit": a.limit or None},
        "dtype": "torch.bfloat16 (transformer, text encoder, VAE)",
        "image": {"size": [a.img_size, a.img_size], "format": "PNG, 8-bit RGB, PIL default", "name": "images/{ID:04d}.png"},
        "versions": {"diffusers": diffusers.__version__, "torch": torch.__version__, "python": platform.python_version()},
        "script": {"sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "machine": {"hostname": socket.gethostname(), "platform": platform.platform(),
                    "gpus": [r.get("gpu_name") for r in ranks], "n_ranks": len(ranks)},
        "wall_time_sec_each_rank": [round(r["wall_sec"], 1) for r in ranks],
        "gpu_seconds_total": round(sum(r["wall_sec"] for r in ranks), 1),
        "sec_per_image_render_only": [round(r["extra"]["sec_per_image"], 3) for r in ranks],
        "prompts_over_max_sequence_length": sum(r.get("over_limit", 0) for r in ranks),
        "model_check": r0.get("keys"),
        "model_inputs_first_image": r0.get("model_inputs"),
        "qc": report,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"[finalize] {out}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["twinflow", "didr"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--twinflow-code", required=True)
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    prompts = bench_prompts(a.bench, a.lang)
    if a.limit:
        prompts = prompts[:a.limit]
    if a.finalize:
        return finalize(a, out, prompts)

    side = a.img_size // 8
    torch.manual_seed(noise_seed(7, a.seed + 1000))
    if not torch.equal(torch.randn((1, 16, side, side))[0], noise_for(7, a.seed + 1000, (16, side, side), "cpu")):
        raise SystemExit("global-generator re-noise differs from noise_for")

    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    mine = [(i, p) for i, p in prompts[a.rank::a.world] if not (images / f"{i:04d}.png").exists()]
    print(f"[render {a.rank}/{a.world}] {len(mine)} prompts to render", flush=True)
    if not mine:
        return 0

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(snapshot(TURBO_REPO, TURBO_REV), "tokenizer"))
    over = sum(len(tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)).input_ids) > MAX_SEQ
               for _, p in mine)
    print(f"[tokens] {over} of {len(mine)} prompts longer than {MAX_SEQ}", flush=True)

    t0 = time.time()
    extra = (render_twinflow if a.model == "twinflow" else render_didr)(a, mine, images)
    rec = {"wall_sec": time.time() - t0, "n": len(mine), "over_limit": over, "extra": extra,
           "gpu_name": torch.cuda.get_device_name(0)}
    (out / "logs" / f"rank_{a.rank}.json").write_text(json.dumps(rec, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

if (( SETUP_ONLY )); then
  "$PY" -m py_compile "$REN" || die "renderer does not compile"
  say "setup done in $ROOT; now on a GPU node: bash $0 --root $ROOT [--gpus ...]"
  exit 0
fi

# ---- 4. render, check ---------------------------------------------------------------------------------
NAMES=()
for m in "${MODEL_LIST[@]}"; do
  n="$(name_of "$m")"; RUN="$RUNS/$n"; NAMES+=("$n")
  if [[ -f "$RUN/run_info.json" ]]; then say "$n: already finished ($RUN)"; continue; fi
  mkdir -p "$RUN/logs"
  if [[ "$m" == twinflow ]]; then PP="$OVERLAY"; else PP=""; fi
  ARGS=(--model "$m" --out "$RUN" --bench "$ROOT/bench" --twinflow-code "$TW" --limit "$LIMIT")
  say "render $n on GPUs ${GG[*]} (log: $RUN/logs/render_<rank>.log)"
  pids=()
  for r in "${!GG[@]}"; do
    CUDA_VISIBLE_DEVICES="${GG[$r]}" PYTHONPATH="$PP" HF_HUB_OFFLINE=1 "$PY" "$REN" "${ARGS[@]}" --rank "$r" --world "${#GG[@]}" \
      > "$RUN/logs/render_$r.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { info "rank $i failed -- $RUN/logs/render_$i.log:"; tail -n 5 "$RUN/logs/render_$i.log" | sed 's/^/      /'; fail=1; }
  done
  (( fail == 0 )) || die "rendering of $n incomplete; re-run the same command to resume"
  for r in "${!GG[@]}"; do grep -h "s/image" "$RUN/logs/render_$r.log" | tail -n 1 | sed 's/^/    /' || true; done
  PYTHONPATH="$PP" HF_HUB_OFFLINE=1 "$PY" "$REN" "${ARGS[@]}" --finalize | sed 's/^/    /' \
    || die "$n failed its checks (see above); nothing uploaded"
done

# ---- 5. upload (optional) -------------------------------------------------------------------------------
if (( UPLOAD )); then
  say "upload to $REPO"
  STAGE="$ROOT/.hf_stage"; rm -rf "$STAGE"
  for n in "${NAMES[@]}"; do
    d="$STAGE/qwen-image-bench/$n$SUFFIX"; mkdir -p "$d"
    cp -al "$RUNS/$n/images" "$d/images"
    for f in input.jsonl generation.json run_info.json contact_sheet_16.png; do cp -l "$RUNS/$n/$f" "$d/"; done
  done
  env -u PYTHONPATH "$PY" - "$REPO" "$STAGE" "$SUFFIX" "${NAMES[@]}" <<'PY' || die "upload failed; re-run the same command to resume"
import sys
from huggingface_hub import HfApi
repo, stage, suffix, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
api = HfApi()
have = {e.path for e in api.list_repo_tree(repo, repo_type="dataset", path_in_repo="qwen-image-bench")}
clash = [n + suffix for n in names if f"qwen-image-bench/{n}{suffix}" in have]
if clash:
    raise SystemExit(f"{repo} already has qwen-image-bench/{clash}: not overwritten; pass another --suffix")
api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=stage,
                        allow_patterns=["qwen-image-bench/*"], num_workers=8, print_report=False)
for n in names:
    imgs = list(api.list_repo_tree(repo, repo_type="dataset", path_in_repo=f"qwen-image-bench/{n}{suffix}/images"))
    print(f"    {n}{suffix}: {len(imgs)} images on the Hub")
PY
  rm -rf "$STAGE"
  say "done: https://huggingface.co/datasets/$REPO/tree/main/qwen-image-bench"
else
  say "done: ${NAMES[*]} in $RUNS (not uploaded; --upload --suffix S to upload)"
fi
