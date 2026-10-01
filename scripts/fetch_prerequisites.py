#!/usr/bin/env python
"""Fetch every *downloadable* prerequisite for FLUX SW-LMMD / iRDM into one designated root.

Everything this repo consumes falls into two classes:

  DOWNLOADABLE -- frozen third-party weights, the COCO source images, and the native
      ``flux2`` package. This script fetches all of them into ``--root`` and writes an
      ``env.sh`` that points ``HF_HOME`` / ``TORCH_HOME`` / ``DREAMSIM_CACHE_DIR`` /
      ``FLUX2_SRC`` at it.

  BUILT LOCALLY -- every artifact under ``data/fid_stats`` (per-encoder reference feature
      pools, the tau(c) text table, the Qwen3 generator-context pool, the Nystrom / joint
      bundles) and the SW-LMMD reference store. These are GPU compute, not downloads;
      nothing hosts them. The script ends by printing the exact build commands, in order,
      with ``--root``-aware paths.

Groups (``--group``, repeatable; default = encoders text flux pickscore flux2src coco assets):

    encoders   the 10 training encoders (timm / HF weights + Inception-FID + DreamSim)
    evalenc    the 4 held-out evaluation encoders (dinov2, siglip_v1, C-RADIOv3-L, FLUX VAE)
    text       the SigLIP2 SO400M text tower that produces tau(c)  [implied by `encoders`]
    flux       FLUX.2 klein-4B MM-DiT + the AE + the Qwen3-4B text encoder (the generator)
    pickscore  PickScore_v1 + its processor: the eval metric and the 02 render curation (3.9 GB)
    flux2src   the native Black Forest Labs ``flux2`` package (git clone -> FLUX2_SRC)
    coco       COCO train2014 images + captions (the reference/prompt source, 13.8 GB)
    assets     the authors' released reference assets (Nystrom bundles + tau(c) table, 1.3 GB)
    student    the RELEASED iRDM one-step student (a baseline / warm-start, 7.8 GB)
    pmfh       the released ImageNet pMF-H generator (ImageNet path only, not SW-LMMD)

    python scripts/fetch_prerequisites.py --root /data/hulk/jiacheng/rdm_assets --dry-run
    python scripts/fetch_prerequisites.py --root /data/hulk/jiacheng/rdm_assets
    python scripts/fetch_prerequisites.py --root ... --group flux --group flux2src
    source /data/hulk/jiacheng/rdm_assets/env.sh

Idempotent and resumable: HF downloads are cache-backed (re-running verifies rather than
refetches) and the COCO zips resume with ``curl -C -``. ``--dry-run`` reports the plan, the
estimated footprint, and the free space on the target filesystem without writing anything.

One item needs a human in the loop: ``black-forest-labs/FLUX.2-dev`` (source of the native
``ae.safetensors`` the flux2 loader wants) is a gated repo. Accept its license once at
https://huggingface.co/black-forest-labs/FLUX.2-dev and run ``hf auth login``; the script
reports this clearly and keeps going rather than aborting the run.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# The registry: every downloadable prerequisite, its source, and its approximate footprint.
# `patterns` keeps HF snapshots to the safetensors copy (timm repos also ship a duplicate
# .bin, which would roughly double the transfer).
# --------------------------------------------------------------------------------------
WEIGHT_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.py", "*.model"]

COCO_TRAIN_ZIP = "http://images.cocodataset.org/zips/train2014.zip"
COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_TRAIN_IMAGES = 82783                       # expected extracted count
INCEPTION_URL = ("https://github.com/toshas/torch-fidelity/releases/download/"
                 "v0.2.0/weights-inception-2015-12-05-6726825d.pth")
FLUX2_GIT = "https://github.com/black-forest-labs/flux2"
# The flux2 commit everything here was built and trained with (2026-03-12). Pinned so a new
# machine gets the same model / text-encoder code, not whatever upstream main is that day.
FLUX2_COMMIT = "50fe5162777813d869182b139e83b10743caef15"


@dataclass
class Item:
    """One downloadable prerequisite."""

    key: str
    group: str
    kind: str                                   # hf | hf_dataset | hf_file | url | git | python
    est_gb: float
    src: str = ""
    filename: str = ""                          # hf_file / url destination basename
    patterns: list = field(default_factory=lambda: list(WEIGHT_PATTERNS))
    note: str = ""
    gated: bool = False
    rev: str = ""                               # git: the commit to check out


ITEMS: list[Item] = [
    # ---- 10 training encoders (registry: rdm/representation/registry.py) ----
    Item("inception",      "encoders", "url", 0.10, INCEPTION_URL, "weights-inception-2015-12-05-6726825d.pth",
         note="Inception-FID (torch-fidelity); cached under TORCH_HOME/hub/checkpoints"),
    Item("convnext",       "encoders", "hf", 0.35, "timm/convnextv2_base.fcmae_ft_in22k_in1k"),
    Item("mae",            "encoders", "hf", 1.22, "timm/vit_large_patch16_224.mae"),
    Item("clip",           "encoders", "hf", 1.71, "timm/vit_large_patch14_clip_224.openai"),
    Item("dinov3_l",       "encoders", "hf", 1.21, "timm/vit_large_patch16_dinov3.lvd1689m"),
    Item("pe_core_l",      "encoders", "hf", 1.27, "timm/vit_pe_core_large_patch14_336.fb"),
    Item("siglip2",        "encoders", "hf", 1.71, "timm/vit_so400m_patch16_siglip_256.v2_webli"),
    Item("aimv2_huge",     "encoders", "hf", 2.72, "timm/aimv2_huge_patch14_224.apple_pt"),
    Item("webssl_dino_1b", "encoders", "hf", 4.54, "facebook/webssl-dino1b-full2b-224"),
    Item("dreamsim",       "encoders", "python", 2.00, note="DreamSim ensemble -> DREAMSIM_CACHE_DIR (needs torch)"),
    # ---- tau(c) text tower (open_clip SigLIP2 SO400M) ----
    Item("siglip2_text",   "text", "hf", 1.75, "timm/ViT-SO400M-16-SigLIP2-256",
         note="text tower for tau(c) (rdm.representation.text_encoder)"),
    # ---- 4 held-out evaluation encoders ----
    Item("dinov2",         "evalenc", "hf", 1.22, "timm/vit_large_patch14_dinov2.lvd142m"),
    Item("siglip_v1",      "evalenc", "hf", 1.71, "timm/vit_so400m_patch14_siglip_384.webli"),
    Item("cradiov3_l",     "evalenc", "hf", 1.28, "nvidia/C-RADIOv3-L"),
    Item("flux_vae",       "evalenc", "hf", 0.34, "black-forest-labs/FLUX.1-schnell",
         patterns=["vae/*"], note="held-out FLUX VAE encoder (subfolder='vae')"),
    # ---- the generator: FLUX.2 klein-4B + AE + Qwen3 text encoder ----
    Item("klein4b",        "flux", "hf_file", 7.75, "black-forest-labs/FLUX.2-klein-4B",
         "flux-2-klein-4b.safetensors", note="the 4-step teacher / student init"),
    Item("flux2_ae",       "flux", "hf_file", 0.34, "black-forest-labs/FLUX.2-dev", "ae.safetensors",
         gated=True, note="native VAE the flux2 loader resolves; GATED repo"),
    Item("qwen3_4b",       "flux", "hf", 8.05, "Qwen/Qwen3-4B",
         note="FLUX.2 prompt encoder (bf16 build; works offline, no fp8 kernels)"),
    # ---- PickScore: the eval metric (eval_checkpoint.sh) and the 02 render curation ----
    Item("pickscore",      "pickscore", "hf", 3.94, "yuvalkirstain/PickScore_v1",
         patterns=["config.json", "model.safetensors"], note="PickScore_v1 (CLIP-H)"),
    Item("pickscore_proc", "pickscore", "hf", 0.01, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
         patterns=["*.json", "*.txt"], note="PickScore's processor: tokenizer + image configs, no weights"),
    # ---- native flux2 package ----
    Item("flux2src",       "flux2src", "git", 0.05, FLUX2_GIT, rev=FLUX2_COMMIT,
         note="-> FLUX2_SRC, pinned"),
    # ---- COCO source data ----
    Item("coco_train",     "coco", "url", 13.51, COCO_TRAIN_ZIP, "train2014.zip",
         note="82,783 reference images (extracted +13.5 GB)"),
    Item("coco_ann",       "coco", "url", 0.25, COCO_ANN_ZIP, "annotations_trainval2014.zip",
         note="captions_train2014.json"),
    # ---- the authors' released reference-side assets ----
    Item("geall_assets", "assets", "hf_dataset", 1.27, "Lanl11/irdm-flux-geall-assets",
         patterns=["*.pt", "*.npy", "*.md"],
         note="10 joint Nystrom bundles (the iRDM baseline arm) + the SigLIP2 tau(c) table, "
              "whose first 82,783 rows are the COCO captions SW-LMMD reads directly"),
    # ---- released checkpoints ----
    Item("student",        "student", "hf_file", 7.75, "epfl-vita/flux2-klein-1step-rdm",
         "model.safetensors", note="released iRDM one-step student (bf16 baseline)"),
    Item("pmfh",           "pmfh", "hf_file", 3.80, "Lanl11/pMF-H-FDSIM-imagenet256-sigma07-4k",
         "model.pth", note="ImageNet pMF-H generator (ImageNet path only)"),
]

DEFAULT_GROUPS = ["encoders", "text", "flux", "pickscore", "flux2src", "coco", "assets"]
ALL_GROUPS = ["encoders", "text", "evalenc", "flux", "pickscore", "flux2src", "coco", "assets",
              "student", "pmfh"]


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def free_gb(path: str) -> float:
    p = path
    while p and not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p or "/").free / 1e9


def say(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], **kw) -> int:
    say("  $ " + " ".join(cmd))
    return subprocess.call(cmd, **kw)


# --------------------------------------------------------------------------------------
# fetchers -- each returns True on success, False on a recoverable failure
# --------------------------------------------------------------------------------------
def fetch_hf_snapshot(item: Item) -> bool:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    try:
        path = snapshot_download(repo_id=item.src, allow_patterns=item.patterns)
        say(f"  ok  {item.src} -> {path}")
        return True
    except GatedRepoError:
        gated_message(item)
        return False
    except RepositoryNotFoundError as e:
        say(f"  FAIL {item.src}: repository not found ({e.__class__.__name__})")
        return False
    except Exception as e:                                   # network / auth / disk
        say(f"  FAIL {item.src}: {type(e).__name__}: {str(e)[:200]}")
        return False


def fetch_hf_dataset(item: Item, dest: str) -> bool:
    """Materialize a dataset repo into a readable directory (not the opaque blob cache).

    These files are read by path from the preprocessing scripts, so they are checked out flat
    rather than left as cache symlinks.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    try:
        snapshot_download(repo_id=item.src, repo_type="dataset", local_dir=dest,
                          allow_patterns=item.patterns)
        say(f"  ok  {item.src} -> {dest}")
        return True
    except GatedRepoError:
        gated_message(item)
        return False
    except (RepositoryNotFoundError, Exception) as e:
        say(f"  FAIL {item.src}: {type(e).__name__}: {str(e)[:200]}")
        return False


def fetch_hf_file(item: Item) -> bool:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    try:
        path = hf_hub_download(repo_id=item.src, filename=item.filename)
        say(f"  ok  {item.src}/{item.filename} -> {path}")
        return True
    except GatedRepoError:
        gated_message(item)
        return False
    except RepositoryNotFoundError as e:
        say(f"  FAIL {item.src}: repository not found ({e.__class__.__name__})")
        return False
    except Exception as e:
        say(f"  FAIL {item.src}/{item.filename}: {type(e).__name__}: {str(e)[:200]}")
        return False


def gated_message(item: Item) -> None:
    say(f"  GATED {item.src} -- accept the license once, then re-run this script:\n"
        f"        1) open https://huggingface.co/{item.src} and accept the terms\n"
        f"        2) hf auth login      (a token with 'read' scope)\n"
        f"        continuing with the other items.")


def fetch_url(item: Item, dest_dir: str) -> bool:
    """Resumable download (curl -C -, wget -c fallback) to ``dest_dir/item.filename``."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, item.filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0.95 * item.est_gb * 1e9:
        say(f"  ok  present {dest} ({os.path.getsize(dest) / 1e9:.2f} GB)")
        return True
    if shutil.which("curl"):
        rc = run(["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "5",
                  "-o", dest, item.src])
    elif shutil.which("wget"):
        rc = run(["wget", "-c", "-O", dest, item.src])
    else:
        say("  FAIL neither curl nor wget available")
        return False
    if rc != 0:
        say(f"  FAIL download exited {rc} -- re-run to resume")
        return False
    say(f"  ok  {dest} ({os.path.getsize(dest) / 1e9:.2f} GB)")
    return True


def _git_head(dest: str) -> str:
    out = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def fetch_git(item: Item, dest: str) -> bool:
    if os.path.isdir(os.path.join(dest, ".git")):
        head = _git_head(dest)
        if item.rev and head != item.rev:
            say(f"  WARN {dest} is at {head[:7]}, not the pinned {item.rev[:7]}: "
                f"git -C {dest} fetch --depth 1 origin {item.rev} && git -C {dest} checkout {item.rev}")
        else:
            say(f"  ok  present {dest} ({head[:7]})")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if run(["git", "clone", "--depth", "1", item.src, dest]) != 0:
        return False
    if item.rev and _git_head(dest) != item.rev:          # upstream moved on: fetch the pinned one
        if (run(["git", "-C", dest, "fetch", "--depth", "1", "origin", item.rev]) != 0
                or run(["git", "-C", dest, "checkout", "-q", item.rev]) != 0):
            return False
    say(f"  ok  {dest} at {_git_head(dest)[:7]}")
    return not item.rev or _git_head(dest) == item.rev


def fetch_dreamsim(cache_dir: str) -> bool:
    """DreamSim pulls its ensemble at first instantiation -- needs torch + the dreamsim pkg."""
    try:
        from dreamsim import dreamsim
    except ImportError:
        say("  SKIP dreamsim: package not importable (pip install -r requirements.txt first); "
            "it self-downloads on first use into DREAMSIM_CACHE_DIR")
        return False
    try:
        os.makedirs(cache_dir, exist_ok=True)
        model, _ = dreamsim(pretrained=True, device="cpu", cache_dir=cache_dir)
        del model
        say(f"  ok  dreamsim ensemble -> {cache_dir}")
        return True
    except Exception as e:
        say(f"  FAIL dreamsim: {type(e).__name__}: {str(e)[:200]}")
        return False


# --------------------------------------------------------------------------------------
# post-download steps
# --------------------------------------------------------------------------------------
def extract_coco(root: str, dry: bool) -> None:
    """Unzip train2014 + annotations (skips when already extracted)."""
    coco = os.path.join(root, "datasets", "coco")
    img_dir = os.path.join(coco, "train2014")
    ann_json = os.path.join(coco, "annotations", "captions_train2014.json")
    for zname, marker, desc in ((("train2014.zip"), img_dir, "images"),
                                (("annotations_trainval2014.zip"), ann_json, "annotations")):
        zpath = os.path.join(coco, zname)
        if os.path.exists(marker):
            say(f"  ok  {desc} already extracted ({marker})")
            continue
        if not os.path.exists(zpath):
            say(f"  SKIP extract {zname}: archive not present")
            continue
        if dry:
            say(f"  would extract {zpath} -> {coco}")
            continue
        say(f"  extracting {zpath} ...")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(coco)
    if os.path.isdir(img_dir):
        n = sum(1 for f in os.scandir(img_dir) if f.name.endswith(".jpg"))
        status = "ok " if n == COCO_TRAIN_IMAGES else "WARN"
        say(f"  {status} train2014: {n} jpgs (expected {COCO_TRAIN_IMAGES})")


def build_coco_pairs(root: str, repo_root: str, dry: bool) -> None:
    """Build the canonical row-aligned pairing everything joint is indexed by."""
    coco = os.path.join(root, "datasets", "coco")
    out = os.path.join(coco, "coco_pairs.npz")
    captions = os.path.join(coco, "annotations", "captions_train2014.json")
    images = os.path.join(coco, "train2014")
    if os.path.exists(out):
        say(f"  ok  present {out}")
        return
    if not (os.path.exists(captions) and os.path.isdir(images)):
        say("  SKIP coco_pairs: extract the archives first")
        return
    if dry:
        say(f"  would build {out}")
        return
    sys.path.insert(0, repo_root)
    try:
        from rdm.data.coco import build_coco_pairs as _build
    except ImportError as e:
        say(f"  SKIP coco_pairs ({type(e).__name__}: needs the torch env). Run later:\n"
            f"        python scripts/prepare_datasets.py coco --captions {captions} "
            f"--images {images} --out {out}")
        return
    pairs = _build(captions, images, out)
    say(f"  ok  {out}: {len(pairs['image_ids'])} image-caption pairs")


def flux2_src_dir(root: str) -> str:
    """The directory to put on ``sys.path`` so ``import flux2`` resolves.

    The upstream clone is a src-layout project (``<clone>/src/flux2/``), so FLUX2_SRC is the
    clone's ``src/`` -- NOT the clone root, which has no importable ``flux2`` package.
    """
    clone = os.path.join(root, "src", "flux2")
    nested = os.path.join(clone, "src")
    return nested if os.path.isdir(os.path.join(nested, "flux2")) else clone


def write_env(root: str, dry: bool, hub_cache: str | None = None) -> str:
    """Write ``<root>/env.sh`` -- the single source of the cache locations.

    HF_HUB_CACHE is exported explicitly, not left to derive from HF_HOME: it OVERRIDES
    HF_HOME when set, so a stray export in the user's shell profile would otherwise send
    every blob somewhere else while HF_HOME quietly appeared to be honoured.
    """
    path = os.path.join(root, "env.sh")
    hub = hub_cache or os.path.join(root, "hf", "hub")
    # rdm/representation/generators/flux_generator.py::_hub_root() resolves the klein-4B / AE
    # weights from HF_HOME ONLY -- it never looks at HF_HUB_CACHE. So HF_HOME must be the
    # PARENT of whatever hub cache actually holds the blobs, or the FLUX path raises
    # FileNotFoundError while huggingface_hub itself is perfectly happy.
    hf_home = os.path.dirname(hub) if os.path.basename(hub) == "hub" else os.path.join(root, "hf")
    body = f"""# Generated by scripts/fetch_prerequisites.py -- `source` before any RDM command.
export RDM_ASSETS="{root}"
export HF_HOME="{hf_home}"
# HF_HUB_CACHE takes PRECEDENCE over HF_HOME -- always set it explicitly, or a shell-profile
# export silently wins and the weights land outside this root.
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$RDM_ASSETS/torch"
export DREAMSIM_CACHE_DIR="$RDM_ASSETS/dreamsim"
# src-layout clone: the importable package is <clone>/src/flux2, so this is <clone>/src
export FLUX2_SRC="{flux2_src_dir(root)}"
export PYTHONPATH="$FLUX2_SRC:${{PYTHONPATH:-}}"
export COCO_ROOT="$RDM_ASSETS/datasets/coco"
"""
    if dry:
        say(f"  would write {path}")
        return path
    os.makedirs(root, exist_ok=True)
    with open(path, "w") as f:
        f.write(body)
    say(f"  ok  wrote {path}")
    return path


def link_checkpoints(root: str, repo_root: str, selected: set, dry: bool) -> None:
    """Link released checkpoints to the flat names the configs' ``load_from`` expect."""
    from huggingface_hub import hf_hub_download
    targets = []
    if "student" in selected:
        targets.append(("epfl-vita/flux2-klein-1step-rdm", "model.safetensors",
                        "flux2_klein_1step_rdm_geallcoco_s180.safetensors"))
    if "pmfh" in selected:
        targets.append(("Lanl11/pMF-H-FDSIM-imagenet256-sigma07-4k", "model.pth", "pMF-H_FD-SIM.pth"))
    if not targets:
        return
    out_dir = os.path.join(repo_root, "checkpoints")
    for repo, fname, flat in targets:
        if dry:
            say(f"  would link checkpoints/{flat}")
            continue
        try:
            src = hf_hub_download(repo_id=repo, filename=fname)
        except Exception as e:
            say(f"  SKIP link {flat}: {type(e).__name__}")
            continue
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, flat)
        if os.path.lexists(dst):
            os.remove(dst)
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copy2(src, dst)
        say(f"  ok  checkpoints/{flat} -> {src}")


# --------------------------------------------------------------------------------------
# next-steps report: what this script CANNOT download
# --------------------------------------------------------------------------------------
def print_next_steps(root: str) -> None:
    coco = os.path.join(root, "datasets", "coco")
    say("""
======================================================================
 Downloads done. The remaining prerequisites are COMPUTE, not downloads
======================================================================
Nothing hosts the reference artifacts -- they are built from the images you
just fetched, by running the frozen encoders over them. In order:

  source {root}/env.sh
  pip install -r requirements.txt                 # torch env (6x RTX 4090 detected here)

  # 1. canonical row-aligned COCO pairing (if the step above skipped it)
  python scripts/prepare_datasets.py coco \\
      --captions {coco}/annotations/captions_train2014.json \\
      --images   {coco}/train2014 \\
      --out      {coco}/coco_pairs.npz

  # 2. per-encoder image features + tau(c) + joint bundles  [GPU-hours; shard by encoder]
  python -m rdm.refprep.run joint --coco-pairs {coco}/coco_pairs.npz --out data/fid_stats
  #    shard across your 6 GPUs with disjoint --encoders on different CUDA_VISIBLE_DEVICES

  # 3. FLUX.2 Qwen3 generator-context pool  [see the size warning below]
  python scripts/build_flux2_ctx.py --captions {coco}/coco_pairs.npz \\
      --out data/fid_stats/flux2/qwen3_ctx_coco.npy --ctx-len 48

  # 4. SW-LMMD reference store (assembles 2+3 into the row-aligned window store)
  #    -- scripts/build_sw_lmmd_reference.py, to be written with the SW-LMMD package

DISK WARNING: step 3 at ctx_len 48 over 82,783 captions is
  82783 x 48 x 7680 x 2 bytes = 61.0 GB
as a single .npy. Plan for it (shard it, put it on the largest volume, or cut
ctx_len) before launching -- it is the largest single artifact in the project.

Verify a config's artifacts at any point with:
  python scripts/check_artifacts.py configs/flux.yaml
""".format(root=root, coco=coco))


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="designated asset root (put it on your largest volume)")
    ap.add_argument("--group", action="append", choices=ALL_GROUPS, default=None,
                    help=f"repeatable; default: {' '.join(DEFAULT_GROUPS)}")
    ap.add_argument("--all", action="store_true", help="every group, including student/pmfh/evalenc")
    ap.add_argument("--dry-run", action="store_true", help="plan + footprint only, write nothing")
    ap.add_argument("--skip-extract", action="store_true", help="download the COCO zips but do not unzip")
    ap.add_argument("--encoders", default=None,
                    help="comma-separated encoder keys: fetch only these from the 'encoders' group, "
                         "and not the implied text tower (e.g. the three a SW-LMMD config trains with)")
    ap.add_argument("--hf-cache", default=None,
                    help="adopt an existing HF hub cache instead of filling <root>/hf/hub "
                         "(e.g. a shared cache that already holds these weights)")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    groups = set(ALL_GROUPS if args.all else (args.group or DEFAULT_GROUPS))
    only_enc = [e for e in (args.encoders or "").split(",") if e]
    known_enc = {i.key for i in ITEMS if i.group == "encoders"}
    if set(only_enc) - known_enc:
        ap.error(f"unknown encoder(s) {sorted(set(only_enc) - known_enc)}; known: {sorted(known_enc)}")
    if "encoders" in groups and not only_enc:
        groups.add("text")                       # tau(c) tower is part of the training path
    plan = [i for i in ITEMS if i.group in groups]
    if only_enc:                                 # SW-LMMD from the prebuilt store: tau is in the store
        plan = [i for i in plan if i.group != "encoders" or i.key in only_enc]

    # ---- caches live under the root; set BEFORE huggingface_hub is imported ----
    # HF_HUB_CACHE / HUGGINGFACE_HUB_CACHE OVERRIDE HF_HOME. Setting HF_HOME alone is not
    # enough: a shell-profile export of HF_HUB_CACHE silently wins and the blobs land
    # outside --root while everything still reports success.
    inherited_hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    hub_cache = os.path.abspath(os.path.expanduser(args.hf_cache)) if args.hf_cache \
        else os.path.join(root, "hf", "hub")
    # `hf auth login` stored the token under the CURRENT HF_HOME (default ~/.cache/huggingface).
    # huggingface_hub derives the token path from HF_HOME at import time, so the redirect below
    # would hide it and the gated FLUX.2-dev download would fail although the user is logged in.
    # Pin the path first (an explicit HF_TOKEN / HF_TOKEN_PATH still wins).
    user_hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "huggingface")
    os.environ.setdefault("HF_TOKEN_PATH",
                          os.path.join(os.path.expanduser(user_hf_home), "token"))
    os.environ["HF_HOME"] = os.path.join(root, "hf")
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
    os.environ["HF_DATASETS_CACHE"] = os.path.join(root, "hf", "datasets")
    os.environ["TORCH_HOME"] = os.path.join(root, "torch")
    os.environ["DREAMSIM_CACHE_DIR"] = os.path.join(root, "dreamsim")

    est = sum(i.est_gb for i in plan)
    extract_extra = 13.6 if "coco" in groups and not args.skip_extract else 0.0
    avail = free_gb(root)
    say(f"root          {root}")
    say(f"groups        {' '.join(sorted(groups))}")
    say(f"items         {len(plan)}")
    say(f"download      ~{est:.1f} GB" + (f"  (+{extract_extra:.1f} GB extracted)" if extract_extra else ""))
    say(f"hub cache     {hub_cache}")
    if inherited_hub and os.path.abspath(inherited_hub) != hub_cache:
        say(f"              (overriding inherited HF_HUB_CACHE={inherited_hub}; pass "
            f"--hf-cache to adopt it instead)")
    say(f"free space    {avail:.1f} GB on that filesystem")
    if avail < est + extract_extra:
        say(f"\nWARNING: short by ~{est + extract_extra - avail:.1f} GB. Pick a bigger --root, "
            f"or fetch fewer groups.")
    say("")
    for i in plan:
        tag = " [GATED]" if i.gated else ""
        say(f"  {i.group:9s} {i.key:16s} {i.est_gb:6.2f} GB  {i.src or i.note}{tag}")
    say("")

    if args.dry_run:
        say("--dry-run: nothing written.")
        write_env(root, dry=True, hub_cache=hub_cache)
        print_next_steps(root)
        return 0

    for d in ("hf", "torch", "dreamsim", os.path.join("datasets", "coco"), "src"):
        os.makedirs(os.path.join(root, d), exist_ok=True)

    ok, failed = [], []
    for item in plan:
        say(f"[{item.group}/{item.key}] {item.note or item.src}")
        if item.kind == "hf":
            good = fetch_hf_snapshot(item)
        elif item.kind == "hf_dataset":
            good = fetch_hf_dataset(item, os.path.join(root, "irdm_geall_assets"))
        elif item.kind == "hf_file":
            good = fetch_hf_file(item)
        elif item.kind == "url" and item.group == "coco":
            good = fetch_url(item, os.path.join(root, "datasets", "coco"))
        elif item.kind == "url":                 # torch.hub checkpoints (Inception-FID)
            good = fetch_url(item, os.path.join(root, "torch", "hub", "checkpoints"))
        elif item.kind == "git":
            good = fetch_git(item, os.path.join(root, "src", "flux2"))
        elif item.kind == "python":
            good = fetch_dreamsim(os.environ["DREAMSIM_CACHE_DIR"])
        else:
            say(f"  FAIL unknown kind {item.kind}")
            good = False
        (ok if good else failed).append(item.key)

    if "coco" in groups and not args.skip_extract:
        say("[coco/extract]")
        extract_coco(root, dry=False)
        say("[coco/pairs]")
        build_coco_pairs(root, repo_root, dry=False)

    say("[env]")
    write_env(root, dry=False, hub_cache=hub_cache)
    say("[checkpoints]")
    link_checkpoints(root, repo_root, groups, dry=False)

    say(f"\nfetched {len(ok)}/{len(plan)} items.")
    if failed:
        say(f"incomplete: {', '.join(failed)}  -- re-run to resume (downloads are cached).")
    print_next_steps(root)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
