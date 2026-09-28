# scripts/

Two entry points, in order. Everything else is either a helper they call or an upstream
one-off.

```bash
bash scripts/setup_env.sh                                  # 1. conda env (python 3.12 + torch)
bash scripts/download_all.sh --root /data/<you>/rdm-sets   # 2. every download (~50 GB)
source /data/<you>/rdm-sets/env.sh
bash scripts/preprocess_all.sh                             # 3. build what nobody hosts
```

## Entry points

| script | what it does |
|---|---|
| `setup_env.sh` | Creates the conda env. Python 3.12 (flux2 caps at `<3.13`), torch pinned from the CUDA index *before* `requirements.txt` so the bulk install cannot replace the build. |
| `download_all.sh` | **All downloads, one command.** Encoders, FLUX.2 klein-4B + AE + Qwen3, the flux2 package, COCO, and the authors' released reference assets. Writes `<root>/env.sh`. |
| `preprocess_all.sh` | **All local compute**, steps 01 → 02 → 03 (below). Each step is independently resumable. |

## The preprocessing chain

Strictly sequential — 02 conditions on 01's context, 03 encodes 02's renders.

| step | produces | cost |
|---|---|---|
| `preprocess_01_ctx.sh` | Qwen3 generator context, `(N, 48, 7680)` fp16 | 1 GPU, ~61 GB out |
| `preprocess_02_render.sh` | 4-step teacher renders, `SEEDS` per prompt | 3.6 img/s per 4090 |
| `preprocess_03_features.sh` | per-row encoder features + the assembled reference store | 134–289 img/s per encoder |

`preprocess_config.sh` holds every parameter; the steps read it and are not meant to be edited.
`_preprocess_*.py` are the per-rank workers the steps launch — not called directly.

```bash
MAX_PROMPTS=64 SEEDS=1 NUM_GPUS=2 bash scripts/preprocess_all.sh    # small pilot
```

## Why step 02 renders at all

The reference is **teacher output, not real photographs**. The student is initialised from the
4-step klein-4B teacher, so photographs are a target neither model reaches and the objective
stops being distillation. `SEEDS` renders per prompt keep each prompt's reference a small
*sample* rather than a single point; with one render per prompt the local MMD degenerates
toward per-prompt regression and diversity collapses.

## What is downloaded vs. what is built

The authors released the reference in **compressed** form: joint Nyström bundles, i.e. the
reference mean embedding, with row identity integrated out. That is everything iRDM needs —
its loss only ever asks "how close is this batch to the reference *as a whole*".

SW-LMMD asks a different question: "how close is the student on *these* prompts to the
reference for *those same* prompts". That needs `y_j` per row, which no bundle can answer and
nobody hosts — hence step 03. The one piece that *is* reusable is the SigLIP2 τ(c) table,
whose first 82,783 rows are the COCO captions (verified by re-encoding: matched-row cosine
0.99998); `preprocess_03` slices it instead of running the text tower.

## Upstream one-offs

| script | when |
|---|---|
| `prepare_datasets.py` | Build the canonical COCO image–caption pairing (`download_all.sh` already runs it). |
| `build_flux2_ctx.py` | The Qwen3 context builder that step 01 wraps; call directly for eval prompt sets. |
| `run_refprep.sh` | The **iRDM** reference build (ImageNet banks / joint Nyström bundles). Not part of the SW-LMMD chain. |
| `download_checkpoints.py` | Fetches just the two released generator checkpoints. Superseded by `download_all.sh` for this project. |
| `check_artifacts.py` | Fails fast if a config's artifacts are missing. |
| `train.sh` | torchrun launcher for `rdm.train.launch`. |
| `fetch_prerequisites.py` | The download engine `download_all.sh` drives; use directly for per-group control. |

## Two environment traps these scripts encode

1. **`HF_HUB_CACHE` overrides `HF_HOME`.** A shell-profile export of the former silently sends
   every blob elsewhere while the latter appears honoured.
2. **`flux_generator.py::_hub_root()` reads `HF_HOME` only.** So `HF_HOME` must be the *parent*
   of the hub cache that actually holds the blobs, or the FLUX path raises `FileNotFoundError`
   while `huggingface_hub` itself is perfectly happy. The generated `env.sh` keeps them
   consistent.
