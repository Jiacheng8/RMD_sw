<div align="center">

# SW-LMMD

**Sliding-Window Prompt-Aligned Local MMD Distillation**

*A fork of [vita-epfl/RDM](https://github.com/vita-epfl/RDM) — one-step distillation at a fraction of the per-step cost.*

</div>

---

## What this fork is

Upstream **iRDM** distills a one-step image generator by matching generated and reference
feature distributions under a battery of frozen encoders. Its loss compares a *large fresh
rollout* — 10,240 generated images per optimizer step for the FLUX run — against a reference
compressed once into a global Nyström kernel-mean embedding.

This fork replaces that objective with **SW-LMMD**, which

1. keeps **every** reference row uncompressed and prompt-addressable, and
2. compares an **exact** two-sample MMD inside an overlapping window of `K = 1024`
   prompt-aligned rows, of which only the newest `B = 128` are freshly generated and
   back-propagated.

Per-step fresh rollouts drop **10,240 → 128**. Everything else — the encoder battery, the
joint image-text feature, GradCache, the generator wrapper — is reused unchanged.

> **Status.** The method layer is implemented and proven (124 tests, incl. the two invariants
> below). It has **never been run on the real 4B model**: every test uses a toy generator and a
> mock battery, and the reference store is still being built. See [Status](#status).

---

## The method

### Global vs. local

Write the reference as a mixture over prompt-indexed components, `Q = Σ_s π_s Q_s`, and the
student likewise. With `d_s = μ_{P_s} − μ_{Q_s}` the two objectives are

```
L_global = ‖ Σ_s π_s d_s ‖²           (iRDM: one global mean embedding)
L_local  =   Σ_s π_s ‖ d_s ‖²          (SW-LMMD: per-component)
```

Jensen gives `L_global ≤ L_local`. The gap is exactly the error *cancellation* the global
objective permits: two prompt subsets can be wrong in opposite directions and still produce a
small global MMD. The local objective forbids that trade.

This is why the reference cannot be a Nyström bundle. `α = K_ZZ⁻¹ μ_Zr` with
`μ_Zr[k] = mean_j k(Z_k, r_j)` has already summed over `j` — row identity is integrated out, and
`Z` is k-means centroids, not any real image's feature. It answers "how close is this batch to
the reference *as a whole*", which is precisely `L_global`. SW-LMMD needs `y_j` for the specific
rows whose prompts the student just generated from, so it reads per-row features instead.

### The window

Window `t` is `K` consecutive positions of a fixed seeded permutation, starting at `t·B`:

```
step t     [ retained  K−B = 896 rows ][ active B = 128 ]
              from cache, detached        fresh, grad
step t+1        [ retained 896 ]............[ active 128 ]
                 └─ slid right by exactly B ─┘
```

Sliding by exactly `B` makes `previous.all_ids[B:] == current.retained_ids` an identity, which
the cache relies on — it is asserted on every transition, cyclic wrap included. Reference and
generated sides index the **same rows**, so both distributions sit on the same prompt support.

Order is a **seeded random permutation**, not a semantic one: "local" means *this finite
subset*, not *semantically close*. Semantic orderings shrink within-window diversity and invite
sequential forgetting, so they are a controlled ablation rather than the default.

### The force

The full local MMD's gradient at a generated point is

```
∂L/∂x_i = (2/K²) Σ_{j∈W} [ ∇ k(x_i, x_j) − ∇ k(x_i, y_j) ]
```

Back-propagating through the `B` active rows alone reproduces it exactly via

```
L_force = (2/(B·K)) Σ_{i∈A} Σ_{j∈W} [ k(x_i, sg(x_j)) − k(x_i, y_j) ]
```

Three things this encodes, each a documented way to get it wrong:

- **The second kernel argument is always detached** — including where it is one of this step's
  own active rows. Letting both carry gradient and adding a symmetry factor of 2 double-counts.
- **The `K/B` correction is already inside `1/(B·K)`.** Relative to the full MMD's `1/K²`,
  dividing by `B·K` *is* that factor. Multiplying the loss by it again applies it twice.
- **The forward scalar is not an MMD.** It is a surrogate whose *gradient* is right; it is
  signed and its magnitude is not comparable across `K`. The monitored `mmd2` is logged
  separately and never back-propagated.

### What this trades away

The retained `K−B` features were produced by older parameters — at `K=1024, B=128` the oldest is
`⌈K/B⌉−1 = 7` optimizer steps stale. That bias is the method's central research risk, so drift
is a first-class metric, bucketed by cache age:

```
Δ = ‖x_cached − x_refreshed‖ / (‖x_refreshed‖ + ε)
```

Because latents are derived per **row id**, a re-roll reuses that row's original noise, so `Δ`
isolates parameter drift instead of mixing it with a different noise draw.

### Two invariants, both proven

**1. The force identity** — `∇_active L_force == (K/B) · ∇_x MMD²|_last B`, verified in float64
across three `(K,B)` shapes. This is what licenses back-propagating 128 rows instead of 1024.

**2. World-size invariance** — identical parameter gradients at world 1 / 2 / 4 (`rtol=1e-8`)
over real gloo processes, and across micro-batch 1 / 2 / 4. It rests on four properties:
per-row noise derivation, contiguous rank partition, *detached* all-gather of the context, and
per-rank `1/(B_local·K)` normalization combined with a **mean** reduction, which telescopes
exactly:

```
(1/R) Σ_r  2/((B/R)·K) Σ_{i∈A_r} (·)   ==   2/(B·K) Σ_{i∈A} (·)
```

So one config ports from a single GPU to a 32-rank job unchanged. `B = 128` divides 1, 2, 4, 8,
16 and 32.

### Optional: an adversarial term

`gan.enabled: true` adds a conditional critic (`rdm/sw_lmmd/adversarial.py`) on the same pooled
features the force compares. It has one projection head per encoder, conditioned on `τ(c)`, and
standardized by fixed reference statistics. The critic lives in GradCache's middle phase, so
it costs no rollout and no encoder forward, and passes 1 and 2 are unchanged. Each step:

```
pass 1  -> cached active features
critic  -> one step on the K-row window: reference rows (real) vs generated context (fake),
           same prompts on both sides, K/R rows per rank, gradients mean-reduced
loss    =  force + λ · adversarial(active rows, scored by the UPDATED critic, frozen)
pass 2  -> generator gradient;   λ = weight · |∂force/∂φ| / |∂adv/∂φ|   (global norms)
```

λ is adaptive because the raw scales are unrelated. On the real store against the base 1-step
student, `|∂adv/∂φ| / |∂force/∂φ| ≈ 2000–2800`, so `weight` is a gradient-norm ratio in feature
space rather than a raw loss multiplier. World-size parity, GradCache exactness, FSDP and
bit-exact resume are all tested with the critic on.

Two measured facts to plan around:

- **The store and the critic's fakes must come from the same encoder pipeline.** The store was
  extracted with fp32 encoder weights. With `memory.battery_bf16: true`, a held-out linear probe
  separates stored from re-encoded features of the *same* 4096 teacher images at 53–56 %. With
  fp32 weights it is at chance (50.0–50.2 %). GAN runs therefore use `battery_bf16: false`
  (`configs/sw_lmmd_train_h100_2gpu_gan.yaml`, ~72 GB/card estimated); the launcher warns
  otherwise.
- **The critic's gradient is nearly orthogonal to the force's** (cosine ≈ 0). Moving fakes
  along it barely changes the window MMD² (0.00483 → 0.00476, against 0.00345 along the force).
  It is a complementary signal, so judge it on GenEval/PickScore rather than on `raw_mmd2`, and
  sweep `weight`.

The critic sees one pooled vector per image, as the MMD does. A token-level critic (ADD-style)
would see more, but it needs the teacher renders online and the critic inside `encode_fn`.

---

## The reference

**The reference is teacher output, not real photographs.** The student is initialised from the
4-step klein-4B teacher, so photographs are a target neither model reaches and the objective
stops being distillation.

Following the paper's COCO block: render `SEEDS = 24` candidates per caption, keep the
**PickScore top-`KEEP`**. Curation is the mechanism, not a detail — it makes the target the
teacher's *best* output rather than its average, which is how a one-step student can exceed the
four-step teacher it came from. `KEEP > 1` also keeps each prompt's reference a *sample* rather
than a point; with one render per prompt the local MMD degenerates toward per-prompt regression
and diversity collapses.

| | paper | this fork |
|---|---:|---:|
| COCO block | 248,349 (82,783 × top-3) | **331,132** (82,783 × top-4) |
| GenEval block | 53,800 | *omitted* |
| total | 302,149 | 331,132 |

`KEEP=4` rather than 3 because all 24 candidates are rendered either way — keeping a fourth
costs no extra compute, and it partly offsets the omitted GenEval block. That block is skipped
because it needs the external `djghosh13/geneval` mmdet scorer, and the repo's own reported
numbers do not show it moving GenEval much (0.803 with it vs 0.805 without, from two different
in-repo sources — suggestive, not conclusive).

**Reusable from upstream's release:** the SigLIP2 τ(c) table — its first 82,783 rows are the
COCO captions, row-aligned to `coco_pairs.npz`, verified by re-encoding (matched-row cosine
0.99998, off-diagonal max 0.87). The pipeline slices it instead of running the text tower.

---

## Pipeline

```bash
bash scripts/setup_env.sh                                   # conda env (py3.12 + torch 2.8)
bash scripts/download_all.sh --root /data/<you>/rdm-sets    # ~50 GB of weights + COCO
source /data/<you>/rdm-sets/env.sh
bash scripts/preprocess_all.sh                              # build what nobody hosts
GPUS=2 bash scripts/train_sw_lmmd.sh configs/sw_lmmd_h100_2gpu.yaml
```

`scripts/README.md` documents each script. Preprocessing is three resumable steps:

| step | produces | measured on 1× RTX 4090 |
|---|---|---|
| `preprocess_01_ctx.sh` | Qwen3 generator context `(N, 48, 7680)` | ~61 GB out, single GPU |
| `preprocess_02_render.sh` | 4-step teacher renders + PickScore curation | **3.6 img/s** → 1,986,792 renders ≈ 26 h on 6 GPUs |
| `preprocess_03_features.sh` | per-row encoder features + the assembled store | **134–289 img/s** per encoder |

Curation is **streamed** — candidates live in RAM and only the kept images are written, so the
24-seed recipe never needs the ~795 GB that writing every candidate first would cost.

---

## Moving to a new machine

The expensive artifact is the reference store, not the code. **Copy it rather than rebuild it**
— it is ~30 GPU-hours to regenerate and 70 GB to move, and the 132 GB of teacher PNGs are
*not* needed at train time (only their features are).

```bash
# ---- on the new machine ----
git clone <this fork> && cd RDM
bash scripts/setup_env.sh                                  # conda env: py3.12 + torch 2.8 + cu126
bash scripts/download_all.sh --root /data/<you>/rdm-sets   # ~50 GB weights + COCO
conda activate rdm && source /data/<you>/rdm-sets/env.sh
```

Then either **copy the store** (fast) or **rebuild it** (`bash scripts/preprocess_all.sh`, ~30 h).

### Copying the store

Two directories, 70 GB total:

| what | size | needed at train time |
|---|---:|:--:|
| `sw_lmmd/reference_store/` | ~9 GB | ✅ |
| `sw_lmmd/qwen3_ctx_coco.npy` | 61 GB | ✅ (the store symlinks it) |
| `sw_lmmd/teacher_renders/` | 132 GB | ❌ only needed to re-extract features |

```bash
rsync -aP --exclude teacher_renders \
    old-host:/data/old/rdm-sets/sw_lmmd/ /data/<you>/rdm-sets/sw_lmmd/
```

**Then re-point the context symlink** — `qwen_context.npy` inside the store is an absolute
symlink written on the old host, so it arrives broken:

```bash
STORE=/data/<you>/rdm-sets/sw_lmmd/reference_store
ln -sf /data/<you>/rdm-sets/sw_lmmd/qwen3_ctx_coco.npy "$STORE/qwen_context.npy"
```

### Verify before launching

This opens the store through the real reader and checks a full window's worth of rows, which
catches a broken symlink, a truncated copy and a row-count mismatch in one shot:

```bash
python -c "
from rdm.sw_lmmd import ReferenceFeatureStore
import json, os
root = os.environ['STORE']
names = list(json.load(open(root + '/metadata.json'))['encoder_feature_dims'])
s = ReferenceFeatureStore(root, names)
rows = s.row_order()[:1024]
print(s.num_rows, 'rows /', s.num_prompts, 'prompts')
for n in names:
    print(' ', n, tuple(s.reference_joint_features(n, rows).shape))
print(' ctx', tuple(s.generator_context(rows[:4]).shape))"
```

### Launch

No YAML editing needed — the paths are overridable:

```bash
REFERENCE_ROOT=/data/<you>/rdm-sets/sw_lmmd/reference_store \
GPUS=2 bash scripts/train_sw_lmmd.sh configs/sw_lmmd_h100_2gpu.yaml
```

`GPUS` must divide `B` (128 by default; the H100 configs use `K = 128, B = 32`).
`FSDP=1` if the card cannot hold the training state,
`MICRO_BATCH=<n>` to trade throughput for memory, `STEPS=<n>` for a short gate run. The
preflight prints the resolved window, batching, memory estimate and store path before
torchrun starts.

### What must match, and what need not

- **Must:** `ctx_len` (48) between the store and `flux_ctx_len` at eval — the student would
  otherwise be evaluated at a sequence geometry it never trained on. The store's
  `metadata.json` records it along with `row_order_hash`.
- **Need not:** GPU count, GPU model, micro-batch. World-size parity is a tested property, so
  a 2-GPU run and a 32-GPU run optimize the same objective.
- **Watch:** `HF_HOME` must be the *parent* of the hub cache holding the blobs —
  `flux_generator.py::_hub_root()` reads `HF_HOME` alone and ignores `HF_HUB_CACHE`. The
  generated `env.sh` keeps them consistent; a stray `HF_HUB_CACHE` in your shell profile is the
  usual way this breaks.

---

## Hardware

`shard: none` is **data parallel**: every rank holds a full copy of the training state, so
adding GPUs buys throughput, never headroom. Per-card budget for klein-4B (3.875 B params):

| | training state | per card | 80 GB | 24 GB |
|---|---:|---:|:--:|:--:|
| fp32 + AdamW | 62.0 GB | ~76 GB | tight | ✗ |
| **bf16 + AdamW** | **46.5 GB** | **~66 GB** | ✅ | ✗ |
| bf16 + 8-bit Adam | 23.2 GB | ~43 GB | ✅ | ✗ |
| FSDP ×4 + 8-bit | 5.8 GB | ~26 GB | ✅ | tight |

Note what SW-LMMD itself costs: the feature cache is `10 × 1024 × ~2700 × 4 B` ≈ **111 MB**, and
the reference window the same. **0.36 % of the total.** SW-LMMD is a *compute* optimization —
peak memory is dominated by the 4B model's training state either way.

`FSDP=1` shards params/grads/optimizer state (ZeRO-3). Needed only where the training state does
not fit one card; an 80 GB card is **faster without it**, since every MM-DiT block costs an extra
all-gather. Three settings in the wrapper are load-bearing: per-block auto-wrap (a flat shard
would all-gather all 3.9 B params at once), `reduce_dtype=float32` (bf16 gradient reduction
loses precision across thousands of values), and `FULL_STATE_DICT` (the default writes per-rank
shards no evaluator can load). Gradient clipping is FSDP-aware — plain `clip_grad_norm_` would
compute each rank's *shard* norm and silently rescale the step.

---

## Layout

```
rdm/sw_lmmd/          this fork's method
  window_schedule.py    overlapping windows over a seeded permutation (+ the overlap assertion)
  cache.py              cross-step generated-feature cache, per-row ages, row-identity checks
  local_mmd.py          exact biased local MMD + the active-only force
  sharding.py           per-row noise, contiguous partition, detached all-gather
  reference_store.py    mmap row-addressed reference (sharded ctx, row→prompt indirection)
  trainer.py            bootstrap → GradCache two-pass → force → reduce → slide
  refresh.py            staleness probe and refresh
  adversarial.py        optional conditional critic on the cached features (gan: block)
  launch.py             config → objects, training loop, entry point

rdm/compare/          upstream: kernels, Nyström, the iRDM loss + ablation distances
rdm/representation/   upstream: the 14-encoder battery, generators (pMF-H, FLUX.2), joint feature
rdm/train/            upstream: the iRDM loop, GradCache (reused verbatim), PID controller
rdm/eval/             upstream: SW_r14, MMDr14, GenEval, PickScore
scripts/              setup → download → preprocess → train
```

---

## Status

| | |
|---|---|
| method layer (schedule, cache, force, sharding, store, trainer) | ✅ implemented, 124 tests |
| force identity + world-size parity | ✅ proven |
| FSDP, 8-bit Adam, configs for 4090 / 2×H100 / 8×H100 | ✅ written, arithmetic checked by test |
| reference store | ⏳ building (~30 h) |
| **any run on the real 4B model** | ❌ **never executed** |
| semantic row-alignment check on image features | ❌ needs the store |
| GenEval reference block | ❌ needs the external mmdet scorer |

The first real run is an integration gate, not a training run. The likeliest failures are dtype
consistency under bf16, the activation estimate (the one term in the memory table that is
estimated rather than computed), and encoder behaviour on bf16 inputs.

---

## Attribution

This is a fork. The method, code, released checkpoints and the `configs/flux*.yaml`,
`rdm/{compare,representation,train,eval,toy,refprep}` trees are the work of the original authors:

```bibtex
@article{feng2026irdm,
  title={Representation Distribution Matching for One-Step Visual Generation},
  author={Feng, Lan and Li, Wuyang and Zablocki, {\'E}loi and Cord, Matthieu and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2607.02375},
  year={2026}
}
```

[Project page](https://alan-lanfeng.github.io/rdm/) · [arXiv](https://arxiv.org/abs/2607.02375) ·
[Checkpoints](https://huggingface.co/epfl-vita/flux2-klein-1step-rdm)

Upstream docs are preserved under `docs/` — `reproduction_map.md` (artifact → command → paper
table), `flux_reference.md` (the FLUX reference build), `flux_geall_assets.md` (the headline
reference spec), `method_notes.md` (design log and pitfalls).

MIT (Copyright 2026 Lan Feng). Vendored third-party components and their licenses are in
`THIRD_PARTY.md`.
