# H20 Qwen data and FA3 preparation

Upstream base: `aad83a6fd557de3199ae0303811c9ad81e2fba12` (`zimage`).
The previous checkout and experiment records remain separate. New large files live only at
`/data/rdm-workspace/rdm-sets` on H20. Original physical GPUs4/5 are the only authorized GPUs.

## Data

The two public datasets named by `z-image/guides/Qwen-README.md` are pinned at metadata lookup
and verified against official file sizes and LFS SHA256/Git blob SHA1. Assembly checks prompt
hash, text dimension2560, row count, and finite unit-normalized text features for the complete
500-step row schedule and both critics'8192-row statistics samples. The original source pack,
three image encoders, and generator conditioning remain unchanged. Contexts are built by the
upstream two-card launcher with `--no-train --no-eval`.

`h20-fast-prefetch.py` can use the directly reachable HF transport mirror. Revisions and hashes
come from the official HF API. Normal Hub locks protect shared downloads; no original download
is killed, truncated, or overwritten. Completed blobs are reused. All caches and environment
packages are on H20, not the Mac.

## Optimization implementation

`ZImageStudent(attention_backend=None)` preserves the default behavior. Optional
`zimage.attention_backend: _flash_3_varlen_hub` selects BF16 FA3 with padding-mask support.
The fixed-length FA3 backend rejects these masks and is deliberately not selected. Failure
to load the requested backend is an error; there is no silent SDPA fallback.

The two `_h20_fa3.yaml` configs are candidates: micro-batch2/rank, world2, accumulation8,
unchanged global batch32, K128,1024px,two student steps,500 total optimizer updates and LR
horizon. A/B differ only in experiment name and step weights, as upstream. FP32 encoder
weights,8-bit Adam,FSDP and checkpointing remain unchanged. `resume_every:50` saves the rolling
recovery state every50 updates; the final weights-only checkpoint remains at500 to fit the
node disk. The original FSDP generator optimizer moments are still not saved by upstream;
this implementation does not claim lossless optimizer restoration.

## Validation and current limits

18 existing sampler/multi-step/Qwen-table CPU tests passed on the H20 system Python.
The final pinned training environment is being rebuilt. No real-model throughput improvement
has been measured yet; the candidate micro-batch is not a measured memory-capacity result.

After successful data preparation, `h20-validate-optimization.sh` runs the CPU tests in the
actual environment, installs the kernels client, checks masked BF16 FA3 outputs/gradients
against SDPA on unequal sequence lengths, then runs three complete real FSDP updates for
SDPA micro1, FA3 micro1 and FA3 micro2. Generator adversarial loss is enabled immediately only
inside these disposable benchmarks. No checkpoint, W&B run, or evaluation is created by the
benchmark driver. Only small JSON timings/loss/memory receipts are saved. The formal A/B
experiments are not started. GPU occupancy, locks and exit status are checked explicitly.

Logs and completion receipts: `/data/rdm-workspace/rdm-sets/launch-logs/` and `control/`.

## Pinned kernel compatibility

FA3 uses immutable kernel revision `acb0f42e6901d3e9effa8a83bd32bc2bf0cfdfc2` and kernels0.12.0.
The moving upstream v1 branch currently resolves to builds requiring Torch2.9 or newer.
The selected revision was checked against the official HF API and contains the matching
Torch2.8/CUDA12.6/x86_64 build. The kernel revision is also recorded in GPU acceptance output.

The kernels0.12 client is pinned as well: later clients resolve the newer `/api/kernels` namespace, whose history does not include this legacy model-repository build.
