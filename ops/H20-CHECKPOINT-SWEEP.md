# H20 checkpoint/capacity sweep

Authorized scope: test only physical GPUs4/5, FA3, 1024px, global batch32, two-step generator;
no formal A/B run. Original data/source and all previous benchmark logs are preserved.

`checkpoint_group_size:2` groups consecutive DiT blocks in the existing non-reentrant
checkpoint callback. Each block remains its own FSDP unit. Canonical state_dict numbering is
unchanged. No-grad bootstrap/refresh and checkpoint-disabled paths retain individual iteration.
The adapter is for eager training, rejects ControlNet residual indexing and makes no compile
compatibility claim. VAE checkpointing remains enabled throughout these tests.

The isolated source is installed at `/data/rdm-workspace/RDM-qwen-perf`; it imports the existing
pinned training environment and cached assets without changing the earlier source checkout.
The launcher waits for the original micro2 validation to exit successfully and for GPUs to be
idle, then holds the task run lock. It runs CPU parity tests and a tiny real two-rank FSDP
output/gradient comparison before any full-model grouped update.

The sweep tests pair checkpointing at micro2/4/8/16, then per-block at micro4/8/16, then
DiT checkpointing disabled at micro1/2/4/8/16. Accumulation is16/micro, so global batch32 stays
fixed. Every successful candidate completes at least2 real updates (first excluded from speed)
and the original drift-probe/retained-refresh path; pair micro2 and the repeated winner get3.
An OOM ends that mode's ascending search. Failed cases remain on disk with exit status, exact
command, bounded error tail and observed GPU memory. Only the owned test process group can
be terminated on timeout. Other users' jobs are never stopped or modified.

The summary reports the highest observed feasible capacity and fastest repeatable point,
which need not be the same. The recommended winner reserves at least5% of device memory if
any tested configuration provides that margin. NVML peaks are polled separately from Torch
allocated/reserved peaks. OOM bounds are empirical; a short gate does not prove every future
sequence/allocator pattern will fit. Formal training remains off pending the user's changes.

Status: `/data/rdm-workspace/rdm-sets/control/checkpoint-sweep-latest.json`.
Per-case logs/results: directory named by its `output` field. Entry: `ops/h20-run-checkpoint-sweep.sh`.

The initial external guard was retired after it misidentified a relative-path controller
and interrupted micro4 after two complete updates but before refresh. Its failure logs remain
under the first scan directory. It is not started by the launcher. The current controller
handles TERM/errors/timeouts in-process and cleans exact owned descendants, including ranks
with independent sessions. Successful earlier cases can be reused via `--resume-summary`;
partial or failed cases are never admitted as passed. Probe/refresh values must also be finite,
and a repeated winner must satisfy the5% margin. BF16/FP32 two-rank parity diagnostics (four
accumulated micros, with and without shared conditioning) passed with zero gradient error.

The first micro8 attempt hit Torch2.8's nearest-NHWC backward INT_MAX index limit at
`[8,256,1024,1024]`; this was not an OOM. The VAE adapter now uses the existing NCHW
nearest kernel for at-risk input/output element counts, forcing contiguous layout in both
forward and backward. Smaller tensors call the original Diffusers method verbatim.
Logical DiT/VAE batch, parameters, checkpoint keys and interpolation values are unchanged.
CPU forward/backward/second-gradient tests and a small CUDA FP32/BF16 adapter gate run
before the full model. `--only-case 2 16` prioritizes the user's batch16/accumulation1
request with three updates plus probe/refresh, without declaring the whole sweep complete.
