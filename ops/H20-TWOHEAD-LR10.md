# Corrected DISC and generator LR tenfold restart

Latest user instructions supersede the earlier choice to keep the legacy job.
The user requested stopping/restarting, G base LR x10, real and generated images
both through Qwen-VL/DINOv3, and four GPUs. GPUs6/7 were occupied by unrelated
`/data/qyl` jobs; the user then explicitly chose to use our two GPUs instead.
Only the owned legacy launcher was stopped. It ended after317 complete steps;
its step300 resume, preserved150/200 weights, source and all logs remain intact.
SIGTERM produced old exit1/Elastic SignalException, an intentional user stop.

The new fresh500-step configuration is
`z-image/configs/zimage_2step_1024_twohead_glr10_h20_fa3.yaml`.
G base LR is2.83e-5, with the original100-step warmup/cosine schedule. D base LR
is1e-5 with the prepared adaptive policy. Both GAN heads use NS/softplus.
The DISC has exactly two image branches: DINOv3 and Qwen3-VL. Both reference PNG
pixels and generated PNG-equivalent pixels pass through the same frozen1024px
spatial encoders. Each map is fused with the existing Qwen3-4B caption condition.
No AIM or SigLIP GAN head is used; the GAN averages its two heads with1/2.
The original MMD image streams remain AIM/DINOv3/SigLIP, with their original
Qwen text vectors, kernels, weights and preprocessing. No original run is resumed.

Clip threshold4 remains: in the stopped317-step job, the previous316-step
read-only observation showed only6 norms above4 and one in the last100.
The user rejected extra gradient/update telemetry. Only the existing global
`grad_norm` and existing GAN/MMD logs remain; there are no new per-layer norm,
parameter-copy, update sampling or BF16-effectiveness operations in training.

`h20-validate-twohead-lr10.sh` runs the CPU/Gloo suite, including an end-to-end
test that captures reference and generated pixels entering both spatial image
encoders, excludes AIM GAN, and retains all three MMD streams. It then uses only
physical GPUs4/5 to measure micro2 and micro4, checkpoint groups2 and FA3. Every
successful case must complete3 finite non-skipped real D/G updates plus probe64
and refresh96 on both global/spatial caches. Timings exclude the first update.
An actual CUDA OOM ends the capacity search; unrelated failures stop validation.
All failure logs and receipts stay remote. No weights are saved by benchmarks.

The selected measured configuration is written to
`ROOT/control/twohead-lr10-selected.json`. After validation exits0, launch the
already-authorized fresh job with `bash ops/h20-run-twohead-lr10-a.sh MICRO`,
using its measured selected microbatch. The entry verifies the source-bound
receipt, fresh output directory, remote data/output/cache paths and GPU4/5 idle
state, then runs to500 with save interval50. It records stdout/stderr, exact
command, PID and true exit under `twohead-lr10-A-500`. G optimizer serialization
retains the existing FSDP limitation; this restart creates a fresh optimizer.
Models and all full outputs remain remote. No other jobs or GPU6/7 are touched.

When student/resume writes coincide, replace the rolling resume first, then write
the student. Contents and intervals are unchanged. This frees the previous
rolling inode before adding the final student, avoiding the extra simultaneous
disk requirement while retaining all old experiment artifacts. Completion still
requires both actual step500 payloads and the true clean exit.

`ROOT` means `/data/rdm-workspace/rdm-sets`. Verify at least one full real formal
iteration and a second real observation after launching; the full goal remains
active until actual500 artifacts and clean exit are audited.
