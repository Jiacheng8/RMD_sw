# Original Turbo two/nine-step Qwen-Image-Bench

Latest human request: evaluate original Z-Image-Turbo at2 and9 steps using other
GPUs on this same H20 node. On2026-11-05 physical6/7 were measured idle,0MiB
compute use. Existing MMD2 training remains unchanged on4/5.

Entry: `bash ops/h20-run-qib-baselines.sh`. Independent lock/control prefix
`qib-turbo2-turbo9-gpu67`; never acquires the active training lock. Checks6/7
again immediately before loading; refuses occupied cards. No allocations on0..5.
No new model downloads or environment changes.

Run names `turbo2-zh-1024-seed0-v1` and `turbo9-zh-1024-seed0-v1`. Each has1000
official Chinese prompts, one1024x1024 PNG per expected ID, seed0, original
pretrained snapshot f332072aa78be7aecdf3ee76d5c247082da564a6 and native shifted
Euler schedule (`renoise=False`). The trained350 run uses `renoise=True`; the
native Turbo2 baseline is therefore distinct from pretrained0 previews using
the student's re-noising sampler. Text contexts/native BF16 model/VAE and
per-ID noise use the existing evaluator implementation.

Both cards split generation evenly, adaptive batch starts32, allocator cap95%,
finite pixels and atomic PNG writes. Generate and finish judging Turbo2, then
Turbo9. Each evaluation reuses the same pinned benchmark, Q-Judger, thinking
enabled, BF16, original judge.py and aggregation as350. Eight125-image shards,
two judge copies on6/7, batch80 (previously completed on this GPU model), bounded
OOM backoff only; no arbitrary inference restarts on observation timeouts.

Acceptance requires all1000 correct-resolution ID images, actual successful
generator/split/eight judge+thinking/merge/report exits,4105 raw and4105 parsed
L1 answers with exact metadata dimension sets, finite official scores, and a
full output archive final-named SHA readback on same-cluster object storage.
Source outputs remain remote; Mac only gets code/docs/small summaries.
Comparison report includes both original baselines and the verified350 run.

All subprocess commands/PIDs, source commit, real exits, failures/tracebacks and
stdout/stderr are durable under each run/logs and ROOT/control. Wrapper final
exit: ROOT/control/qib-turbo2-turbo9-gpu67.exit. Completed files/failed logs are
preserved; wrapper refuses a duplicate launch. No GPU occupation placeholders.

Recovery2026-11-05: first attempt produced all1000 Turbo2 images, both generator
exits0, but both judges died with actual SIGBUS(-7) while safetensors loaded the
same offloaded49.825GB object-mount shard. No completed judge batch/part;9-step
had not started. Preserve all old control records and logs before a fresh
workflow attempt; generation resumes existing verified PNGs without recompute.

The recovery stages that single shard by sequential reads into remote `/dev/shm`
(measured1.03TB free,1.9TB system RAM available), checks its pinned official SHA
during copy and rereads the final named RAM file for SHA again. The judge gets
an explicit model directory with this ordinary tmpfs file; other files link to
the unchanged original model. No mmap directly on the offloaded object file,
no node-local disk duplicate that would compete with training checkpoints,
no Mac weights and no changes to shared model paths or running training.
This addresses the failing storage access path; it does not claim an unobserved
kernel-level cause. After recovery require actual completed inference batches,
not merely a successful weight load, before reporting effective progress.
