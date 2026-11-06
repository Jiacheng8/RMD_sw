# Read-only verification of the running two-head A experiment

The active training checkout remains /data/rdm-workspace/RDM-twohead-lr10-v1,
commit a25e4c42794b13d5c881d3f1994944929b019190. This branch adds operational
readers only. Do not deploy it over the active checkout or restart training.

ops/h20-observe-twohead-a.py is a one-shot system-Python reader of the fixed
formal command/PID/exit, actual JSONL iterations, original grad_norm, errors,
save-completion logs, real file stats, GPU state and free disk. It launches no
GPU computation, inference, evaluation or training, and writes nothing.

ops/h20-audit-twohead-checkpoint.py --step 50 uses the actual Torch environment
with CUDA hidden and mmap on CPU. It checks payload/schedule positions, full
model key/shape/dtype schema, exact two-GAN contract, saved D Adam counters and
LR/controller agreement, original six MMD streams, four spatial streams and
row/age alignment. While training runs it reads metadata and small counters only;
it does not scan weight values, compute gradient/parameter norms, sample updates,
or measure BF16 changes. No training hooks or extra tensor copies are introduced.

For actual final completion, after the formal wrapper exits0:
--step 500 --full verifies the rolling resume, then
--step 500 --student --full verifies the student. Full mode checks saved tensor
finiteness and file SHA256 only after clean completion; it is rejected during
training. All checkpoints remain remote and only the small JSON receipt is
returned. Source binding and final step must match this fresh run; the preserved
legacy300/three-head checkpoint is a negative control and cannot pass.

No training modules, configs, optimizers, save intervals or cache policy are
changed. The existing FSDP generator-optimizer serialization limit remains:
the full generator Adam is absent from resume; D optimizers/controller are saved.

Observer output retains only the latest four detailed probe records, with full
probe/refresh step lists and mismatch counts over every JSONL iteration. This
bounds its output at500 and prevents transport/output truncation while retaining
complete counter, phase, unexpected-AIM and row-count acceptance evidence.
