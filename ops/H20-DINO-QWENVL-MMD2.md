# Fresh0 two-encoder MMD successor

Latest human instructions: finish full350 Qwen-Image-Bench, start the successor
from the official pretrained generator at0, replace SigLIP with Qwen-VL and remove
AIM from MMD. Training during judge download was permitted, then evaluation when
weights finish. Judge weights are now verified, so active350 evaluation keeps
priority; reference preparation runs concurrently in bounded spare memory. New
full-memory training follows actual evaluation completion. The old418 stop does
not pause this newly authorized successor and the old500 objective must not cause
the old experiment to resume. Both350 evaluation and new500 training remain work.

The successor config is `z-image/configs/zimage_2step_1024_dino_qwenvl_mmd2_from0.yaml`.
MMD encoders are exactly DINOv3 and Qwen3-VL; GAN retains its two1024px spatial
PatchGAN heads DINOv3/Qwen-VL. Text remains Qwen3-4B2560D. The existing MMD Qwen
adapter uses the pretrained vision tower+merger final-token mean2560D, with256px
internal input after the shared ADM512 preprocessing. The existing GAN Qwen
spatial feature contract, including its language-decoder path, is unchanged.
No new image-feature L2 normalization is introduced: raw image features and the
existing per-encoder RBF calibration are retained. Text remains L2 normalized.

`bash ops/h20-run-dino-qwenvl-reference.sh` builds a separate reference view from
the already retained125808 canonical1024px teacher PNGs. Every input PNG is
checked against the canonical index SHA; source stores remain immutable. Each
GPU4/5 worker has a3.5GiB allocator budget, FP32 frozen vision weights and BF16
compute, and commits disjoint float16 rows with resumable chunk hashes. The
encoder's finite nonzero pixel gradient and frozen parameters are checked once.
DINO same-row alignment keeps the existing relative<=.03/cosine>=.999 gates.
Publication requires full125808-row finite coverage, consistent contracts, a
fixed sigma=.25*median over2000 deterministic stored image rows, beta=sigma/s_txt,
finite required500-step/statistic text rows with unit norm, and valid reference
statistics. The view contains only DINO/Qwen image streams and shares the original
text, contexts, row order and prompt identities through verified links.

Generator base2.83e-5, D base1e-5, softplus, adaptive D policy, clip4, global32,
micro2/accum8, FA3 and checkpoint groups2 remain the current settings. Student
weights must retain every50..500 milestone with final-named SHA readback. The
old resume400 stays local. During judging, `h20-stage-judge-cache-offload.py` copies
the49.8GB first judge shard to shared object storage and verifies its final SHA.
The active source path remains untouched until actual full350 benchmark exit0,
all8 parts and4105 answers, and judge subprocess exit. Only then replace that
verified local cache payload with its remote symlink, recovering enough space
for the new rolling resume and a student snapshot plus margin. Never release an
unverified checkpoint. Require actual350 benchmark exit0, complete
new reference, sufficient remote checkpoint/cache space and a real two-GPU
D/G-active smoke gate. Reinitialize from official pretrained0 after smoke; do not
carry its updates or warm from350/400. Normal commits/main push and full Mac
code/docs backup are required; all weights/features/environments stay remote.

`bash ops/h20-run-dino-qwenvl-mmd2.sh` is the source-bound handoff controller.
It runs changed-path CPU checks with CUDA hidden while350 judging continues.
It waits for real benchmark exit0/full output, complete new reference and verified
judge-cache offload, then requires62GB local staging headroom and acquires the
existing run lock. It runs only the existing group2/micro2 real smoke:3 complete
D/G-active iterations and retained-window probe/refresh. It validates exact MMD2
and GAN2 identities, finite non-skipped updates and measured memory headroom.
The smoke process exits before formal fresh0 initialization. No smoke updates
carry into training, whose GAN warmup remains50 and whose full budget remains500.
The archiver starts at50 and preserves every50..500 student with full final-named
SHA verification. Failure exits retain command, stage exit and logs; the controller
does not retry/restart formal training. Final full-resume audit and reporting are
separate from proving training/weight retention complete.

Final acceptance is CPU-only after actual training exit0, never added runtime
telemetry. `h20-audit-mmd2-final.py` binds the exact source/formal command, replays
all500 actual adaptive-D decisions, checks both D Adam states, global MMD2 and
spatial cache row/age/shape alignment, and scans the saved tensors for finite
values. It SHA-loads archived student500 to remote RAM and compares all521 FP32
tensors exactly with the rolling resume500. The original FSDP G-Adam serialization
limitation is retained and recorded honestly.
`h20-preserve-qib350.py` requires actual full benchmark exit0, all8 successful
thinking checks, merged1000 unique IDs,4105 raw answers and official scoring/report
before streaming the complete run to shared storage and checking final-named SHA.
The source run remains on the node. `h20-finalize-qib350-mmd2.py` waits for these
existing processes and runs preservation/final integrity in sequence; it cannot
launch or restart training/evaluation. Final curves and a read-back of actual
scores/integrity are still required before declaring the overall goal complete.

Loss exports for this successor use the separate read-only scalar reader
`ops/local-control/h20-mmd2-loss-snapshot.py` through the existing SSH helper and
`ops/h20-plot-mmd2-loss.py` locally. The reader binds immutable trainingv3/source
and passed fresh0 preflight, includes only complete actual JSONL iterations and
records its config/log-prefix hashes. A not-started/initializing response does
not produce a graph. The stdlib contract checks actual MMD and GAN metric names
in every row, rejecting historical SigLIP/AIM data even if metadata was relabeled.
Both generation steps show softplus D, active-only softplus G and two-encoder
MMD, plus G training surrogate and the existing global grad_norm. Warmup proxy
values never appear as active G losses. Moving means are trailing10 iterations;
vertical D LR markers come only from actual controller actions. Output is a
standalone PNG/PDF; no extra training metrics/hooks or fake new-run plots.

The first real two-GPU successor smoke (trainingv3/e863fe5) exited1 before any
optimizer update: SpatialGANRuntime still required the old three-MMD panel.
The corrected runtime accepts exactly either the legacy3 panel or the new
DINO/Qwen2 panel, with no change to spatial heads, softplus, head averaging or
learning rates. Legacy initialization/RNG behavior remains covered. A new CPU
integration case constructs the real critic from a DINO/Qwen-only store and
executes finite D and G updates with no SigLIP/AIM feature access. This integration
test is now part of the launch CPU gate, followed by the same real3-iteration
FSDP smoke. Failed command/exit/logs are preserved before a fresh attempt; formal
training never started in the failed attempt. The finalizer reuses the already
full-SHA-verified350 archive, and scalar snapshots bind the actual immutable fresh0
formal command/preflight source instead of the superseded failed smoke source.
