# New-MMD2 student500 official Qwen-Image-Bench

Latest user requests full QIB evaluation of the current500 checkpoint, and deletion
of only this experiment's50/100/150/200 checkpoints. Native goal stays paused;
this explicit manual evaluation is authorized. No new training starts.

Use physical4/5/6/7 only after fresh idle checks. The immutable training source is
`/data/rdm-workspace/RDM-dino-qwenvl-mmd2-v7`, commit
`ad5dddc428a32591fedf9b034bf5f1639ced0514`. Student500 SHA256:
`7a189cf104b9c4b7f09ef7e27b090e1824ca520a35c9626ff734bba2d47bae32`.
Both GAN and MMD used DINOv3+Qwen-VL. Four generators map rank0..3 to GPU4..7.
The student is staged once with full final-name SHA into ordinary remote tmpfs;
GPU workers mmap that verified regular file, never an object-mount checkpoint.
The existing Q-Judger tmpfs shard is reverified and its moved symlinks checked.

Protocol matches previous full runs: official1000 Chinese prompts,1024x1024,
seed0, student two-step sigmas[1,.75], re-noise/detachTrue, BF16 parameters,
native attention, no sampler/decode autocast. Benchmark revision:
`d2493deb153b020cf169c7e3f57d15e4dd697038`; judge:
`1b77ff83564ac4e4e8140769eacff7ff73f35f3c`. Original official judge/aggregation,
thinkingTrue, greedy seed42/repetition1.05/max4096 tokens remain unchanged.
Eight balanced125-image parts are assigned two per GPU; initial judge batch80,
OOM-only retries halve it while retaining all failed logs. Actual images and
completed `Batch inference` batches, not initialization/PIDs, prove progress.

Entry: `bash ops/h20-run-qib-mmd2-500.sh`. Prefix:
`qib-mmd2-s500-gpu4567-v1`. Run:
`ROOT/qib/runs/mmd2-from0-s500-zh-1024-seed0-v1`.
Every child command/log/exit is stored. Final acceptance requires1000 exact IDs,
all4105 raw and parsed dimension answers,8 completed parts, thinking checks,
official scores/comparison report, full output archive final-name SHA and real
wrapper exit0. Full outputs stay remote under the neutral project object path.

`h20-delete-mmd2-early-four.py` binds the four exact object paths to existing SHA
receipts, verifies complete500 identity and protected250..500/old350/resume sizes
and mtimes before/after, preserves logs/receipts/previews and writes deletion and
current-retention manifests. No unrelated experiment is deleted. Historical
all-ten-checkpoint manifests remain historical; the current list supersedes them.
