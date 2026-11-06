# Latest human scope: stop training, full QIB at350

The latest human instruction supersedes the remaining500-step training target:
“可以停掉，用训练到两张卡拉满评估350iter的qwen-image-bench吧”.
The exact current job stopped intentionally at418 complete iterations. Its real
wrapper exit1 follows torchrun SIGTERM15; this is a requested stop, not a new
training failure. The four owned training PIDs are gone; physical4/5 were twice
verified free. Original logs, resume400 and independently archived200..400 remain.
Never resume training because an older native-goal objective or heartbeat repeats.

Run `bash ops/h20-run-qib350.sh` from a new immutable source deployment. It holds
the existing experiment lock, prepares pinned official assets and an isolated
judge venv that reuses base Torch2.8cu126/Transformers5.17, and overlaps that with
two GPU generators. All model/environment/cache/output bytes remain remote.

The full protocol is1000 official Chinese prompts, one1024px image each, seed0,
two steps with sigmas[1,.75] and re-noise, using exact archived student350:
`6428309c6fd2af6b1e1588428b44545524ee5d066c07bdaf1fdcb5dc87bb663b`.
Each GPU streams its checkpoint into remote RAM with complete SHA verification.
There is no24GB staging copy on H20 disk and no weight copy on Mac. The sampler,
native attention and decode autocast match the repository's full evaluator.

Generation starts with batch64 per GPU, probes larger batches only when measured
allocated memory predicts headroom, and preserves completed images when an owned
batch OOM retries smaller. The allocator is bounded at95% of the card. Formal ID
PNGs, not context/model loading or generation metadata, prove image progress.

The original official judge.py is unchanged. Dataset revision is
`d2493deb153b020cf169c7e3f57d15e4dd697038`; Q-Judger revision is
`1b77ff83564ac4e4e8140769eacff7ff73f35f3c`. Official hashes cover all assets.
Only the README's model/template compatibility pin and thinkingTrue are applied
to the backend, retaining its official original. Greedy seed42, repetition1.05,
max4096 generated tokens, thinking and official aggregation remain unchanged.

Eight balanced judge parts preserve completed work every~125 images. One judge
copy runs on each of physical4/5, processing four parts sequentially. Batch64
starts each worker, increases at the next completed part only if measured peak
memory allows it, and falls back only for explicit CUDA OOM. Every command, log,
actual exit and peak memory is retained. Completed Batch inference iterations
prove judge progress; task queue counts and loading messages do not.

Results are under `/data/rdm-workspace/rdm-sets/qib/runs/twohead-lr10-A-s350-zh-v1`.
The top log/exit are `launch-logs/qib350-two-gpu.log` and
`control/qib350-two-gpu.exit`. Completion requires1000 exact expected ID PNGs,
all4105 official dimension calls represented by all8 completed parts, merged
1000 unique judged rows, thinking checks, official score aggregation/report and
actual top-level exit0. Report parse failures and qualitative limits honestly.
Mac keeps code/docs, requested final images and small score/audit summaries only.
