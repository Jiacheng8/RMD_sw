# Authorized200 handoff and independent milestone weights

The user requested a resume after200 and independent weights every50, explicitly
including200,250,300,350,400. This supersedes the earlier500-only student retention.
Training still targets500 total updates on physical4/5, micro2, accumulation8.
GAN/MMD, LR schedules and adaptive-D behavior are unchanged.

`h20-stop-at200.py` waits for the actual200 save-completion line, reads the payload
using the existing CPU metadata auditor, checks the exact owned launcher identity,
records the current user authorization, and sends SIGTERM to that launcher only.
The resulting old-run signal exit is an intentional handoff, not an unexplained failure.

`h20-resume200-a.sh` verifies that handoff, checkpoint inode metadata, idle GPUs,
and unchanged model/algorithm files against the previously validated source. It resumes
the same data/cache/controller/D-Adam at200 to500, with `save_freq=50` and
`resume_every=50`. The old FSDP snapshot lacks G Adam; the user was explicitly told
that this authorized resume resets G Adam. No loss/LR/precision changes are bundled.

The local `/data` filesystem has approximately55GB free at handoff. Independent
FP32 students are about24.62GB each, so seven200..500 students cannot all stay there.
The remote mounted object-store destination is:

`/mnt/airs-business/shaoshitong/rdm-sets/checkpoints/zimage-2step-1024-glr2p83e-5-twohead-softplus-adaptive-A`

The mount passed actual64MiB write/fsync/rename/SHA readback. Its synthetic72PB
capacity from `df` is not a verified bucket quota and is never reported as such.

`h20-export-student200.py` exports the actual200 FP32 model, without cache/critics,
from the audited resume. Later students use the original trainer's checkpoint function.
`rdm.sw_lmmd.checkpoint_archive` works concurrently on CPU, validates each saved
model schema and step, copies to a private partial object, hashes during copy,
fully re-reads the remote SHA, publishes by rename, persists two receipts, and only
then removes that particular completed local student staging file. Resume checkpoints
are never deleted by the archiver. Existing conflicting destinations are never replaced.

Before any new save, the optional disk-space gate waits for archived staging files
to be released. It raises before writing if the worker failed or space cannot be
recovered within20minutes. The archiver closes the training-lock descriptor so an
archive process cannot accidentally keep the training lock after a later stop.

The first resume startup was rejected before any training iteration because the
existing FSDP guard requires `load_from == resume_from`. The corrected v2 entry
passes both paths, keeps that failed attempt's logs, and reuses its already-running
verified-destination archiver instead of launching a duplicate copy job.

A redundant tiny mount probe in the v2 preflight hit temporary ENOSPC while the
200 object was being archived. No new training iteration executed. After200 had
completed a full24.62GB verified archive and a fresh tiny probe passed, v3 uses
that stronger actual-archive receipt instead of repeating the redundant write.
`h20-verify-archived-weights.py` independently reads every final-named200..500 object
and checks its full SHA as well, with byte-progress and true-exit receipts.
The first independent final-object pass matched its complete SHA but rejected a
ks3fs mtime-cache refresh. The verifier now requires stable exact size plus the
complete expected content SHA; synthetic object-store inode/mtime changes alone
do not invalidate matching bytes. Local resume files retain strict inode/mtime checks.

Logs: `launch-logs/twohead-lr10-resume200-500-v3.log`,
`launch-logs/twohead-retain50-archive.log`.
True exits: `control/twohead-lr10-resume200-500-v3.exit`,
`control/twohead-retain50-archive/checkpoint-archive-worker.exit`.
Per-stage byte/SHA receipts: `control/twohead-retain50-archive/checkpoint-archive-N.json`.
All weights remain remote; Mac receives only source/docs and small audit summaries.

Read-only observer/auditor and portrait followers bind the resumed segment using
`H20_CONTROL_PREFIX`, `H20_TRAIN_SOURCE`, `H20_TRAIN_SHA` and the separate
`H20_PORTRAIT_FOLLOW_PREFIX`. Old signal exits and the failed startup remain in
their original logs. Cumulative iteration/counter checks still cover the single
continuous1..500 scalar log. The final student auditor accepts the authorized
object-store path only with its exact per-step archive receipt and full SHA match.
Its final archived-student pass uses the real200-validated sequential RAM loader,
which hashes the entire actual stream before deserialization, then scans every
saved floating tensor/scalar for finite values. Local rolling-resume audits continue
to use local mmap. Full scans still require actual500 and a clean training exit0.
Standalone CPU validators explicitly set `TMPDIR` to the existing remote
`/data/rdm-workspace/rdm-sets/.cache/tmp` before importing Torch, because the node's
root filesystem has no usable default temporary space. Training already uses that
explicit remote temp directory; its environment is not changed by this fix.

At250, both training ranks still mapped the unlinked original200 resume inode,
retaining31.72GB after replacement. The save-space gate correctly waited before
writing the independent250 student. An inactive24.62GB legacy schema-reference
weight is therefore moved to a separate `checkpoints/legacy-spatial-A` namespace,
with full final-object SHA verification, durable exact key/shape/dtype JSON and
an original-path symlink. The reference is not current twohead150. No current
training process, optimizer or checkpoint is changed. The auditor uses the verified
schema JSON so it does not mmap that archived reference through ks3fs.

## Reproducible local read-only observations

`ops/local-control/h20-watch-twohead-progress.py --target 400 --ssh-helper <existing-helper>
--output-dir <local-small-summary-directory>` polls the existing job only. Targets450/500
are supported as well; `--once` validates one real observation without waiting for a save.
The query is frozen and SHA-recorded at startup. A target-specific local lock prevents
duplicate observers, and PID/command/status/true observer-exit files retain provenance.
Each invocation also retains its own run directory. After acquiring that lock, a later
invocation moves the previous target's command/PID/status/exit/history into that new
run's `previous` directory before writing fresh status. Old observations and failures
therefore remain readable and cannot be mistaken for a new observer's exit. Unexpected
Python tracebacks are persisted as `exception.log` in the owning run directory.
Transient SSH failures retry the same helper up to three times; query failure or timeout
is explicitly distinct from training exit and never triggers a training restart.
The observer retains compact history, all cumulative numerical/counter/trajectory checks,
archive and native-preview evidence, and actual post-save iterations. A clean500 exit
may legitimately have no live wrapper. Its successful observation is still not the final
goal acceptance: full saved tensor/SHA and source readbacks remain required. At final500,
inspect both complete training logs for all save markers; the one-shot observer's last12
save lines do not cover the entire50..500 history.

## Detached final CPU readbacks

The owned SSH helper times out a single query after110s, while the measured24.62GB
archived200 RAM read/SHA/deserialization took112.82s. Final audits therefore run through
`ops/h20-run-checkpoint-audit.sh`, detached using the existing helper, with stdin closed
and launch output redirected. No new relay or connection configuration is needed.
Every invocation has its own `control/twohead-checkpoint-audits/<UTC>-<PID>-stepN/`
with command, wrapper/child PIDs, raw stdout/stderr, launcher log and actual exit.
The `latest.json` pointer is only a locator; acceptance reads the matching run directory.
A separate CPU-audit lock serializes scans without taking the training lock. The audit
has a3600s timeout; query timeout, audit timeout and formal training exit remain separate.

Validate the runner on the actual400 rolling checkpoint with `400`. After clean500 exit0,
run `500 --full`, then `500 --student --full --file <verified-archived-student500.pth>`.
The underlying auditor still rejects premature full/student scans. Poll small log/status
reads instead of holding an SSH query open. Archived full scans emit RAM-read progress
before the final integrity JSON, so parse the final result after actual audit exit0.

## Shared preview lock and six-stage comparison

An extra authorized200 preview held the GPU6 preview lock when the retained
portrait follower reached400. The original nonblocking lock raised
`BlockingIOError: [Errno 11] Resource temporarily unavailable` before model loading;
the stage and follower both returned1. Training continued normally. Preserve the
failed command, exit, last30 lines, GPU/disk state and full original logs before
relaunching this failed preview helper. Do not restart or modify training.

The portrait preview now queues on the same lock, before importing Torch or using
GPU memory. Its existing1800s wrapper timeout bounds the wait. Deploy this fix to
a separate cold source and resume the follower against verified independent
archives; already verified stages are read back and skipped. Preserve historical
failed exit files under a unique attempt directory before publishing new ones.

The latest requested qualitative comparison is0/200/250/300/350/400 on the same
eight QIB cases and seeds0/1. After actual400 image/exit verification, run
`CUDA_VISIBLE_DEVICES= python ops/h20-export-qib-six-stages.py` in the remote Torch
environment. It verifies all96 source PNGs and matching prompt/sampling contracts,
then exports four6144x4416 lossless pages. Each1024x1024 tile is pixel-identical to
its source, with all six stages available. This is no full benchmark/judge score.

## Exact adaptive-controller acceptance

`ops/h20-replay-twohead-controller.py` replays every recorded raw global D loss,
actual active G loss and pre-D proxy through the actual immutable controller.
It compares full decisions/statistics plus the used and next D learning rates at
every iteration. It checks controller source identity across the200 resume and
round-trips the actual200 history boundary. This is a CPU-only log audit, without
new training hooks, gradients, model loads or runtime metric collection.

A real410-row replay passed with all410 decisions equal. Its explicit `--final`
guard correctly refused acceptance before actual500 and training exit0. At final
acceptance, run this published helper with `--final` and retain the real command,
stdout/stderr and exit; it supplements full checkpoint/source integrity checks.
