# H20 project storage and naming

The 2026-11-06 user request supersedes personal-name directory defaults. The
canonical node workspace is `/data/rdm-workspace`; persistent new assets belong
under `/mnt/airs-business/rdm-experiments`. `/data/ready_ckpt_0423` is not ours and
must not be changed. Existing historical object archives keep their original
names and hashes; no foreign data is moved.

`ops/h20-storage-relocate.py` only runs after the actual 500-step training exit
and full final integrity receipt. It copies all 87 canonical image tars and the
final resume to object storage, verifies full SHA256 after final naming, and
retains durable per-file receipts before removing node copies. Original indices,
logs and failed attempts remain. The new `reference_images_patchgan_object_v1`
view preserves every image identity and offset, changing only paths/stat records
and the resulting SQLite hash. It tests PNG SHA reads from every tar. Model
weights are never transferred through the Mac.

The `.cache/qwen3vl-teacher-staging` name is historical: these 173 GB of tars are
actual GAN reference data, not disposable download cache. Future configs must
use the migrated object view. Object reads use bounded pread; they may be slower
than node-local reads. Stage into an explicitly budgeted local cache if needed.
Do not mmap model/resume files directly from the object mount. Verified student
checkpoints remain in their existing archive; a resume must be staged and SHA
verified onto an ordinary remote filesystem before mmap loading.

Keep frequently read contexts, model snapshots and the Python environment on
node storage. Preserve all 50-step students. Global `/data` availability is
shared with other users; measure it before launching and before checkpoint saves.
A cumulative process write counter is not retained disk usage.


Completion is driven by `h20-storage-driver.py` from a small neutral path outside
the tree being renamed. It waits for the archive's real zero exit, atomically
renames the workspace on the same filesystem, repairs absolute symlinks and text
entrypoints (with original environment text retained), and exercises Torch2.8,
PIL, the canonical image reader, pip and torchrun from the new prefix. The old
personal-name path is not retained as an alias. Historical source and provenance
are not rewritten; current observers explicitly project old recorded paths into
the new namespace while preserving original commit binding.

An optional distro rclone client was extracted and checksum-verified while
investigating transfer speed; direct credentials are not available from the
mounted account's current credential file. No credentials were extracted from
process memory, no authentication/configuration was changed, and all actual
migration uses the existing object mount. No client package was stored on Mac.


Before a future resume, run `python3 ops/h20-stage-resume500.py` on H20. It copies
only the selected existing final resume into `ROOT/staging`, requires64GiB spare
headroom after the copy, and checks full SHA on the staged final filename. It
starts no training. Pass that ordinary-filesystem path for both load_from and
resume_from; the original snapshot did not save G Adam history. The archival
resume pointer must not be passed directly to a mmap loader.


The user prioritized the naming correction during transfer. The first storage
worker and waiting controller were deliberately terminated with SIGTERM (recorded
as a controlled interruption, not a training failure). Workspace naming and
prefix repairs were then completed immediately, with no old-name alias. The
resumable archiver continues in the new namespace; existing complete objects
are rehashed before use, and incomplete objects never authorize source removal.
The completion controller also supports naming already completed in this order.


For a controlled namespace-only interruption, set `H20_REUSE_VERIFIED_OBJECTS=1`
to reuse existing full-final-name SHA receipts in the unchanged object namespace.
Exact destination/official SHA/size must match, and the object mtime must predate
the original full verification. Mismatches or later modifications require a full
hash read; no original verification timestamp is fabricated. New objects always
receive full final-name SHA readback. Seven failure/reuse tests cover this path.


The interrupted shard73 left an old `.partial` object. Reopening that object for
writing returned ENOSPC while `/data` still had120GB free (the root filesystem
had558MB free). The exact remote command, true exit1, last30, GPU and filesystem
snapshot are retained in `control/storage-relocation-20261106/failure-enospc-v6.json`.
This does not establish the backend's exact ENOSPC cause. New attempts use unique
PID-suffixed temporary objects, retaining the old failed object and diagnostics;
no shared mount/cache configuration is changed. Eight tests include stale partial
preservation. New temporary names still require a complete final-name SHA readback.
