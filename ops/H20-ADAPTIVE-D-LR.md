# Spatial A adaptive discriminator learning rate

This document describes the original legacy three-head adaptive configuration
and its migration entry. For the corrected two-GAN-head candidate and its
separate resume contract, see `H20-TWOHEAD-ADAPTIVE.md`.

This optional policy implements the updated A500 instruction. The base D LR is
1e-5; the G base LR remains2.83e-6 with its original warmup/cosine schedule.
The MMD, GAN weighting, initializers, inputs, batch sizes, FA3 and save policy
are unchanged. `test_spatial_configs.py` compares every old A setting.

`adaptive_d_lr` is a separate top-level block, disabled when absent. The adaptive
A config enables it. Both generation-step D optimizers receive the same LR,
effective from the next iteration; changing LR retains their Adam moments.

The control signals are detached, globally row-weighted raw GAN losses. Actual
G is measured on active rows after D update; D is measured on its window before
D update. Each series is compared with its own past, never with the other's
absolute scale. A free G proxy from the existing pre-D logits is audit evidence,
and is not labelled as actual G. G logging now includes both ranks without
changing the differentiable local loss or FSDP gradient averaging.

Default decisions:

- Compare two consecutive10-point valid windows. Require both an absolute0.05
  and relative20% median change. At least2 of3 heads in each step must agree,
  and both steps must agree. Opposing heads or a clearly opposing proxy hold.
- Actual G rising with D falling halves D LR. Actual G falling with D rising
  multiplies D LR by1.25. Trends require actual G observations after warmup;
  first50-step missing G values never enter as zeros.
- Sustained large oscillation takes priority and halves LR: recent10-point
  p90/p10 at least4, absolute range at least0.5, at least3 adjacent2x changes,
  in at least2 heads of a step. Before G activation, only D can trigger this.
- LR stays within[1e-7,2e-5]. After a change, discard old history and hold for
  the next20 valid observations. Skips/nonfinite data hold and clear history.

Rank0 decides and broadcasts decision plus controller state. Logs include
`gan/d_lr_used`, `gan/d_lr_next`, all head inputs/window statistics, action and
reason. The rolling resume serializes the effective optimizer LRs, D moments,
controller history/phase/cooldown/action counter. Policy mismatches fail closed.
The initial immutable critic contract retains base1e-5, not the changing LR.

## Existing running51a job and migration boundary

The running51a process has no LR-control reload interface. Updating another
checkout does not change that process. No source hot edit or debugger injection
is used. Its existing FSDP rolling resume omits the generator AdamW8bit state;
it contains model, schedule, both windows, both Ds and D Adam states. Therefore
switching implementations at a saved step would restart G Adam history. This
is not a lossless optimizer continuation and needs explicit user acceptance.

Keep the old job running while preparing/verifying the new immutable source.
Do not stop it merely because the controller or its tests are ready. After
explicit acceptance, use an actual completed50-step snapshot, stop only the
verified owned old launcher, and verify its children/lock are terminal.

`h20-freeze-adaptive-resume.py` refuses an active run lock, a live old wrapper,
missing acceptance, an unexpected checkpoint step, or a skipped/gapped prefix.
It hard-links the closed resume into `ROOT/control/spatial-adaptive-migrations`,
hashes it remotely, and preserves the full original JSONL before selecting the
saved prefix for append. No weights pass through the Mac. An acceptance receipt
must state `accept_generator_optimizer_reset:true`, `approved_utc` and the actual
`user_instruction`.

    ROOT/envs/rdm/bin/python ops/h20-freeze-adaptive-resume.py \
      --expected-step 50 --old-launcher-pid OWNED_PID --approval-file ROOT/control/APPROVAL.json
    bash ops/h20-run-spatial-adaptive-a.sh FROZEN_CHECKPOINT 50

`ROOT` in the example means `/data/rdm-workspace/rdm-sets`. Both `load_from` and
`resume_from` point at the same frozen file, never the mutable rolling filename.
The entry appends from the saved cursor through the original total500, saves
every50, retains stdout/stderr and real exit under `spatial-A-500-adaptive`,
and uses only physical GPUs4/5. B/evaluation are not started. Follow five real
post-switch iterations and continue the full budget; do not mark completion at
the first gate. Validate final step500 weights/resume contents and real exit0.

`h20-validate-adaptive.sh` runs the targeted CPU/Gloo suite in the actual remote
training environment, without taking the run lock or allocating a GPU. This is
not a substitute for actual post-switch training or the final500-step audit.
