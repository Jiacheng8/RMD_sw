# Actual training prompt comparison at iteration 100

User requested inspection of training-set prompts while training continues.
`h20-run-training-prompts100.sh 6` runs the separate bounded inference job.
It never mutates training code, settings, processes or checkpoints, and never
copies/pins the rolling checkpoint. It requires actual saved step 100 and checks
the checkpoint inode/size/mtime again afterwards. Its timeout is 1800 seconds.

Before looking at outputs, select active slot zero at iterations
1, 15, 30, 45, 60, 75, 90 and 100 using the checkpoint's exact schedule. All eight
iterations must exist and be non-skipped in the actual training JSONL. Canonical
row/prompt identity and PNG hashes are checked against the reference image index.
This is a fixed qualitative training-sample inspection, not a held-out evaluation.

Each row compares the canonical nine-step teacher image, the original pretrained
model with the two-step student sampler, and the actual iteration-100 model.
The two student columns use identical cached contexts, actual row noise
`seed=0, epoch=0`, sigmas `[1.0,0.75]`, re-noising and FA3. The reference uses its
dataset seed, which differs from the student noise; this distinction is recorded.
No extra training telemetry is installed. The inference model uses BF16 weights.

Only GPUs 6/7 are permitted for this command; two preflights require at least
48 GiB free and the allocator is capped at 32 GiB. Training remains on 4/5.
Full PNGs remain under the remote `visualizations/twohead-lr10-A/` directory;
only final requested contact sheets, prompts and small receipts go to the Mac.
The shell wrapper saves its real exit code and combined stdout/stderr.

The user initially waived Git push (“不推送就行”), then explicitly requested
publication to `shaoshitong/rdm_lmmd` after reviewing these images. Preserve the
Mac code/docs snapshot and publish the latest commit, with remote SHA readback
and file hashes checked against the deployed scripts. This publication does not
change the running training checkout.
