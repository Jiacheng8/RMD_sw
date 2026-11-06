# H20 spatial PatchGAN

Latest user overrides (2026-11-03): use non-saturating softplus for all GAN heads,
D learning rate1e-5 and G base learning rate2.83e-6. Keep the original MMD,
encoder/step/GAN weights,100-step warmup and cosine schedule. Formal A runs500
steps; every50 steps atomically replace the latest resume state, with a separate
student weights file at500. The resume state includes model, cache, cursor and
both critics/critic optimizers; the existing FSDP generator optimizer is not
serialized by this launcher. Training uses the configured G-GAN start at50,
unlike the earlier bounded test's explicit start-at0 override.

Implements the Qwen3-VL / DINOv3 spatial PatchGAN Page specification
`page_3bd51524f3008191b740abb3cea05cd3`, including the requested concatenation
of the existing 2560-dimensional Qwen3-4B caption feature into each region.

Both GAN image branches receive the same 1024 RGB quantization for real and fake
images. Qwen3-VL-4B-Instruct uses the fixed instruction before the image, native
merger, image mask, causal attention and DeepStack; it executes only the first
24 language decoder blocks and reads their output before the final norm as a
`B×2560×32×32` map. DINOv3-L supplies the final normalized spatial patches,
without CLS/register tokens, as `B×1024×64×64`.

Each branch has a trainable nonoverlapping convolution (Qwen kernel/stride4,
DINO kernel/stride8), producing an 8×8 region grid. Its shared regional critic
concatenates the broadcast caption feature, applies bidirectional attention with
2D row/column RoPE, and returns 64 raw logits per image. The current softplus
losses are `mean(softplus(-real))+mean(softplus(fake))` for D and
`mean(softplus(-fake))` for G, averaged over batch and region together without an
additional division by64 or2. The original Page's hinge remains available as a
compatible helper, but the latest experiment configs explicitly select `ns`.
The original encoder1/3 and generation-step weights still apply.

The old global DINO/SigLIP/AIM MMD definitions, global reference features and
ADM512 preprocessing remain separate. The original AIM global GAN is retained.
Apart from the explicitly changed D rate/objective, the schedule, GAN
coefficient, adaptive coefficient formula, step weights, global batch32 and
checkpoint intervals stay in the saved config.
Adaptive gradient norms use decoded pixels as the common space for the global
MMD and spatial GAN. Two-level pixel GradCache replays encoders and generator
separately; frozen encoder weights still transmit image gradients. D updates
detach image features; G uses the updated, frozen critic and fixed spectral-norm
buffers. Aggregator outputs are never cached.

## Data and source roles

All large files and caches stay under `/data/rdm-workspace/rdm-sets` on H20.
The complete Qwen3-VL snapshot is pinned to
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. Teacher images are pinned to dataset
revision `3195cc9e58d83dfe17a6b5d89f9886121872586f`: 87 retained tars,
173074565120 bytes, and125808 canonical rows. The final manifest lives at
`zimage/zimage-ref-v1/reference_images_patchgan/manifest.json` under that root.
The index verifies complete PNG decoding, prompt/image identities, render seed
and settings, official tar hashes, and byte offsets. Its source-hash contract
must remain unchanged while the index is being assembled.

The asset downloader has its own published branch `codex/h20-patchgan-assets`.
It relays only HF metadata requests through the existing loopback18791 proxy;
signed CDN payload URLs are passed to aria2 on stdin, with per-URI `dir`/`out`,
and the payload downloads directly on H20. Originals, partials, failed receipts
and controlled replacement history are retained. The training/validation source
is published separately on `codex/h20-spatial-patchgan`.

Real spatial features are computed online from the canonical teacher images,
with a bounded CPU LRU. Generated spatial maps are cached only for the current
sliding window in BF16; feature-gradient leaves are FP32. A full frozen spatial
reference cache would require about1.715 TB and is not deployed.

## Bounded acceptance

`ops/h20-run-patchgan-validate.sh` uses only physical GPU4/5 and never starts
formal500-step A/B runs, saves their weights, or starts evaluation/W&B.
It runs actual-environment CPU tests, two-process Gloo normalization checks,
actual-weight single-image forward/backward checks, and then releases GPU
contexts and the run lock while waiting for the complete teacher-image manifest.
It subsequently checks an actual indexed teacher PNG and runs three complete
dual-GPU student iterations plus probe/refresh. The candidate uses FA3,
checkpoint groups of2 and microbatch2; only a verified CUDA OOM permits a
microbatch1 fallback at the same global batch32.
Before each GPU stage, an empty device may wait up to5 seconds for NVML's previous
utilization sample to expire; a device with more than32 MiB allocated is still
rejected immediately. This handles the just-finished single-image oracle without
ignoring a live memory user.

For this bounded test only, `g_start_step=0` exercises the generator GAN path
without burning50 formal warmup steps. The saved experiment configs keep their
original warmup. Each accepted iteration must have one D update per generation
step, no D/G skips, all three heads' D/G losses, finite positive D/G gradient
norms and per-step pixel adversarial norms. Timings exclude the first update
when reporting steady throughput. Logs retain per-rank CUDA allocation/reservation
and the controller records both cards' NVML peaks. Probe/refresh is checked too.

For interpreting logs, D losses/logits are from the pre-update window and G
losses/logits from post-update active rows. They are labeled by `gan/d_phase`
and `gan/g_phase`. Softplus G loss is not minus the mean fake logit: `g_fake`
is measured directly. `loss_g_adv_raw`, `pixel_adv_lambda` and
`loss_g_adv_weighted` distinguish the reported raw loss from its contribution.
`ops/h20-spatial-critic-diagnose.py` compares hinge/1e-4, softplus/1e-4 and
softplus/1e-5 with identical fixed real/fake features and initial D state.
It performs no generator updates, bootstrap, formal training or checkpointing.

`ops/h20-run-spatial-a.sh` is the formal A entry. It keeps the existing remote
environment and GPU4/5 lock, checks fresh output/cache paths, and appends stdout
and stderr to `launch-logs/spatial-A-500.log` with a real exit-code receipt in
`control/spatial-A-500.exit`. It never starts B or evaluation. Its config filename
retains the historical `dlr1e-4` name; the actual pinned D rate is1e-5 and the
output experiment is `zimage-2step-1024-dlr1e-5-spatial-patchgan-softplus`.

Final completion requires the full image manifest, actual-PNG GPU receipt,
three valid full iterations and probe/refresh, a current-source
`control/patchgan-acceptance.json`, and `control/patchgan-validate.exit=0`.
CPU tests or a single-image GPU check alone do not establish completion.
