# Two GAN image branches with adaptive D learning rate

This combines the prepared two-head correction with the previously implemented
adaptive D learning-rate controller. GAN has only DINOv3 and Qwen3-VL image
features, each concatenated with the frozen Qwen3-4B caption vector. Its two
heads are averaged with weight 1/2. The three MMD image streams remain AIMv2,
DINOv3 and SigLIP; their existing joint text condition, kernels, statistics,
weights, cache, learning rate and data remain unchanged.

The standalone candidate configuration is
`z-image/configs/zimage_2step_1024_twohead_patchgan_adaptive_h20_fa3.yaml`.
It preserves the two-head non-adaptive settings except for a separate experiment
name and the explicit adaptive policy. It requires four actual global GAN
step/head metric entries, not the legacy six. Both branches of both steps must
agree on trends; one flat/opposing branch prevents a trend adjustment. Large
volatility requires both branches of a step. Thresholds, cooldown, base D LR
1e-5 and bounds are the existing policy. G base LR stays 2.83e-6 with its original
warmup/cosine schedule. No MMD loss controls the GAN learning rate.

The controller receives the actual runtime head contract. It serializes those
head keys, validates them on restore, and rejects a mismatched two/three-head
controller state. The existing critic contract independently rejects loading
three-head D/Adam state into the two-head runtime. Legacy three-head behavior
and state remain supported under explicitly legacy configurations.

`ops/h20-validate-twohead-adaptive.sh` runs CPU tests in the actual remote Torch
environment, including real tiny-encoder GAN/MMD trainer steps and two-process
Gloo checks for both head counts. The distributed oracle checks unequal rank
row counts, global G metrics, local pixel gradients, absence of AIM GAN
gradients, rank-0-only decisions, matching rank state, LR changes, D moments,
G optimizer invariance, and resumed decision trajectories. These CPU checks
do not establish a full-model GPU training acceptance result for this candidate.

The current immutable 51a A500 training remains running unchanged. The user
explicitly chose to keep that job and later asked to keep training during the
checkpoint previews. This candidate is not injected into it, and no new formal
run is launched. The current run has legacy extra AIM GAN and fixed D LR. The
previews at 150/200 belong to that current run, not this corrected candidate.
Its FSDP resume omits G Adam history; switching is not lossless. In addition,
the changed GAN head contract cannot load the legacy D/Adam state unchanged.
Do not use the inherited legacy adaptive migration entry as a two-head migration
command. A future explicitly chosen transition must account for both boundaries.
