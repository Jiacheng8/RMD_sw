# Method notes & pitfalls

A short design log for the load-bearing choices — the things that are easy to get subtly
wrong and that the code pins deliberately.

## The estimator (comparison axis)

- **Exact repulsion, frozen attraction.** The two MMD terms sum over different sets and are
  estimated differently: the within-batch repulsion `k_gg` is the exact (biased) B×B kernel
  mean (cheap, no reference); the attraction `k_gr` compares against the *whole* training set,
  so it is frozen once into a Nyström kernel-mean embedding (resampling the reference each step
  injects noise that grows as σ shrinks). The data term is constant in θ and dropped from the
  gradient; it is kept as the value constant `k_rr` so the loss stays ≥ 0 for the self-norm.
- **Biased estimator.** `k_gg` includes the `i = i` diagonal and divides by `N²` (not the
  unbiased `N(N−1)` off-diagonal U-statistic). This is what the released runs use.
- **Nyström, two equivalent forms.** Production ships `k_gr = mean_i k(g_i, Z)·α` with the
  precomputed `α = K_ZZ⁻¹ μ̄` (Cholesky solve); the toy uses the explicit `ψ(x) = K_ZZ^{-1/2}
  k(x,Z)` eigh feature map. `ψ(g)ᵀμ̄ = k(g,Z)·α` — identical cross term.
- **No feature normalization.** Kernels operate on the raw encoder embeddings; the per-encoder
  bandwidth is the median heuristic `σ = sqrt(median(d²))`, held at a single scale (×0.25 for
  the cold / joint kernel).
- **SW bias.** The Sliced-Wasserstein metric resamples directions each call; do not "freeze"
  them across the gen/floor/checkpoint comparisons except the single shared seed that makes the
  ratios comparable.

## Gradient balancing & the controller (representation axis)

- **Self-normalization, not RMS tracking.** Per-encoder losses span orders of magnitude;
  `raw / (|raw.detach()| + ε)` (a ∇log surrogate) equalizes their gradient scales. `|raw|` (not
  `raw`) keeps the descent sign when a biased MMD² dips slightly negative. Near a value of zero
  this denominator can blow up — keep the floor `ε` small but nonzero (`1e-7`).
- **PID gate.** The proportional Lagrangian upweights the *worst* (farthest-from-floor) encoder
  and gates satisfied ones to λ = 0 (anti-overfitting). It needs the rollout to score the live
  per-encoder MMD on the *same* scale as the floor `b_phi` (same `k_rr`).

## Systems

- **Fresh, large batches.** The generated side moves every step and must be sampled fresh; a
  stale EMA buffer biases the gradient off-policy. The optimum batch is above 2048.
- **Gradient caching.** The batch-coupled `k_gg` does not decompose per-sample, so plain
  accumulation is invalid. GradCache takes the exact full-batch gradient at one-chunk memory
  (bit-exact under deterministic fp32 with equal pass-1/pass-2 chunk sizes; a first-order
  approximation under bf16 with unequal sizes — still a descent direction).
- **Distributed.** No DDP wrapper: per-rank features are gathered with a gradient-preserving
  all-gather (with a load-bearing stream sync before the NCCL collective), the loss is the
  global-batch mean, and per-parameter grads are combined with an `all_reduce` average; the
  config learning rates assume the `scale = 1/grad_accum` + AVG convention.
- **pMF-H.** The released network is vendored so the checkpoint loads; `convert_pmf_checkpoint`
  maps the flax-style keys (and skips the on-the-fly `rope_freqs` buffer).

## The adversarial term (SW-LMMD, `rdm/sw_lmmd/adversarial.py`)

- **The critic belongs in the loss, not in `encode_fn`.** It scores GradCache's cached features
  in the middle phase, so pass 1 and pass 2 stay deterministic whatever the critic does in
  between. A critic inside `encode_fn` would need identical weights, spectral-norm vectors and
  augmentations in both passes.
- **Frozen in the generator's graph.** The adversarial term is built with the critic's
  parameters set to `requires_grad=False`. Otherwise the generator's backward also accumulates
  into the critic.
- **Same prompts on both sides.** Real and fake are the same window rows, so they share their
  `τ(c)` and the critic cannot score the caption distribution instead of the images.
- **Same encoder pipeline on both sides.** The store's features came from fp32 encoder weights.
  A bf16-weight battery (`battery_bf16: true`) is a signature a linear probe already detects on
  identical images (53–56 % held out, vs 50 % with fp32 weights, 4096 teacher renders,
  2026-10-03), and the generator cannot remove it.
- **λ is a gradient ratio.** The raw `|∂adv/∂φ| / |∂force/∂φ|` is ~2000–2800 on the real store
  against the base 1-step student, and it moves as the critic sharpens. A fixed λ would need to
  sit near 1e-4 and would drift away from it. The norms cover the image columns only (the
  `β·τ` block has no path to the generator) and are summed over ranks, so every rank applies
  the same λ.
- **Hinge saturation.** On a fixed 512-row set, the critic separates train rows perfectly within
  10 steps and then stops learning (zero hinge gradient). The sliding window keeps it supplied
  with unseen rows: its accuracy on not-yet-seen active rows rose 0.68 → 0.95 over 24 steps.
- **Step checkpoints never carry the critic** (they are what gets evaluated). `resume.pth` does,
  including its AdamW moments, under FSDP too, since the critic is replicated rather than
  sharded.
