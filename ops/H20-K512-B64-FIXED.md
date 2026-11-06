# Fresh pretrained0, K512/B64, fixed GAN weights

Latest manual instructions supersede the earlier e-6 LR interpretation. This is a new 500-update run, not a continuation of the preceding model.

- Physical GPUs **6,7** only, wait for exclusive idle capacity; never stop foreign jobs.
- K512 sliding window,64 active rows/global update,448 retained; two ranks × microbatch2 × accumulation16. Each generation step has its own aligned window. D still trains on all512 window rows; G differentiates64 new rows.
- GAN and MMD both use frozen DINOv3 and Qwen3-VL image backbones; no AIM or SigLIP. Student original text conditioning is unchanged.
- **No gradient-norm GAN balancing.** Fixed GAN loss coefficient5e-5 (chosen near the previous run's terminal effective coefficient); no gradient ratio, independent MMD/GAN norm VJPs, or per-step lambda controller. Feature cotangents combine before one encoder VJP. GAN-G starts at51 after the existing50-update D warmup.
- G linear warmup100 to2.83e-5, cosine to exactly1.415e-5 on update500; D cosine1e-5→5e-6 on updates1→500. Both floors stay fixed afterward. Loss-driven D LR adaptation disabled.
- G global/FSDP gradient norm cap4 remains. D cap4 applies **after** full chunk accumulation and distributed mean. Nonfinite updates skip. Original losses remain visible: clipping bounds update gradients rather than falsifying/clamping the reported loss. No additional per-layer gradients, parameter deltas, or BF16 change telemetry.
- FA3 and2-block activation checkpointing retained. CPU tests do not establish K512 GPU capacity: launch requires three complete bounded D+G updates plus probe/refresh, then a separate process reloads pretrained0 for formal training. Bounded smoke overrides only GAN warmup; formal job keeps50.
- Retain each50-step student checkpoint on `/mnt/airs-business/rdm-experiments/checkpoints/`; rolling full resume remains on node. Existing FSDP resume format omits G Adam history, and must never be described as an exact optimizer resume.

Two-group entry: `bash ops/h20-run-k512-b64-fixed.sh all` runs MMD-only first, then a separate fresh0 MMD+GAN run. Use `mmd-only` or `mmd-gan` to select one. The MMD-only config changes only the experiment name and disables both GAN flags; no D or spatial GAN battery is constructed. The no-argument entry retains Group2 as its default. Immutable deployment `SOURCE_COMMIT` binds source/config/scripts. Logs and actual child/wrapper exit codes live under remoteROOT `launch-logs`/`control`. GPU waiting is not training progress. The controller never retries a failed formal job or overwrites existing output.

Cleanup entry: `ops/h20-delete-prior-except-mmd2-500.py`. Exact six older checkpoint paths only (new250/300/350/400/450 and old350); preserves second-training500 plus its full resume, all pretrained assets, data, metrics, images, logs, code and manifests. Previous50/100/150/200 had already been deleted. Deletion has a persistent receipt and refuses open candidate files.
