# Current A previews and corrected branch assignment

Latest user clarification: GAN image branches are DINOv3 and Qwen3-VL only.
MMD image branches remain AIMv2, DINOv3 and SigLIP, with the same Qwen3-4B
caption vectors and all original MMD inputs, weights and mathematics.

The new `zimage_2step_1024_twohead_patchgan_h20_fa3.yaml` explicitly sets
`spatial_gan.global_gan_encoders: []`. GAN averages its two branches with1/2;
it does not infer GAN normalization from the three MMD streams. Legacy configs
retain their explicitly historical three-head behavior for checkpoint integrity.
The two-head critic contract rejects a three-head optimizer/cache resume.

This is a prepared correction. It has NOT hot-edited, restarted or changed the
running immutable51a A job, which still includes legacy AIM GAN. User asked to
keep that training running while inspecting images; previews are labelled as
results of that existing job. No claim that these are two-head-trained weights.

At step196 the existing step150 rolling inode was hard-linked before step200
could replace it. Step50 and100 no longer have retained weights: only their
small content/hash audits remain. They cannot be represented by fabricated or
substituted images. The preview worker pins the actual closed step200 checkpoint
and validates its saved step before rendering.

The published preview entry runs150 on physical GPU6 and200 on physical GPU7,
independently from training GPUs4/5. Each uses the same12 prompts, seeds0/1,
1024px output and the checkpoint's two-step sigmas/re-noising. All24 images and
two contact sheets stay remote; only final viewing sheets are copied to Mac.
Optional step0 uses the original pretrained weights under this same student
sampling contract; it is a reference baseline, not a replacement for missing50/100.
