# Requested 0/50/100/150 comparison

This concerns the fresh two-head, generator-LR10x run at training commit a25e4c4.
The rolling step50 and step100 weights were overwritten by later saves. Their
actual previews remain, but use different prompt sets. Therefore there is no
fully matched four-stage row; unavailable intersections must not be invented.

`h20-run-compare-four-stages150.sh 6` loads and strictly checks actual step150,
then renders 24 handwritten-prompt images and eight previously selected actual
training-prompt images. It uses the existing immutable sampler/model. Group A
matches the exact native-attention, online-Qwen, noise_for/batch2 protocol of
the saved 0/50 images. Group B matches the actual cached training contexts,
row_noise seed0/epoch0, FA3 and batch2 protocol of the saved 0/100 images.
Even VAE autocast placement is preserved separately for those two protocols.
The common step0 is the original pretrained model with the two-step sampler;
the nine-step teacher images are not substituted for initialization.

Four final JPEG sheets compare matched0/50/150 (seeds0/1, six fixed prompts)
and matched0/100/150 (eight fixed training prompts, two pages). The receipt
records missing combinations, original source PNG hashes, actual snapshot
identity, GPU memory and true exit status. Full PNGs remain remote.

The separate GPU6/7 process holds a preview lock, requires 48GiB headroom twice,
caps its allocator at32GiB, and has an1800-second deadline. It creates no
checkpoint copy/hardlink, checks unchanged input inode/size/mtime after rendering,
and leaves active training on4/5 running. This command does not launch another
training job or modify the running source. Publish the comparison source on
`main`, the only authorized remote branch, and retain the Mac code/docs copy.

## Native-resolution four-column export

`h20-run-native-four-columns.sh` performs CPU-only assembly of existing source
PNGs into columns0/50/100/150. Every available tile remains1024x1024, canvas width
is4096, and labels occupy separate header/footer strips. Missing combinations
are visibly marked. It performs no inference or resizing and uses lossless PNG
encoding. After saving, every one of the60available tiles is read back and its
pixels compared exactly to the source PNG; all must be identical.

`local-control/download-visual-original.py` streams the finished PNGs in binary
over the existing owned ControlMaster, verifies SHA256, and retains partials if
a transfer fails. It creates no relay and copies no credentials or weights.
