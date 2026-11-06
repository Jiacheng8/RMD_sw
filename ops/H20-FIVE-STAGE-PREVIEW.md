# Matched native previews through iteration200

Run `bash ops/h20-run-compare-five-stages200.sh 6` from a separate cold deployment.
This never restarts or edits training. It binds iteration200 to the independently
archived FP32 student and its final-named full SHA receipt, streams it into remote
CPU RAM with96GiB headroom, and checks payload/sampling identity before generation.
No weights are staged on either Mac or H20 disk. GPU6 shares the existing preview
lock with portrait previews, uses a32GiB allocator cap and48GiB free preflight.
The wrapper records its real exit in control and stdout/stderr in launch-logs.

The hand-written cases retain original prompt-index/seed noise, online Qwen text,
native attention and VAE decode outside autocast, matching0/50/150. The eight
actual training rows retain original cached contexts, row_noise seed0 epoch0, FA3
and the original decode autocast, matching0/100/150. These are separate methods;
only images in the same row are compared. There is no matched row at all five
stages because hand-written100 and training50 previews/weights are unavailable.
These cells remain explicitly labeled unavailable. No stages are synthesized.

After32 real1024px images finish, the CPU exporter creates four5120px-wide lossless
PNG pages with columns0/50/100/150/200. All80 available tiles are checked byte-for-
byte at the pixel level against verified source PNGs;20 missing cells are labeled.
Pages preserve native1024px tiles without resizing. This is qualitative comparison,
not a full benchmark or a claim of monotonic quality improvement.
