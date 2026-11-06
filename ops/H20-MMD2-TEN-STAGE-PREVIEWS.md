# Fresh MMD2 experiment:0/50/.../450 preview comparisons

User2026-11-06 explicitly chose images plus complete loss curves, rather than
ten full Qwen-Image-Bench evaluations. Source training is immutablev7/ad5dddc,
experiment `zimage-2step-1024-dino-qwenvl-mmd2-from0`. Never substitute old350 or
the original native Euler baseline runs. GPU4/5 training remains unchanged.

Entry `bash ops/h20-run-mmd2-ten-stage-preview.sh`, independent lock and real
wrapper/GPU6/GPU7 exits under ROOT/control/mmd2-ten-stage-portraits.*. GPU6
generates0/50/100/150/200, GPU7 generates250/300/350/400/450. Each acquires the
existing preview GPU lock and checks it is idle. Existing frozen prompts:
QIB IDs5,8,47,81,172,312,393,849; exactEnglish prompts; seeds0/1. These are fixed
before rendering and are qualitative examples, not benchmark scores.

Every stage uses identical native1024 outputs, batch4, native attention, BF16
parameters without sampler/decode autocast, same prompt contexts, same per-ID
seed noise, sampling[1,.75] with renoise=True/detach=True. Stage0 is original
pretrained weights before any checkpoint load, on exactly this student sampler.
This is a weight-only comparison with native Euler not mixed into the grid.

Each trained stage is bound to newexperiment/checkpoint-archive-N.json, requires
passed/final_named_sha_readback, correct final destination, bytes,521 FP32 model
tensors and6,154,908,736 elements; streams the full archived object into remote
process RAM with completeSHA before torch.load(mmap=False). No mmap on FUSE,
no weight copies to Mac/node disk. Payload stage/schedule/Adam-controller heads
and sampler validated before strict model loading. Finite pixels,16 exactPNG
names and per-PNG SHA/native dimensions required before stage proof.

CPU export validates identical generation contracts and all160 inputPNG hashes.
Four10240x4416 panoramas contain ten native1024 columns, four cases per page,
two seeds. Eight5120x2240 case sheets put0..200 across the first row and250..450
across the second, seed0. All240 pasted tiles are checked pixel-equal to their
source: no image resampling or JPEG compression. Mac receives only these
requested final PNGs and small receipts; originals and weights remain remote.
