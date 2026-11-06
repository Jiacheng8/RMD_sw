"""Real frozen vision weights: PNG/ADM online parity and finite pixel gradients."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import transformers

from rdm.data.imagenet import center_crop_arr
from rdm.representation.backbones.qwen3_vl_backbone import Qwen3VLBackbone
from zimage_distill.model import reference_pixels

parser = argparse.ArgumentParser()
parser.add_argument('--snapshot', required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
assert torch.cuda.device_count() == 1, 'Expose only physical GPU4 for this gate'
torch.manual_seed(3407)
started = time.monotonic()
model = Qwen3VLBackbone(device='cuda', snapshot_dir=args.snapshot)
assert all(not parameter.requires_grad for parameter in model.parameters())
# The same pipeline postprocess used when making the released teacher PNGs.
raw = torch.randn(2, 3, 1024, 1024, dtype=torch.bfloat16)
decoded = (raw.float() * 0.5 + 0.5).clamp(0, 1).cuda().requires_grad_()
png = torch.round(255 * (raw / 2 + 0.5).clamp(0, 1).float()).to(torch.uint8)
offline = torch.stack([
    torch.from_numpy(np.array(center_crop_arr(
        Image.fromarray(image.permute(1, 2, 0).numpy()), 512)).copy()).permute(2, 0, 1).float()/255
    for image in png
]).cuda()
online = reference_pixels(decoded, factor=2, quantize=True)
# CPU and CUDA divide-by255 can differ by one FP32 ULP. The underlying PNG/BOX
# integers must match exactly; allow only that representation difference.
assert torch.equal(torch.round(online.detach()*255).to(torch.uint8),
                   torch.round(offline*255).to(torch.uint8))
torch.testing.assert_close(online, offline, rtol=0, atol=1e-7)
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    reference = model(offline)
with torch.autocast('cuda', dtype=torch.bfloat16):
    actual = model(online)
assert actual.shape == (2, 2560) and torch.isfinite(actual).all()
torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)
actual.float().square().mean().backward()
gradient = decoded.grad
assert gradient is not None and torch.isfinite(gradient).all()
assert all(value > 0 for value in gradient.abs().flatten(1).sum(1))
assert all(parameter.grad is None for parameter in model.parameters())
torch.cuda.synchronize()
result = {'passed': True, 'snapshot': args.snapshot, 'contract': model.feature_contract(),
          'torch': torch.__version__, 'transformers': transformers.__version__,
          'device': torch.cuda.get_device_name(), 'image_shape': list(decoded.shape),
          'feature_shape': list(actual.shape), 'png_adm_uint8_pixels_exact': True,
          'png_adm_float_max_abs_difference': (online.detach()-offline).abs().max().item(),
          'feature_max_abs_difference': (actual.float()-reference.float()).abs().max().item(),
          'input_gradient_abs_sum_per_image': gradient.abs().flatten(1).sum(1).tolist(),
          'frozen_parameters_have_no_grad': True,
          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
          'seconds': time.monotonic()-started,
          'adapter_sha256': hashlib.sha256(Path(__import__(
              'rdm.representation.backbones.qwen3_vl_backbone', fromlist=['x']).__file__).read_bytes()).hexdigest()}
args.out.parent.mkdir(parents=True, exist_ok=True)
temporary = args.out.with_suffix('.tmp')
temporary.write_text(json.dumps(result, indent=2)+'\n')
temporary.replace(args.out)
print(json.dumps(result), flush=True)
