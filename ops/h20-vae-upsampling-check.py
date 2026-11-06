"""Small CUDA BF16 check for the large-element VAE nearest2D fallback."""
import argparse
import copy
import json
from pathlib import Path

import torch
from diffusers.models.upsampling import Upsample2D
from zimage_distill.vae_upsample import install_safe_vae_upsampling

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
args = parser.parse_args()
torch.manual_seed(31)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
records = []
for dtype in (torch.float32, torch.bfloat16):
    original = Upsample2D(4, use_conv=True).to(device='cuda', dtype=dtype)
    guarded = copy.deepcopy(original)
    report = install_safe_vae_upsampling(guarded, element_limit=1)
    data = torch.randn(2, 4, 5, 7, device='cuda', dtype=dtype).contiguous(memory_format=torch.channels_last)
    probe = torch.randn(2, 4, 10, 14, device='cuda', dtype=dtype).contiguous(memory_format=torch.channels_last)
    outputs = []
    for module in (original, guarded):
        x = data.detach().clone().requires_grad_()
        y = module(x)
        y.backward(probe)
        outputs.append((y.detach(), x.grad, [p.grad for p in module.parameters()]))
    differences = []
    reference, actual = outputs
    for a, b in zip([reference[0], reference[1], *reference[2]], [actual[0], actual[1], *actual[2]]):
        assert torch.isfinite(b).all()
        relative = ((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-12)).item()
        assert relative < (0.015 if dtype == torch.bfloat16 else 1e-5), relative
        differences.append(relative)
    assert actual[1].abs().sum() > 0
    records.append({'dtype': str(dtype), 'relative_l2_output_input_parameter_gradients': differences,
                    'adapter': report})
torch.cuda.synchronize()
result = {'passed': True, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(), 'checks': records}
Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
