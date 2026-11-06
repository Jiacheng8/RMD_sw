"""Validate masked BF16 FA3 forward/backward against SDPA on one authorized GPU."""
import json
from pathlib import Path
import torch
from diffusers.models.attention_dispatch import attention_backend, dispatch_attention_fn
from zimage_distill.attention import FA3_REVISION, pin_attention_kernel

torch.manual_seed(13)
assert torch.cuda.get_device_capability() == (9, 0)
backend = '_flash_3_varlen_hub'
pin_attention_kernel(backend)
results = []
for length in (128, 257):
    tensors = [torch.randn(2, length, 4, 128, device='cuda', dtype=torch.bfloat16)
               for _ in range(3)]
    mask = torch.arange(length, device='cuda')[None, :] < torch.tensor(
        [length, length - 37], device='cuda')[:, None]
    mask = mask[:, None, None, :]
    ref_inputs = [x.detach().clone().requires_grad_() for x in tensors]
    fa_inputs = [x.detach().clone().requires_grad_() for x in tensors]
    reference = torch.nn.functional.scaled_dot_product_attention(
        *[x.transpose(1, 2) for x in ref_inputs], attn_mask=mask).transpose(1, 2)
    reference.float().square().mean().backward()
    with attention_backend(backend):
        actual = dispatch_attention_fn(*fa_inputs, attn_mask=mask)
        actual.float().square().mean().backward()
    def error(a, b):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-20))
    row = {'length': length, 'output_relative_l2': error(actual, reference),
           'gradient_relative_l2': [error(a.grad, b.grad) for a, b in zip(fa_inputs, ref_inputs)]}
    assert row['output_relative_l2'] < 0.02, row
    assert max(row['gradient_relative_l2']) < 0.03, row
    results.append(row)
record = {'backend': backend, 'kernel_revision': FA3_REVISION, 'device': torch.cuda.get_device_name(),
          'torch': torch.__version__, 'checks': results, 'passed': True}
path = Path('/data/rdm-workspace/rdm-sets/control/fa3-kernel-check.json')
path.write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
