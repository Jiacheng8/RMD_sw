"""Attention kernel revision paired with the README's Torch2.8/CUDA12.6 environment."""

FA3_REVISION = 'acb0f42e6901d3e9effa8a83bd32bc2bf0cfdfc2'


def pin_attention_kernel(backend: str) -> None:
    if backend not in ('_flash_3_hub', '_flash_3_varlen_hub'):
        return
    # The moving v1 branch dropped its Torch2.8 build. Pin an immutable revision
    # containing torch28-cxx11-cu126-x86_64-linux instead of upgrading Torch.
    # This registry interface is paired with the pinned Diffusers0.40.0 package.
    from diffusers.models.attention_dispatch import AttentionBackendName, _HUB_KERNELS_REGISTRY
    for name in ('_flash_3_hub', '_flash_3_varlen_hub'):
        config = _HUB_KERNELS_REGISTRY[AttentionBackendName(name)]
        config.revision = FA3_REVISION
        config.version = None
