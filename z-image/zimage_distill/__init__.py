"""Z-Image-Turbo (8-step teacher) -> few-step student with SW-LMMD + one critic per step.

Lazy on purpose: ``import zimage_distill.store`` must not pull in diffusers / transformers.
"""
