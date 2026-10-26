"""Z-Image conditioning extension of the existing RDM feature-store reader.

Add both process/ and the RDM checkout to PYTHONPATH before importing this module.
"""
from pathlib import Path

import torch
from rdm.sw_lmmd import ReferenceFeatureStore

from common import prompt_dir, read_json


class ZImageReferenceStore(ReferenceFeatureStore):
    def __init__(self, root, encoder_names):
        super().__init__(str(root), encoder_names, require_context=False)
        context = read_json(Path(root) / "z_context.json")
        if context["prompt_hash"] != self.metadata["prompt_hash"]:
            raise ValueError("Context and feature prompt hashes disagree")
        self.render_root = (Path(root) / context["root"]).resolve()

    def generator_context(self, row_ids, device="cpu", dtype=None):
        """Return actual-length tensors for ZImagePipeline(prompt_embeds=...), preserving masks."""
        result = []
        for p in self.prompt_rows(row_ids).reshape(-1):
            value = torch.load(prompt_dir(self.render_root, int(p)) / "context.pt",
                               map_location="cpu", weights_only=True)
            result.append(value.to(device=device, dtype=dtype or value.dtype))
        return result
