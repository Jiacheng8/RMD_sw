"""Frozen Qwen3-VL-4B vision tower -> mean of final merged image tokens.

The pinned official checkpoint is read from an existing local snapshot only. This module
never downloads weights and never constructs the language decoder. Set
``RDM_QWEN3VL_SNAPSHOT`` to the prepared snapshot directory, or pre-cache the pinned
files in the normal Hugging Face cache. All large files belong on remote storage.

The caller supplies [0, 1] NCHW pixels after its shared reference/training crop. Both
offline extraction and training use this same differentiable 256px preprocessing and
final-token mean. Weights are frozen, but input-pixel gradients remain enabled.

API verified against transformers v5.17.0: Qwen3VLVisionModel.forward accepts flattened
patch pixels and grid_thw; its pooler_output is the *unpooled* final-merger token tensor.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
WEIGHT_FILE = "model-00002-of-00002.safetensors"
WEIGHT_SIZE = 3_908_490_048
WEIGHT_SHA256 = "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6"
VISION_PREFIX = "model.visual."
VISION_KEY_COUNT = 315
TARGET_SIZE = 256
FEATURE_DIM = 2560
_EXPECTED_VISION = {
    "depth": 24, "hidden_size": 1024, "intermediate_size": 4096,
    "hidden_act": "gelu_pytorch_tanh", "num_heads": 16, "in_channels": 3,
    "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
    "out_hidden_size": FEATURE_DIM, "num_position_embeddings": 2304,
    "deepstack_visual_indexes": [5, 11, 17], "initializer_range": 0.02,
}


def patchify_images(images: torch.Tensor, *, patch_size: int = 16,
                    merge_size: int = 2, temporal_patch_size: int = 2):
    """Official Qwen2VL/Qwen3VL single-image patch order, without detaching pixels.

    Spatial tokens are grouped by merge_size before flattening, with channel, repeated
    temporal frame, patch-height, patch-width inside each token. Returns concatenated
    image patches and a per-image [1, grid_h, grid_w] tensor on the input device.
    """
    if images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("Qwen3-VL patchify expects a nonempty NCHW batch")
    if min(patch_size, merge_size, temporal_patch_size) <= 0:
        raise ValueError("Qwen3-VL patch and merge sizes must be positive")
    batch, channels, height, width = images.shape
    factor = patch_size * merge_size
    if height == 0 or width == 0 or height % factor or width % factor:
        raise ValueError(f"Qwen3-VL image dimensions must be positive multiples of {factor}")
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = images.reshape(batch, channels, grid_h // merge_size, merge_size,
                             patch_size, grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    patches = patches.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1,
                                           temporal_patch_size, -1, -1)
    pixels = patches.reshape(batch * grid_h * grid_w,
                             channels * temporal_patch_size * patch_size * patch_size)
    grid = torch.tensor([[1, grid_h, grid_w]], device=images.device,
                        dtype=torch.long).expand(batch, -1).contiguous()
    return pixels, grid


def prepare_pixel_values(images: torch.Tensor):
    """[0,1] RGB -> fixed 256px bicubic-AA, [-1,1], official packed patches.

    Float32 preprocessing is explicit so battery weight casting cannot change the
    resize/normalization contract. The vision patch embed performs its own dtype cast.
    This adds no PNG quantization or ADM crop; those are the caller's shared pipeline.
    """
    if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] == 0:
        raise ValueError("Qwen3-VL expects a nonempty (B,3,H,W) RGB batch")
    if not images.is_floating_point():
        raise TypeError("Qwen3-VL expects floating-point [0,1] pixels")
    images = images.float()
    if images.shape[-2:] != (TARGET_SIZE, TARGET_SIZE):
        images = F.interpolate(images, size=(TARGET_SIZE, TARGET_SIZE), mode="bicubic",
                               align_corners=False, antialias=True)
    return patchify_images((images - 0.5) / 0.5)


def _cached_files(snapshot_dir: str | Path | None):
    names = ("config.json", "model.safetensors.index.json", WEIGHT_FILE)
    directory = snapshot_dir or os.environ.get("RDM_QWEN3VL_SNAPSHOT")
    if directory:
        files = {name: Path(directory).expanduser() / name for name in names}
    else:
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError as exc:
            raise FileNotFoundError(
                "Prepare the pinned Qwen3-VL snapshot remotely and set RDM_QWEN3VL_SNAPSHOT"
            ) from exc
        files = {}
        for name in names:
            found = try_to_load_from_cache(MODEL_ID, name, revision=REVISION)
            if not isinstance(found, str):
                raise FileNotFoundError(
                    f"{MODEL_ID}@{REVISION}/{name} is not cached; this adapter never downloads"
                )
            files[name] = Path(found)
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached {name}: {path}; this adapter never downloads")
    return files


def _check_vision_config(config: dict):
    vision = config.get("vision_config", {})
    for key, expected in _EXPECTED_VISION.items():
        if vision.get(key) != expected:
            raise ValueError(f"Pinned Qwen3-VL-4B vision_config.{key}: expected {expected!r}, "
                             f"got {vision.get(key)!r}")
    extra = set(vision) - set(_EXPECTED_VISION) - {"model_type"}
    if extra or vision.get("model_type") not in ("qwen3_vl", "qwen3_vl_vision"):
        raise ValueError("Pinned Qwen3-VL-4B vision configuration contains unexpected overrides")
    # The released config uses the old qwen3_vl type; transformers 5.17's full-model
    # config fixes this in __post_init__. Apply the same fix for vision-only loading.
    return {**vision, "model_type": "qwen3_vl_vision"}


def _visual_keys(index: dict):
    weight_map = index.get("weight_map", {})
    keys = [key for key in weight_map if key.startswith(VISION_PREFIX)]
    if len(keys) != VISION_KEY_COUNT or any(weight_map[key] != WEIGHT_FILE for key in keys):
        raise ValueError("Pinned Qwen3-VL index must map all 315 visual keys to the second shard")
    return keys


def _verify_weight_file(path: Path):
    if path.stat().st_size != WEIGHT_SIZE:
        raise ValueError(f"Qwen3-VL shard size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError(f"Qwen3-VL shard SHA-256 mismatch: {path}")


def _load_vision(snapshot_dir, device, attn_implementation):
    files = _cached_files(snapshot_dir)
    vision_config = _check_vision_config(json.loads(files["config.json"].read_text()))
    keys = _visual_keys(json.loads(files["model.safetensors.index.json"].read_text()))
    _verify_weight_file(files[WEIGHT_FILE])
    from safetensors import safe_open
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = Qwen3VLVisionConfig(**vision_config)
    config._attn_implementation = attn_implementation
    # Construct only the vision model, in fp32 like the other frozen battery encoders.
    # Keeping construction on CPU avoids a transient random-weight GPU allocation.
    model = Qwen3VLVisionModel(config).float()
    with safe_open(files[WEIGHT_FILE], framework="pt", device="cpu") as handle:
        actual = {key for key in handle.keys() if key.startswith(VISION_PREFIX)}
        if actual != set(keys):
            raise ValueError("Qwen3-VL shard visual keys disagree with the pinned index")
        state = {key.removeprefix(VISION_PREFIX): handle.get_tensor(key) for key in keys}
        model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


class Qwen3VLBackbone(nn.Module):
    """Pinned vision-only Qwen3-VL-4B -> raw 2560D final-token mean."""

    has_logits = False

    def __init__(self, device="cuda", model_id=MODEL_ID, target_size=TARGET_SIZE,
                 snapshot_dir=None, attn_implementation=None):
        super().__init__()
        if model_id != MODEL_ID or target_size != TARGET_SIZE:
            raise ValueError("qwen3vl_4b requires the pinned 4B-Instruct model and 256px input")
        self.attn_implementation = (attn_implementation or
                                   os.environ.get("RDM_QWEN3VL_ATTN_IMPLEMENTATION", "sdpa"))
        self.model = _load_vision(snapshot_dir, device, self.attn_implementation)
        self.feat_dim = FEATURE_DIM
        self.target_size = TARGET_SIZE
        self.eval()

    def train(self, mode=True):
        # Battery.train() must never enable stochastic/training behavior in the frozen tower.
        super().train(False)
        return self

    def feature_contract(self):
        """Provenance for reference receipts; weights dtype reflects battery casting."""
        return {
            "model_id": MODEL_ID, "revision": REVISION, "weight_sha256": WEIGHT_SHA256,
            "input_res": TARGET_SIZE, "resize": "torch-bicubic-antialias-align_corners_false",
            "preprocess_dtype": "float32", "image_mean": [0.5] * 3, "image_std": [0.5] * 3,
            "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
            "pool": "final_merger_token_mean", "dim": FEATURE_DIM, "l2_normalize": False,
            "weight_dtype": str(next(self.model.parameters()).dtype),
            "attention_backend": self.attn_implementation,
        }

    def forward(self, images):
        pixels, grid = prepare_pixel_values(images)
        output = self.model(pixels, grid_thw=grid, return_dict=True)
        tokens = output.pooler_output
        count = (TARGET_SIZE // 16 // 2) ** 2
        expected = (images.shape[0] * count, FEATURE_DIM)
        if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != expected:
            raise RuntimeError(f"Unexpected Qwen3-VL final-merger tokens; expected {expected}")
        # No no_grad/detach or L2: generator gradients pass through frozen vision weights.
        return tokens.reshape(images.shape[0], count, FEATURE_DIM).mean(dim=1)
