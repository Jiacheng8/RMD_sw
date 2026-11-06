"""Fixed-instruction Qwen3-VL-4B spatial features after language block 24.

1024 RGB -> 4096 ViT patches -> 1024 native merged tokens -> the first 24
language blocks, including native causal masks, mRoPE and DeepStack injection.
Only image-token positions are returned as B x 2560 x 32 x 32. There is no LM
head, final language RMSNorm, resizing, ADM crop or global pooling in this path.

Weights/tokenizer must already be cached on the target machine. No download is
performed. Frozen parameters do not disable the generated image's autograd path.
The implementation targets the inspected transformers 5.17.0 native model API.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import torch
import torch.nn as nn

from .qwen3_vl_backbone import MODEL_ID, REVISION, _check_vision_config, patchify_images


IMAGE_SIZE = 1024
FEATURE_DIM = 2560
SPATIAL_SIZE = 32
DECODER_BLOCKS = 24
ORIGINAL_DECODER_BLOCKS = 36
FIXED_INSTRUCTION = (
    "Examine the image that follows, attending to both its overall composition\n"
    "and its local visual details.\n"
    "\n"
    "Identify the visible objects, their attributes, colors, materials, shapes,\n"
    "and spatial relationships. Pay attention to what appears in the top-left,\n"
    "top-right, bottom-left, bottom-right, and central regions, including small\n"
    "objects and background details.\n"
    "\n"
    "Examine local structures, boundaries, textures, lighting, and the spatial\n"
    "consistency of visible objects. If text is present, identify the clearly\n"
    "legible characters and their locations. Do not guess unreadable text or\n"
    "invent objects, details, or content that are not visibly supported.\n"
    "\n"
    "Base the analysis entirely on the image.\n"
)
INSTRUCTION_SHA256 = "31882d15af308a6c920cf57fc1f4c7ab4e1e591783bbe9a89797c9275998f839"
WEIGHT_FILES = {
    "model-00001-of-00002.safetensors": (
        4_967_229_296, "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9"),
    "model-00002-of-00002.safetensors": (
        3_908_490_048, "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6"),
}


def prepare_spatial_pixels(images: torch.Tensor):
    """Differentiable normalized 1024px patches; explicitly reject the old 256/512 path."""
    if images.ndim != 4 or images.shape[0] == 0 or tuple(images.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("Qwen spatial encoder requires nonempty (B,3,1024,1024) RGB pixels")
    if not images.is_floating_point():
        raise TypeError("Qwen spatial encoder expects floating-point [0,1] RGB pixels")
    # Both operations are exact binary affine operations on the supplied fp32 pixels.
    return patchify_images(images.float() * 2.0 - 1.0)


def _snapshot_directory(snapshot_dir=None):
    location = (snapshot_dir or os.environ.get("RDM_QWEN3VL_SPATIAL_SNAPSHOT") or
                os.environ.get("RDM_QWEN3VL_SNAPSHOT"))
    if location is None:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(MODEL_ID, "config.json", revision=REVISION)
        if not isinstance(cached, str):
            raise FileNotFoundError("Pinned Qwen3-VL spatial snapshot is not cached; no download is attempted")
        location = Path(cached).parent
    directory = Path(location).expanduser()
    for name in ("config.json", "model.safetensors.index.json", *WEIGHT_FILES):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing cached Qwen3-VL spatial file: {directory / name}")
    return directory


def _verify_shards(directory):
    for name, (size, expected_hash) in WEIGHT_FILES.items():
        path = directory / name
        if path.stat().st_size != size:
            raise ValueError(f"Pinned Qwen3-VL shard size mismatch: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"Pinned Qwen3-VL shard SHA-256 mismatch: {name}")


def _prefix_weight_map(index, depth=DECODER_BLOCKS):
    """Select official visual + text embedding + first depth blocks; no lm_head/rest blocks."""
    selected = {}
    layers = set()
    for name, shard in index.get("weight_map", {}).items():
        match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\..+", name)
        if match:
            layer = int(match.group(1))
            layers.add(layer)
            include = layer < depth
        else:
            include = (name.startswith("model.visual.") or name in (
                "model.language_model.embed_tokens.weight", "model.language_model.norm.weight"))
        if include:
            if shard not in WEIGHT_FILES:
                raise ValueError(f"Unexpected Qwen3-VL shard in prefix weight map: {shard}")
            selected[name] = shard
    if layers != set(range(ORIGINAL_DECODER_BLOCKS)):
        raise ValueError("Pinned Qwen3-VL index must contain the original 36 language layers")
    if sum(name.startswith("model.visual.") for name in selected) != 315:
        raise ValueError("Pinned Qwen3-VL index must contain all 315 native visual/DeepStack keys")
    for name in ("model.language_model.embed_tokens.weight", "model.language_model.norm.weight"):
        if name not in selected:
            raise ValueError(f"Missing necessary Qwen3-VL prefix weight: {name}")
    return selected


def truncate_language_prefix(model, depth):
    """Early-stop a native model and return pre-finalnorm hidden_states[depth].

    Applying the original final RMSNorm at this newly truncated depth would not be
    the original 36-layer model's intermediate state. Native DeepStack injection
    remains inside the native language forward, including after its first 3 blocks.
    """
    language = model.language_model
    if depth < 1 or len(language.layers) < depth:
        raise ValueError("Requested language prefix is outside the available decoder layers")
    language.layers = nn.ModuleList(list(language.layers[:depth]))
    language.config.num_hidden_layers = depth
    language.norm = nn.Identity()
    return model


def _load_prefix_model(directory, parameter_dtype, attention_backend):
    import transformers
    if transformers.__version__.split("+")[0] != "5.17.0":
        raise RuntimeError("Qwen spatial encoder requires the validated transformers==5.17.0 API")
    from safetensors import safe_open
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    raw = json.loads((directory / "config.json").read_text())
    _check_vision_config(raw)
    text = raw.get("text_config", {})
    expected = {"num_hidden_layers": 36, "hidden_size": 2560, "intermediate_size": 9728,
                "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                "vocab_size": 151936, "rms_norm_eps": 1e-6, "attention_dropout": 0.0}
    if any(text.get(name) != value for name, value in expected.items()):
        raise ValueError("Cached Qwen3-VL text config differs from the pinned 4B model")
    if raw.get("image_token_id") != 151655 or raw.get("vision_start_token_id") != 151652:
        raise ValueError("Cached Qwen3-VL image special tokens differ from the pinned model")
    selected = _prefix_weight_map(json.loads((directory / "model.safetensors.index.json").read_text()))
    _verify_shards(directory)
    state = {}
    for shard in WEIGHT_FILES:
        with safe_open(directory / shard, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name, filename in selected.items():
                if filename == shard:
                    if name not in available:
                        raise ValueError(f"Pinned index references absent weight: {name}")
                    state[name.removeprefix("model.")] = handle.get_tensor(name)
    config = Qwen3VLConfig.from_dict(raw)
    # Instantiate 24 blocks rather than allocating the discarded 12 or any LM head.
    config.text_config.num_hidden_layers = DECODER_BLOCKS
    config.text_config.use_cache = False
    expected_keys = set(state)
    model, loading = Qwen3VLModel.from_pretrained(
        None, config=config, state_dict=state, dtype=parameter_dtype,
        attn_implementation=attention_backend, output_loading_info=True, local_files_only=True,
    )
    # from_pretrained may report rather than raise missing/unexpected keys: reject either.
    if any(loading.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Qwen3-VL prefix load was not strict: {loading}")
    if set(model.state_dict()) != expected_keys:
        raise ValueError("Qwen3-VL prefix model/state keys differ after loading")
    truncate_language_prefix(model, DECODER_BLOCKS)
    return model.eval().requires_grad_(False)


def _load_processor(directory):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)


def fixed_sequence_inputs(processor, image_grid=None):
    """Official chat template and official image placeholder expansion, without image I/O."""
    if hashlib.sha256(FIXED_INSTRUCTION.encode()).hexdigest() != INSTRUCTION_SHA256:
        raise RuntimeError("Fixed spatial instruction changed without a new feature contract")
    grid = torch.tensor([[1, 64, 64]], dtype=torch.long) if image_grid is None else image_grid
    message = [{"role": "user", "content": [
        {"type": "text", "text": FIXED_INSTRUCTION}, {"type": "image"}]}]
    template = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
    image_token = processor.image_token
    if template.count(image_token) != 1 or template.count(FIXED_INSTRUCTION) != 1:
        raise ValueError("Official template must contain fixed L followed by exactly one image placeholder")
    if template.index(FIXED_INSTRUCTION) >= template.index(image_token):
        raise ValueError("Fixed instruction must precede image positions under causal attention")
    # This is the native processor's expansion method, not a hand-written token count.
    replacement = processor.replace_image_token({"image_grid_thw": grid}, 0)
    expanded = template.replace(image_token, replacement, 1)
    encoded = processor.tokenizer(expanded, add_special_tokens=False, return_tensors="pt",
                                  return_attention_mask=True)
    ids = encoded["input_ids"].long()
    mask = encoded["attention_mask"].long()
    image_positions = ids.eq(processor.image_token_id)
    expected_count = int(grid.prod() // 4)
    if ids.shape[0] != 1 or int(image_positions.sum()) != expected_count:
        raise ValueError("Official image placeholder/tokenizer did not produce the required image positions")
    if not torch.all(mask[image_positions] == 1):
        raise ValueError("Image positions must not be padding")
    # Qwen3VLProcessor's native convention is 0=text, 1=image, 2=video; this has no video.
    mm_types = image_positions.long()
    contract = {
        "fixed_instruction": FIXED_INSTRUCTION, "instruction_sha256": INSTRUCTION_SHA256,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "expanded_input_ids_sha256": hashlib.sha256(
            json.dumps(ids.tolist(), separators=(",", ":")).encode()).hexdigest(),
        "instruction_before_image": True, "add_generation_prompt": False,
        "sequence_length": ids.shape[1], "image_tokens": expected_count,
    }
    return {"input_ids": ids, "attention_mask": mask, "mm_token_type_ids": mm_types}, contract


def image_positions_to_spatial(hidden, input_ids, image_token_id, image_grid_thw,
                               expected_grid=(SPATIAL_SIZE, SPATIAL_SIZE)):
    """Gather image IDs only, then restore native merger row-major coordinates."""
    if hidden.ndim != 3 or input_ids.shape != hidden.shape[:2]:
        raise ValueError("Hidden states must align with input token positions")
    batch = hidden.shape[0]
    if tuple(image_grid_thw.shape) != (batch, 3):
        raise ValueError("Each spatial sample requires exactly one image grid")
    height, width = expected_grid
    expected_raw = torch.tensor([1, height * 2, width * 2], device=image_grid_thw.device)
    if not torch.all(image_grid_thw == expected_raw):
        raise ValueError("Actual image grid differs from the required spatial contract")
    mask = input_ids.eq(image_token_id)
    if not torch.all(mask.sum(-1) == height * width):
        raise ValueError("Each sample must contain exactly its spatial grid's image positions")
    tokens = hidden[mask].reshape(batch, height, width, hidden.shape[-1])
    return tokens.permute(0, 3, 1, 2).contiguous()


class Qwen3VLSpatialEncoder(nn.Module):
    """Frozen native VL prefix; generated-image gradients survive all 24 blocks."""

    has_logits = False
    feat_dim = FEATURE_DIM

    def __init__(self, snapshot_dir=None, device="cuda", parameter_dtype=torch.bfloat16,
                 attention_backend="sdpa"):
        super().__init__()
        directory = _snapshot_directory(snapshot_dir)
        processor = _load_processor(directory)
        sequence, self.sequence_contract = fixed_sequence_inputs(processor)
        self.image_token_id = int(processor.image_token_id)
        if self.image_token_id != 151655:
            raise ValueError("Processor image token ID does not match the pinned Qwen3-VL model")
        self.model = _load_prefix_model(directory, parameter_dtype, attention_backend).to(device)
        self.attention_backend = attention_backend
        for name, value in sequence.items():
            self.register_buffer(name, value.to(device), persistent=False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def feature_contract(self):
        return {
            **self.sequence_contract, "model_id": MODEL_ID, "revision": REVISION,
            "weight_sha256": {name: pair[1] for name, pair in WEIGHT_FILES.items()},
            "input_res": IMAGE_SIZE, "resize": None, "adm_crop": None,
            "normalization": "float32_rgb_2x_minus_1", "patch_size": 16,
            "temporal_patch_size": 2, "spatial_merge_size": 2,
            "vit_tokens": 4096, "image_grid_thw": [1, 64, 64], "image_tokens": 1024,
            "original_decoder_blocks": ORIGINAL_DECODER_BLOCKS, "decoder_blocks": DECODER_BLOCKS,
            "readout": "after_language_block_24_before_final_norm", "final_norm": False,
            "lm_head": False, "deepstack_decoder_layers": [0, 1, 2],
            "output_shape": [FEATURE_DIM, SPATIAL_SIZE, SPATIAL_SIZE],
            "spatial_order": "native_merger_row_major", "l2_normalize": False,
            "weight_dtype": str(next(self.model.parameters()).dtype),
            "attention_backend": self.attention_backend, "transformers_api": "5.17.0",
        }

    def forward(self, images):
        pixels, grid = prepare_spatial_pixels(images)
        batch = images.shape[0]
        sequence = {name: getattr(self, name).expand(batch, -1).contiguous()
                    for name in ("input_ids", "attention_mask", "mm_token_type_ids")}
        output = self.model(**sequence, pixel_values=pixels, image_grid_thw=grid,
                            use_cache=False, return_dict=True)
        hidden = output.last_hidden_state
        if hidden.shape[-1] != FEATURE_DIM:
            raise RuntimeError("Unexpected Qwen language-prefix feature dimension")
        return image_positions_to_spatial(hidden, sequence["input_ids"],
                                          self.image_token_id, grid)
