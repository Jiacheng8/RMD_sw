"""Z-Image-Turbo as a few-step student: the DiT, the VAE, the Qwen3 prompt encoder, the sampler.

Conventions (read off ``diffusers.ZImagePipeline``, diffusers 0.40):

* latents are ``(B, 16, H/8, W/8)`` in the VAE's scaled space; pixels = ``vae.decode(z /
  scaling_factor + shift_factor)``, ``[-1, 1]`` -> ``[0, 1]``.
* noise level ``sigma`` runs 1 (noise) -> 0 (data), ``x_sigma = (1 - sigma) x0 + sigma eps``. The
  transformer takes ``t = 1 - sigma`` and returns ``out = -v`` with ``v = eps - x0``, so the
  clean-image estimate at any step is ``x0_hat = x + sigma * out``.
* the Turbo schedule is ``linspace(1, 1/N, N)`` shifted by 3 (``shift * s / (1 + (shift - 1) s)``),
  then 0. ``N = 2`` gives ``[1.0, 0.75]``; the official ``N = 9`` gives the teacher's schedule.
* the prompt is wrapped in Qwen3's chat template (``enable_thinking=True``), padded to 512
  tokens, and the second-to-last hidden state is kept for the unpadded tokens only, so every
  prompt carries a different number of context tokens.

The student (:class:`ZImageStudent`) returns one ``x0_hat`` per step. Between steps it either
re-noises (``renoise``: ``x_{k+1} = (1 - s_{k+1}) sg(x0_hat_k) + s_{k+1} eps_{k+1}``, every step's
output is a full image) or takes the Euler ODE step (the teacher's sampler). With the teacher's
schedule and ``renoise=False`` the last ``x0_hat`` IS the pipeline's output, which is what
``check_teacher.py`` verifies.
"""
from __future__ import annotations

import glob
import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rdm")

ZIMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
VAE_DOWNSAMPLE = 8
LATENT_CHANNELS = 16
TURBO_SHIFT = 3.0


def resolve_snapshot(repo: str = ZIMAGE_REPO) -> str:
    """A local directory: ``repo`` itself if it is one, else its newest HF-cache snapshot."""
    if os.path.isdir(repo):
        return repo
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    matches = sorted(glob.glob(os.path.join(hub, "models--" + repo.replace("/", "--"),
                                            "snapshots", "*", "model_index.json")))
    if not matches:
        raise FileNotFoundError(f"{repo} not found under {hub}; download it with "
                                f"`hf download {repo}` or pass a local directory")
    return os.path.dirname(matches[-1])


def turbo_sigmas(num_steps: int, shift: float = TURBO_SHIFT) -> list[float]:
    """The pipeline's sigmas for ``num_inference_steps=num_steps``, without the final 0.

    Bit-exact: fp32 linspace, the shift in fp32 arithmetic, as ``FlowMatchEulerDiscreteScheduler``.
    """
    base = torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float32)
    return (shift * base / (1 + (shift - 1) * base)).tolist()


class ZImageStudent(nn.Module):
    """``ZImageTransformer2DModel`` with the few-step contract ``forward(noise, cap_feats, args)``.

    ``noise`` is ``(B, S, 16, h, w)``: slot 0 is the initial latent, slot ``k`` the fresh noise
    of step ``k`` (used only with ``renoise``). ``cap_feats`` is the list of per-prompt Qwen3
    contexts. Returns the ``S`` per-step ``x0_hat`` latents, in fp32 (or the noise's dtype if wider).

    Under FSDP this module is the root unit, so the whole rollout runs inside its ``forward``;
    the transformer blocks (refiners and main layers) are the shard units, see
    :meth:`fsdp_unit_types`.
    """

    def __init__(self, image_resolution: int = 1024, repo: str = ZIMAGE_REPO,
                 param_dtype: torch.dtype = torch.float32, gradient_checkpointing: bool = True,
                 transformer: nn.Module | None = None):
        """``transformer``: a prebuilt model to wrap instead of loading Z-Image-Turbo (tests)."""
        super().__init__()
        if image_resolution % (2 * VAE_DOWNSAMPLE):
            raise ValueError(f"image_resolution {image_resolution} must be a multiple of 16")
        if transformer is None:
            from diffusers import ZImageTransformer2DModel
            path = os.path.join(resolve_snapshot(repo), "transformer")
            logger.info("[zimage] loading transformer from %s", path)
            transformer = ZImageTransformer2DModel.from_pretrained(path, torch_dtype=param_dtype)
        self.transformer = transformer.to(dtype=param_dtype)
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.image_resolution = int(image_resolution)
        self.in_channels = LATENT_CHANNELS
        self.input_size = self.image_resolution // VAE_DOWNSAMPLE

    def fsdp_unit_types(self) -> set:
        """The transformer block class: FSDP's shard unit, one block gathered at a time."""
        from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock
        return {ZImageTransformerBlock}

    def velocity_out(self, x: torch.Tensor, sigma: float, cap_feats: list) -> torch.Tensor:
        """The transformer's raw output ``-v`` at noise level ``sigma``, ``(B, 16, h, w)`` in x's dtype.

        ``t`` is computed as the pipeline does, ``(1000 - fp32(sigma) * 1000) / 1000`` in fp32: the
        bf16 transformer is that sensitive (``1 - sigma`` in fp64 is one fp32 ulp off at sigma
        0.96 and moves the output by ~4 %). Outside autocast the input is cast to the weights'
        dtype, as the pipeline does, so the teacher check can compare bit for bit.
        """
        ts = torch.full((x.shape[0],), float(sigma), device=x.device, dtype=torch.float32) * 1000
        t = (1000 - ts) / 1000
        x_in = x if torch.is_autocast_enabled(x.device.type) else \
            x.to(next(self.transformer.parameters()).dtype)
        out = self.transformer(list(x_in.unsqueeze(2).unbind(0)), t, cap_feats, return_dict=False)[0]
        return torch.stack(out, 0).squeeze(2).to(x.dtype)

    def forward(self, noise: torch.Tensor, cap_feats: list, sampling_args: dict) -> list:
        sigmas = [float(s) for s in sampling_args["sigmas"]]
        renoise = bool(sampling_args.get("renoise", True))
        detach = bool(sampling_args.get("detach_between_steps", True))
        if noise.dim() != 5 or noise.shape[1] < (len(sigmas) if renoise else 1):
            raise ValueError(f"noise must be (B, S, C, h, w) with S >= {len(sigmas)} slots for "
                             f"{len(sigmas)} re-noised steps; got {tuple(noise.shape)}")
        if len(cap_feats) != noise.shape[0]:
            raise ValueError(f"{len(cap_feats)} prompt contexts for {noise.shape[0]} latents")
        dtype = torch.promote_types(noise.dtype, torch.float32)     # latents stay >= fp32
        x = noise[:, 0].to(dtype)
        outs = []
        for k, sigma in enumerate(sigmas):
            out = self.velocity_out(x, sigma, cap_feats)
            x0 = x + sigma * out
            outs.append(x0)
            if k + 1 == len(sigmas):
                break
            nxt = sigmas[k + 1]
            if renoise:
                start = x0.detach() if detach else x0
                x = (1.0 - nxt) * start + nxt * noise[:, k + 1].to(dtype)
            else:
                x = x - (nxt - sigma) * out              # Euler: x + (s_next - s) * v, v = -out
                if detach:
                    x = x.detach()
        return outs


class ZImageVAE(nn.Module):
    """The frozen Z-Image VAE (FLUX.1's AutoencoderKL): scaled latents -> ``[0, 1]`` pixels.

    Gradients flow through the decode (training renders through it); the weights are frozen.
    ``gradient_checkpointing`` recomputes the decoder blocks in the backward, which is what
    makes a 1024 px decode with autograd fit next to the student.
    """

    def __init__(self, repo: str = ZIMAGE_REPO, dtype: torch.dtype = torch.bfloat16,
                 device="cuda", gradient_checkpointing: bool = True, vae: nn.Module | None = None):
        super().__init__()
        if vae is None:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(os.path.join(resolve_snapshot(repo), "vae"),
                                                torch_dtype=dtype)
        self.vae = vae.to(device=device, dtype=dtype).requires_grad_(False).eval()
        if gradient_checkpointing and hasattr(self.vae, "enable_gradient_checkpointing"):
            self.vae.enable_gradient_checkpointing()
        self.scaling_factor = float(self.vae.config.scaling_factor)
        self.shift_factor = float(self.vae.config.shift_factor)
        self.dtype = dtype

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # the pipeline's order: cast to the VAE dtype first, then unscale in that dtype
        z = z.to(self.dtype) / self.scaling_factor + self.shift_factor
        img = self.vae.decode(z, return_dict=False)[0]
        return (img.float() * 0.5 + 0.5).clamp(0.0, 1.0)


def reference_pixels(x: torch.Tensor, factor: int = 2, quantize: bool = True) -> torch.Tensor:
    """Decoded ``[0, 1]`` pixels -> what the process pipeline's encoders saw for a reference.

    The references went: VAE decode (bf16) -> ``ZImagePipeline`` postprocess ``(x / 2 + 0.5)`` in
    bf16 -> ``round(255 x)`` -> PNG -> ``rdm.data.imagenet.center_crop_arr`` at ``1024 / factor``,
    i.e. PIL's 2x BOX reduce, which is two passes (horizontal, then vertical) of a pairwise mean
    rounded half up in uint8. The forward value here is exactly that (checked pixel for pixel on
    the smoke store); the gradient is the plain ``factor x factor`` average pool's
    (straight-through). ``quantize=False`` is the average pool alone.
    """
    smooth = F.avg_pool2d(x, factor) if factor > 1 else x
    if not quantize:
        return smooth
    if factor & (factor - 1):
        raise ValueError(f"PIL's BOX reduce emulation needs a power-of-two factor, got {factor}")
    with torch.no_grad():
        # x = clamp(float(img_bf16) * 0.5 + 0.5): recover img_bf16 exactly, redo the bf16 postprocess
        u = torch.round(255 * ((x * 2 - 1).to(torch.bfloat16) / 2 + 0.5).clamp(0, 1).float())
        f = factor
        while f > 1:
            u = torch.floor((u[..., 0::2] + u[..., 1::2] + 1) / 2)
            u = torch.floor((u[..., 0::2, :] + u[..., 1::2, :] + 1) / 2)
            f //= 2
        exact = (u / 255).to(x.dtype)
    return smooth + (exact - smooth).detach()


class QwenPromptEncoder:
    """Frozen Qwen3 prompt encoder, exactly ``ZImagePipeline._encode_prompt``, with a small cache.

    GradCache encodes every active row twice (pass 1 and pass 2) and a refresh re-encodes the
    retained rows, so contexts are cached by prompt string (LRU, ``cache_size`` prompts on the
    encoder's device). The encoder is frozen and deterministic, so a hit is exact.
    """

    def __init__(self, repo: str = ZIMAGE_REPO, device="cuda", dtype: torch.dtype = torch.bfloat16,
                 max_sequence_length: int = 512, cache_size: int = 512, batch: int = 16):
        from transformers import AutoModel, AutoTokenizer
        root = resolve_snapshot(repo)
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(root, "tokenizer"))
        self.model = AutoModel.from_pretrained(os.path.join(root, "text_encoder"),
                                               torch_dtype=dtype).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.max_sequence_length = int(max_sequence_length)
        self.cache_size = int(cache_size)
        self.batch = int(batch)
        self._cache: OrderedDict = OrderedDict()

    @torch.no_grad()
    def _encode_uncached(self, prompts: list) -> list:
        text = [self.tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True)
                for p in prompts]
        tok = self.tokenizer(text, padding="max_length", max_length=self.max_sequence_length,
                             truncation=True, return_tensors="pt")
        ids = tok.input_ids.to(self.device)
        mask = tok.attention_mask.to(self.device).bool()
        hidden = self.model(input_ids=ids, attention_mask=mask,
                            output_hidden_states=True).hidden_states[-2]
        return [hidden[i][mask[i]] for i in range(len(prompts))]

    def __call__(self, prompts: list) -> list:
        missing = list(dict.fromkeys(p for p in prompts if p not in self._cache))
        for i in range(0, len(missing), self.batch):
            chunk = missing[i:i + self.batch]
            for p, ctx in zip(chunk, self._encode_uncached(chunk)):
                self._cache[p] = ctx
        out = []
        for p in prompts:
            self._cache.move_to_end(p)
            out.append(self._cache[p])
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return out


class ZImageGenerator:
    """Student + VAE (+ prompt encoder) behind ``sample_steps(noise, ...) -> [pixels per step]``.

    The condition is either the prompts' precomputed Qwen3 contexts (``contexts=``, from the
    store) or their strings (``prompts=``, encoded by ``self.text``). ``self.model`` is the
    :class:`ZImageStudent` or its FSDP wrapper; it is always called through ``__call__`` so FSDP
    gathers the root unit (the same rule as the FLUX generator).
    """

    def __init__(self, model, vae: ZImageVAE, text: QwenPromptEncoder | None, sampling_args: dict,
                 amp_dtype: torch.dtype = torch.bfloat16, enable_amp: bool = True):
        self.model = model
        self.vae = vae
        self.text = text
        self.sampling_args = dict(sampling_args)
        self.amp_dtype = amp_dtype
        self.enable_amp = bool(enable_amp)

    @property
    def num_steps(self) -> int:
        return len(self.sampling_args["sigmas"])

    def latents(self, noise: torch.Tensor, prompts: list | None = None,
                contexts: list | None = None) -> list:
        if contexts is None:
            if self.text is None:
                raise RuntimeError("no prompt encoder loaded: pass the precomputed contexts")
            contexts = self.text(prompts)
        cap = [c.to(noise.device) for c in contexts]
        with torch.autocast(noise.device.type, dtype=self.amp_dtype, enabled=self.enable_amp):
            return self.model(noise, cap, self.sampling_args)

    def sample_steps(self, noise: torch.Tensor, prompts: list | None = None,
                     contexts: list | None = None) -> list:
        """``[0, 1]`` pixels of every step's ``x0_hat``, first step first."""
        return [self.vae.decode(z) for z in self.latents(noise, prompts, contexts)]

    def sample(self, noise: torch.Tensor, prompts: list | None = None,
               contexts: list | None = None) -> torch.Tensor:
        """The final step's pixels (the image the student delivers)."""
        return self.vae.decode(self.latents(noise, prompts, contexts)[-1])


def noise_for(prompt: int, seed: int, shape, device) -> torch.Tensor:
    """A latent that is a pure function of (prompt, seed), for evaluation renders."""
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(prompt))
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device)
