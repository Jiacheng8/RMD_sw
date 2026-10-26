"""The Z-Image reference store, and the per-step view the multi-step trainer reads it through.

The store is built by ``/data/ironman/jiacheng/RDM/z-image/process`` (``launch.py all``): the
SW-LMMD layout (:mod:`rdm.sw_lmmd.reference_store`) inside ``<root>/reference_store``, with the
prompts and the per-prompt Qwen3 contexts left in the render root next to it:

    <root>/
      prompts.jsonl, prepare.json          frozen prompt table ({"prompt": ...} per prompt id)
      render_config.json                   teacher settings (steps, size, candidates, keep, ...)
      renders/{p//1000:04d}/{p:08d}/
        context.pt                         the prompt's Qwen3 context, (L_p, 2560) bf16
        k0.png ...                         the kept teacher renders
      reference_store/
        metadata.json  bandwidths.json  row_order.npy  prompt_ids.npy  group_ids.npy
        text_features.npy                  (P, 1152) SigLIP2 text tau(c), unscaled
        encoder_features/<enc>.npy         (P * keep, d), from k{r}.png ADM-cropped to --img-size
        encoder_features/<enc>.done.json   the extraction contract (img_size, ...)
        z_context.json                     {"root": "..", "prompt_hash": ...}

The context is exactly ``ZImagePipeline.encode_prompt(prompt, max_sequence_length=512)``, so the
trainer feeds it to the student directly and loads no Qwen3 encoder. A training pack
(``scripts/pack_reference.sh``) ships without the renders and their 64 GB of contexts; the new
machine rebuilds them into ``<root>/contexts/`` (:mod:`zimage_distill.contexts`), which is read
in preference to ``renders/``. A store with a
``prompts.jsonl`` of its own and no ``z_context.json`` (a hand-built one) falls back to encoding
the prompts online (:class:`zimage_distill.model.QwenPromptEncoder`).

Streams: every student step ``k`` (1-based) gets its own copy of each encoder's feature
stream, named ``s{k}/{encoder}``. The trainer's window, cache, refresh and drift probe are
keyed by stream, so each step has its own generated window and its own cache. Every stream of
encoder ``e`` is compared against the same reference features under ``e``'s bandwidth:
:class:`StepStreamStore` maps a stream name back to its encoder for those reads.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from rdm.sw_lmmd.reference_store import ReferenceFeatureStore


def stream_name(step: int, encoder: str) -> str:
    """Step ``step`` (1-based) of encoder ``encoder``."""
    return f"s{int(step)}/{encoder}"


def split_stream(name: str) -> tuple[int, str]:
    """``s2/dinov3_l`` -> ``(2, "dinov3_l")``."""
    head, encoder = name.split("/", 1)
    if not head.startswith("s") or not head[1:].isdigit():
        raise ValueError(f"not a step stream name: {name!r}")
    return int(head[1:]), encoder


def stream_names(num_steps: int, encoders) -> list[str]:
    """Step-major: every encoder of step 1, then of step 2, ..."""
    return [stream_name(k, e) for k in range(1, num_steps + 1) for e in encoders]


def read_prompts(path) -> list[str]:
    """``prompts.jsonl`` (``{"prompt": ...}`` per line) or a plain text file, one prompt per line."""
    path = Path(path)
    out = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            out.append(json.loads(line)["prompt"] if path.suffix == ".jsonl" else line)
    return out


def prompt_dir(render_root, p: int) -> Path:
    """``renders/{p//1000:04d}/{p:08d}`` -- the process pipeline's per-prompt folder."""
    return Path(render_root) / "renders" / f"{p // 1000:04d}" / f"{p:08d}"


class ZImageReferenceStore(ReferenceFeatureStore):
    """:class:`ReferenceFeatureStore` whose generator condition is the prompt's Qwen3 context
    (``context.pt`` per prompt, preferred) or its string (``prompts.jsonl``)."""

    def __init__(self, root: str, encoder_names):
        super().__init__(root, encoder_names, require_context=False)
        self.render_root = None
        self._flat = None
        zc = self.root / "z_context.json"
        if zc.exists():
            zc = json.loads(zc.read_text())
            if zc.get("prompt_hash") != self.metadata.get("prompt_hash"):
                raise ValueError(f"{self.root}: z_context.json and metadata.json disagree on the "
                                 f"prompt hash")
            self.render_root = (self.root / zc["root"]).resolve()
            prompts_path = self.render_root / "prompts.jsonl"
            flat = self.render_root / "contexts" / "manifest.json"
            if flat.exists():
                from .contexts import FlatContexts
                self._flat = FlatContexts(flat.parent)
                if self._flat.manifest.get("prompt_hash") not in (None, zc.get("prompt_hash")):
                    raise ValueError(f"{flat}: built for another prompt table")
        else:
            prompts_path = self.root / "prompts.jsonl"
        self._prompts = read_prompts(prompts_path)
        if len(self._prompts) != len(self.text):
            raise ValueError(f"{prompts_path} has {len(self._prompts)} prompts, the text table "
                             f"{len(self.text)} rows; both are prompt-indexed")

    @property
    def has_context(self) -> bool:
        """Precomputed Qwen3 contexts exist: rebuilt into ``<root>/contexts/`` (a training pack,
        :mod:`zimage_distill.contexts`) or the render's own ``renders/.../context.pt``."""
        if self._flat is not None:
            return True
        return self.render_root is not None and (self.render_root / "renders").is_dir()

    def prompts(self, row_ids) -> list[str]:
        """The prompt string of each reference row."""
        return [self._prompts[int(p)] for p in self.prompt_rows(row_ids)]

    def generator_context(self, row_ids, device="cpu", dtype=None) -> list:
        """The rows' Qwen3 contexts: a list of ``(L_p, 2560)`` tensors (actual token length)."""
        if not self.has_context:
            raise RuntimeError(f"{self.root} has no precomputed context (no z_context.json); "
                               f"encode store.prompts(row_ids) with the prompt encoder")
        out = []
        for p in self.prompt_rows(row_ids).reshape(-1):
            if self._flat is not None:
                ctx = self._flat.get(int(p))
            else:
                ctx = torch.load(prompt_dir(self.render_root, int(p)) / "context.pt",
                                 map_location="cpu", weights_only=True)
            out.append(ctx.to(device=device, dtype=dtype or ctx.dtype))
        return out

    def feature_image_size(self) -> int | None:
        """The ADM crop the reference features were extracted at (``done.json``), if recorded."""
        sizes = set()
        for name in self.image_features:
            done = self.root / "encoder_features" / f"{name}.done.json"
            if done.exists():
                sizes.add(int(json.loads(done.read_text())["contract"]["img_size"]))
        if len(sizes) > 1:
            raise ValueError(f"{self.root}: encoders extracted at different sizes {sorted(sizes)}")
        return sizes.pop() if sizes else None


class StepStreamStore:
    """The store, read by stream name: ``s{k}/{enc}`` reads encoder ``enc``'s reference.

    Everything else (text features, prompts, row order, ...) passes through unchanged.
    """

    def __init__(self, store):
        self.store = store

    @staticmethod
    def _encoder(name: str) -> str:
        return split_stream(name)[1] if "/" in name else name

    def sigma(self, name: str) -> float:
        return self.store.sigma(self._encoder(name))

    def beta(self, name: str) -> float:
        return self.store.beta(self._encoder(name))

    def dim(self, name: str) -> int:
        return self.store.dim(self._encoder(name))

    def reference_image_features(self, name: str, row_ids, device="cpu", dtype=None, **kw):
        kw = kw if dtype is None else {**kw, "dtype": dtype}
        return self.store.reference_image_features(self._encoder(name), row_ids, device, **kw)

    def reference_joint_features(self, name: str, row_ids, device="cpu", dtype=None, **kw):
        kw = kw if dtype is None else {**kw, "dtype": dtype}
        return self.store.reference_joint_features(self._encoder(name), row_ids, device, **kw)

    def __getattr__(self, name):
        return getattr(self.store, name)
