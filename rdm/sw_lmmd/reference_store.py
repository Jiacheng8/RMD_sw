"""Row-addressed access to the frozen reference: image features, tau(c), generator context.

SW-LMMD never compresses the reference. It keeps all ``M`` rows on disk and reads only the
``K`` rows the current window names, so the store is memory-mapped and indexed by row id
rather than loaded. That matters concretely: the FLUX generator context is
``M x ctx_len x 7680`` -- 61 GB at ``M=82783, ctx_len=48`` -- and a window touches 0.0004 of it.

    root/
      metadata.json               num_rows, per-encoder dims, provenance hashes
      bandwidths.json             {encoder: {"sigma": .., "beta": ..}}
      row_order.npy               (M,)      the canonical permutation
      text_features.npy           (M, d_txt)  frozen tau(c), NOT pre-scaled by beta
      encoder_features/<enc>.npy  (M, d_img)  frozen phi_e(reference image)
      qwen_context.npy            (M, L, d)   generator conditioning -- or a shard directory
      prompt_ids.npy              (M,)        OPTIONAL row -> prompt-table row indirection

``prompt_ids.npy`` exists when several reference rows share one prompt -- the standard case,
since a single teacher render per prompt degenerates the local MMD into per-prompt regression
(spec sec. 22.2), so the reference carries several teacher seeds per caption. The image features
then have one row per (prompt, seed) pair while the text table and the generator context keep
one row per *prompt*, and this array maps between them. Replicating the context instead would
cost 4x 61 GB for a 4-seed reference.

The row-alignment invariant is the whole basis of the method: for every ``row_id`` the prompt,
its text embedding, its generator context and every encoder's reference image feature must
describe the *same pair*, because the student rolls out row ``i``'s prompt and is compared
against row ``i``'s reference. Shapes are checked on construction; the semantic check (re-encode
a sample of rows and compare) belongs in the builder and its test.

``beta`` is per-encoder (``sigma_img / s_txt``), so the text block is stored unscaled and
scaled at read time -- the same convention as :mod:`rdm.representation.joint_feature`, which
keeps the offline joint bundles and the online features bit-comparable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class _ShardedRows:
    """Read-only row-addressable view over a list of memory-mapped shards.

    A single 61 GB ``.npy`` is legal but awkward to build, copy and checksum; the context pool
    is therefore allowed to be a directory of shards with a manifest. Fancy indexing returns
    rows in the order requested, exactly like indexing one array.
    """

    def __init__(self, manifest_path: str):
        root = Path(manifest_path).parent
        manifest = json.loads(Path(manifest_path).read_text())
        self.parts, self.starts = [], []
        cursor = 0
        for entry in manifest["shards"]:
            arr = np.load(root / entry["path"], mmap_mode="r")
            if "rows" in entry and int(entry["rows"]) != arr.shape[0]:
                raise ValueError(f"shard {entry['path']}: manifest says {entry['rows']} rows, "
                                 f"array has {arr.shape[0]}")
            self.parts.append(arr)
            self.starts.append(cursor)
            cursor += arr.shape[0]
        if not self.parts:
            raise ValueError(f"{manifest_path}: no shards listed")
        self.starts = np.asarray(self.starts, dtype=np.int64)
        self._num_rows = cursor
        self.shape = (cursor, *self.parts[0].shape[1:])
        self.dtype = self.parts[0].dtype
        for p in self.parts[1:]:
            if p.shape[1:] != self.parts[0].shape[1:]:
                raise ValueError("context shards disagree on the trailing shape")

    def __len__(self) -> int:
        return self._num_rows

    def __getitem__(self, row_ids) -> np.ndarray:
        ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
        which = np.searchsorted(self.starts, ids, side="right") - 1
        out = np.empty((ids.size, *self.shape[1:]), dtype=self.dtype)
        for shard_idx in np.unique(which):
            sel = which == shard_idx
            local = ids[sel] - self.starts[shard_idx]
            out[sel] = self.parts[shard_idx][local]
        return out


def _open_rows(root: Path, stem: str):
    """Open ``<stem>.npy`` as a memmap, or ``<stem>/manifest.json`` as a sharded view."""
    single = root / f"{stem}.npy"
    if single.exists():
        return np.load(single, mmap_mode="r")
    manifest = root / stem / "manifest.json"
    if manifest.exists():
        return _ShardedRows(str(manifest))
    return None


class ReferenceFeatureStore:
    """Memory-mapped, row-addressed access to one SW-LMMD reference directory."""

    def __init__(self, root: str, encoder_names, require_context: bool = True):
        self.root = Path(root)
        self.encoder_names = list(encoder_names)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        self.num_rows = int(self.metadata["num_rows"])

        bw_path = self.root / "bandwidths.json"
        self.bandwidths = json.loads(bw_path.read_text()) if bw_path.exists() else {}

        self.text = _open_rows(self.root, "text_features")
        if self.text is None:
            raise FileNotFoundError(f"{self.root}/text_features.npy is required")
        self.context = _open_rows(self.root, "qwen_context")
        if self.context is None and require_context:
            raise FileNotFoundError(
                f"{self.root}/qwen_context.npy (or qwen_context/manifest.json) is required; "
                f"pass require_context=False for image-marginal runs")

        feat_dir = self.root / "encoder_features"
        self.image_features = {}
        for name in self.encoder_names:
            path = feat_dir / f"{name}.npy"
            if not path.exists():
                raise FileNotFoundError(f"missing reference features for encoder {name!r}: {path}")
            self.image_features[name] = np.load(path, mmap_mode="r")

        pid = self.root / "prompt_ids.npy"
        self.prompt_ids = np.load(pid).astype(np.int64) if pid.exists() else None
        self._validate()

    # ---------------------------------------------------------------- validation
    @property
    def num_prompts(self) -> int:
        """Rows in the prompt-indexed tables (== num_rows when there is no indirection)."""
        return len(self.text)

    def prompt_rows(self, row_ids) -> np.ndarray:
        """Map reference row ids to prompt-table row ids (identity without ``prompt_ids``)."""
        ids = np.asarray(row_ids, dtype=np.int64)
        return ids if self.prompt_ids is None else self.prompt_ids[ids]

    def _validate(self) -> None:
        if self.prompt_ids is not None:
            if self.prompt_ids.shape[0] != self.num_rows:
                raise ValueError(f"prompt_ids has {self.prompt_ids.shape[0]} entries for "
                                 f"{self.num_rows} reference rows")
            if int(self.prompt_ids.max()) >= len(self.text):
                raise ValueError(f"prompt_ids indexes row {int(self.prompt_ids.max())} but the "
                                 f"text table has only {len(self.text)} rows")
        elif len(self.text) != self.num_rows:
            raise ValueError(f"text_features has {len(self.text)} rows, metadata says "
                             f"{self.num_rows} (and there is no prompt_ids.npy indirection)")
        if self.context is not None and len(self.context) != len(self.text):
            raise ValueError(f"qwen_context has {len(self.context)} rows but the text table has "
                             f"{len(self.text)}; both are prompt-indexed and must agree")
        declared = self.metadata.get("encoder_feature_dims", {})
        for name, arr in self.image_features.items():
            if arr.shape[0] != self.num_rows:
                raise ValueError(f"{name}: {arr.shape[0]} reference rows, metadata says "
                                 f"{self.num_rows}")
            if name in declared and int(declared[name]) != arr.shape[1]:
                raise ValueError(f"{name}: feature dim {arr.shape[1]} != declared "
                                 f"{declared[name]} -- offline and online extraction disagree")
            if name not in self.bandwidths:
                raise ValueError(f"{name}: no bandwidth in bandwidths.json; SW-LMMD uses a "
                                 f"fixed per-encoder sigma, it is never re-estimated at train time")

    def row_order(self, seed: int | None = None) -> np.ndarray:
        """The canonical permutation, from ``row_order.npy`` or freshly built from ``seed``."""
        path = self.root / "row_order.npy"
        if path.exists():
            order = np.load(path).astype(np.int64)
            if order.size != self.num_rows or np.unique(order).size != self.num_rows:
                raise ValueError("row_order.npy is not a full permutation of the reference rows")
            return order
        from .window_schedule import build_row_order
        if seed is None:
            raise FileNotFoundError(f"{path} not found and no seed given to build one")
        return build_row_order(self.num_rows, seed)

    # ---------------------------------------------------------------- per-encoder constants
    def sigma(self, encoder_name: str) -> float:
        return float(self.bandwidths[encoder_name]["sigma"])

    def beta(self, encoder_name: str) -> float:
        """Text-block scale ``sigma_img / s_txt``; 0 keeps the objective image-marginal."""
        return float(self.bandwidths[encoder_name].get("beta", 0.0))

    def dim(self, encoder_name: str) -> int:
        return int(self.image_features[encoder_name].shape[1])

    # ---------------------------------------------------------------- row reads
    @staticmethod
    def _take(array, row_ids, device, dtype: torch.dtype) -> torch.Tensor:
        ids = np.asarray(row_ids, dtype=np.int64)
        # .copy() materializes the mmap'd rows contiguous and writable before torch wraps them
        value = np.asarray(array[ids]).copy()
        return torch.from_numpy(value).to(device=device, dtype=dtype)

    def text_features(self, row_ids, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """Frozen ``tau(c)`` rows, **unscaled** (apply the per-encoder beta at coupling time)."""
        return self._take(self.text, self.prompt_rows(row_ids), device, dtype)

    def generator_context(self, row_ids, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """The generator's conditioning rows (FLUX.2 Qwen3 context)."""
        if self.context is None:
            raise RuntimeError("this store was opened without a generator context")
        return self._take(self.context, self.prompt_rows(row_ids), device, dtype)

    def reference_image_features(self, encoder_name: str, row_ids, device="cpu",
                                 dtype=torch.float32) -> torch.Tensor:
        return self._take(self.image_features[encoder_name], row_ids, device, dtype)

    def reference_joint_features(self, encoder_name: str, row_ids, device="cpu",
                                 dtype=torch.float32, joint: bool = True) -> torch.Tensor:
        """``[phi_e(r) | beta_e * tau(c)]`` for the given rows (image-only when ``joint`` is off)."""
        image = self.reference_image_features(encoder_name, row_ids, device, dtype)
        if not joint:
            return image
        text = self.text_features(row_ids, device, dtype)
        return torch.cat([image, self.beta(encoder_name) * text], dim=-1)


def write_reference_store(root: str, *, encoder_features: dict, text_features: np.ndarray,
                          bandwidths: dict, context: np.ndarray | None = None,
                          row_order: np.ndarray | None = None, extra_metadata: dict | None = None,
                          context_shards: int = 0, prompt_ids: np.ndarray | None = None) -> str:
    """Write a store directory in the layout :class:`ReferenceFeatureStore` expects.

    Used by the builder script and by the tests; keeping the writer next to the reader is what
    stops the two from drifting apart.
    """
    root = Path(root)
    (root / "encoder_features").mkdir(parents=True, exist_ok=True)
    num_rows = int(text_features.shape[0] if prompt_ids is None else len(prompt_ids))
    if prompt_ids is not None:
        np.save(root / "prompt_ids.npy", np.ascontiguousarray(prompt_ids, dtype=np.int64))
    for name, feat in encoder_features.items():
        if feat.shape[0] != num_rows:
            raise ValueError(f"{name}: {feat.shape[0]} rows vs {num_rows} text rows")
        np.save(root / "encoder_features" / f"{name}.npy", np.ascontiguousarray(feat))
    np.save(root / "text_features.npy", np.ascontiguousarray(text_features))
    if row_order is not None:
        np.save(root / "row_order.npy", np.ascontiguousarray(row_order, dtype=np.int64))

    if context is not None:
        if context_shards and context_shards > 1:
            shard_dir = root / "qwen_context"
            shard_dir.mkdir(exist_ok=True)
            bounds = np.linspace(0, num_rows, context_shards + 1).astype(int)
            shards = []
            for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
                name = f"shard_{i:04d}.npy"
                np.save(shard_dir / name, np.ascontiguousarray(context[lo:hi]))
                shards.append({"path": name, "rows": int(hi - lo)})
            (shard_dir / "manifest.json").write_text(json.dumps({"shards": shards}, indent=2))
        else:
            np.save(root / "qwen_context.npy", np.ascontiguousarray(context))

    (root / "bandwidths.json").write_text(json.dumps(bandwidths, indent=2))
    meta = {"num_rows": num_rows, "num_prompts": int(text_features.shape[0]),
            "text_dim": int(text_features.shape[1]),
            "encoder_feature_dims": {k: int(v.shape[1]) for k, v in encoder_features.items()}}
    meta.update(extra_metadata or {})
    (root / "metadata.json").write_text(json.dumps(meta, indent=2))
    return str(root)
