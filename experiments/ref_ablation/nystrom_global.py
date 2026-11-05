"""The ``global`` attraction: a frozen Nystrom kernel mean embedding of the WHOLE reference.

    psi(x)        = K_mm^{-1/2} ( k(x, l_1), ..., k(x, l_m) )^T          (symmetric K_mm^{-1/2})
    mu_bar        = (1/N) sum_{n=1}^{N} psi(y_n)                         (all N reference rows, once)
    attraction(x) = psi(x)^T mu_bar  ~=  (1/N) sum_n k(x, y_n)

At train time it is evaluated as ``k(x, L) . v`` with ``v = K_mm^{-1/2} mu_bar`` -- the same number
(``psi(x)^T mu_bar = k(x, L) K_mm^{-1/2} mu_bar``); :meth:`GlobalNystrom.psi` keeps the explicit
form for the tests and the diagnostics. The kernel is the training kernel: the same
``[phi | beta * tau(c)]`` joint feature, the same per-encoder ``sigma`` and ``beta``.

The bundles are built by :mod:`experiments.ref_ablation.build_landmarks` from THIS store, and every
load re-derives :func:`store_fingerprint` and refuses a bundle built from another store, another
bandwidth or another text weight -- the official iRDM bundles (M=8192, a different reference with a
GenEval block, slightly different sigmas) would be refused for exactly that reason.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from rdm.compare.kernels import gamma_from_sigma, gaussian_gram

#: Bump when the bundle layout changes; a loader refuses other versions.
BUNDLE_VERSION = 1


def bundle_path(bundle_dir, name: str) -> Path:
    return Path(bundle_dir) / f"{name}.pt"


def store_fingerprint(store, name: str, joint: bool = True, probe_rows: int = 256) -> dict:
    """What a bundle must agree with: the store's size, the encoder's sigma / beta / dims, and a
    hash of ``probe_rows`` evenly spaced joint features (catches a different store of equal shape)."""
    n = int(store.num_rows)
    rows = np.unique(np.linspace(0, n - 1, min(int(probe_rows), n)).astype(np.int64))
    probe = store.reference_joint_features(name, rows, dtype=torch.float32, joint=joint).numpy()
    ext = store.extension
    return {"num_rows": n, "num_prompts": int(store.num_prompts),
            "row_order_hash": store.metadata.get("row_order_hash"),
            "extension": None if ext is None else {"rows": int(ext["rows"]), "kind": ext["kind"]},
            "sigma": float(store.sigma(name)), "beta": float(store.beta(name)) if joint else 0.0,
            "d_img": int(store.dim(name)), "d_txt": int(store.text.shape[1]) if joint else 0,
            "joint": bool(joint),
            "probe_sha1": hashlib.sha1(np.ascontiguousarray(probe).tobytes()).hexdigest()[:16]}


def check_against_store(bundle: dict, store, name: str, joint: bool = True) -> None:
    """Raise unless ``bundle`` was built from this store with this encoder's sigma and beta."""
    if int(bundle.get("version", 0)) != BUNDLE_VERSION:
        raise ValueError(f"{name}: bundle version {bundle.get('version')} != {BUNDLE_VERSION}; rebuild it")
    if bundle.get("encoder") != name:
        raise ValueError(f"bundle for encoder {bundle.get('encoder')!r} used as {name!r}")
    built, here = bundle["store"], store_fingerprint(store, name, joint=joint)
    diff = {k: (built.get(k), here[k]) for k in here if built.get(k) != here[k]}
    if diff:
        raise ValueError(f"{name}: the Nystrom bundle does not match this store / kernel "
                         f"(field: (bundle, store)) {diff}. Rebuild it with "
                         f"python -m experiments.ref_ablation.build_landmarks <config>")


class GlobalNystrom:
    """Per-encoder frozen ``(landmarks, v = K_mm^{-1/2} mu_bar)``; ``psi(x)^T mu_bar`` on demand."""

    def __init__(self, bundles: dict, device="cpu"):
        self.device = torch.device(device)
        self.bundles = {}
        for name, b in bundles.items():
            if int(b.get("version", 0)) != BUNDLE_VERSION:
                raise ValueError(f"{name}: bundle version {b.get('version')} != {BUNDLE_VERSION}")
            self.bundles[name] = b
        self._tensors_cache: dict = {}
        self._fingerprints: dict = {}

    @classmethod
    def load(cls, bundle_dir, names, store=None, joint: bool = True, device="cpu") -> "GlobalNystrom":
        """Load ``<bundle_dir>/<name>.pt`` for every name; with ``store``, check each against it."""
        bundles = {}
        for name in names:
            path = bundle_path(bundle_dir, name)
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found -- build it with "
                                        f"python -m experiments.ref_ablation.build_landmarks <config>")
            b = torch.load(path, map_location="cpu", weights_only=False)
            if store is not None:
                check_against_store(b, store, name, joint=joint)
            bundles[name] = b
        return cls(bundles, device=device)

    @staticmethod
    def available(bundle_dir, names) -> bool:
        return bool(bundle_dir) and all(bundle_path(bundle_dir, n).is_file() for n in names)

    @property
    def names(self) -> list:
        return list(self.bundles)

    def gamma(self, name: str) -> float:
        return gamma_from_sigma(self.bundles[name]["sigma"])

    def num_landmarks(self, name: str) -> int:
        return int(self.bundles[name]["landmarks"].shape[0])

    def _tensors(self, name: str, dtype: torch.dtype, device) -> tuple:
        key = (name, dtype, str(device))
        if key not in self._tensors_cache:
            b = self.bundles[name]
            self._tensors_cache[key] = (b["landmarks"].to(device=device, dtype=dtype),
                                        b["v"].to(device=device, dtype=dtype))
        return self._tensors_cache[key]

    def kernel_to_landmarks(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """``(n, m)`` ``k(x_i, l_k)``, differentiable in ``x`` (the training kernel arithmetic)."""
        L, _ = self._tensors(name, x.dtype, x.device)
        return gaussian_gram(x, L, self.gamma(name))

    def attraction_rows(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """``(n,)`` ``psi(x_i)^T mu_bar`` -- each row's approximate mean similarity to all N rows."""
        _, v = self._tensors(name, x.dtype, x.device)
        return self.kernel_to_landmarks(name, x) @ v

    def attraction(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """``(1/n) sum_i psi(x_i)^T mu_bar`` -- the bracket of the force, averaged over rows."""
        return self.attraction_rows(name, x).mean()

    def psi(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """The explicit feature map ``psi(x) = K_mm^{-1/2} k(L, x)``, as ``(n, m)`` rows."""
        W = self.bundles[name]["Kmm_inv_sqrt"].to(device=x.device, dtype=x.dtype)
        return self.kernel_to_landmarks(name, x) @ W          # W is symmetric

    def mu_bar(self, name: str, dtype=torch.float64, device="cpu") -> torch.Tensor:
        return self.bundles[name]["mu_bar"].to(device=device, dtype=dtype)

    def fingerprint(self, name: str) -> str:
        """Content hash of what the attraction uses (landmarks and v): a resume must see the same."""
        if name not in self._fingerprints:
            b = self.bundles[name]
            h = hashlib.sha1()
            for key in ("landmarks", "v"):
                h.update(b[key].detach().cpu().contiguous().numpy().tobytes())
            self._fingerprints[name] = h.hexdigest()[:16]
        return self._fingerprints[name]

    def fingerprints(self) -> dict:
        return {name: self.fingerprint(name) for name in self.names}
