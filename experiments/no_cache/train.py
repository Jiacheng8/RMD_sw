"""No-cache control: K = B, every window row generated fresh and back-propagated.

    torchrun --nproc_per_node=2 -m experiments.no_cache.train \
        experiments/no_cache/configs/nocache_k128_b128.yaml      # K = B = 128
        (or experiments/no_cache/configs/nocache_k32_b32.yaml)   # K = B = 32
    (or: bash experiments/no_cache/run_nocache.sh [--k 32])

With K = B = 128, each step rolls out 128 fresh rows (128 prompts, one seed each, ungrouped as K128 · GAN) and matches
them against those same 128 rows' references with the exact biased MMD. No retained rows, so no
generated cache, no staleness probe and no refresh. With K = B the active force

    (2 / (B K)) sum_{i in A} sum_{j in W} [k(x_i, sg(x_j)) - k(x_i, y_j)]

is exactly the gradient of the full-window MMD (the K/B factor is 1): this is plain minibatch MMD.

``force_norm_window: N`` (config) multiplies the force by ``K / N``, so its normalization is
``2 / (B N)`` instead of ``2 / (B K)``. The K = B = 32 control sets N = 128, K128 · GAN's
normalization. Without the key, N = K and nothing changes. The kernel bandwidths are the store's
fixed per-encoder sigmas either way; the monitored MMD^2 is not scaled.

Nothing in ``rdm`` is changed. The trainer, the training loop, the logs and the checkpoints are the
existing ones (:func:`rdm.sw_lmmd.launch.build_trainer`, :func:`rdm.sw_lmmd.launch.train`); the only
override is :meth:`NoCacheTrainer.bootstrap`. The stock bootstrap encodes the first window's retained
rows, and with K = B there are none. Here the cache starts EMPTY (zero rows, correct feature width),
so the stock ``step`` concatenates zero retained rows to the fresh ones and ``cache.advance`` keeps
zero of them, step after step.
"""
from __future__ import annotations

import logging

import torch

from rdm.sw_lmmd.launch import apply_override, build_trainer, train
from rdm.sw_lmmd.trainer import SWLMMDTrainer

logger = logging.getLogger("rdm")


class ScaledForceMMD:
    """``mmd`` with its ``active_force`` (and the force-unit stats) multiplied by ``scale``."""

    def __init__(self, mmd, scale: float):
        self.mmd, self.scale = mmd, float(scale)

    def active_force(self, *args, **kwargs):
        force, stats = self.mmd.active_force(*args, **kwargs)
        return force * self.scale, {k: v * self.scale for k, v in stats.items()}

    def __getattr__(self, name):                       # monitor, block_size, ...
        return getattr(self.mmd, name)


class NoCacheTrainer(SWLMMDTrainer):
    """:class:`SWLMMDTrainer` for a window with no retained rows (K == B)."""

    @classmethod
    def from_trainer(cls, base: SWLMMDTrainer,
                     force_norm_window: int | None = None) -> "NoCacheTrainer":
        """Same object state as ``base`` (model, battery, store, schedule, optimizer, critic).

        ``force_norm_window`` N scales the force by ``K / N`` (normalization ``2 / (B N)``).
        """
        k = base.schedule.window_size
        if k != base.schedule.stride:
            raise ValueError(f"no-cache control needs K == B, got K={k} B={base.schedule.stride}")
        t = cls.__new__(cls)
        t.__dict__.update(base.__dict__)
        if force_norm_window is not None and int(force_norm_window) != k:
            if int(force_norm_window) <= 0:
                raise ValueError(f"force_norm_window must be positive, got {force_norm_window}")
            t.mmd = ScaledForceMMD(base.mmd, k / int(force_norm_window))
        return t

    def bootstrap(self) -> dict:
        """Take the first window and start an EMPTY cache: there is nothing to retain."""
        window = self.schedule.next()
        if window.retained_ids.size:
            raise RuntimeError("no-cache control got a window with retained rows")
        self._pending_window = window
        probe = window.all_ids[:1]                        # feature width only, no generation
        empty = {name: self._reference_window(name, probe)[:0].to(self.cache.store_dtype)
                 for name in self.encoder_names}
        self.cache.initialize(window.retained_ids, empty, step=-1)
        return {"bootstrap_rows": 0, "bootstrap_rows_per_rank": 0}


def check_config(cfg) -> None:
    """Fail before the model loads if the config is not the no-cache control."""
    w = dict(getattr(cfg, "window", None) or {})
    c = dict(getattr(cfg, "cache", None) or {})
    if int(w.get("size", 0)) != int(w.get("stride", -1)):
        raise SystemExit(f"window.size must equal window.stride (K = B), got {w}")
    if int(getattr(cfg, "probe_every", 0) or 0) or int(c.get("refresh_every_windows", 0) or 0) \
            or float(c.get("drift_threshold", 0) or 0):
        raise SystemExit("no-cache control: set probe_every 0, cache.refresh_every_windows 0 and "
                         "cache.drift_threshold 0 (there are no retained rows to probe or refresh)")
    if w.get("group_by_prompt"):
        raise SystemExit("no-cache control is ungrouped, as K128 · GAN")
    n = getattr(cfg, "force_norm_window", None)
    if n is not None and (not isinstance(n, int) or n <= 0):
        raise SystemExit(f"force_norm_window must be a positive integer, got {n!r}")


def main() -> None:
    """Same as :func:`rdm.sw_lmmd.launch.main`, with the :class:`NoCacheTrainer`."""
    import argparse

    from rdm.train.launch import load_config
    from rdm.utils.distributed import setup_distributed
    from rdm.utils.logging import setup_logging
    from rdm.utils.seed import fix_random_seeds

    ap = argparse.ArgumentParser(description="SW-LMMD no-cache control (K = B)")
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for override in args.set:
        apply_override(cfg, override)
    check_config(cfg)

    rank, world, local_rank = setup_distributed()
    setup_logging(rank=rank)
    fix_random_seeds(getattr(cfg, "seed", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    norm = getattr(cfg, "force_norm_window", None)
    trainer = NoCacheTrainer.from_trainer(build_trainer(cfg, device=device), force_norm_window=norm)
    k = trainer.schedule.stride
    logger.info("[no-cache] K=B=%d: every row fresh and back-propagated, no cache / probe / refresh; "
                "force normalization 2/(B x %d)%s", k, norm or k,
                f" (force x {k}/{norm})" if norm and norm != k else "")
    train(cfg, device=device, trainer=trainer)


if __name__ == "__main__":
    main()
