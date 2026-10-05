"""SW-LMMD: sliding-window prompt-aligned local MMD distillation.

Where the global iRDM objective matches one large fresh rollout against a Nystrom-compressed
reference mean, SW-LMMD keeps every reference row and matches an *exact* two-sample MMD inside
an overlapping window of ``K`` prompt-aligned rows, of which only the newest ``B`` are freshly
generated and back-propagated. The remaining ``K - B`` generated features are reused from the
previous windows, detached.

    L_local = sum_s pi_s MMD^2(P_s, Q_s)   >=   MMD^2(P, Q) = L_global   (Jensen)

so the local objective forbids the per-subset errors from cancelling against each other, and
each optimizer step pays for ``B`` fresh samples instead of the whole batch.

Layering (see :mod:`rdm.sw_lmmd.config`): :mod:`window_schedule`, :mod:`cache` and
:mod:`local_mmd` are pure method and know nothing about ranks, dtypes or sharding;
:mod:`sharding` holds the distribution contract that makes the parameter gradient identical at
any world size; :mod:`trainer` orchestrates.
"""
from .adversarial import AdversarialCritic, FeatureCritic, build_critic
from .cache import EncoderCacheEntry, GeneratedWindowCache
from .config import CacheConfig, GANConfig, KernelConfig, MemoryPolicy, WindowConfig
from .local_mmd import ExactLocalMMD, MMDMonitor
from .reference_store import ReferenceFeatureStore, write_reference_store
from .refresh import probe_drift, refresh_retained, should_refresh
from .sharding import all_gather_detached, partition_rows, row_noise, row_seed
from .trainer import SWLMMDTrainer
from .window_schedule import (SlidingWindowSchedule, WindowBatch, build_grouped_row_order,
                              build_row_order, order_hash)

__all__ = [
    "AdversarialCritic", "CacheConfig", "EncoderCacheEntry", "ExactLocalMMD", "FeatureCritic",
    "GANConfig", "GeneratedWindowCache", "KernelConfig", "MMDMonitor", "MemoryPolicy",
    "ReferenceFeatureStore", "SWLMMDTrainer", "SlidingWindowSchedule", "WindowBatch",
    "WindowConfig", "all_gather_detached", "build_critic", "build_grouped_row_order",
    "build_row_order", "order_hash",
    "partition_rows", "probe_drift", "refresh_retained", "row_noise", "row_seed",
    "should_refresh", "write_reference_store",
]
