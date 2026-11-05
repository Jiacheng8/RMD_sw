"""Reference-side ablation of the SW-LMMD attraction (see README.md).

Three choices of the attraction's reference side, everything else unchanged:

* ``local``  -- the method: the window's own K reference rows, prompt-aligned with the generated rows;
* ``random`` -- K rows drawn uniformly from the whole reference each step, each with its own caption;
* ``global`` -- the whole reference through a frozen Nystrom kernel mean embedding (m landmarks).
"""
from .config import REF_MODES, RefAblationConfig, ref_ablation_from_config
from .nystrom_global import GlobalNystrom, store_fingerprint
from .random_reference import RandomReferenceSampler
from .ref_loss import RefAblationLoss
from .trainer import RefAblationTrainer

__all__ = ["REF_MODES", "GlobalNystrom", "RandomReferenceSampler", "RefAblationConfig",
           "RefAblationLoss", "RefAblationTrainer", "ref_ablation_from_config", "store_fingerprint"]
