"""Only the current generated window's frozen spatial maps live in CPU memory."""
from __future__ import annotations

import numpy as np
import torch

from .cache import GeneratedWindowCache, EncoderCacheEntry


class SpatialWindowCache(GeneratedWindowCache):
    def __init__(self, names):
        super().__init__(names, store_dtype=torch.bfloat16)

    def _cpu(self, features):
        return {name: value.detach().to('cpu', dtype=self.store_dtype, copy=True)
                for name, value in features.items()}

    @staticmethod
    def _ids(row_ids):
        ids = np.asarray(row_ids)
        if ids.ndim != 1 or ids.size and ids.dtype.kind not in 'iu':
            raise ValueError('Spatial cache row IDs must be a one-dimensional integer sequence')
        ids = ids.astype(np.int64, copy=True)
        if (ids < 0).any() or len(np.unique(ids)) != len(ids):
            raise ValueError('Spatial cache requires unique nonnegative row IDs')
        return ids

    def _validate_features(self, ids, features, *, existing=False):
        if set(features) != set(self.encoder_names):
            raise ValueError('Spatial cache encoder keys differ from the configured streams')
        for name, value in features.items():
            if not isinstance(value, torch.Tensor) or value.ndim != 4 \
                    or not value.is_floating_point() or value.shape[0] != len(ids) \
                    or min(value.shape[1:]) <= 0:
                raise ValueError(f'{name}: expected row-aligned floating(B,C,H,W) spatial maps')
            if existing and value.shape[1:] != self.entries[name].features.shape[1:]:
                raise ValueError(f'{name}: spatial map shape changed within the generated window')

    def initialize(self, row_ids, features, step=-1):
        ids = self._ids(row_ids)
        self._validate_features(ids, features)
        super().initialize(ids, self._cpu(features), step=step)

    def advance(self, full_row_ids, retained_count, active_features, step):
        ids = self._ids(full_row_ids)
        if not isinstance(retained_count, int) or not 0 <= retained_count <= len(ids):
            raise ValueError('Invalid spatial retained row count')
        self._validate_features(ids[retained_count:], active_features, existing=True)
        for name in self.encoder_names:
            self.retained(name, ids[:retained_count])  # Check identity before mutating any stream.
        super().advance(ids, retained_count, self._cpu(active_features), step)

    def refresh_retained(self, encoder_name, row_ids, features, step):
        entry = self.entries[encoder_name]
        ids = self._ids(row_ids)
        if features.shape != (len(ids), *entry.features.shape[1:]) or not features.is_floating_point():
            raise ValueError('Spatial refresh shape differs from the stored map contract')
        positions = self._positions_of(entry.row_ids, ids)
        fresh = features.detach().to('cpu', dtype=self.store_dtype)
        old = entry.features[positions]
        drift = ((old.float()-fresh.float()).flatten(1).norm(dim=1) /
                 fresh.float().flatten(1).norm(dim=1).clamp_min(1e-8))
        entry.features[positions] = fresh
        entry.generated_at_steps[positions] = int(step)
        return drift

    def load_state_dict(self, state, device='cpu'):
        if not state:
            self.clear()
            return
        if set(state) != set(self.encoder_names):
            raise ValueError('Spatial checkpoint stream keys differ from the configured cache')
        features = {name: value['features'] for name, value in state.items()}
        ids = self._ids(next(iter(state.values()))['row_ids'])
        self._validate_features(ids, features)
        entries = {}
        for name, value in state.items():
            row_ids = self._ids(value['row_ids'])
            ages = value['generated_at_steps']
            if not np.array_equal(ids, row_ids) or not isinstance(ages, torch.Tensor) \
                    or ages.shape != (len(ids),) or ages.dtype not in (
                        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError('Spatial checkpoint row order or generation-step shape differs')
            entries[name] = EncoderCacheEntry(row_ids, features[name].detach().to(
                'cpu', dtype=self.store_dtype, copy=True).contiguous(), ages.detach().to(
                    'cpu', dtype=torch.int64, copy=True))
        self.entries = entries
