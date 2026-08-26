"""Synthetic detector: ground-truth target pixel measurement.

M8: optional seeded detection dropouts — with probability ``dropout_prob``
per step the detector misses (measurement reported invisible) even though
the target is truly in frame. Episode termination is unaffected (it uses
true visibility); controllers see a one-step detection gap and coast.
"""

from __future__ import annotations

import numpy as np


class SyntheticDetector:
    def __init__(self, dropout_prob: float = 0.0, rng: np.random.Generator | None = None):
        self.dropout_prob = dropout_prob
        self.rng = rng

    def measure(self, uv: np.ndarray, visible: bool) -> tuple[np.ndarray, bool]:
        if visible and self.dropout_prob > 0.0 and self.rng is not None:
            if self.rng.random() < self.dropout_prob:
                visible = False
        if not visible:
            return np.full(2, np.nan), False
        return np.asarray(uv, dtype=float), True
