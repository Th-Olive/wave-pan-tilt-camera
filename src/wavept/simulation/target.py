"""Targets: a single fixed 3D point (a planar marker is future work)."""

from __future__ import annotations

import numpy as np


class PointTarget:
    def __init__(self, position_W):
        self._p = np.asarray(position_W, dtype=float)

    def position(self, t: float) -> np.ndarray:
        return self._p
