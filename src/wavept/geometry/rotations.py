"""Thin wrappers around scipy rotations with the project's conventions.

All angles are radians. Rotations are right-handed about the named axis.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def Rx(angle: float) -> Rotation:
    return Rotation.from_euler("x", angle)


def Ry(angle: float) -> Rotation:
    return Rotation.from_euler("y", angle)


def Rz(angle: float) -> Rotation:
    return Rotation.from_euler("z", angle)


def rpy_to_rotation(roll: float, pitch: float, yaw: float = 0.0) -> Rotation:
    """Platform attitude R_WB = Rz(yaw) * Ry(pitch) * Rx(roll) (intrinsic ZYX)."""
    return Rotation.from_euler("ZYX", [yaw, pitch, roll])


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float(-((-angle + np.pi) % (2.0 * np.pi) - np.pi))
