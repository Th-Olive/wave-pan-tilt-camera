"""Coordinate frame chain — the FIXED convention.

Frames
------
- ``W`` world: right-handed, **z up**; platform reference position at origin;
  underwater targets have z < 0.
- ``B`` platform: coincides with W at zero disturbance;
  ``R_WB = Rz(yaw) * Ry(pitch) * Rx(roll)`` (intrinsic ZYX; yaw = 0 in the core
  project). Roll is about x_B, pitch about y_B.
- ``P`` pan: ``R_BP = Rz(q_p)`` — positive pan is counterclockwise seen from
  above (right-hand rule about the platform vertical axis).
- ``T`` tilt: ``R_PT = Ry(q_t)`` — right-hand rule about y_P. For a target
  below the platform, q_t equals the off-nadir angle (q_t = 0 looks straight
  down).
- ``C`` camera: computer-vision convention (x right, y down, z = optical axis
  forward); fixed extrinsic ``R_TC = Rx(pi)``.

Full chain: ``R_WC = R_WB * Rz(q_p) * Ry(q_t) * Rx(pi)``.
With zero attitude and zero joints the camera looks straight down (-z_W) and
the image u axis aligns with +x_W.

Nadir singularity
-----------------
For a target exactly below the camera, pan reduces to a rotation about the
optical axis and cannot move the target in the image (the visual Jacobian is
singular at nadir). The default scenario therefore offsets the target from
nadir; `aim_at` yields the well-conditioned configuration.

Image-motion signs at the aim configuration (locked by tests/test_frames.py):
pan and tilt are *not* axis-aligned with u/v for a downward camera — at the
aim point, +tilt moves the target toward +u (radially, away from nadir), and
+pan moves it toward -v (tangentially), with pan gain proportional to
sin(q_t).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from wavept.geometry.rotations import Rx, Ry, Rz, rpy_to_rotation, wrap_to_pi

R_TC = Rx(math.pi)


def camera_rotation(roll: float, pitch: float, q_p: float, q_t: float) -> Rotation:
    """R_WC for a given platform attitude and joint configuration."""
    return rpy_to_rotation(roll, pitch) * Rz(q_p) * Ry(q_t) * R_TC


def camera_position(
    roll: float,
    pitch: float,
    platform_pos_W=(0.0, 0.0, 0.0),
    cam_offset_B=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """Camera optical-center position in W (lever arm rotates with the platform)."""
    return np.asarray(platform_pos_W, dtype=float) + rpy_to_rotation(roll, pitch).apply(
        np.asarray(cam_offset_B, dtype=float)
    )


def world_to_camera(p_W, R_WC: Rotation, cam_pos_W) -> np.ndarray:
    """p_C = R_WC^T (p_W - cam_pos_W)."""
    return R_WC.inv().apply(np.asarray(p_W, dtype=float) - np.asarray(cam_pos_W, dtype=float))


def aim_at(
    target_W,
    roll: float = 0.0,
    pitch: float = 0.0,
    platform_pos_W=(0.0, 0.0, 0.0),
    cam_offset_B=(0.0, 0.0, 0.0),
) -> tuple[float, float]:
    """Joint angles (q_p, q_t) that put the target on the optical axis.

    Derivation: with d the camera-to-target vector in the platform frame,
    requiring ``(Rz(q_p) Ry(q_t) Rx(pi))^T d = (0, 0, |d|)`` gives

        q_p = wrap(atan2(d_y, d_x) + pi)
        q_t = atan2(hypot(d_x, d_y), -d_z)

    Valid for targets below the platform horizon (d_z < 0); q_t is then the
    off-nadir angle in [0, pi/2).
    """
    cam = camera_position(roll, pitch, platform_pos_W, cam_offset_B)
    d = rpy_to_rotation(roll, pitch).inv().apply(np.asarray(target_W, dtype=float) - cam)
    if d[2] >= 0.0:
        raise ValueError(f"target is not below the platform horizon (d_z={d[2]:.3f})")
    q_p = wrap_to_pi(math.atan2(d[1], d[0]) + math.pi)
    q_t = math.atan2(math.hypot(d[0], d[1]), -d[2])
    return q_p, q_t
