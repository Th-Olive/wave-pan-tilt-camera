"""Reactive visual servo baselines (M2). Actions are joint velocities (rad/s).

Sign convention (locked by tests/test_frames.py): at the aim configuration of
this downward camera the image-motion map is ANTI-diagonal —
    d(u_bar)/d(q_tilt) = +2f/W          (tilt moves the target radially, +u)
    d(v_bar)/d(q_pan)  = -2f/H*sin(q_t) (pan moves it tangentially, -v)
so the P controller couples u-error to TILT and v-error to PAN:
    a_pan  = +k_pan  * v_bar
    a_tilt = -k_tilt * u_bar
(a plain element-wise ``-Kp * e`` would have zero loop gain here).
The Jacobian controller derives the full 2x2 map from nominal geometry
instead and is valid away from the aim configuration.
"""

from __future__ import annotations

import numpy as np

from wavept.config import NominalParams
from wavept.models.nominal import numeric_uv_jacobian


class _ActionShaper:
    """Velocity-command shaping: optional velocity->effort conversion, then a
    first-order low-pass (alpha = new-sample weight; 1.0 disables) and clip.

    Controllers are tuned in the velocity domain. With ``effort_scale`` set
    (nominal k_d/k_u, the steady-state effort per unit velocity), the same
    gains drive the second-order effort-mode mechanism; the clip is then the
    effort bound 1.0 instead of v_max.
    """

    def __init__(self, alpha: float, v_max: float, effort_scale: float | None = None):
        self.alpha = alpha
        self.effort_scale = effort_scale
        self.a_max = 1.0 if effort_scale is not None else v_max
        self.prev = np.zeros(2)

    def reset(self) -> None:
        self.prev = np.zeros(2)

    def __call__(self, a_cmd: np.ndarray) -> np.ndarray:
        if self.effort_scale is not None:
            a_cmd = a_cmd * self.effort_scale
        a = (1.0 - self.alpha) * self.prev + self.alpha * a_cmd
        a = np.clip(a, -self.a_max, self.a_max)
        self.prev = a
        return a


class PGainController:
    name = "pgain"

    def __init__(
        self,
        k_pan: float = 1.5,
        k_tilt: float = 1.5,
        alpha: float = 0.6,
        v_max: float = 2.0,
        effort_scale: float | None = None,
    ):
        self.k_pan = k_pan
        self.k_tilt = k_tilt
        self.shaper = _ActionShaper(alpha, v_max, effort_scale)

    def reset(self) -> None:
        self.shaper.reset()

    def act(self, obs: dict) -> np.ndarray:
        if not obs["target_visible"]:
            return np.zeros(2)
        e = obs["target_uv_norm"]
        a_cmd = np.array([self.k_pan * e[1], -self.k_tilt * e[0]])
        return self.shaper(a_cmd)


class JacobianController:
    """q_dot = -lam * pinv(J_nom) * e, with J_nom = d(uv_norm)/d(q) from
    central finite differences of the nominal geometric projection at the
    current state (nominal camera, assumed scene depth)."""

    name = "jacobian"

    def __init__(
        self,
        nominal: NominalParams,
        lam: float = 2.0,
        alpha: float = 1.0,
        fd_step: float = 1e-4,
        v_max: float = 2.0,
        effort_scale: float | None = None,
    ):
        self.nominal = nominal
        self.lam = lam
        self.fd_step = fd_step
        self.shaper = _ActionShaper(alpha, v_max, effort_scale)

    def reset(self) -> None:
        self.shaper.reset()

    def act(self, obs: dict) -> np.ndarray:
        if not obs["target_visible"]:
            return np.zeros(2)
        e = obs["target_uv_norm"]
        J = numeric_uv_jacobian(
            obs["target_uv"],
            obs["joint_angle"],
            obs["platform_attitude"],
            self.nominal,
            fd_step=self.fd_step,
        )
        a_cmd = -self.lam * (np.linalg.pinv(J) @ e)
        return self.shaper(a_cmd)
