"""Pan-tilt mechanism dynamics.

Modes (PanTiltParams.mode):
- ``kinematic`` (M1-M2): action = desired joint velocity (rad/s), clipped to
  +/-v_max and integrated directly.
- ``second_order`` (M3+): action = normalized motor effort in [-1, 1] driving
  ``qdd = k_u * a - k_d * qdot`` (semi-implicit Euler at dt/n_substeps), with
  velocity limit, dead zone, and a command-delay FIFO of n_cmd_delay control
  steps (zero-initialized at reset).

Both modes clamp joint angles to their limits and zero the velocity of an
axis pinned at a stop.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from wavept.config import PanTiltParams


class PanTiltMechanism:
    def __init__(self, params: PanTiltParams, n_substeps: int = 4):
        if params.mode not in ("kinematic", "second_order"):
            raise ValueError(f"unknown pan_tilt mode '{params.mode}'")
        self.params = params
        self.n_substeps = n_substeps
        self.q = np.zeros(2)
        self.qdot = np.zeros(2)
        self._cmd_queue: deque = deque()

    def reset(self, q0) -> None:
        self.q = self._clip_angles(np.asarray(q0, dtype=float).copy())
        self.qdot = np.zeros(2)
        self._cmd_queue = deque(
            [np.zeros(2)] * self.params.n_cmd_delay, maxlen=max(1, self.params.n_cmd_delay)
        )

    def _clip_angles(self, q: np.ndarray) -> np.ndarray:
        p = self.params
        return np.array(
            [
                np.clip(q[0], p.pan_limits[0], p.pan_limits[1]),
                np.clip(q[1], p.tilt_limits[0], p.tilt_limits[1]),
            ]
        )

    def _delayed(self, action: np.ndarray) -> np.ndarray:
        if self.params.n_cmd_delay == 0:
            return action
        executed = self._cmd_queue[0]
        self._cmd_queue.append(action)  # maxlen pops the executed command
        return executed

    def step(self, action, dt: float) -> None:
        action = np.asarray(action, dtype=float)
        if self.params.mode == "kinematic":
            self._step_kinematic(self._delayed(action), dt)
        else:
            self._step_second_order(self._delayed(action), dt)

    def _step_kinematic(self, action: np.ndarray, dt: float) -> None:
        v = np.clip(action, -self.params.v_max, self.params.v_max)
        q_new = self._clip_angles(self.q + v * dt)
        self.qdot = (q_new - self.q) / dt  # zero on axes pinned at a limit
        self.q = q_new

    def _step_second_order(self, action: np.ndarray, dt: float) -> None:
        p = self.params
        a = np.clip(action, -1.0, 1.0)
        a = np.where(np.abs(a) < p.dead_zone, 0.0, a)
        dt_sub = dt / self.n_substeps
        for _ in range(self.n_substeps):
            qdd = p.k_u * a - p.k_d * self.qdot
            self.qdot = np.clip(self.qdot + qdd * dt_sub, -p.qdot_max, p.qdot_max)
            q_new = self._clip_angles(self.q + self.qdot * dt_sub)
            pinned = q_new != self.q + self.qdot * dt_sub
            self.qdot = np.where(pinned, 0.0, self.qdot)
            self.q = q_new
