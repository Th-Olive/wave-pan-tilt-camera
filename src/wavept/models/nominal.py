"""Nominal (controller-side) visual model.

Everything here uses only ``NominalParams`` and the observation — never
simulator truth. M2 provides target reconstruction and the numeric visual
Jacobian; M4 adds ``NominalModel`` — the vectorized rollout used by the
predictive controller.

The nominal model assumes a zero camera lever arm; a real offset is a
Mismatch-tier modelling error (M8).

Delay handling (M4): the observation is ``n_obs_delay`` steps stale and
commands take ``n_cmd_delay`` steps to execute, so a rollout starting from
the observed joint state must first replay the controller's own recently
issued actions (``queued_actions``, provided by the MPC from its action
history): the first ``n_obs_delay`` entries advance the observed state to
the current time, the next ``n_cmd_delay`` entries execute during the first
future steps before candidate actions take effect. Platform attitude is
extrapolated at constant rate (no oracle future disturbance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavept.config import NominalParams
from wavept.geometry.camera import PinholeCamera
from wavept.geometry.frames import camera_rotation, world_to_camera
from wavept.geometry.rotations import rpy_to_rotation


def reconstruct_target_W(uv, joint_angle, attitude, nominal: NominalParams) -> np.ndarray:
    """Back-project a pixel to a world point on the assumed scene plane.

    The ray through the pixel is scaled so the reconstructed point lies at
    world depth z = -assumed_depth (flat scene below the platform — the
    natural nominal assumption for downward underwater inspection).
    """
    c = nominal.camera
    ray_C = np.array([(uv[0] - c.cx) / c.f, (uv[1] - c.cy) / c.f, 1.0])
    R_WC = camera_rotation(attitude[0], attitude[1], joint_angle[0], joint_angle[1])
    ray_W = R_WC.apply(ray_C)
    if abs(ray_W[2]) < 1e-9:
        raise ValueError("view ray is horizontal; cannot intersect scene plane")
    s = -nominal.assumed_depth / ray_W[2]  # camera at origin (nominal lever arm = 0)
    return s * ray_W


def project_nominal(p_W, joint_angle, attitude, nominal: NominalParams):
    """Project a world point with the nominal camera. Returns (uv, in_front)."""
    R_WC = camera_rotation(attitude[0], attitude[1], joint_angle[0], joint_angle[1])
    p_C = world_to_camera(p_W, R_WC, np.zeros(3))
    return PinholeCamera(nominal.camera).project(p_C)


def numeric_uv_jacobian(
    uv, joint_angle, attitude, nominal: NominalParams, fd_step: float = 1e-4
) -> np.ndarray:
    """J = d(uv_norm)/d(q_pan, q_tilt) (2x2) by central finite differences.

    The target is first reconstructed at the assumed depth, then reprojected
    under perturbed joint angles with the nominal camera.
    """
    cam = PinholeCamera(nominal.camera)
    p_W = reconstruct_target_W(uv, joint_angle, attitude, nominal)
    J = np.zeros((2, 2))
    q = np.asarray(joint_angle, dtype=float)
    for axis in range(2):
        dq = np.zeros(2)
        dq[axis] = fd_step
        uv_plus, ok_p = project_nominal(p_W, q + dq, attitude, nominal)
        uv_minus, ok_m = project_nominal(p_W, q - dq, attitude, nominal)
        if not (ok_p and ok_m):
            raise ValueError("target behind nominal camera during Jacobian evaluation")
        J[:, axis] = (cam.normalize(uv_plus) - cam.normalize(uv_minus)) / (2.0 * fd_step)
    return J


# ---------------------------------------------------------------------------
# M4: vectorized nominal rollout model for predictive control
# ---------------------------------------------------------------------------


@dataclass
class ModelState:
    """Everything the model needs at planning time, built by the MPC from the
    (possibly stale) observation and its own action history."""

    q: np.ndarray                 # (2,) observed joint angles
    qdot: np.ndarray              # (2,) observed joint velocities
    uv: np.ndarray                # (2,) observed target pixels
    attitude: np.ndarray          # (2,) observed roll, pitch
    attitude_rate: np.ndarray     # (2,) rad/s, finite-differenced by the MPC
    queued_actions: np.ndarray    # (n_obs_delay + n_cmd_delay, 2) issued actions


class NominalModel:
    """Vectorized nominal rollout: joint dynamics + geometric reprojection.

    ``rollout(state, candidates)`` with candidates (K, N, 2) returns
    (uv_norm (K, N, 2), in_front (K, N)): the predicted normalized target
    position after each of the N future control steps. Candidate action k
    executes at future step k + n_cmd_delay; candidates beyond N - n_cmd_delay
    never execute inside the horizon (the cost still regularizes them).
    """

    def __init__(self, nominal: NominalParams, n_substeps: int = 4, rate_decay: float = 1.0):
        """rate_decay < 1 geometrically damps the extrapolated attitude rate
        (shrinkage toward 'attitude holds'), capping the effective look-ahead
        at ~dt/(1-rate_decay) — constant-rate extrapolation overshoots badly
        at wave crests over a long horizon."""
        self.nominal = nominal
        self.n_substeps = n_substeps
        self.rate_decay = rate_decay

    @property
    def n_queued(self) -> int:
        pt = self.nominal.pan_tilt
        return pt.n_obs_delay + pt.n_cmd_delay

    def rollout(self, state: ModelState, candidates: np.ndarray, correction=None):
        """correction (optional, M6): callable
        ``(q, qdot, uv_prev_norm, att_pre, rate, a_exec, a_exec_prev) -> (K,2)``
        returning the learned per-step visual-increment residual, evaluated on
        the model's own rolled-out state and ACCUMULATED along the horizon
        (compounding assumption: total drift ~ sum of one-step increment
        errors)."""
        nom, pt, cam = self.nominal, self.nominal.pan_tilt, self.nominal.camera
        K, N, _ = candidates.shape
        d_obs = pt.n_obs_delay
        dt_sub = nom.dt / self.n_substeps

        target_W = reconstruct_target_W(state.uv, state.q, state.attitude, nom)

        q = np.broadcast_to(state.q, (K, 2)).copy()
        qdot = np.broadcast_to(state.qdot, (K, 2)).copy()
        lo = np.array([pt.pan_limits[0], pt.tilt_limits[0]])
        hi = np.array([pt.pan_limits[1], pt.tilt_limits[1]])

        uv_norm = np.empty((K, N, 2))
        in_front = np.empty((K, N), dtype=bool)
        queued = state.queued_actions
        # damped-rate look-ahead: after j+1 steps the attitude has advanced by
        # rate * dt * sum_{i=0..j} decay^i
        lookahead = np.cumsum(self.rate_decay ** np.arange(d_obs + N)) * nom.dt

        if correction is not None:
            cum = np.zeros((K, 2))
            uv_prev = np.broadcast_to(
                np.array(
                    [
                        2.0 * (state.uv[0] - cam.width / 2.0) / cam.width,
                        2.0 * (state.uv[1] - cam.height / 2.0) / cam.height,
                    ]
                ),
                (K, 2),
            )
            a_prev = None

        for j in range(d_obs + N):
            a = queued[j] if j < len(queued) else candidates[:, j - len(queued)]
            a = np.clip(np.atleast_2d(a), -1.0, 1.0)
            if pt.dead_zone > 0.0:
                a = np.where(np.abs(a) < pt.dead_zone, 0.0, a)
            if correction is not None:
                a_full = np.broadcast_to(a, (K, 2))
                att_pre = state.attitude + state.attitude_rate * (
                    lookahead[j - 1] if j > 0 else 0.0
                )
                cum = cum + correction(
                    q, qdot, uv_prev, att_pre, state.attitude_rate, a_full,
                    a_full if a_prev is None else a_prev,
                )
                a_prev = a_full
            for _ in range(self.n_substeps):
                qdd = pt.k_u * a - pt.k_d * qdot
                qdot = np.clip(qdot + qdd * dt_sub, -pt.qdot_max, pt.qdot_max)
                q_new = np.clip(q + qdot * dt_sub, lo, hi)
                qdot = np.where(q_new != q + qdot * dt_sub, 0.0, qdot)
                q = q_new
            if correction is not None or j >= d_obs:
                att = state.attitude + state.attitude_rate * lookahead[j]
                uv_j, ok_j = self._project(target_W, q, att, cam)
                if correction is not None:
                    uv_j = uv_j + cum
                    uv_prev = uv_j
                if j >= d_obs:
                    k = j - d_obs
                    uv_norm[:, k], in_front[:, k] = uv_j, ok_j
        return uv_norm, in_front

    @staticmethod
    def _project(target_W, q, attitude, cam):
        """Project one world point for K joint configurations (shared attitude).

        p_C = Rx(pi)^T Ry(-q_t) Rz(-q_p) d_B with d_B = R_WB^T target
        (camera at origin — nominal zero lever arm)."""
        d_B = rpy_to_rotation(attitude[0], attitude[1]).inv().apply(target_W)
        cp, sp = np.cos(q[:, 0]), np.sin(q[:, 0])
        ct, st = np.cos(q[:, 1]), np.sin(q[:, 1])
        x1 = cp * d_B[0] + sp * d_B[1]
        y1 = -sp * d_B[0] + cp * d_B[1]
        z1 = d_B[2]
        x2 = ct * x1 - st * z1
        z2 = st * x1 + ct * z1
        X, Y, Z = x2, -y1, -z2
        in_front = Z > cam.z_min
        Zs = np.where(in_front, Z, 1.0)  # avoid div-by-zero; masked by in_front
        u = cam.f * X / Zs + cam.cx
        v = cam.f * Y / Zs + cam.cy
        uv_norm = np.stack(
            [2.0 * (u - cam.width / 2.0) / cam.width, 2.0 * (v - cam.height / 2.0) / cam.height],
            axis=-1,
        )
        return uv_norm, in_front
