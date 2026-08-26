"""Gymnasium-like pan-tilt tracking environment (no gymnasium dependency).

API:
    obs, info = env.reset(seed=seed)
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()

- ``terminated`` — target truly left the field of view (episode failure);
  judged on current truth even when the served observation is delayed.
- ``truncated`` — scenario duration reached.
- ``info`` is current truth for logging/evaluation only; controllers must not
  read it.
- ``reward`` — negative normalized centering error (-2.0 when lost); defined
  for API completeness, controllers do not use it.
- With ``n_obs_delay > 0`` the served observation (including the measured
  attitude and timestamp) is n_obs_delay control steps stale.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from wavept.config import Scenario, SimParams
from wavept.geometry.camera import PinholeCamera
from wavept.geometry.frames import aim_at, camera_position, camera_rotation, world_to_camera
from wavept.perception.synthetic_detector import SyntheticDetector
from wavept.simulation.pan_tilt import PanTiltMechanism
from wavept.simulation.platform_motion import WaveDisturbance
from wavept.simulation.target import PointTarget

TRAIL_LENGTH = 50
LOST_REWARD = -2.0


class PanTiltEnv:
    def __init__(self, sim: SimParams, scenario: Scenario):
        self.sim = sim
        self.scenario = scenario
        self.camera = PinholeCamera(sim.camera)
        self.mechanism = PanTiltMechanism(sim.pan_tilt, n_substeps=sim.n_substeps)
        self.detector = SyntheticDetector()
        self.target = PointTarget(scenario.target_W)
        self.disturbance: WaveDisturbance | None = None
        self.t = 0.0
        self.trail: deque = deque(maxlen=TRAIL_LENGTH)
        self._obs_queue: deque = deque()
        self._renderer = None
        self._last: dict = {}

    # ------------------------------------------------------------------ API

    def reset(self, seed: int | None = None):
        if seed is None:
            seed = self.scenario.seed
        rng = np.random.default_rng(seed)
        self.disturbance = WaveDisturbance(
            self.scenario.roll_waves,
            self.scenario.pitch_waves,
            rng,
            impulses=self.scenario.impulses,
        )
        # child rng so dropout draws never disturb wave-phase determinism
        self.detector = SyntheticDetector(
            self.sim.obs_dropout_prob, np.random.default_rng(rng.integers(2**32))
        )
        self.t = 0.0
        self.mechanism.reset(self._initial_joints())
        self.trail.clear()
        self._obs_queue = deque(maxlen=self.sim.pan_tilt.n_obs_delay + 1)
        obs, info = self._measure()
        # pre-fill so early steps serve the t=0 measurement until real ones age in
        while len(self._obs_queue) < self._obs_queue.maxlen:
            self._obs_queue.append(obs)
        return self._obs_queue[0], info

    def step(self, action):
        if self.disturbance is None:
            raise RuntimeError("call reset() before step()")
        self.mechanism.step(action, self.sim.dt)
        self.t += self.sim.dt
        obs_now, info = self._measure()
        self._obs_queue.append(obs_now)
        obs = self._obs_queue[0]  # served (possibly stale) observation
        terminated = not info["true_visible"]
        truncated = self.t >= self.scenario.duration - 1e-9
        if obs["target_visible"]:
            reward = -float(np.linalg.norm(obs["target_uv_norm"]))
        else:
            reward = LOST_REWARD
        return obs, reward, terminated, truncated, info

    def render(self, predicted_uv=None, label: str = "") -> np.ndarray:
        """predicted_uv: optional (N, 2) pixel path (e.g. MPC prediction overlay).
        label: optional panel caption (controller name in comparison videos)."""
        from wavept.visualization.render import Renderer, RenderState

        if self._renderer is None:
            self._renderer = Renderer(self.sim.camera)
        s = self._last
        state = RenderState(
            t=self.t,
            target_uv=s["uv"],
            target_visible=s["visible"],
            trail=list(self.trail),
            joint_angle=self.mechanism.q.copy(),
            attitude=s["attitude"],
            predicted_uv=predicted_uv,
            title=label,
        )
        return self._renderer.render(state)

    # ------------------------------------------------------------- internals

    def _initial_joints(self) -> np.ndarray:
        roll, pitch = self.disturbance.attitude(0.0)
        if self.scenario.initial_joints == "aim_at":
            q0 = np.array(
                aim_at(self.scenario.target_W, roll, pitch, cam_offset_B=self.sim.cam_offset_B)
            )
        else:
            q0 = np.asarray(self.scenario.initial_joints, dtype=float)
        offset = np.asarray(self.scenario.initial_offset_norm, dtype=float)
        if np.any(offset != 0.0):
            q0 = self._solve_offset_joints(q0, offset, roll, pitch)
        return q0

    def _solve_offset_joints(self, q0, offset_norm, roll, pitch) -> np.ndarray:
        """Newton iterations (true geometry) placing the target at the given
        normalized image offset at t=0."""
        q = q0.copy()
        target = self.target.position(0.0)
        cam_pos = camera_position(roll, pitch, cam_offset_B=self.sim.cam_offset_B)
        eps = 1e-5
        for _ in range(6):
            def uv_norm_at(qq):
                R = camera_rotation(roll, pitch, qq[0], qq[1])
                uv, ok = self.camera.project(world_to_camera(target, R, cam_pos))
                return self.camera.normalize(uv) if ok else None

            cur = uv_norm_at(q)
            if cur is None:
                break
            err = offset_norm - cur
            if np.linalg.norm(err) < 1e-6:
                break
            J = np.zeros((2, 2))
            for axis in range(2):
                dq = np.zeros(2)
                dq[axis] = eps
                plus, minus = uv_norm_at(q + dq), uv_norm_at(q - dq)
                if plus is None or minus is None:
                    return q
                J[:, axis] = (plus - minus) / (2 * eps)
            q = q + np.linalg.solve(J, err)
        return q

    def _measure(self):
        roll, pitch = self.disturbance.attitude(self.t)
        R_WC = camera_rotation(roll, pitch, self.mechanism.q[0], self.mechanism.q[1])
        cam_pos = camera_position(roll, pitch, cam_offset_B=self.sim.cam_offset_B)
        p_C = world_to_camera(self.target.position(self.t), R_WC, cam_pos)
        true_uv, visible = self.camera.observe(p_C)
        meas_uv, meas_visible = self.detector.measure(true_uv, visible)

        if meas_visible:
            self.trail.append(meas_uv.copy())
        obs = {
            "target_uv": meas_uv,
            "target_uv_norm": self.camera.normalize(meas_uv)
            if meas_visible
            else np.full(2, np.nan),
            "target_visible": meas_visible,
            "joint_angle": self.mechanism.q.copy(),
            "joint_velocity": self.mechanism.qdot.copy(),
            "platform_attitude": np.array([roll, pitch]),
            "timestamp": self.t,
        }
        info = {
            "true_uv": true_uv,
            "true_uv_norm": self.camera.normalize(true_uv) if visible else np.full(2, np.nan),
            "true_visible": visible,
            "margin_px": self.camera.margin_px(true_uv) if visible else float("-inf"),
            "p_C": p_C,
        }
        self._last = {"uv": meas_uv, "visible": meas_visible, "attitude": (roll, pitch)}
        return obs, info
