"""Random-shooting receding-horizon controller.

Model-agnostic: any object with ``rollout(ModelState, candidates) ->
(uv_norm, in_front)`` and ``n_queued`` works (nominal now, residual-corrected
with a learned residual). Replans every step; only the first action is applied.

Sampling: piecewise-constant blocks, Gaussian around the warm-started
previous best (shifted one step), plus a fraction of pure uniform
exploration; sampling is seeded, so episodes are reproducible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from wavept.control.costs import CostWeights, trajectory_cost
from wavept.models.nominal import ModelState


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 10
    n_samples: int = 256
    block: int = 2
    sigma: float = 0.3
    explore_frac: float = 0.25
    seed: int = 12345


class RandomShootingMPC:
    name = "mpc"

    def __init__(
        self,
        model,
        cfg: MPCConfig = MPCConfig(),
        weights: CostWeights = CostWeights(),
        dt: float = 0.04,
        name: str = "mpc",
    ):
        self.model = model
        self.cfg = cfg
        self.weights = weights
        self.dt = dt
        self.name = name
        self.n_blocks = int(np.ceil(cfg.horizon / cfg.block))
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)
        self.history: deque = deque(
            [np.zeros(2)] * self.model.n_queued, maxlen=max(1, self.model.n_queued)
        )
        self.prev_best: np.ndarray | None = None   # (n_blocks, 2)
        self.prev_action = np.zeros(2)
        self.prev_attitude: np.ndarray | None = None
        self.predicted_uv_norm: np.ndarray | None = None  # (N, 2) debug/overlay

    # ------------------------------------------------------------------ act

    def act(self, obs: dict) -> np.ndarray:
        if not obs["target_visible"]:
            return self._issue(np.zeros(2))

        attitude = np.asarray(obs["platform_attitude"], dtype=float)
        if self.prev_attitude is None:
            attitude_rate = np.zeros(2)
        else:
            attitude_rate = (attitude - self.prev_attitude) / self.dt
        self.prev_attitude = attitude.copy()

        state = ModelState(
            q=np.asarray(obs["joint_angle"], dtype=float),
            qdot=np.asarray(obs["joint_velocity"], dtype=float),
            uv=np.asarray(obs["target_uv"], dtype=float),
            attitude=attitude,
            attitude_rate=attitude_rate,
            queued_actions=np.array(self.history),
        )

        blocks = self._sample_blocks()                      # (K, n_blocks, 2)
        candidates = np.repeat(blocks, self.cfg.block, axis=1)[:, : self.cfg.horizon]
        uv_norm, in_front = self.model.rollout(state, candidates)
        costs = trajectory_cost(uv_norm, in_front, candidates, self.prev_action, self.weights)
        best = int(np.argmin(costs))

        self.prev_best = blocks[best]
        self.predicted_uv_norm = uv_norm[best]
        return self._issue(candidates[best, 0].copy())

    # ------------------------------------------------------------ internals

    def _issue(self, action: np.ndarray) -> np.ndarray:
        if self.model.n_queued > 0:
            self.history.append(action.copy())
        self.prev_action = action
        return action

    def _sample_blocks(self) -> np.ndarray:
        K, B = self.cfg.n_samples, self.n_blocks
        if self.prev_best is None:
            base = np.zeros((B, 2))
        else:
            base = np.roll(self.prev_best, -1, axis=0)
            base[-1] = self.prev_best[-1]
        blocks = base + self.rng.normal(0.0, self.cfg.sigma, size=(K, B, 2))
        n_explore = int(self.cfg.explore_frac * K)
        if n_explore:
            blocks[:n_explore] = self.rng.uniform(-1.0, 1.0, size=(n_explore, B, 2))
        blocks[K - 1] = base  # always include the pure warm start
        return np.clip(blocks, -1.0, 1.0)
