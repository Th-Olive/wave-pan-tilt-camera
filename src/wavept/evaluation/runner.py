"""Single episode runner — the one choke point used by
demos, benchmarks and dataset collection."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from wavept.evaluation.metrics import episode_metrics
from wavept.simulation.environment import PanTiltEnv


@dataclass
class EpisodeResult:
    metrics: dict
    frames: list | None = None
    transitions: list | None = None


def _predicted_px(env: PanTiltEnv, controller) -> np.ndarray | None:
    """MPC debug overlay: denormalize the controller's predicted target path."""
    pred = getattr(controller, "predicted_uv_norm", None)
    if pred is None:
        return None
    cam = env.sim.camera
    return np.stack(
        [(pred[:, 0] + 1.0) * cam.width / 2.0, (pred[:, 1] + 1.0) * cam.height / 2.0],
        axis=-1,
    )


def run_episode(
    env: PanTiltEnv,
    controller,
    seed: int | None = None,
    record_frames: bool = False,
    record_transitions: bool = False,
    label: str = "",
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    controller.reset()

    frames = [env.render(label=label)] if record_frames else None
    transitions = [] if record_transitions else None
    # metrics use TRUE visibility/margins from info (obs may be delayed)
    visible, uv_norm, margins, actions, compute_ms = (
        [info["true_visible"]],
        [obs["target_uv_norm"].copy()],
        [info["margin_px"]],
        [],
        [],
    )
    terminated = truncated = False
    while not (terminated or truncated):
        t0 = time.perf_counter()
        action = np.asarray(controller.act(obs), dtype=float)
        compute_ms.append((time.perf_counter() - t0) * 1e3)
        prev_obs, prev_info = obs, info
        obs, _, terminated, truncated, info = env.step(action)
        visible.append(info["true_visible"])
        uv_norm.append(obs["target_uv_norm"].copy())
        margins.append(info["margin_px"])
        actions.append(action)
        if record_frames:
            frames.append(
                env.render(predicted_uv=_predicted_px(env, controller), label=label)
            )
        if record_transitions:
            transitions.append(
                {
                    "obs": prev_obs,       # served (possibly stale) observation
                    "action": action,
                    "next_obs": obs,
                    "prev_info": prev_info,  # truth at act time
                    "info": info,            # truth after the step
                }
            )

    metrics = episode_metrics(
        dt=env.sim.dt,
        visible=np.array(visible),
        uv_norm=np.array(uv_norm),
        margin_px=np.array(margins),
        actions=np.array(actions),
        v_max=env.sim.pan_tilt.v_max,
        compute_ms=np.array(compute_ms),
        terminated_early=bool(terminated),
    )
    return EpisodeResult(metrics=metrics, frames=frames, transitions=transitions)
