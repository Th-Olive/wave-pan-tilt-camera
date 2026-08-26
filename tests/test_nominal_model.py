"""Cornerstone M4 test: with zero mismatch and zero disturbance, the nominal
model rollout must match the simulator to < 0.5 px over 10 steps — including
when command and observation delays are active (delay-replay logic)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from wavept.config import load_config, make_nominal
from wavept.models.nominal import ModelState, NominalModel
from wavept.simulation.environment import PanTiltEnv

DYN = Path(__file__).resolve().parents[1] / "configs" / "sim_dynamics.yaml"
N = 10


def calm_config(n_cmd_delay=0, n_obs_delay=0):
    """sim_dynamics variant: no waves, no mismatch, chosen delays."""
    config = load_config(DYN)
    pan_tilt = dataclasses.replace(
        config.sim.pan_tilt, n_cmd_delay=n_cmd_delay, n_obs_delay=n_obs_delay
    )
    return dataclasses.replace(
        config,
        sim=dataclasses.replace(config.sim, pan_tilt=pan_tilt),
        scenario=dataclasses.replace(config.scenario, roll_waves=(), pitch_waves=()),
        mismatch=dataclasses.replace(config.mismatch, k_u_scale=1.0),
    )


def rollout_vs_sim(config, past_actions, candidate):
    """Feed past_actions to the env, snapshot obs, roll out the model with the
    candidate sequence, and compare against the simulator's true pixels."""
    env = PanTiltEnv(config.sim, config.scenario)
    obs, info = env.reset(seed=0)
    for a in past_actions:
        obs, _, _, _, info = env.step(a)

    nominal = make_nominal(config.sim, config.mismatch, config.scenario)
    model = NominalModel(nominal, n_substeps=config.sim.n_substeps)
    d = model.n_queued
    queued = np.array(past_actions[-d:]) if d else np.zeros((0, 2))
    state = ModelState(
        q=obs["joint_angle"],
        qdot=obs["joint_velocity"],
        uv=obs["target_uv"],
        attitude=obs["platform_attitude"],
        attitude_rate=np.zeros(2),
        queued_actions=queued,
    )
    uv_pred, in_front = model.rollout(state, candidate[None])
    assert in_front.all()

    cam = env.camera
    errors = []
    for k in range(N):
        _, _, _, _, info = env.step(candidate[k])
        true_uv = info["true_uv"]
        pred_px = np.array(
            [
                (uv_pred[0, k, 0] + 1.0) * cam.width / 2.0,
                (uv_pred[0, k, 1] + 1.0) * cam.height / 2.0,
            ]
        )
        errors.append(np.linalg.norm(pred_px - true_uv))
    return max(errors)


@pytest.fixture()
def candidate():
    rng = np.random.default_rng(7)
    return np.clip(rng.normal(0.0, 0.4, size=(N, 2)), -1, 1)


def test_rollout_matches_simulator_no_delay(candidate):
    rng = np.random.default_rng(1)
    past = list(np.clip(rng.normal(0.0, 0.3, size=(5, 2)), -1, 1))
    err = rollout_vs_sim(calm_config(), past, candidate)
    assert err < 0.5  # px over 10 steps


def test_rollout_matches_simulator_with_delays(candidate):
    rng = np.random.default_rng(2)
    past = list(np.clip(rng.normal(0.0, 0.3, size=(8, 2)), -1, 1))
    err = rollout_vs_sim(calm_config(n_cmd_delay=3, n_obs_delay=2), past, candidate)
    assert err < 0.5


def test_rollout_vectorization_consistent(candidate):
    """K-batched rollout equals per-sample rollouts."""
    config = calm_config(n_cmd_delay=2, n_obs_delay=1)
    env = PanTiltEnv(config.sim, config.scenario)
    obs, _ = env.reset(seed=0)
    nominal = make_nominal(config.sim, config.mismatch, config.scenario)
    model = NominalModel(nominal)
    state = ModelState(
        q=obs["joint_angle"], qdot=obs["joint_velocity"], uv=obs["target_uv"],
        attitude=obs["platform_attitude"], attitude_rate=np.array([0.05, -0.02]),
        queued_actions=np.zeros((3, 2)),
    )
    rng = np.random.default_rng(3)
    batch = np.clip(rng.normal(0.0, 0.4, size=(16, N, 2)), -1, 1)
    uv_batch, ok_batch = model.rollout(state, batch)
    for i in [0, 5, 15]:
        uv_i, ok_i = model.rollout(state, batch[i][None])
        np.testing.assert_allclose(uv_batch[i], uv_i[0], atol=1e-12)
        np.testing.assert_array_equal(ok_batch[i], ok_i[0])
