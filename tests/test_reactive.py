"""Closed-loop tests for the M2 reactive baselines."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from wavept.config import load_config, make_nominal
from wavept.control.reactive import JacobianController, PGainController
from wavept.evaluation.runner import run_episode
from wavept.models.nominal import numeric_uv_jacobian, reconstruct_target_W
from wavept.simulation.environment import PanTiltEnv

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "sim_basic.yaml"

# uncompensated mean error on sim_basic seed 0 is ~0.104 (M1 result)
UNCONTROLLED_MEAN_ERR = 0.104


@pytest.fixture()
def config():
    return load_config(CONFIG)


@pytest.fixture()
def nominal(config):
    return make_nominal(config.sim, config.mismatch, config.scenario)


def run(config, controller):
    env = PanTiltEnv(config.sim, config.scenario)
    return run_episode(env, controller, seed=config.scenario.seed).metrics


def test_pgain_tracks_mild_waves(config):
    m = run(config, PGainController())
    assert m["retained"]
    assert m["visible_time_ratio"] == 1.0
    assert m["mean_center_err_norm"] < UNCONTROLLED_MEAN_ERR / 2


def test_jacobian_tracks_mild_waves(config, nominal):
    m = run(config, JacobianController(nominal))
    assert m["retained"]
    assert m["mean_center_err_norm"] < UNCONTROLLED_MEAN_ERR / 2


def test_controllers_zero_action_when_lost(nominal):
    obs = {
        "target_uv": np.full(2, np.nan),
        "target_uv_norm": np.full(2, np.nan),
        "target_visible": False,
        "joint_angle": np.zeros(2),
        "joint_velocity": np.zeros(2),
        "platform_attitude": np.zeros(2),
        "timestamp": 0.0,
    }
    for c in [PGainController(), JacobianController(nominal)]:
        c.reset()
        np.testing.assert_array_equal(c.act(obs), np.zeros(2))


def test_run_episode_deterministic(config):
    m1 = run(config, PGainController())
    m2 = run(config, PGainController())
    for key in ["mean_center_err_norm", "min_margin_px", "action_abs_integral"]:
        assert m1[key] == m2[key]


def test_reconstruction_recovers_target(config, nominal):
    """At the aim configuration the reconstructed world point matches the true
    target (assumed depth is exact for sim_basic: flat-scene z = target z)."""
    from wavept.geometry.frames import aim_at

    q = np.array(aim_at(config.scenario.target_W))
    p_W = reconstruct_target_W(np.array([320.0, 240.0]), q, np.zeros(2), nominal)
    np.testing.assert_allclose(p_W, config.scenario.target_W, atol=1e-6)


def test_numeric_jacobian_matches_locked_signs(config, nominal):
    """J columns must reproduce the M1 sign structure at the aim config:
    pan -> -v_bar (u_bar ~ 0), tilt -> +u_bar (v_bar ~ 0)."""
    from wavept.geometry.frames import aim_at

    q = np.array(aim_at(config.scenario.target_W))
    J = numeric_uv_jacobian(np.array([320.0, 240.0]), q, np.zeros(2), nominal)
    assert abs(J[0, 0]) < 0.05          # d u_bar / d pan ~ 0
    assert J[1, 0] < -0.3               # d v_bar / d pan < 0
    assert J[0, 1] > 1.0                # d u_bar / d tilt ~ 2f/W = 1.56
    assert abs(J[1, 1]) < 0.05          # d v_bar / d tilt ~ 0
