import time
from pathlib import Path

import numpy as np
import pytest

from wavept.config import load_config
from wavept.control.factory import build_controller
from wavept.evaluation.runner import run_episode
from wavept.simulation.environment import PanTiltEnv

DYN = Path(__file__).resolve().parents[1] / "configs" / "sim_dynamics.yaml"


@pytest.fixture()
def config():
    return load_config(DYN)


def test_mpc_tracks_medium_dev_scenario(config):
    env = PanTiltEnv(config.sim, config.scenario)
    mpc = build_controller("mpc_nominal", config)
    m = run_episode(env, mpc, seed=0).metrics
    assert m["retained"]
    assert m["mean_center_err_norm"] < 0.15  # at least reactive-level tracking


def test_mpc_step_under_budget(config):
    env = PanTiltEnv(config.sim, config.scenario)
    mpc = build_controller("mpc_nominal", config)
    obs, _ = env.reset(seed=1)
    mpc.reset()
    mpc.act(obs)  # warm-up (allocations, first-call overhead)
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        action = mpc.act(obs)
        times.append((time.perf_counter() - t0) * 1e3)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    assert np.mean(times) < 40.0, f"mean MPC step {np.mean(times):.1f} ms"


def test_mpc_deterministic(config):
    metrics = []
    for _ in range(2):
        env = PanTiltEnv(config.sim, config.scenario)
        mpc = build_controller("mpc_nominal", config)
        metrics.append(run_episode(env, mpc, seed=3).metrics)
    for key in ["mean_center_err_norm", "min_margin_px", "action_abs_integral"]:
        assert metrics[0][key] == metrics[1][key]


def test_mpc_zero_action_when_lost(config):
    mpc = build_controller("mpc_nominal", config)
    mpc.reset()
    obs = {
        "target_uv": np.full(2, np.nan),
        "target_uv_norm": np.full(2, np.nan),
        "target_visible": False,
        "joint_angle": np.zeros(2),
        "joint_velocity": np.zeros(2),
        "platform_attitude": np.zeros(2),
        "timestamp": 0.0,
    }
    np.testing.assert_array_equal(mpc.act(obs), np.zeros(2))


def test_mpc_actions_within_bounds(config):
    env = PanTiltEnv(config.sim, config.scenario)
    mpc = build_controller("mpc_nominal", config)
    obs, _ = env.reset(seed=4)
    mpc.reset()
    for _ in range(20):
        a = mpc.act(obs)
        assert np.all(np.abs(a) <= 1.0)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break


def test_mpc_residual_runs_when_checkpoint_present(config):
    """Integration check for the M7 controller (skipped if the gitignored
    checkpoint has not been trained on this machine)."""
    ckpt = Path(__file__).resolve().parents[1] / "outputs" / "models" / "residual_v1.pt"
    if not ckpt.exists():
        pytest.skip("residual checkpoint not trained")
    env = PanTiltEnv(config.sim, config.scenario)
    mpc = build_controller("mpc_residual", config)
    m = run_episode(env, mpc, seed=0).metrics
    assert m["retained"]
    assert m["mean_compute_ms"] < 40.0
