import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from wavept.config import Scenario, WaveComponent, load_config
from wavept.simulation.environment import PanTiltEnv

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "sim_basic.yaml"


@pytest.fixture()
def config():
    return load_config(CONFIG)


def make_env(config, **scenario_overrides):
    scenario = dataclasses.replace(config.scenario, **scenario_overrides)
    return PanTiltEnv(config.sim, scenario)


def test_reset_deterministic(config):
    rng = np.random.default_rng(123)
    actions = rng.uniform(-1.0, 1.0, size=(10, 2))
    traces = []
    for _ in range(2):
        env = make_env(config)
        obs, _ = env.reset(seed=7)
        trace = [obs["target_uv"].copy()]
        for a in actions:
            obs, *_ = env.step(a)
            trace.append(obs["target_uv"].copy())
        traces.append(np.array(trace))
    np.testing.assert_array_equal(traces[0], traces[1])


def test_step_advances_time_and_obs_fields(config):
    env = make_env(config)
    obs, info = env.reset()
    assert obs["timestamp"] == pytest.approx(0.0)
    assert obs["target_visible"]  # aim_at start -> centered
    np.testing.assert_allclose(obs["target_uv"], [320.0, 240.0], atol=0.5)
    assert info["margin_px"] > 200.0
    obs, _, _, _, _ = env.step([0.0, 0.0])
    assert obs["timestamp"] == pytest.approx(config.sim.dt)
    assert obs["joint_angle"].shape == (2,)
    assert obs["platform_attitude"].shape == (2,)


def test_uncontrolled_mild_waves_move_target_but_keep_visible(config):
    env = make_env(config)
    obs, _ = env.reset()
    max_err = 0.0
    while True:
        obs, _, terminated, truncated, _ = env.step([0.0, 0.0])
        if obs["target_visible"]:
            max_err = max(max_err, float(np.linalg.norm(obs["target_uv"] - [320, 240])))
        if terminated or truncated:
            break
    assert truncated and not terminated  # mild waves: never lost
    assert max_err > 20.0  # but clearly moving in the image


def test_target_loss_terminates(config):
    # violent deterministic waves, no control -> target must leave the frame
    waves = (WaveComponent(amplitude=math.radians(35.0), freq_hz=0.5, phase=0.0),)
    env = make_env(config, roll_waves=waves, pitch_waves=waves, duration=10.0)
    env.reset()
    terminated = truncated = False
    reward = 0.0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step([0.0, 0.0])
    assert terminated and not truncated
    assert not obs["target_visible"]
    assert np.all(np.isnan(obs["target_uv"]))
    assert reward == pytest.approx(-2.0)


def test_render_returns_frame(config):
    env = make_env(config)
    env.reset()
    frame = env.render()
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
