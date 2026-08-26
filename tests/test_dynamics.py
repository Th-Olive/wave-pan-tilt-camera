"""M3 tests: second-order motor dynamics, limits, delays, dead zone."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from wavept.config import PanTiltParams, load_config
from wavept.simulation.environment import PanTiltEnv
from wavept.simulation.pan_tilt import PanTiltMechanism

DYN = Path(__file__).resolve().parents[1] / "configs" / "sim_dynamics.yaml"
DT = 0.04


def make_mech(**overrides):
    params = PanTiltParams(mode="second_order", **overrides)
    mech = PanTiltMechanism(params, n_substeps=4)
    mech.reset(np.zeros(2))
    return mech


def analytic_step(k_u, k_d, a, t):
    """q(t), qdot(t) for constant effort from rest, unlimited."""
    v_inf = k_u * a / k_d
    qdot = v_inf * (1.0 - math.exp(-k_d * t))
    q = v_inf * (t - (1.0 - math.exp(-k_d * t)) / k_d)
    return q, qdot


def test_step_response_matches_analytic():
    k_u, k_d, a = 20.0, 8.0, 0.5
    mech = make_mech(k_u=k_u, k_d=k_d, qdot_max=100.0, pan_limits=(-10, 10), tilt_limits=(-10, 10))
    n = int(1.0 / DT)
    for _ in range(n):
        mech.step([a, a], DT)
    q_ref, qdot_ref = analytic_step(k_u, k_d, a, n * DT)
    np.testing.assert_allclose(mech.q, [q_ref, q_ref], rtol=2e-2)
    np.testing.assert_allclose(mech.qdot, [qdot_ref, qdot_ref], rtol=2e-2)


def test_velocity_limit():
    mech = make_mech(k_u=100.0, k_d=1.0, qdot_max=2.5, pan_limits=(-100, 100), tilt_limits=(-100, 100))
    for _ in range(50):
        mech.step([1.0, 1.0], DT)
    assert np.all(np.abs(mech.qdot) <= 2.5 + 1e-12)
    assert mech.qdot[0] == pytest.approx(2.5)


def test_angle_clamp_zeroes_velocity():
    mech = make_mech(tilt_limits=(-0.17, 0.1))  # tight tilt stop
    for _ in range(100):
        mech.step([0.0, 1.0], DT)
    assert mech.q[1] == pytest.approx(0.1)
    assert mech.qdot[1] == pytest.approx(0.0)


def test_command_delay():
    mech = make_mech(n_cmd_delay=2)
    mech.step([1.0, 1.0], DT)  # executes zero-init queue
    mech.step([1.0, 1.0], DT)  # still zero
    assert np.all(mech.q == 0.0) and np.all(mech.qdot == 0.0)
    mech.step([0.0, 0.0], DT)  # executes the first real command
    assert np.all(mech.qdot > 0.0)


def test_dead_zone():
    mech = make_mech(dead_zone=0.1)
    for _ in range(10):
        mech.step([0.05, -0.09], DT)
    assert np.all(mech.q == 0.0)
    mech.step([0.2, 0.0], DT)
    assert mech.q[0] > 0.0


def test_effort_clipped_to_unit():
    strong = make_mech(qdot_max=100.0, pan_limits=(-100, 100), tilt_limits=(-100, 100))
    weak = make_mech(qdot_max=100.0, pan_limits=(-100, 100), tilt_limits=(-100, 100))
    for _ in range(10):
        strong.step([5.0, 5.0], DT)
        weak.step([1.0, 1.0], DT)
    np.testing.assert_allclose(strong.q, weak.q)


def test_obs_delay_serves_stale_measurement():
    config = load_config(DYN)  # n_obs_delay = 1
    env = PanTiltEnv(config.sim, config.scenario)
    obs0, _ = env.reset()
    assert obs0["timestamp"] == pytest.approx(0.0)
    obs1, *_ = env.step([0.0, 0.0])
    # first step still serves the t=0 measurement
    assert obs1["timestamp"] == pytest.approx(0.0)
    obs2, *_ = env.step([0.0, 0.0])
    assert obs2["timestamp"] == pytest.approx(config.sim.dt)  # one step stale


def test_second_order_env_deterministic():
    config = load_config(DYN)
    rng = np.random.default_rng(5)
    actions = rng.uniform(-1, 1, size=(20, 2))
    traces = []
    for _ in range(2):
        env = PanTiltEnv(config.sim, config.scenario)
        env.reset(seed=42)
        trace = []
        for a in actions:
            obs, *_ = env.step(a)
            trace.append(np.concatenate([obs["joint_angle"], obs["target_uv"]]))
        traces.append(np.array(trace))
    np.testing.assert_array_equal(traces[0], traces[1])


def test_initial_offset_places_target():
    config = load_config(DYN)
    scenario = dataclasses.replace(config.scenario, initial_offset_norm=(0.7, 0.0))
    env = PanTiltEnv(config.sim, scenario)
    obs, _ = env.reset(seed=3)
    np.testing.assert_allclose(obs["target_uv_norm"], [0.7, 0.0], atol=1e-4)


def test_impulse_adds_transient():
    from wavept.config import Impulse
    from wavept.simulation.platform_motion import WaveDisturbance

    rng = np.random.default_rng(0)
    d = WaveDisturbance((), (), rng, impulses=(Impulse("roll", 0.1, t0=1.0, duration=0.4),))
    assert d.attitude(0.5)[0] == pytest.approx(0.0)
    assert d.attitude(1.2)[0] == pytest.approx(0.1)  # half-sine peak
    assert d.attitude(1.5)[0] == pytest.approx(0.0, abs=1e-12)
    assert d.attitude(2.0)[0] == pytest.approx(0.0)
