"""M8 tests: observation dropouts and sweep construction."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from wavept.config import load_config
from wavept.evaluation.benchmark import load_benchmark
from wavept.evaluation.robustness import build_sweep
from wavept.simulation.environment import PanTiltEnv

BENCH = Path(__file__).resolve().parents[1] / "configs" / "benchmark_v1.yaml"
DYN = Path(__file__).resolve().parents[1] / "configs" / "sim_dynamics.yaml"


def make_env(dropout):
    config = load_config(DYN)
    sim = dataclasses.replace(config.sim, obs_dropout_prob=dropout)
    return PanTiltEnv(sim, config.scenario)


def run_visible_ratio(env, seed, n=200):
    env.reset(seed=seed)
    served = []
    truncated = terminated = False
    for _ in range(n):
        obs, _, terminated, truncated, info = env.step([0.0, 0.0])
        served.append(obs["target_visible"])
        assert info["true_visible"]  # dropouts never fake a true loss
        if terminated or truncated:
            break
    assert not terminated
    return np.mean(served)


def test_dropout_rates():
    assert run_visible_ratio(make_env(0.0), seed=0) == 1.0
    assert run_visible_ratio(make_env(1.0), seed=0) == 0.0  # never detected, never lost
    r = run_visible_ratio(make_env(0.3), seed=1)
    assert 0.55 < r < 0.85  # ~70% detections


def test_dropout_deterministic_and_phase_preserving():
    """Dropout draws must not perturb the wave phases (child rng)."""
    trails = []
    for dropout in (0.0, 0.3):
        env = make_env(dropout)
        env.reset(seed=5)
        atts = []
        for _ in range(20):
            obs, *_ = env.step([0.1, -0.1])
            atts.append(obs["platform_attitude"].copy())
        trails.append(np.array(atts))
    np.testing.assert_array_equal(trails[0], trails[1])


def test_build_sweep_isolates_axes():
    base = {t.name: t for t in load_benchmark(BENCH).tiers}["medium"].config
    conditions = build_sweep(base)
    axes = {c.axis for c in conditions}
    assert len(axes) == 11
    for c in conditions:
        # every condition still runs the second-order mechanism and base camera
        assert c.config.sim.pan_tilt.mode == "second_order"
        assert c.config.sim.camera.f == base.sim.camera.f
        if c.axis == "target_depth_m":
            assert c.config.mismatch.assumed_depth == pytest.approx(3.0)
            assert c.config.scenario.target_W[2] == pytest.approx(-c.level)
        if c.axis == "delay_error_steps":
            assert c.config.sim.pan_tilt.n_cmd_delay == base.sim.pan_tilt.n_cmd_delay + int(c.level)
            assert c.config.mismatch.extra_cmd_delay == int(c.level)
    # base level of the amplitude axis equals the base scenario
    amp0 = next(c for c in conditions if c.axis == "wave_amplitude" and c.level == 1.0)
    assert amp0.config.scenario.roll_waves == base.scenario.roll_waves
