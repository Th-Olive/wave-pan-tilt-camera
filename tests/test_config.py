import math
from pathlib import Path

import pytest

from wavept.config import (
    Mismatch,
    Scenario,
    SimParams,
    load_config,
    make_nominal,
    to_dict,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "sim_basic.yaml"


@pytest.fixture()
def config():
    return load_config(CONFIG)


def test_degrees_converted_to_radians(config):
    wave = config.scenario.roll_waves[0]
    assert wave.amplitude == pytest.approx(math.radians(3.0))
    assert wave.freq_hz == pytest.approx(0.20)
    assert config.sim.pan_tilt.pan_limits == pytest.approx(
        (-math.radians(170.0), math.radians(170.0))
    )


def test_basic_values(config):
    assert isinstance(config.sim, SimParams)
    assert isinstance(config.scenario, Scenario)
    assert config.sim.dt == pytest.approx(0.04)
    assert config.sim.camera.width == 640
    assert config.scenario.target_W == (-1.0, -0.5, -3.0)
    assert config.scenario.initial_joints == "aim_at"


def test_nominal_is_independent_of_sim(config):
    nominal = make_nominal(config.sim, config.mismatch, config.scenario)
    assert nominal.camera is not config.sim.camera
    assert nominal.pan_tilt is not config.sim.pan_tilt
    # no mismatch -> same values, defaults for depth
    assert nominal.camera.f == pytest.approx(config.sim.camera.f)
    assert nominal.assumed_depth == pytest.approx(3.0)


def test_mismatch_applied(config):
    mismatch = Mismatch(focal_scale=1.1, k_u_scale=0.8, assumed_depth=2.5)
    nominal = make_nominal(config.sim, mismatch, config.scenario)
    assert nominal.camera.f == pytest.approx(config.sim.camera.f * 1.1)
    assert nominal.pan_tilt.k_u == pytest.approx(config.sim.pan_tilt.k_u * 0.8)
    assert nominal.assumed_depth == pytest.approx(2.5)
    # truth untouched
    assert config.sim.camera.f == pytest.approx(500.0)


def test_to_dict_round_trip(config):
    d = to_dict(config)
    assert d["sim"]["camera"]["f"] == pytest.approx(500.0)
    assert d["scenario"]["roll_waves"][0]["amplitude"] == pytest.approx(math.radians(3.0))
