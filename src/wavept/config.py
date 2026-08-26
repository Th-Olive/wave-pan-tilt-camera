"""Configuration dataclasses and YAML loading.

Conventions:
- All angles are radians, lengths meters, times seconds internally.
- YAML keys suffixed ``_deg`` hold degrees; they are converted to radians at
  load time and the suffix is stripped (works element-wise on lists).
- ``SimParams`` is what the simulator actually uses ("truth").
  ``NominalParams`` is what controllers/models believe; it is derived from
  ``SimParams`` + a ``Mismatch`` spec and never shares instances with it.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEG_SUFFIX = "_deg"


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraParams:
    f: float = 500.0          # focal length, px
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    margin_px: float = 64.0   # warning margin -> safe region |u_bar|<=0.8, |v_bar|<=0.733
    z_min: float = 0.05       # points with Z_C below this are invisible


@dataclass(frozen=True)
class PanTiltParams:
    mode: str = "kinematic"                    # "kinematic" (M1-M2) | "second_order" (M3+)
    v_max: float = 2.0                         # rad/s velocity clip
    pan_limits: tuple[float, float] = (-math.radians(170), math.radians(170))
    tilt_limits: tuple[float, float] = (-math.radians(10), math.radians(80))
    k_u: float = 20.0                          # rad/s^2 per unit effort (second_order)
    k_d: float = 8.0                           # 1/s damping (second_order)
    qdot_max: float = 2.5                      # rad/s velocity limit (second_order)
    n_cmd_delay: int = 0                       # command delay, control steps
    n_obs_delay: int = 0                       # observation delay, control steps
    dead_zone: float = 0.0                     # |a| below this -> 0 (second_order)


@dataclass(frozen=True)
class SimParams:
    """Ground-truth simulator parameters. Controllers must never receive this."""

    camera: CameraParams = field(default_factory=CameraParams)
    pan_tilt: PanTiltParams = field(default_factory=PanTiltParams)
    dt: float = 0.04                           # control period (25 Hz)
    n_substeps: int = 4                        # physics substeps per control step
    cam_offset_B: tuple[float, float, float] = (0.0, 0.0, 0.0)  # lever arm, platform frame
    obs_dropout_prob: float = 0.0              # per-step detection miss probability (M8)


@dataclass(frozen=True)
class WaveComponent:
    amplitude: float                           # rad
    freq_hz: float
    phase: float | None = None                 # None -> drawn from episode seed at reset


@dataclass(frozen=True)
class Impulse:
    """Transient attitude bump: a half-sine of given peak amplitude."""

    axis: str                                  # "roll" | "pitch"
    amplitude: float                           # rad, peak
    t0: float                                  # s, onset
    duration: float                            # s


@dataclass(frozen=True)
class Scenario:
    # default keeps nominal pan (+26.6 deg) far from the +/-170 deg stops
    target_W: tuple[float, float, float] = (-1.0, -0.5, -3.0)
    duration: float = 20.0
    seed: int = 0
    roll_waves: tuple[WaveComponent, ...] = ()
    pitch_waves: tuple[WaveComponent, ...] = ()
    impulses: tuple[Impulse, ...] = ()
    # "aim_at" -> initial joints computed to center the target; or explicit (q_p, q_t) rad
    initial_joints: str | tuple[float, float] = "aim_at"
    # start the target at this normalized image offset instead of the center
    initial_offset_norm: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class Mismatch:
    """How the controllers' nominal belief deviates from simulator truth."""

    focal_scale: float = 1.0                   # nominal f = true f * focal_scale
    k_u_scale: float = 1.0
    k_d_scale: float = 1.0
    assumed_depth: float | None = None         # None -> true |target z|
    extra_cmd_delay: int = 0                   # delay steps unknown to the controller


@dataclass(frozen=True)
class NominalParams:
    """What controllers and predictive models believe. Derived, never shared."""

    camera: CameraParams
    pan_tilt: PanTiltParams
    assumed_depth: float
    dt: float


@dataclass(frozen=True)
class Config:
    sim: SimParams
    scenario: Scenario
    mismatch: Mismatch = field(default_factory=Mismatch)


# ---------------------------------------------------------------------------
# Degree -> radian conversion of raw YAML
# ---------------------------------------------------------------------------


def _deg_to_rad(value: Any) -> Any:
    if isinstance(value, list):
        return [_deg_to_rad(v) for v in value]
    return math.radians(value)


def _convert_degrees(node: Any) -> Any:
    """Recursively strip ``_deg`` suffixes, converting values to radians."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key.endswith(_DEG_SUFFIX):
                new_key = key[: -len(_DEG_SUFFIX)]
                if new_key in node:
                    raise ValueError(f"both '{key}' and '{new_key}' present")
                out[new_key] = _deg_to_rad(value)
            else:
                out[key] = _convert_degrees(value)
        return out
    if isinstance(node, list):
        return [_convert_degrees(v) for v in node]
    return node


# ---------------------------------------------------------------------------
# Construction from dicts
# ---------------------------------------------------------------------------


def _as_tuple(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _build_sim(raw: dict) -> SimParams:
    raw = dict(raw)
    camera = CameraParams(**raw.pop("camera", {}))
    pan_tilt_raw = {k: _as_tuple(v) for k, v in raw.pop("pan_tilt", {}).items()}
    pan_tilt = PanTiltParams(**pan_tilt_raw)
    raw = {k: _as_tuple(v) for k, v in raw.items()}
    return SimParams(camera=camera, pan_tilt=pan_tilt, **raw)


def _build_scenario(raw: dict) -> Scenario:
    raw = dict(raw)
    roll = tuple(WaveComponent(**c) for c in raw.pop("roll_waves", []))
    pitch = tuple(WaveComponent(**c) for c in raw.pop("pitch_waves", []))
    impulses = tuple(Impulse(**c) for c in raw.pop("impulses", []))
    raw = {k: _as_tuple(v) for k, v in raw.items()}
    return Scenario(roll_waves=roll, pitch_waves=pitch, impulses=impulses, **raw)


def convert_degrees(raw: dict) -> dict:
    """Public wrapper: strip ``_deg`` suffixes, converting values to radians."""
    return _convert_degrees(raw)


def config_from_dict(raw: dict) -> Config:
    """Build a Config from an already-parsed (degree-converted) dict tree."""
    return Config(
        sim=_build_sim(raw.get("sim", {})),
        scenario=_build_scenario(raw.get("scenario", {})),
        mismatch=Mismatch(**raw.get("mismatch", {})),
    )


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return config_from_dict(_convert_degrees(raw))


def make_nominal(sim: SimParams, mismatch: Mismatch, scenario: Scenario) -> NominalParams:
    """Derive the controllers' belief from truth + mismatch (fresh instances)."""
    camera = dataclasses.replace(sim.camera, f=sim.camera.f * mismatch.focal_scale)
    pan_tilt = dataclasses.replace(
        sim.pan_tilt,
        k_u=sim.pan_tilt.k_u * mismatch.k_u_scale,
        k_d=sim.pan_tilt.k_d * mismatch.k_d_scale,
        # the controller only knows the delay it was told about
        n_cmd_delay=max(0, sim.pan_tilt.n_cmd_delay - mismatch.extra_cmd_delay),
    )
    depth = mismatch.assumed_depth
    if depth is None:
        depth = abs(scenario.target_W[2])
    return NominalParams(camera=camera, pan_tilt=pan_tilt, assumed_depth=depth, dt=sim.dt)


def to_dict(obj: Any) -> Any:
    """Dataclass tree -> plain dict/list tree (for manifests / YAML dump)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj
