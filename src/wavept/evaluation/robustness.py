"""M8 robustness study: one-parameter-at-a-time sweeps around the frozen
benchmark's medium tier.

Each axis varies a single physical or belief parameter; level 0 is (or is
closest to) the base condition. Seeds 6000+ are reserved for sweeps (disjoint
from benchmark dev/test and dataset seeds). Same seeds across all conditions
and controllers for comparability.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from wavept.config import Config, Impulse, WaveComponent
from wavept.control.factory import build_controller
from wavept.evaluation.runner import run_episode
from wavept.simulation.environment import PanTiltEnv

SWEEP_SEEDS = (6000, 6001, 6002, 6003, 6004)


@dataclass(frozen=True)
class SweepCondition:
    axis: str
    level: float          # numeric level value (for plots)
    label: str            # short human-readable level label
    config: Config


def _scale_waves(waves, amp_scale=1.0, freq_scale=1.0):
    return tuple(
        WaveComponent(w.amplitude * amp_scale, w.freq_hz * freq_scale, w.phase) for w in waves
    )


def _with(base: Config, *, scenario=None, sim=None, pan_tilt=None, mismatch=None) -> Config:
    cfg = base
    if pan_tilt:
        sim = dict(sim or {}, pan_tilt=dataclasses.replace(cfg.sim.pan_tilt, **pan_tilt))
    if sim:
        cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, **sim))
    if scenario:
        cfg = dataclasses.replace(cfg, scenario=dataclasses.replace(cfg.scenario, **scenario))
    if mismatch:
        cfg = dataclasses.replace(cfg, mismatch=dataclasses.replace(cfg.mismatch, **mismatch))
    return cfg


def build_sweep(base: Config) -> list[SweepCondition]:
    sc = base.scenario
    conditions: list[SweepCondition] = []

    def add(axis, level, label, cfg):
        conditions.append(SweepCondition(axis, float(level), label, cfg))

    for s in (1.0, 1.33, 1.66, 2.0):
        add("wave_amplitude", s, f"x{s:g}", _with(base, scenario={
            "roll_waves": _scale_waves(sc.roll_waves, amp_scale=s),
            "pitch_waves": _scale_waves(sc.pitch_waves, amp_scale=s)}))
    for s in (1.0, 1.25, 1.5, 1.75):
        add("wave_frequency", s, f"x{s:g}", _with(base, scenario={
            "roll_waves": _scale_waves(sc.roll_waves, freq_scale=s),
            "pitch_waves": _scale_waves(sc.pitch_waves, freq_scale=s)}))
    for a in (0.0, 6.0, 10.0, 14.0):
        imp = () if a == 0.0 else (Impulse("roll", np.radians(a), t0=8.0, duration=0.3),)
        add("impulse_deg", a, f"{a:g}°", _with(base, scenario={"impulses": imp}))
    for v in (0.9, 1.0, 1.15, 1.3):
        add("focal_scale", v, f"{v:g}", _with(base, mismatch={"focal_scale": v}))
    for v in (0.6, 0.75, 0.9, 1.1):
        add("k_u_scale", v, f"{v:g}", _with(base, mismatch={"k_u_scale": v}))
    for v in (0.0, 0.05, 0.1, 0.15):
        add("dead_zone", v, f"{v:g}", _with(base, pan_tilt={"dead_zone": v}))
    for n in (0, 2, 4, 5):
        add("cmd_delay_steps", n, f"{n}", _with(base, pan_tilt={"n_cmd_delay": n}))
    for k in (0, 1, 2, 3):
        # true delay fixed (base=2 + k extra), controller told only about 2
        add("delay_error_steps", k, f"+{k}",
            _with(base, pan_tilt={"n_cmd_delay": base.sim.pan_tilt.n_cmd_delay + k},
                  mismatch={"extra_cmd_delay": k}))
    for p in (0.0, 0.05, 0.15, 0.30):
        add("obs_dropout", p, f"{int(p*100)}%", _with(base, sim={"obs_dropout_prob": p}))
    for z in (2.0, 3.0, 4.0, 5.0):
        # true depth varies; controller belief pinned at the base 3.0 m
        add("target_depth_m", z, f"{z:g}m", _with(base,
            scenario={"target_W": (sc.target_W[0], sc.target_W[1], -z)},
            mismatch={"assumed_depth": 3.0}))
    for i, off in enumerate([(0.0, 0.0, 0.0), (0.2, 0.0, -0.1), (0.4, 0.1, -0.2)]):
        add("cam_offset_m", float(np.linalg.norm(off)), f"{np.linalg.norm(off):.2f}m",
            _with(base, sim={"cam_offset_B": off}))
    return conditions


def run_sweep(
    conditions: list[SweepCondition],
    controllers: list[str],
    seeds=SWEEP_SEEDS,
    subset_controllers: dict[str, tuple[str, ...]] | None = None,
    progress: bool = False,
) -> list[dict]:
    """subset_controllers: controller -> axes it should run on (None = all)."""
    rows = []
    for cond in conditions:
        for name in controllers:
            allowed = (subset_controllers or {}).get(name)
            if allowed is not None and cond.axis not in allowed:
                continue
            for seed in seeds:
                scenario = dataclasses.replace(cond.config.scenario, seed=seed)
                env = PanTiltEnv(cond.config.sim, scenario)
                controller = build_controller(name, cond.config)
                m = run_episode(env, controller, seed=seed).metrics
                rows.append({"axis": cond.axis, "level": cond.level, "label": cond.label,
                             "controller": name, "seed": seed, **m})
            if progress:
                sub = [r for r in rows if r["axis"] == cond.axis and r["label"] == cond.label
                       and r["controller"] == name]
                ret = np.mean([r["retained"] for r in sub])
                print(f"  {cond.axis:18s} {cond.label:6s} {name:12s} retention={ret:.2f}",
                      flush=True)
    return rows


def build_extension(base: Config) -> list[SweepCondition]:
    """Extreme levels + a combined-stress axis, added after the core sweep
    found no failures under any single perturbation (retention 1.00
    everywhere): the failure envelope is combination-driven, so we push the
    single axes further and scale all stressors together."""
    sc = base.scenario
    conditions: list[SweepCondition] = []

    def add(axis, level, label, cfg):
        conditions.append(SweepCondition(axis, float(level), label, cfg))

    for s in (2.5, 3.0):
        add("wave_amplitude", s, f"x{s:g}", _with(base, scenario={
            "roll_waves": _scale_waves(sc.roll_waves, amp_scale=s),
            "pitch_waves": _scale_waves(sc.pitch_waves, amp_scale=s)}))
    for s in (2.0, 2.5):
        add("wave_frequency", s, f"x{s:g}", _with(base, scenario={
            "roll_waves": _scale_waves(sc.roll_waves, freq_scale=s),
            "pitch_waves": _scale_waves(sc.pitch_waves, freq_scale=s)}))
    for a in (20.0, 28.0):
        add("impulse_deg", a, f"{a:g}°", _with(base, scenario={
            "impulses": (Impulse("roll", np.radians(a), t0=8.0, duration=0.3),)}))
    for v in (0.7, 1.5):
        add("focal_scale", v, f"{v:g}", _with(base, mismatch={"focal_scale": v}))
    for v in (0.45, 0.3):
        add("k_u_scale", v, f"{v:g}", _with(base, mismatch={"k_u_scale": v}))
    for v in (0.2, 0.3):
        add("dead_zone", v, f"{v:g}", _with(base, pan_tilt={"dead_zone": v}))
    for k in (4, 6):
        add("delay_error_steps", k, f"+{k}",
            _with(base, pan_tilt={"n_cmd_delay": base.sim.pan_tilt.n_cmd_delay + k},
                  mismatch={"extra_cmd_delay": k}))
    for p in (0.5, 0.7):
        add("obs_dropout", p, f"{int(p*100)}%", _with(base, sim={"obs_dropout_prob": p}))
    for z in (1.5, 7.0):
        add("target_depth_m", z, f"{z:g}m", _with(base,
            scenario={"target_W": (sc.target_W[0], sc.target_W[1], -z)},
            mismatch={"assumed_depth": 3.0}))
    off = (0.6, 0.2, -0.3)
    add("cam_offset_m", float(np.linalg.norm(off)), f"{np.linalg.norm(off):.2f}m",
        _with(base, sim={"cam_offset_B": off}))

    # combined stress: all stressors scale together (s=1 ~ hard-tier-like)
    for s in (0.5, 1.0, 1.5, 2.0):
        cfg = _with(
            base,
            scenario={
                "roll_waves": _scale_waves(sc.roll_waves, amp_scale=1 + s, freq_scale=1 + 0.4 * s),
                "pitch_waves": _scale_waves(sc.pitch_waves, amp_scale=1 + s, freq_scale=1 + 0.4 * s),
                "impulses": (Impulse("roll", np.radians(8.0 * s), t0=8.0, duration=0.3),),
                "initial_offset_norm": (min(0.35 * s, 0.75), 0.0),
            },
            pan_tilt={
                "n_cmd_delay": base.sim.pan_tilt.n_cmd_delay + round(s),
                "n_obs_delay": base.sim.pan_tilt.n_obs_delay + round(s),
            },
        )
        add("combined_stress", s, f"s={s:g}", cfg)
    return conditions
