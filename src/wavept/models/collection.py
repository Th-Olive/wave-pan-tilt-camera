"""Dataset collection (M5): conditions, split plan, and mixture policies
for the learned residual model.

Splits are by complete episode AND condition:
- train / val / test_id share the ID conditions (calm, base, rough) with
  disjoint seed ranges;
- test_ood uses distinct physical conditions (unseen wave frequencies,
  focal_scale 1.2, k_u_scale 0.7, assumed depth 4.0), disjoint seeds.

All conditions share the dataset's fixed delays (n_cmd_delay=2, n_obs_delay=1)
so the recorded action-history length is constant.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wavept.config import Config, Mismatch, WaveComponent, load_config, make_nominal, to_dict
from wavept.control.factory import build_controller
from wavept.evaluation.runner import run_episode
from wavept.models.datasets import (
    ACTION_HISTORY_LEN,
    add_nominal_predictions,
    episode_to_arrays,
    write_dataset,
)
from wavept.models.nominal import NominalModel
from wavept.simulation.environment import PanTiltEnv

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = REPO_ROOT / "configs" / "sim_dynamics.yaml"

ID_CONDITIONS = ("calm", "base", "rough")
OOD_CONDITIONS = ("ood_freq", "ood_model")

# boundary-skimming start offsets, cycled per episode
BOUNDARY_OFFSETS = ((0.7, 0.0), (-0.7, 0.3), (0.5, -0.6), (-0.5, 0.55))

SEED_BASE = {"train": 2000, "val": 3000, "test_id": 4000, "test_ood": 5000}


def _waves(spec):
    return tuple(WaveComponent(amplitude=np.radians(a), freq_hz=f) for a, f in spec)


def build_conditions(base: Config) -> dict[str, Config]:
    def variant(roll, pitch, mismatch):
        scenario = dataclasses.replace(
            base.scenario, roll_waves=_waves(roll), pitch_waves=_waves(pitch)
        )
        return dataclasses.replace(base, scenario=scenario, mismatch=mismatch)

    return {
        "calm": variant([(3.0, 0.20)], [(2.0, 0.13)], Mismatch()),
        "base": dataclasses.replace(base),  # sim_dynamics: medium waves, k_u 0.9
        "rough": variant(
            [(13.0, 0.18), (6.5, 0.35), (4.2, 0.55)],
            [(10.7, 0.22), (6.5, 0.40), (3.7, 0.55)],
            Mismatch(k_u_scale=0.8, focal_scale=1.1, assumed_depth=2.5),
        ),
        "ood_freq": variant(
            [(5.0, 0.65), (3.0, 0.80)], [(4.0, 0.68), (2.5, 0.78)], Mismatch()
        ),
        "ood_model": dataclasses.replace(
            base, mismatch=Mismatch(focal_scale=1.2, k_u_scale=0.7, assumed_depth=4.0)
        ),
    }


# split -> condition -> policy -> number of episodes
DEFAULT_PLAN = {
    "train": {c: {"ou": 8, "pgain": 6, "mpc": 6, "boundary": 4} for c in ID_CONDITIONS},
    "val": {c: {"ou": 1, "pgain": 1, "mpc": 1, "boundary": 1} for c in ID_CONDITIONS},
    "test_id": {c: {"ou": 1, "pgain": 1, "mpc": 1, "boundary": 1} for c in ID_CONDITIONS},
    "test_ood": {c: {"ou": 2, "pgain": 2, "mpc": 2} for c in OOD_CONDITIONS},
}


class OUPolicy:
    """Smooth random effort: Ornstein-Uhlenbeck around zero."""

    name = "ou"

    def __init__(self, seed: int, theta: float = 4.0, sigma: float = 0.5, dt: float = 0.04):
        self.seed, self.theta, self.sigma, self.dt = seed, theta, sigma, dt
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.a = np.zeros(2)

    def act(self, obs) -> np.ndarray:
        self.a = (
            self.a
            - self.theta * self.a * self.dt
            + self.sigma * np.sqrt(self.dt) * self.rng.normal(size=2)
        )
        self.a = np.clip(self.a, -1.0, 1.0)
        return self.a.copy()


@dataclass(frozen=True)
class EpisodeSpec:
    split: str
    condition: str
    policy: str
    seed: int
    offset: tuple[float, float] = (0.0, 0.0)


def make_plan(plan: dict = None) -> list[EpisodeSpec]:
    plan = plan or DEFAULT_PLAN
    specs = []
    for split, conditions in plan.items():
        counter = 0
        for condition, policies in conditions.items():
            for policy, n in policies.items():
                for j in range(n):
                    offset = (
                        BOUNDARY_OFFSETS[j % len(BOUNDARY_OFFSETS)]
                        if policy == "boundary"
                        else (0.0, 0.0)
                    )
                    specs.append(
                        EpisodeSpec(split, condition, policy, SEED_BASE[split] + counter, offset)
                    )
                    counter += 1
    return specs


def _make_controller(spec: EpisodeSpec, config: Config):
    if spec.policy == "ou":
        return OUPolicy(seed=spec.seed * 31 + 7, dt=config.sim.dt)
    if spec.policy == "mpc":
        return build_controller("mpc_nominal", config)
    return build_controller("pgain", config)  # pgain and boundary


def collect(
    out_dir: Path,
    plan: dict = None,
    duration: float | None = None,
    progress: bool = False,
) -> Path:
    """Run the plan, convert episodes to arrays with nominal one-step
    predictions, and write the dataset."""
    base = load_config(BASE_CONFIG)
    conditions = build_conditions(base)
    specs = make_plan(plan)

    models = {
        name: NominalModel(
            make_nominal(cfg.sim, cfg.mismatch, cfg.scenario),
            n_substeps=cfg.sim.n_substeps,
            rate_decay=1.0,  # audit/training corrects the undamped nominal model
        )
        for name, cfg in conditions.items()
    }

    split_episodes: dict[str, list] = {s: [] for s in {sp.split for sp in specs}}
    episode_id = 0
    for spec in specs:
        config = conditions[spec.condition]
        scenario = dataclasses.replace(
            config.scenario, seed=spec.seed, initial_offset_norm=spec.offset
        )
        if duration is not None:
            scenario = dataclasses.replace(scenario, duration=duration)
        run_config = dataclasses.replace(config, scenario=scenario)
        env = PanTiltEnv(run_config.sim, run_config.scenario)
        controller = _make_controller(spec, run_config)
        result = run_episode(env, controller, seed=spec.seed, record_transitions=True)
        arrays = episode_to_arrays(result.transitions, dt=config.sim.dt)
        add_nominal_predictions(arrays, models[spec.condition])
        meta = {
            "episode_id": episode_id,
            "condition": spec.condition,
            "policy": spec.policy,
            "seed": spec.seed,
            "retained": result.metrics["retained"],
        }
        split_episodes[spec.split].append((meta, arrays))
        if progress:
            print(
                f"  ep {episode_id:3d} {spec.split:8s} {spec.condition:9s} "
                f"{spec.policy:8s} seed={spec.seed} steps={len(arrays['t'])}"
            )
        episode_id += 1

    from wavept.manifest import git_info

    manifest = {
        "git": git_info(),
        "action_history_len": ACTION_HISTORY_LEN,
        "conditions": {name: to_dict(cfg) for name, cfg in conditions.items()},
        "plan": plan or DEFAULT_PLAN,
        "seed_base": SEED_BASE,
        "n_episodes": episode_id,
        "duration_override": duration,
    }
    return write_dataset(out_dir, split_episodes, manifest)
