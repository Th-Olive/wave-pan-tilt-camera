"""Controller factory: build controllers by name for a given Config.

In ``second_order`` (effort) mode, velocity-domain controllers are adapted via
the NOMINAL steady-state map effort = (k_d/k_u) * velocity, so gains stay in
the velocity domain and motor-gain mismatch degrades them realistically. The
oracle uses true simulator parameters by design (upper bound, not fair).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from wavept.config import Config, make_nominal
from wavept.control.costs import CostWeights
from wavept.control.predictive import MPCConfig, RandomShootingMPC
from wavept.control.reactive import JacobianController, PGainController
from wavept.geometry.frames import aim_at
from wavept.models.nominal import NominalModel

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTROLLERS_YAML = REPO_ROOT / "configs" / "controllers" / "reactive.yaml"
MPC_NOMINAL_YAML = REPO_ROOT / "configs" / "controllers" / "predictive_nominal.yaml"
MPC_RESIDUAL_YAML = REPO_ROOT / "configs" / "controllers" / "predictive_residual.yaml"


class ZeroController:
    name = "none"

    def act(self, obs):
        return np.zeros(2)

    def reset(self):
        pass


class OracleController:
    """Servos toward the aim_at joints using true target position (and true
    motor parameters in effort mode). Geometry oracle / sanity reference: it
    is NOT delay-aware, so under command delay it is no longer an upper bound
    (deadbeat commands go unstable; effort mode therefore uses a moderate
    proportional tracking gain instead)."""

    name = "oracle"

    def __init__(
        self,
        target_W,
        dt,
        effort_scale: float | None = None,
        v_max: float = 2.0,
        k_track: float = 4.0,
    ):
        self.target_W = target_W
        self.dt = dt
        self.effort_scale = effort_scale
        self.v_max = v_max
        self.k_track = k_track

    def act(self, obs):
        roll, pitch = obs["platform_attitude"]
        q_des = np.array(aim_at(self.target_W, roll, pitch))
        err = q_des - obs["joint_angle"]
        if self.effort_scale is not None:
            return np.clip(self.k_track * err * self.effort_scale, -1.0, 1.0)
        return np.clip(err / self.dt, -self.v_max, self.v_max)  # deadbeat (kinematic)

    def reset(self):
        pass


def build_controller(
    name: str, config: Config, controllers_yaml: Path = DEFAULT_CONTROLLERS_YAML
):
    pt = config.sim.pan_tilt
    nominal = make_nominal(config.sim, config.mismatch, config.scenario)
    effort_mode = pt.mode == "second_order"
    nominal_scale = nominal.pan_tilt.k_d / nominal.pan_tilt.k_u if effort_mode else None

    if name == "none":
        return ZeroController()
    if name == "oracle":
        true_scale = pt.k_d / pt.k_u if effort_mode else None
        return OracleController(
            np.array(config.scenario.target_W), config.sim.dt,
            effort_scale=true_scale, v_max=pt.v_max,
        )
    if name in ("mpc_nominal", "mpc_residual"):
        if not effort_mode:
            raise ValueError(f"{name} requires the second_order (effort) mechanism")
        spec_path = MPC_NOMINAL_YAML if name == "mpc_nominal" else MPC_RESIDUAL_YAML
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        model_spec = dict(spec.get("model", {}))
        checkpoint = model_spec.pop("checkpoint", None)
        model = NominalModel(nominal, n_substeps=config.sim.n_substeps, **model_spec)
        if name == "mpc_residual":
            from wavept.models.residual import CorrectedModel, ResidualCorrector

            ckpt_path = REPO_ROOT / checkpoint
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"residual checkpoint missing: {ckpt_path} — run scripts/train_residual.py"
                )
            model = CorrectedModel(model, ResidualCorrector(ckpt_path))
        return RandomShootingMPC(
            model=model,
            cfg=MPCConfig(**spec["mpc"]),
            weights=CostWeights(**spec["cost"]),
            dt=config.sim.dt,
            name=name,
        )
    spec = yaml.safe_load(Path(controllers_yaml).read_text(encoding="utf-8"))
    if name == "pgain":
        return PGainController(v_max=pt.v_max, effort_scale=nominal_scale, **spec["pgain"])
    if name == "jacobian":
        return JacobianController(
            nominal, v_max=pt.v_max, effort_scale=nominal_scale, **spec["jacobian"]
        )
    raise ValueError(f"unknown controller '{name}'")
