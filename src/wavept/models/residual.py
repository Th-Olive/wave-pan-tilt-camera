"""Learned residual visual model (M6).

The residual corrects the ALIGNED single-step visual map: from the true state
at time tau and the action executed during [tau, tau+1], the nominal model
predicts uv(tau+1); the network learns the error of that increment,

    y = uv_true(tau+1) - uv_nominal_one_step(tau+1 | state tau, a_exec)

in normalized image units. Delays are structural and exactly known, so they
stay in the nominal replay machinery; the residual only sees aligned states.
Inside a rollout (``CorrectedModel``) the per-step residual is evaluated on
the model's own rolled-out state and accumulated — total drift is treated as
the sum of one-step increment errors (first-order compounding assumption).

Known honest limitation: at rollout time the attitude features are the
model's damped extrapolation, while training used true attitudes — residual
quality degrades where extrapolation is poor (documented in the model card).

Feature vector (14): q(2), qdot(2), uv_norm(2), attitude(2),
attitude_rate(2), a_exec(2), a_exec_prev(2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from wavept.config import NominalParams

FEATURE_NAMES = (
    "q_pan", "q_tilt", "qdot_pan", "qdot_tilt", "u_norm", "v_norm",
    "roll", "pitch", "roll_rate", "pitch_rate",
    "a_pan", "a_tilt", "a_pan_prev", "a_tilt_prev",
)
N_FEATURES = len(FEATURE_NAMES)


def make_features(q, qdot, uv_norm, att, rate, a_exec, a_prev) -> np.ndarray:
    """Stack (K,2) blocks (broadcasting (2,) inputs) into (K, 14) float32."""
    K = len(np.atleast_2d(q))
    blocks = [np.broadcast_to(np.atleast_2d(b), (K, 2)) for b in
              (q, qdot, uv_norm, att, rate, a_exec, a_prev)]
    return np.concatenate(blocks, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Vectorized aligned one-step nominal prediction (training-pair construction)
# ---------------------------------------------------------------------------


def _R_WC_batch(att: np.ndarray, q: np.ndarray) -> Rotation:
    n = len(q)
    R_WB = Rotation.from_euler(
        "ZYX", np.column_stack([np.zeros(n), att[:, 1], att[:, 0]])
    )
    # scipy >= 1.18 requires shape (n, 1) for batched single-axis sequences
    return (
        R_WB
        * Rotation.from_euler("z", q[:, [0]])
        * Rotation.from_euler("y", q[:, [1]])
        * Rotation.from_euler("x", np.full((n, 1), np.pi))
    )


def one_step_nominal_batch(
    q, qdot, uv_norm, att, att_next, a_exec, nominal: NominalParams, n_substeps: int = 4
):
    """Nominal single-step prediction from exact states, vectorized over rows.

    Returns (uv_norm_next (n,2), valid (n,)). Mirrors NominalModel exactly:
    flat-scene reconstruction at the assumed depth, second-order joint step,
    reprojection at att_next."""
    cam, pt = nominal.camera, nominal.pan_tilt
    q = np.asarray(q, float)
    n = len(q)

    u = (np.asarray(uv_norm, float)[:, 0] + 1.0) * cam.width / 2.0
    v = (np.asarray(uv_norm, float)[:, 1] + 1.0) * cam.height / 2.0
    ray_C = np.column_stack([(u - cam.cx) / cam.f, (v - cam.cy) / cam.f, np.ones(n)])
    R0 = _R_WC_batch(np.asarray(att, float), q)
    ray_W = R0.apply(ray_C)
    valid = ray_W[:, 2] < -1e-6
    s = -nominal.assumed_depth / np.where(valid, ray_W[:, 2], -1.0)
    target_W = ray_W * s[:, None]

    a = np.clip(np.asarray(a_exec, float), -1.0, 1.0)
    if pt.dead_zone > 0.0:
        a = np.where(np.abs(a) < pt.dead_zone, 0.0, a)
    qn = q.copy()
    qdn = np.asarray(qdot, float).copy()
    lo = np.array([pt.pan_limits[0], pt.tilt_limits[0]])
    hi = np.array([pt.pan_limits[1], pt.tilt_limits[1]])
    dt_sub = nominal.dt / n_substeps
    for _ in range(n_substeps):
        qdd = pt.k_u * a - pt.k_d * qdn
        qdn = np.clip(qdn + qdd * dt_sub, -pt.qdot_max, pt.qdot_max)
        qn_new = np.clip(qn + qdn * dt_sub, lo, hi)
        qdn = np.where(qn_new != qn + qdn * dt_sub, 0.0, qdn)
        qn = qn_new

    p_C = _R_WC_batch(np.asarray(att_next, float), qn).inv().apply(target_W)
    valid &= p_C[:, 2] > cam.z_min
    z = np.where(valid, p_C[:, 2], 1.0)
    u1 = cam.f * p_C[:, 0] / z + cam.cx
    v1 = cam.f * p_C[:, 1] / z + cam.cy
    uv_next = np.column_stack(
        [2.0 * (u1 - cam.width / 2.0) / cam.width, 2.0 * (v1 - cam.height / 2.0) / cam.height]
    )
    return uv_next, valid


# ---------------------------------------------------------------------------
# Training pairs from an M5 dataset split
# ---------------------------------------------------------------------------


def build_pairs(
    data: dict,
    index: pd.DataFrame,
    split: str,
    nominals: dict[str, NominalParams],
    d_cmd: int = 2,
    d_obs: int = 1,
    dt: float = 0.04,
    n_substeps: int = 4,
):
    """Aligned (X, Y) pairs for one split. The measurement series is stale by
    d_obs, so the state at time tau is read from row tau + d_obs; truth series
    come from the stored per-row ground truth."""
    X_all, Y_all, cond_all = [], [], []
    for _, ep in index[index.split == split].iterrows():
        idx = np.flatnonzero(data["episode_id"] == ep.episode_id)
        T = len(idx)
        if T < d_obs + 4:
            continue
        uv_true = data["true_uv_norm_t"][idx]
        vis = data["true_visible_t"][idx]
        label = data["true_uv_norm_t1"][idx]
        label_vis = data["true_visible_t1"][idx]
        actions = data["action"][idx]
        # aligned state series: measurement at tau lives in row tau + d_obs
        q_s = data["q"][idx][d_obs:]
        qdot_s = data["qdot"][idx][d_obs:]
        att_s = data["attitude"][idx][d_obs:]
        n_tau = T - d_obs  # states available for tau = 0 .. n_tau-1

        taus = np.arange(1, min(n_tau, T - 1))  # need rate (tau>=1) and label at tau
        if not len(taus):
            continue
        q_t, qdot_t, att_t = q_s[taus], qdot_s[taus], att_s[taus]
        rate_t = (att_s[taus] - att_s[taus - 1]) / dt
        a_exec = np.array([actions[t - d_cmd] if t >= d_cmd else np.zeros(2) for t in taus])
        a_prev = np.array(
            [actions[t - d_cmd - 1] if t >= d_cmd + 1 else np.zeros(2) for t in taus]
        )
        pred, valid = one_step_nominal_batch(
            q_t, qdot_t, uv_true[taus], att_t, att_t + rate_t * dt, a_exec,
            nominals[ep.condition], n_substeps,
        )
        mask = valid & vis[taus] & label_vis[taus]
        if not mask.any():
            continue
        X_all.append(
            make_features(q_t, qdot_t, uv_true[taus], att_t, rate_t, a_exec, a_prev)[mask]
        )
        Y_all.append((label[taus] - pred).astype(np.float32)[mask])
        cond_all.extend([ep.condition] * int(mask.sum()))
    X = np.concatenate(X_all) if X_all else np.zeros((0, N_FEATURES), np.float32)
    Y = np.concatenate(Y_all) if Y_all else np.zeros((0, 2), np.float32)
    return X, Y, np.array(cond_all)


# ---------------------------------------------------------------------------
# Torch model, corrector, corrected rollout model
# ---------------------------------------------------------------------------


def build_mlp(in_dim: int = N_FEATURES, hidden: int = 64, n_hidden: int = 2):
    import torch.nn as nn

    layers, d = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, 2))
    return nn.Sequential(*layers)


def save_checkpoint(path, net, x_mean, x_std, y_mean, y_std, y_clip, meta: dict) -> None:
    import torch

    torch.save(
        {
            "state_dict": net.state_dict(),
            "x_mean": x_mean, "x_std": x_std,
            "y_mean": y_mean, "y_std": y_std, "y_clip": y_clip,
            "hidden": meta.get("hidden", 64), "n_hidden": meta.get("n_hidden", 2),
            "feature_names": FEATURE_NAMES,
            "meta": meta,
        },
        path,
    )


class ResidualCorrector:
    """Loads a checkpoint; callable on stacked feature blocks, returns the
    per-step residual (normalized image units), clipped to the training-label
    range (guard against implausible extrapolation)."""

    def __init__(self, checkpoint_path):
        import torch

        self.torch = torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.net = build_mlp(N_FEATURES, ckpt["hidden"], ckpt["n_hidden"])
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.x_mean, self.x_std = ckpt["x_mean"], ckpt["x_std"]
        self.y_mean, self.y_std, self.y_clip = ckpt["y_mean"], ckpt["y_std"], ckpt["y_clip"]
        self.meta = ckpt.get("meta", {})

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = (features - self.x_mean) / self.x_std
        with self.torch.no_grad():
            y = self.net(self.torch.from_numpy(x.astype(np.float32))).numpy()
        y = y * self.y_std + self.y_mean
        return np.clip(y, -self.y_clip, self.y_clip)

    def __call__(self, q, qdot, uv_norm, att, rate, a_exec, a_prev) -> np.ndarray:
        return self.predict(make_features(q, qdot, uv_norm, att, rate, a_exec, a_prev))


class CorrectedModel:
    """Nominal rollout + accumulated learned per-step residual (Model protocol)."""

    def __init__(self, nominal_model, corrector: ResidualCorrector):
        self.nominal_model = nominal_model
        self.corrector = corrector

    @property
    def n_queued(self) -> int:
        return self.nominal_model.n_queued

    def rollout(self, state, candidates):
        return self.nominal_model.rollout(state, candidates, correction=self.corrector)
