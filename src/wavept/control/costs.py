"""Trajectory cost for predictive control.

J = sum_k [ w_c ||s_k||^2 + w_b * boundary(s_k) + w_a ||a_k||^2
            + w_s ||a_k - a_{k-1}||^2 ]           (+ hard out-of-frame penalty)

with s in normalized image coordinates. The boundary term is a squared
softplus of the excess beyond the safe region, smooth so that approaching the
warning boundary is penalized before it is crossed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


@dataclass(frozen=True)
class CostWeights:
    w_center: float = 1.0
    w_boundary: float = 20.0
    tau_boundary: float = 0.05
    safe_u: float = 0.8       # |u_bar| beyond this is inside the warning band
    safe_v: float = 0.733
    w_effort: float = 0.02
    w_smooth: float = 0.05
    oob_penalty: float = 1.0e4


def trajectory_cost(
    uv_norm: np.ndarray,      # (K, N, 2) predicted normalized target positions
    in_front: np.ndarray,     # (K, N)
    actions: np.ndarray,      # (K, N, 2) candidate action sequences
    prev_action: np.ndarray,  # (2,) last applied action (for smoothness at k=0)
    w: CostWeights,
) -> np.ndarray:
    center = np.sum(uv_norm**2, axis=(1, 2))

    safe = np.array([w.safe_u, w.safe_v])
    excess = (np.abs(uv_norm) - safe) / w.tau_boundary
    boundary = np.sum(softplus(excess) ** 2, axis=(1, 2))

    oob = (np.abs(uv_norm[..., 0]) >= 1.0) | (np.abs(uv_norm[..., 1]) >= 1.0) | ~in_front
    oob_cost = w.oob_penalty * np.sum(oob, axis=1)

    effort = np.sum(actions**2, axis=(1, 2))
    deltas = np.diff(
        np.concatenate([np.broadcast_to(prev_action, (len(actions), 1, 2)), actions], axis=1),
        axis=1,
    )
    smooth = np.sum(deltas**2, axis=(1, 2))

    return (
        w.w_center * center
        + w.w_boundary * boundary
        + oob_cost
        + w.w_effort * effort
        + w.w_smooth * smooth
    )
