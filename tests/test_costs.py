import numpy as np
import pytest

from wavept.control.costs import CostWeights, trajectory_cost

W = CostWeights()


def cost_at(uv, actions=None, in_front=None, prev=None):
    uv = np.asarray(uv, dtype=float)[None]        # (1, N, 2)
    n = uv.shape[1]
    actions = np.zeros((1, n, 2)) if actions is None else np.asarray(actions)[None]
    in_front = np.ones((1, n), dtype=bool) if in_front is None else np.asarray(in_front)[None]
    prev = np.zeros(2) if prev is None else np.asarray(prev)
    return float(trajectory_cost(uv, in_front, actions, prev, W)[0])


def test_cost_increases_toward_boundary():
    costs = [cost_at([[u, 0.0]]) for u in [0.0, 0.4, 0.7, 0.85, 0.95]]
    assert all(b > a for a, b in zip(costs, costs[1:]))


def test_boundary_band_dominates_centering():
    inside = cost_at([[0.5, 0.0]])
    at_band = cost_at([[0.9, 0.0]])
    assert at_band > inside + W.w_boundary  # boundary term clearly active


def test_out_of_frame_is_catastrophic():
    near = cost_at([[0.95, 0.0]])
    out = cost_at([[1.05, 0.0]])
    behind = cost_at([[0.0, 0.0]], in_front=[False])
    assert out > near + 0.5 * W.oob_penalty
    assert behind > 0.5 * W.oob_penalty


def test_effort_and_smoothness():
    still = cost_at([[0.0, 0.0]] * 4)
    effortful = cost_at([[0.0, 0.0]] * 4, actions=[[1.0, 1.0]] * 4)
    assert effortful == pytest.approx(still + W.w_effort * 8 + W.w_smooth * 2)
    jerky = cost_at(
        [[0.0, 0.0]] * 4, actions=[[1, 0], [-1, 0], [1, 0], [-1, 0]]
    )
    smooth_same_effort = cost_at([[0.0, 0.0]] * 4, actions=[[1, 0]] * 4)
    assert jerky > smooth_same_effort
