import numpy as np
import pytest

from wavept.evaluation.metrics import aggregate, episode_metrics


def make_metrics(**overrides):
    kwargs = dict(
        dt=0.04,
        visible=np.array([True, True, True, False]),
        uv_norm=np.array([[0.0, 0.0], [0.3, 0.4], [0.6, 0.8], [np.nan, np.nan]]),
        margin_px=np.array([240.0, 100.0, 20.0, -np.inf]),
        actions=np.array([[0.5, -0.5], [2.0, 0.0], [1.0, 1.0]]),
        v_max=2.0,
        compute_ms=np.array([1.0, 2.0, 3.0]),
        terminated_early=True,
    )
    kwargs.update(overrides)
    return episode_metrics(**kwargs)


def test_episode_metrics_values():
    m = make_metrics()
    assert m["steps"] == 4
    assert m["retained"] is False
    assert m["visible_time_ratio"] == pytest.approx(0.75)
    assert m["min_margin_px"] == pytest.approx(20.0)  # -inf (lost step) excluded
    # errors over visible steps: 0.0, 0.5, 1.0
    assert m["mean_center_err_norm"] == pytest.approx(0.5)
    assert m["rms_center_err_norm"] == pytest.approx(np.sqrt((0.0 + 0.25 + 1.0) / 3))
    assert m["max_center_err_norm"] == pytest.approx(1.0)
    assert m["action_abs_integral"] == pytest.approx((1.0 + 2.0 + 2.0) * 0.04)
    assert m["n_saturations"] == 1  # only the [2.0, 0.0] step hits v_max
    assert m["mean_compute_ms"] == pytest.approx(2.0)
    assert m["max_compute_ms"] == pytest.approx(3.0)


def test_aggregate():
    a = make_metrics()
    b = make_metrics(
        visible=np.array([True, True, True, True]),
        uv_norm=np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]]),
        margin_px=np.array([240.0, 230.0, 220.0, 210.0]),
        terminated_early=False,
    )
    agg = aggregate([a, b])
    assert agg["n_episodes"] == 2
    assert agg["retention_rate"] == pytest.approx(0.5)
    assert agg["mean_min_margin_px"] == pytest.approx((20.0 + 210.0) / 2)
    assert agg["max_compute_ms"] == pytest.approx(3.0)
