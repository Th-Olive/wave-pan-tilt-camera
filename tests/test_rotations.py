import math

import numpy as np
import pytest

from wavept.geometry.rotations import Rx, Ry, Rz, rpy_to_rotation, wrap_to_pi


def _mx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _my(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _mz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_rz_90_maps_x_to_y():
    np.testing.assert_allclose(Rz(math.pi / 2).apply([1, 0, 0]), [0, 1, 0], atol=1e-12)


def test_elementary_rotations_match_hand_rolled():
    a = 0.37
    np.testing.assert_allclose(Rx(a).as_matrix(), _mx(a), atol=1e-12)
    np.testing.assert_allclose(Ry(a).as_matrix(), _my(a), atol=1e-12)
    np.testing.assert_allclose(Rz(a).as_matrix(), _mz(a), atol=1e-12)


def test_rpy_is_rz_ry_rx_product():
    roll, pitch, yaw = 0.1, -0.2, 0.3
    expected = _mz(yaw) @ _my(pitch) @ _mx(roll)
    np.testing.assert_allclose(
        rpy_to_rotation(roll, pitch, yaw).as_matrix(), expected, atol=1e-12
    )


def test_wrap_to_pi():
    assert wrap_to_pi(0.0) == pytest.approx(0.0)
    assert wrap_to_pi(math.pi) == pytest.approx(math.pi)
    assert wrap_to_pi(-math.pi) == pytest.approx(math.pi)  # (-pi, pi]
    assert wrap_to_pi(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap_to_pi(-3 * math.pi / 2) == pytest.approx(math.pi / 2)
