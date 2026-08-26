"""Geometry checklist tests.

These tests LOCK the frame conventions and image-motion signs. Do not change
expected signs without updating frames.py and the README together.
"""

import math

import numpy as np
import pytest

from wavept.config import CameraParams
from wavept.geometry.camera import PinholeCamera
from wavept.geometry.frames import aim_at, camera_rotation, world_to_camera

TARGET = np.array([1.0, 0.5, -3.0])
CAM = PinholeCamera(CameraParams())


def project(roll, pitch, q_p, q_t, target=TARGET):
    R_WC = camera_rotation(roll, pitch, q_p, q_t)
    p_C = world_to_camera(target, R_WC, np.zeros(3))
    return CAM.project(p_C)


def test_zero_pose_looks_straight_down():
    R_WC = camera_rotation(0.0, 0.0, 0.0, 0.0)
    optical_axis_W = R_WC.apply([0.0, 0.0, 1.0])
    np.testing.assert_allclose(optical_axis_W, [0.0, 0.0, -1.0], atol=1e-12)
    # image u axis (camera x) aligns with world +x
    np.testing.assert_allclose(R_WC.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)


def test_aim_at_centers_target():
    q_p, q_t = aim_at(TARGET)
    uv, in_front = project(0.0, 0.0, q_p, q_t)
    assert in_front
    np.testing.assert_allclose(uv, [320.0, 240.0], atol=0.5)
    # lock the numeric convention for the default target
    assert math.degrees(q_p) == pytest.approx(-153.43, abs=0.05)
    assert math.degrees(q_t) == pytest.approx(20.44, abs=0.05)  # off-nadir angle


def test_aim_at_compensates_attitude():
    roll, pitch = math.radians(5.0), math.radians(3.0)
    q_p, q_t = aim_at(TARGET, roll, pitch)
    uv, _ = project(roll, pitch, q_p, q_t)
    np.testing.assert_allclose(uv, [320.0, 240.0], atol=0.5)


def test_pan_tilt_image_motion_signs():
    """At the aim configuration (downward camera): +tilt moves the target
    toward +u (radially from nadir); +pan moves it toward -v (tangentially,
    gain ~ f*sin(q_t))."""
    q_p, q_t = aim_at(TARGET)
    uv0, _ = project(0.0, 0.0, q_p, q_t)
    d = 0.05

    uv_tilt, _ = project(0.0, 0.0, q_p, q_t + d)
    du, dv = uv_tilt - uv0
    assert du > 10.0  # ~ f*d = 25 px
    assert abs(dv) < abs(du) * 0.1

    uv_pan, _ = project(0.0, 0.0, q_p + d, q_t)
    du, dv = uv_pan - uv0
    assert dv < -5.0  # ~ -f*sin(q_t)*d = -8.7 px
    assert abs(du) < abs(dv) * 0.1


def test_roll_pitch_move_target():
    q_p, q_t = aim_at(TARGET)
    uv0, _ = project(0.0, 0.0, q_p, q_t)
    for att in [(0.05, 0.0), (0.0, 0.05)]:
        uv, _ = project(att[0], att[1], q_p, q_t)
        assert np.linalg.norm(uv - uv0) > 5.0  # disturbance visibly moves the target


def test_nadir_pan_degeneracy():
    """A target exactly at nadir cannot be moved by pan (documented singularity)."""
    nadir = np.array([0.0, 0.0, -3.0])
    q_p, q_t = aim_at(nadir)
    assert q_t == pytest.approx(0.0, abs=1e-12)
    uv0, _ = project(0.0, 0.0, q_p, q_t, target=nadir)
    uv1, _ = project(0.0, 0.0, q_p + 0.3, q_t, target=nadir)
    np.testing.assert_allclose(uv1, uv0, atol=1e-6)


def test_aim_at_rejects_target_above_horizon():
    with pytest.raises(ValueError):
        aim_at([1.0, 0.0, 1.0])
