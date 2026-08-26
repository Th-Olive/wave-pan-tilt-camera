import numpy as np
import pytest

from wavept.config import CameraParams
from wavept.geometry.camera import PinholeCamera

CAM = PinholeCamera(CameraParams())


def test_point_on_axis_projects_to_center():
    uv, in_front = CAM.project([0.0, 0.0, 2.0])
    assert in_front
    np.testing.assert_allclose(uv, [320.0, 240.0], atol=1e-12)


def test_point_behind_camera_invisible():
    uv, in_front = CAM.project([0.0, 0.0, -1.0])
    assert not in_front
    assert np.all(np.isnan(uv))
    _, visible = CAM.observe([0.0, 0.0, -1.0])
    assert not visible


def test_point_outside_frame_not_visible():
    # X/Z = 1 -> u = 500 + 320 = 820 > width
    uv, visible = CAM.observe([2.0, 0.0, 2.0])
    assert not visible
    assert uv[0] > CAM.width


def test_projection_scale():
    uv, _ = CAM.project([1.0, 0.5, 5.0])
    np.testing.assert_allclose(uv, [320.0 + 100.0, 240.0 + 50.0], atol=1e-9)


def test_normalize():
    np.testing.assert_allclose(CAM.normalize([320.0, 240.0]), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(CAM.normalize([640.0, 480.0]), [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(CAM.normalize([0.0, 0.0]), [-1.0, -1.0], atol=1e-12)


def test_margin_px():
    assert CAM.margin_px([320.0, 240.0]) == pytest.approx(240.0)
    assert CAM.margin_px([10.0, 240.0]) == pytest.approx(10.0)
    assert CAM.margin_px([630.0, 240.0]) == pytest.approx(10.0)
    assert CAM.margin_px([-5.0, 240.0]) == pytest.approx(-5.0)  # negative outside
