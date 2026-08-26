"""Pinhole camera: projection, visibility, normalized coordinates, margins."""

from __future__ import annotations

import numpy as np

from wavept.config import CameraParams


class PinholeCamera:
    def __init__(self, params: CameraParams):
        self.params = params

    @property
    def width(self) -> int:
        return self.params.width

    @property
    def height(self) -> int:
        return self.params.height

    def project(self, p_C) -> tuple[np.ndarray, bool]:
        """Project a camera-frame point. Returns (uv, in_front).

        uv is (nan, nan) when the point is behind the camera (Z_C <= z_min);
        otherwise it is the pixel position, which may lie outside the frame.
        """
        p = np.asarray(p_C, dtype=float)
        if p[2] <= self.params.z_min:
            return np.full(2, np.nan), False
        c = self.params
        return np.array([c.f * p[0] / p[2] + c.cx, c.f * p[1] / p[2] + c.cy]), True

    def in_frame(self, uv) -> bool:
        u, v = float(uv[0]), float(uv[1])
        return 0.0 <= u < self.params.width and 0.0 <= v < self.params.height

    def observe(self, p_C) -> tuple[np.ndarray, bool]:
        """(uv, visible): visible means in front of the camera and inside the frame."""
        uv, in_front = self.project(p_C)
        return uv, bool(in_front and self.in_frame(uv))

    def normalize(self, uv) -> np.ndarray:
        """Normalized image coordinates: u_bar = 2(u - W/2)/W, v_bar = 2(v - H/2)/H."""
        c = self.params
        return np.array(
            [2.0 * (uv[0] - c.width / 2.0) / c.width, 2.0 * (uv[1] - c.height / 2.0) / c.height]
        )

    def margin_px(self, uv) -> float:
        """Pixel distance to the nearest image boundary (negative outside)."""
        u, v = float(uv[0]), float(uv[1])
        c = self.params
        return min(u, c.width - u, v, c.height - v)
