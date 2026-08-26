"""Image-plane renderer: synthetic camera view with HUD.

Shows image center, warning-margin rectangle, target dot + trail, joint and
attitude HUD, and a red LOST indicator. No scene texture — the image plane is
the instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from wavept.config import CameraParams


@dataclass
class RenderState:
    t: float
    target_uv: np.ndarray          # (2,), may be nan when lost
    target_visible: bool
    trail: list = field(default_factory=list)
    joint_angle: np.ndarray = None  # (q_p, q_t) rad
    attitude: tuple = (0.0, 0.0)    # (roll, pitch) rad
    predicted_uv: np.ndarray = None  # (N, 2) px, MPC predicted target path
    extra_text: str = ""
    title: str = ""          # panel caption (e.g. controller name) for comparisons


class Renderer:
    def __init__(self, cam: CameraParams, dpi: int = 100):
        self.cam = cam
        self.fig = plt.figure(figsize=(cam.width / dpi, cam.height / dpi), dpi=dpi)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0])

    def render(self, s: RenderState) -> np.ndarray:
        cam, ax = self.cam, self.ax
        ax.clear()
        ax.set_xlim(0, cam.width)
        ax.set_ylim(cam.height, 0)  # image convention: v grows downward
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#0b1d2a")  # dark water-ish background

        m = cam.margin_px
        ax.add_patch(
            plt.Rectangle(
                (m, m), cam.width - 2 * m, cam.height - 2 * m,
                fill=False, edgecolor="#e0a800", linestyle="--", linewidth=1.2,
            )
        )
        cx, cy = cam.width / 2.0, cam.height / 2.0
        ax.plot([cx - 12, cx + 12], [cy, cy], color="#8fb8d8", linewidth=1)
        ax.plot([cx, cx], [cy - 12, cy + 12], color="#8fb8d8", linewidth=1)

        if len(s.trail) >= 2:
            trail = np.asarray(s.trail)
            ax.plot(trail[:, 0], trail[:, 1], color="#4fc3f7", linewidth=1.0, alpha=0.6)
        if s.predicted_uv is not None and len(s.predicted_uv) >= 2:
            pred = np.asarray(s.predicted_uv)
            ax.plot(
                pred[:, 0], pred[:, 1], color="#a5d6a7", linewidth=1.2,
                linestyle=":", marker=".", markersize=3, alpha=0.9,
            )
        if s.target_visible and np.all(np.isfinite(s.target_uv)):
            ax.plot(s.target_uv[0], s.target_uv[1], "o", color="#ff5252", markersize=9)

        roll, pitch = s.attitude
        q = s.joint_angle if s.joint_angle is not None else (float("nan"), float("nan"))
        hud = (
            f"t={s.t:6.2f}s   pan={np.degrees(q[0]):7.2f}\N{DEGREE SIGN}  "
            f"tilt={np.degrees(q[1]):6.2f}\N{DEGREE SIGN}   "
            f"roll={np.degrees(roll):5.2f}\N{DEGREE SIGN}  "
            f"pitch={np.degrees(pitch):5.2f}\N{DEGREE SIGN}"
        )
        ax.text(8, 16, hud, color="white", fontsize=9, family="monospace")
        if s.extra_text:
            ax.text(8, 34, s.extra_text, color="white", fontsize=9, family="monospace")
        if s.title:
            ax.text(
                cx, cam.height - 16, s.title, color="#e8eef4", fontsize=13,
                fontweight="bold", ha="center", family="monospace",
            )

        if not s.target_visible:
            ax.text(
                cx, cy - 40, "TARGET LOST", color="#ff5252", fontsize=18,
                fontweight="bold", ha="center",
            )
            ax.add_patch(
                plt.Rectangle(
                    (2, 2), cam.width - 4, cam.height - 4,
                    fill=False, edgecolor="#ff5252", linewidth=4,
                )
            )

        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        return buf.copy()

    def close(self) -> None:
        plt.close(self.fig)
