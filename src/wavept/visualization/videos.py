"""Video export helpers (mp4 via imageio-ffmpeg, optional gif)."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def save_video(frames, path: str | Path, fps: int = 25) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    return path


def side_by_side(*frame_sequences, pad: int = 4) -> list[np.ndarray]:
    """Horizontally stack synchronized frame sequences (e.g. two controllers on
    the same seed). Shorter sequences are held on their last frame."""
    n = max(len(seq) for seq in frame_sequences)
    divider = None
    out = []
    for i in range(n):
        parts = []
        for k, seq in enumerate(frame_sequences):
            frame = np.asarray(seq[min(i, len(seq) - 1)])
            if k > 0:
                if divider is None:
                    divider = np.full((frame.shape[0], pad, 3), 255, dtype=np.uint8)
                parts.append(divider)
            parts.append(frame)
        out.append(np.hstack(parts))
    return out


def save_gif(frames, path: str | Path, fps: int = 12, stride: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), [np.asarray(f) for f in frames[::stride]], fps=fps, loop=0)
    return path
