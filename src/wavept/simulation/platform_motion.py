"""Wave-like platform disturbance: sum-of-sines roll and pitch, plus optional
half-sine transient impulses (M3 hard tier)."""

from __future__ import annotations

import numpy as np

from wavept.config import Impulse, WaveComponent


class WaveDisturbance:
    """Deterministic attitude(t) once constructed; unset phases drawn from rng."""

    def __init__(
        self,
        roll_waves,
        pitch_waves,
        rng: np.random.Generator,
        impulses: tuple[Impulse, ...] = (),
    ):
        self._roll = self._resolve(roll_waves, rng)
        self._pitch = self._resolve(pitch_waves, rng)
        self._impulses = tuple(impulses)

    @staticmethod
    def _resolve(waves: tuple[WaveComponent, ...], rng: np.random.Generator):
        resolved = []
        for w in waves:
            phase = rng.uniform(0.0, 2.0 * np.pi) if w.phase is None else w.phase
            resolved.append((w.amplitude, 2.0 * np.pi * w.freq_hz, phase))
        return resolved

    @staticmethod
    def _eval(components, t: float) -> float:
        return float(sum(a * np.sin(omega * t + phi) for a, omega, phi in components))

    def _impulse(self, axis: str, t: float) -> float:
        total = 0.0
        for imp in self._impulses:
            if imp.axis == axis and imp.t0 <= t <= imp.t0 + imp.duration:
                total += imp.amplitude * np.sin(np.pi * (t - imp.t0) / imp.duration)
        return total

    def attitude(self, t: float) -> np.ndarray:
        """(roll, pitch) in radians at time t."""
        return np.array(
            [
                self._eval(self._roll, t) + self._impulse("roll", t),
                self._eval(self._pitch, t) + self._impulse("pitch", t),
            ]
        )
