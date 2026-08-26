"""Per-episode and aggregate metrics.

Reporting rule: a controller is not better solely because of lower mean error
if it loses the target more often — retention comes first.
"""

from __future__ import annotations

import numpy as np


def episode_metrics(
    *,
    dt: float,
    visible: np.ndarray,        # (T,) bool
    uv_norm: np.ndarray,        # (T,2), nan when lost
    margin_px: np.ndarray,      # (T,), -inf when lost
    actions: np.ndarray,        # (T,2) commanded actions
    v_max: float,
    compute_ms: np.ndarray,     # (T,)
    terminated_early: bool,
) -> dict:
    visible = np.asarray(visible, dtype=bool)
    uv_norm = np.asarray(uv_norm, dtype=float)
    actions = np.asarray(actions, dtype=float)
    err = np.linalg.norm(uv_norm[visible], axis=1) if visible.any() else np.array([])
    margins = np.asarray(margin_px, dtype=float)[visible]

    sat = np.any(np.abs(actions) >= v_max * (1.0 - 1e-9), axis=1)
    return {
        "steps": int(len(visible)),
        "retained": bool(not terminated_early),
        "visible_time_ratio": float(visible.mean()) if len(visible) else 0.0,
        "min_margin_px": float(margins.min()) if len(margins) else float("-inf"),
        "mean_center_err_norm": float(err.mean()) if len(err) else None,
        "rms_center_err_norm": float(np.sqrt(np.mean(err**2))) if len(err) else None,
        "max_center_err_norm": float(err.max()) if len(err) else None,
        "action_abs_integral": float(np.sum(np.abs(actions)) * dt),
        "action_delta_integral": float(np.sum(np.abs(np.diff(actions, axis=0)))),
        "n_saturations": int(sat.sum()),
        "mean_compute_ms": float(np.mean(compute_ms)) if len(compute_ms) else 0.0,
        "max_compute_ms": float(np.max(compute_ms)) if len(compute_ms) else 0.0,
    }


def aggregate(results: list[dict]) -> dict:
    """Aggregate episode metric dicts (None-safe means)."""

    def mean_of(key):
        vals = [
            r[key]
            for r in results
            if r.get(key) is not None and np.isfinite(r[key])
        ]
        return float(np.mean(vals)) if vals else None

    return {
        "n_episodes": len(results),
        "retention_rate": float(np.mean([r["retained"] for r in results])),
        "mean_visible_time_ratio": mean_of("visible_time_ratio"),
        "mean_min_margin_px": mean_of("min_margin_px"),
        "mean_center_err_norm": mean_of("mean_center_err_norm"),
        "mean_action_abs_integral": mean_of("action_abs_integral"),
        "mean_n_saturations": mean_of("n_saturations"),
        "mean_compute_ms": mean_of("mean_compute_ms"),
        "max_compute_ms": max(r["max_compute_ms"] for r in results) if results else 0.0,
    }
