"""Transition dataset storage.

A dataset directory contains:
  manifest.json   git state, condition configs, split plan, counts
  index.csv       one row per episode (episode_id, split, condition, policy,
                  seed, n_steps) — csv rather than parquet to avoid a pyarrow
                  dependency for a ~100-row index
  <split>.npz     stacked per-transition arrays for that split

Row fields (transition i of an episode; delays are FIXED across the dataset
at n_cmd_delay=2, n_obs_delay=1 so the action history length L=3 is constant):
  t                  time at act
  uv_obs_px (2)      served (stale) target pixels
  uv_norm_obs (2)    served normalized target position
  obs_visible        served visibility flag
  q, qdot (2 each)   served joint state
  attitude (2)       served roll, pitch
  attitude_rate (2)  finite difference of served attitude (0 at episode start)
  action (2)         action issued at step i
  action_history (6) flattened (L,2): actions issued at i-L .. i-1 (0-padded)
  true_uv_norm_t (2), true_visible_t     truth at act time
  true_uv_norm_t1 (2), true_visible_t1   truth after the step (label)
  pred_uv_norm_t1 (2)  nominal one-step prediction (rate_decay=1.0), nan when
                       the observation was invalid
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wavept.models.nominal import ModelState, NominalModel

ACTION_HISTORY_LEN = 3  # n_obs_delay (1) + n_cmd_delay (2), fixed for the dataset

VEC2_FIELDS = [
    "uv_obs_px",
    "uv_norm_obs",
    "q",
    "qdot",
    "attitude",
    "attitude_rate",
    "action",
    "true_uv_norm_t",
    "true_uv_norm_t1",
    "pred_uv_norm_t1",
]
SCALAR_FIELDS = ["t", "obs_visible", "true_visible_t", "true_visible_t1"]


def episode_to_arrays(transitions: list[dict], dt: float, L: int = ACTION_HISTORY_LEN) -> dict:
    """Convert run_episode transitions into the per-row array dict."""
    T = len(transitions)
    a = {f: np.zeros((T, 2), dtype=np.float32) for f in VEC2_FIELDS}
    a["action_history"] = np.zeros((T, L * 2), dtype=np.float32)
    a["t"] = np.zeros(T, dtype=np.float32)
    for f in ["obs_visible", "true_visible_t", "true_visible_t1"]:
        a[f] = np.zeros(T, dtype=bool)

    actions = np.array([tr["action"] for tr in transitions], dtype=np.float32)
    for i, tr in enumerate(transitions):
        obs, info_prev, info = tr["obs"], tr["prev_info"], tr["info"]
        a["t"][i] = obs["timestamp"]
        a["uv_obs_px"][i] = obs["target_uv"]
        a["uv_norm_obs"][i] = obs["target_uv_norm"]
        a["obs_visible"][i] = obs["target_visible"]
        a["q"][i] = obs["joint_angle"]
        a["qdot"][i] = obs["joint_velocity"]
        a["attitude"][i] = obs["platform_attitude"]
        if i > 0:
            prev_att = transitions[i - 1]["obs"]["platform_attitude"]
            a["attitude_rate"][i] = (obs["platform_attitude"] - prev_att) / dt
        a["action"][i] = tr["action"]
        for j in range(L):
            k = i - L + j
            if k >= 0:
                a["action_history"][i, 2 * j : 2 * j + 2] = actions[k]
        a["true_uv_norm_t"][i] = info_prev["true_uv_norm"]
        a["true_visible_t"][i] = info_prev["true_visible"]
        a["true_uv_norm_t1"][i] = info["true_uv_norm"]
        a["true_visible_t1"][i] = info["true_visible"]
    a["pred_uv_norm_t1"][:] = np.nan
    return a


def add_nominal_predictions(arrays: dict, model: NominalModel, L: int = ACTION_HISTORY_LEN) -> None:
    """Fill pred_uv_norm_t1 with the nominal one-step prediction per row.

    With delays, the one-step prediction depends only on the action history
    (the action issued now first takes effect n_cmd_delay steps later), so the
    candidate sequence is a dummy."""
    T = len(arrays["t"])
    dummy = np.zeros((1, 1, 2))
    for i in range(T):
        if not arrays["obs_visible"][i]:
            continue
        state = ModelState(
            q=arrays["q"][i].astype(float),
            qdot=arrays["qdot"][i].astype(float),
            uv=arrays["uv_obs_px"][i].astype(float),
            attitude=arrays["attitude"][i].astype(float),
            attitude_rate=arrays["attitude_rate"][i].astype(float),
            queued_actions=arrays["action_history"][i].reshape(L, 2).astype(float),
        )
        uv_pred, in_front = model.rollout(state, dummy)
        if in_front[0, 0]:
            arrays["pred_uv_norm_t1"][i] = uv_pred[0, 0]


def write_dataset(
    out_dir: Path,
    split_episodes: dict[str, list[tuple[dict, dict]]],  # split -> [(meta, arrays)]
    manifest: dict,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for split, episodes in split_episodes.items():
        stacked: dict[str, list] = {}
        episode_ids = []
        for meta, arrays in episodes:
            n = len(arrays["t"])
            episode_ids.append(np.full(n, meta["episode_id"], dtype=np.int32))
            for k, v in arrays.items():
                stacked.setdefault(k, []).append(v)
            index_rows.append({**meta, "split": split, "n_steps": n})
        if episodes:
            data = {k: np.concatenate(v) for k, v in stacked.items()}
            data["episode_id"] = np.concatenate(episode_ids)
            np.savez_compressed(out_dir / f"{split}.npz", **data)
    pd.DataFrame(index_rows).to_csv(out_dir / "index.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def load_split(dataset_dir: Path, split: str) -> dict:
    with np.load(Path(dataset_dir) / f"{split}.npz") as z:
        return {k: z[k] for k in z.files}


def load_index(dataset_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(dataset_dir) / "index.csv")


def load_manifest(dataset_dir: Path) -> dict:
    return json.loads((Path(dataset_dir) / "manifest.json").read_text(encoding="utf-8"))
