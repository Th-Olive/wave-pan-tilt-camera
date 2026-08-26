"""Audit the nominal model against a collected dataset (M5): one-step and
multi-step prediction errors per split/condition, error structure plots.

Example:
  python scripts/audit_nominal_model.py --dataset outputs/datasets/wavept_v1
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wavept.config import config_from_dict, make_nominal
from wavept.manifest import create_run_dir, git_info
from wavept.models.datasets import ACTION_HISTORY_LEN, load_index, load_manifest, load_split
from wavept.models.nominal import ModelState, NominalModel

REPO_ROOT = Path(__file__).resolve().parents[1]
PX = np.array([320.0, 240.0])  # normalized -> pixel scale (W/2, H/2)
HORIZON = 10


def one_step_residuals_px(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Residual norm (px) per valid row + validity mask."""
    valid = (
        data["obs_visible"]
        & data["true_visible_t1"]
        & np.isfinite(data["pred_uv_norm_t1"]).all(axis=1)
    )
    err = (data["true_uv_norm_t1"] - data["pred_uv_norm_t1"]) * PX
    return np.linalg.norm(err, axis=1), valid


def multi_step_errors(data: dict, models: dict, index: pd.DataFrame, split: str,
                      stride: int = 25) -> np.ndarray:
    """Mean |error| (px) per horizon step, over rollout starts every `stride`
    steps of each episode, using the actually-executed action log."""
    errors = [[] for _ in range(HORIZON)]
    for _, ep in index[index.split == split].iterrows():
        mask = data["episode_id"] == ep.episode_id
        idx = np.flatnonzero(mask)
        model = models[ep.condition]
        for s in range(0, len(idx) - HORIZON - 1, stride):
            i = idx[s]
            if not data["obs_visible"][i]:
                continue
            state = ModelState(
                q=data["q"][i].astype(float),
                qdot=data["qdot"][i].astype(float),
                uv=data["uv_obs_px"][i].astype(float),
                attitude=data["attitude"][i].astype(float),
                attitude_rate=data["attitude_rate"][i].astype(float),
                queued_actions=data["action_history"][i].reshape(ACTION_HISTORY_LEN, 2).astype(float),
            )
            candidates = data["action"][idx[s] : idx[s] + HORIZON].astype(float)[None]
            uv_pred, _ = model.rollout(state, candidates)
            for k in range(HORIZON):
                j = idx[s] + k
                if data["true_visible_t1"][j]:
                    e = (data["true_uv_norm_t1"][j] - uv_pred[0, k]) * PX
                    errors[k].append(np.linalg.norm(e))
    return np.array([np.mean(e) if e else np.nan for e in errors])


def binned_curve(x, y, bins):
    which = np.digitize(x, bins)
    return np.array([np.mean(y[which == b]) if np.any(which == b) else np.nan
                     for b in range(1, len(bins))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=REPO_ROOT / "outputs" / "datasets" / "wavept_v1")
    args = parser.parse_args()

    manifest = load_manifest(args.dataset)
    index = load_index(args.dataset)
    models = {
        name: NominalModel(
            make_nominal(cfg.sim, cfg.mismatch, cfg.scenario),
            n_substeps=cfg.sim.n_substeps, rate_decay=1.0,
        )
        for name, cfg in ((n, config_from_dict(c)) for n, c in manifest["conditions"].items())
    }
    splits = sorted(index.split.unique())
    run_dir = create_run_dir("nominal_audit")

    # ---- one-step stats per split and condition
    rows, split_data = [], {}
    for split in splits:
        data = load_split(args.dataset, split)
        split_data[split] = data
        res, valid = one_step_residuals_px(data)
        conds = pd.Series(index.set_index("episode_id").condition.loc[data["episode_id"]].values)
        for cond in sorted(conds.unique()):
            m = valid & (conds == cond).values
            rows.append({
                "split": split, "condition": cond, "n": int(m.sum()),
                "rmse_px": float(np.sqrt(np.mean(res[m] ** 2))),
                "p95_px": float(np.percentile(res[m], 95)),
            })
        m = valid
        rows.append({"split": split, "condition": "ALL", "n": int(m.sum()),
                     "rmse_px": float(np.sqrt(np.mean(res[m] ** 2))),
                     "p95_px": float(np.percentile(res[m], 95))})
    table = pd.DataFrame(rows)
    print("one-step nominal prediction error (px):")
    print(table.to_string(index=False))

    # ---- error structure: |residual| vs attitude rate and action magnitude
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ["train", "test_ood"]:
        data = split_data[split]
        res, valid = one_step_residuals_px(data)
        rate = np.linalg.norm(data["attitude_rate"], axis=1)[valid]
        act = np.linalg.norm(data["action"], axis=1)[valid]
        r = res[valid]
        bins_rate = np.linspace(0, np.percentile(rate, 99), 12)
        bins_act = np.linspace(0, 1.4, 12)
        axes[0].plot(bins_rate[:-1], binned_curve(rate, r, bins_rate), marker="o", label=split)
        axes[1].plot(bins_act[:-1], binned_curve(act, r, bins_act), marker="o", label=split)
    axes[0].set_xlabel("|attitude rate| (rad/s)"); axes[0].set_ylabel("mean |residual| (px)")
    axes[1].set_xlabel("|action|")
    for ax in axes: ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Nominal one-step error structure")
    fig.tight_layout()
    fig.savefig(run_dir / "error_structure.png", dpi=120)

    # ---- multi-step drift per split
    fig2, ax = plt.subplots(figsize=(6, 4))
    drift = {}
    for split in splits:
        curve = multi_step_errors(split_data[split], models, index, split)
        drift[split] = curve.tolist()
        ax.plot(np.arange(1, HORIZON + 1), curve, marker="o", label=split)
    ax.set_xlabel("horizon step (40 ms each)"); ax.set_ylabel("mean |error| (px)")
    ax.set_title("Nominal rollout drift (executed-action replay)")
    ax.legend(); ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(run_dir / "rollout_drift.png", dpi=120)

    table.to_csv(run_dir / "one_step_errors.csv", index=False)
    (run_dir / "manifest.json").write_text(json.dumps({
        "git": git_info(), "dataset": str(args.dataset),
        "rollout_drift_px": drift,
    }, indent=2), encoding="utf-8")
    print(f"\noutputs: {run_dir}")


if __name__ == "__main__":
    main()
