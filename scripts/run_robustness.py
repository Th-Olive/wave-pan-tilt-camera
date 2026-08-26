"""M8 robustness study: run the parameter sweeps and produce heatmaps, curves,
and a summary table.

Example:
  python scripts/run_robustness.py                    # full sweep (~25-30 min)
  python scripts/run_robustness.py --report-only DIR  # re-plot from episodes.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wavept.evaluation.benchmark import load_benchmark
from wavept.evaluation.robustness import SWEEP_SEEDS, build_sweep, run_sweep
from wavept.manifest import create_run_dir, git_info

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROLLERS = ["mpc_nominal", "mpc_residual", "pgain"]
# reactive reference only on the axes where its behavior is informative
PGAIN_AXES = ("wave_amplitude", "wave_frequency", "cmd_delay_steps", "obs_dropout")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["controller", "axis", "level", "label"], sort=False)
        .agg(
            retention=("retained", "mean"),
            min_margin_px=("min_margin_px", lambda s: np.mean(s[np.isfinite(s)]) if np.isfinite(s).any() else np.nan),
            mean_err=("mean_center_err_norm", "mean"),
            n=("retained", "size"),
        )
        .reset_index()
    )
    return out


def heatmap_figure(summary: pd.DataFrame, controllers: list[str], path: Path) -> None:
    axes_order = list(dict.fromkeys(summary.axis))
    n_cols = int(summary.groupby("axis")["level"].nunique().max())
    fig, axs = plt.subplots(1, len(controllers), figsize=(5.6 * len(controllers), 7),
                            sharey=True)
    for ax, ctrl in zip(np.atleast_1d(axs), controllers):
        sub = summary[summary.controller == ctrl]
        grid = np.full((len(axes_order), n_cols), np.nan)
        labels = np.full((len(axes_order), n_cols), "", dtype=object)
        for i, axis in enumerate(axes_order):
            rows = sub[sub.axis == axis].sort_values("level").reset_index(drop=True)
            for j, r in rows.iterrows():
                grid[i, j] = r.retention
                labels[i, j] = f"{r.label}\n{r.retention:.2f}"
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        for i in range(len(axes_order)):
            for j in range(n_cols):
                if labels[i, j]:
                    ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(n_cols), [f"L{j}" for j in range(n_cols)])
        ax.set_yticks(range(len(axes_order)), axes_order, fontsize=8)
        ax.set_title(ctrl)
    fig.colorbar(im, ax=np.atleast_1d(axs).tolist(), label="retention rate", shrink=0.8)
    fig.suptitle("Robustness sweep: retention per axis level (5 seeds each; levels sorted)")
    fig.savefig(path, dpi=130, bbox_inches="tight")


def curves_figure(summary: pd.DataFrame, path: Path) -> None:
    show = ["wave_amplitude", "delay_error_steps", "impulse_deg", "combined_stress"]
    show = [a for a in show if a in set(summary.axis)]
    summary = summary.sort_values(["axis", "level"])
    fig, axs = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.4))
    for ax, axis in zip(axs, show):
        for ctrl, style in [("mpc_nominal", "-o"), ("mpc_residual", "-s"), ("pgain", "--^")]:
            sub = summary[(summary.controller == ctrl) & (summary.axis == axis)]
            if len(sub):
                ax.plot(sub.level, sub.retention, style, label=ctrl)
        ax.set_xlabel(axis)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("retention rate")
    axs[0].legend(fontsize=8)
    fig.suptitle("Retention vs perturbation level")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", type=Path, default=None,
                        help="existing run dir with episodes.csv (re-plot only)")
    parser.add_argument("--extension", type=Path, default=None, metavar="CORE_DIR",
                        help="run the extended (extreme + combined-stress) conditions and "
                             "merge with the core run's episodes.csv")
    parser.add_argument("--name", default="robustness_v1")
    args = parser.parse_args()

    if args.report_only:
        run_dir = args.report_only
        df = pd.read_csv(run_dir / "episodes.csv")
    else:
        from wavept.evaluation.robustness import build_extension

        benchmark = load_benchmark(REPO_ROOT / "configs" / "benchmark_v1.yaml")
        base = {t.name: t for t in benchmark.tiers}["medium"].config
        conditions = build_extension(base) if args.extension else build_sweep(base)
        print(f"{len(conditions)} conditions x {len(SWEEP_SEEDS)} seeds")
        rows = run_sweep(
            conditions, CONTROLLERS,
            subset_controllers={"pgain": PGAIN_AXES},
            progress=True,
        )
        run_dir = create_run_dir(args.name)
        df = pd.DataFrame(rows)
        if args.extension:
            df = pd.concat([pd.read_csv(args.extension / "episodes.csv"), df],
                           ignore_index=True)
        df.to_csv(run_dir / "episodes.csv", index=False)

    summary = aggregate(df)
    summary.to_csv(run_dir / "summary.csv", index=False)
    heatmap_figure(summary, ["mpc_nominal", "mpc_residual"], run_dir / "retention_heatmap.png")
    curves_figure(summary, run_dir / "retention_curves.png")
    (run_dir / "manifest.json").write_text(
        json.dumps({"git": git_info(), "seeds": list(SWEEP_SEEDS),
                    "controllers": CONTROLLERS, "pgain_axes": list(PGAIN_AXES)}, indent=2),
        encoding="utf-8",
    )
    with pd.option_context("display.width", 160):
        print(summary.pivot_table(index=["axis", "label"], columns="controller",
                                  values="retention", sort=False).to_string())
    print(f"\noutputs: {run_dir}")


if __name__ == "__main__":
    main()
