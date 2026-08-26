"""Benchmark reporting: episode rows -> aggregate tables (csv + markdown)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wavept.evaluation.metrics import aggregate

SUMMARY_COLUMNS = [
    "retention_rate",
    "mean_min_margin_px",
    "mean_center_err_norm",
    "mean_action_abs_integral",
    "mean_n_saturations",
    "mean_compute_ms",
    "max_compute_ms",
]


def summarize(rows: list[dict]) -> pd.DataFrame:
    """Aggregate per-episode rows by (tier, controller)."""
    df = pd.DataFrame(rows)
    out = []
    for (tier, controller), group in df.groupby(["tier", "controller"], sort=False):
        agg = aggregate(group.to_dict("records"))
        out.append({"tier": tier, "controller": controller, **agg})
    return pd.DataFrame(out)


def to_markdown(summary: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    cols = ["tier", "controller", "n_episodes"] + [
        c for c in SUMMARY_COLUMNS if c in summary.columns
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, row in summary[cols].iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(
                float_fmt.format(v) if isinstance(v, float) else str(v)
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_report(rows: list[dict], summary: pd.DataFrame, run_dir: Path) -> dict[str, str]:
    run_dir = Path(run_dir)
    episodes_csv = run_dir / "episodes.csv"
    summary_csv = run_dir / "summary.csv"
    summary_md = run_dir / "summary.md"
    pd.DataFrame(rows).to_csv(episodes_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(to_markdown(summary) + "\n", encoding="utf-8")
    return {
        "episodes_csv": str(episodes_csv),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
    }
