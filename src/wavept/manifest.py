"""Reproducibility manifests: every run records git state,
config, seed, controller, metrics, and output paths."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from wavept.config import Config, to_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "outputs" / "runs"


def git_info() -> dict:
    def _run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": _run("rev-parse", "HEAD"),
            "dirty": bool(_run("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def create_run_dir(name: str, root: Path = RUNS_DIR) -> Path:
    run_dir = root / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(
    run_dir: Path,
    config: Config,
    seed: int,
    controller: str,
    metrics: dict,
    outputs: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> Path:
    manifest = {
        "git": git_info(),
        "config": to_dict(config),
        "seed": seed,
        "controller": controller,
        "metrics": metrics,
        "outputs": outputs,
        **(extra or {}),
    }
    path = Path(run_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
