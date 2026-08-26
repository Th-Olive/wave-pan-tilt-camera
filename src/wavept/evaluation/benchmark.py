"""Benchmark runner: controllers x tiers x seeds from a benchmark YAML.

Benchmark YAML structure (see configs/benchmark_v1.yaml):

    name: benchmark_v1
    tiers:
      easy:
        dev_seeds: [...]     # tuning allowed
        test_seeds: [...]    # FROZEN - only for reported results
        config: {sim: ..., scenario: ..., mismatch: ...}   # full Config dict
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

from wavept.config import Config, config_from_dict, convert_degrees
from wavept.control.factory import build_controller
from wavept.evaluation.runner import run_episode
from wavept.simulation.environment import PanTiltEnv


@dataclass(frozen=True)
class BenchmarkTier:
    name: str
    dev_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    config: Config


@dataclass(frozen=True)
class Benchmark:
    name: str
    tiers: tuple[BenchmarkTier, ...]


def load_benchmark(path: str | Path) -> Benchmark:
    raw = convert_degrees(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    tiers = tuple(
        BenchmarkTier(
            name=tier_name,
            dev_seeds=tuple(spec["dev_seeds"]),
            test_seeds=tuple(spec["test_seeds"]),
            config=config_from_dict(spec["config"]),
        )
        for tier_name, spec in raw["tiers"].items()
    )
    return Benchmark(name=raw["name"], tiers=tiers)


def run_benchmark(
    benchmark: Benchmark,
    controllers: list[str],
    split: str = "test",
    progress: bool = False,
) -> list[dict]:
    """Run every controller on every tier/seed of the split. Returns one row
    (dict of tier, controller, seed + episode metrics) per episode."""
    assert split in ("test", "dev")
    rows = []
    for tier in benchmark.tiers:
        seeds = tier.test_seeds if split == "test" else tier.dev_seeds
        for name in controllers:
            for seed in seeds:
                scenario = dataclasses.replace(tier.config.scenario, seed=seed)
                env = PanTiltEnv(tier.config.sim, scenario)
                controller = build_controller(name, tier.config)
                result = run_episode(env, controller, seed=seed)
                rows.append(
                    {"tier": tier.name, "controller": name, "seed": seed, **result.metrics}
                )
                if progress:
                    m = result.metrics
                    print(
                        f"  {tier.name:8s} {name:9s} seed={seed:5d} "
                        f"retained={str(m['retained']):5s} "
                        f"min_margin={m['min_margin_px']:7.1f}px"
                    )
    return rows
