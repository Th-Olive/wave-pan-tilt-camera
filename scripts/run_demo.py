"""Run demo episodes and export video + reproducibility manifest.

Controllers:
  none      zero action - waves move the target (uncompensated)
  oracle    per-step aim_at on true target position (upper bound, not fair)
  pgain     reactive P controller on normalized image error (M2 baseline)
  jacobian  reactive nominal-Jacobian servo (M2 baseline)

Examples:
  python scripts/run_demo.py --config configs/sim_basic.yaml --controller pgain
  python scripts/run_demo.py --config configs/sim_waves_hard_dev.yaml --compare none,pgain --gif
"""

import argparse
import dataclasses
from pathlib import Path

from wavept.config import load_config
from wavept.control.factory import build_controller
from wavept.evaluation.runner import run_episode
from wavept.manifest import create_run_dir, write_manifest
from wavept.simulation.environment import PanTiltEnv
from wavept.visualization.videos import save_gif, save_video, side_by_side

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "sim_basic.yaml")
    parser.add_argument("--benchmark-tier", default=None, metavar="TIER",
                        help="use a tier config from configs/benchmark_v1.yaml instead of --config")
    parser.add_argument("--controller", default="none")
    parser.add_argument("--compare", default=None, metavar="A,B",
                        help="run several controllers on the same seed, split-screen video")
    parser.add_argument("--seed", type=int, default=None, help="override scenario seed")
    parser.add_argument("--gif", action="store_true", help="also export a downsampled gif")
    parser.add_argument("--name", default=None, help="run directory suffix")
    args = parser.parse_args()

    if args.benchmark_tier:
        from wavept.evaluation.benchmark import load_benchmark

        benchmark = load_benchmark(REPO_ROOT / "configs" / "benchmark_v1.yaml")
        tiers = {t.name: t for t in benchmark.tiers}
        config = tiers[args.benchmark_tier].config
    else:
        config = load_config(args.config)
    if args.seed is not None:
        config = dataclasses.replace(
            config, scenario=dataclasses.replace(config.scenario, seed=args.seed)
        )
    seed = config.scenario.seed
    names = args.compare.split(",") if args.compare else [args.controller]

    results = {}
    for name in names:
        env = PanTiltEnv(config.sim, config.scenario)
        controller = build_controller(name, config)
        # label panels only in comparisons; a single-controller video needs no caption
        panel_label = name if len(names) > 1 else ""
        results[name] = run_episode(
            env, controller, seed=seed, record_frames=True, label=panel_label
        )

    label = "_vs_".join(names)
    run_dir = create_run_dir(args.name or f"demo_{label}")
    fps = round(1.0 / config.sim.dt)
    frames = (
        side_by_side(*(results[n].frames for n in names))
        if len(names) > 1
        else results[names[0]].frames
    )
    outputs = {"video": str(save_video(frames, run_dir / "episode.mp4", fps=fps))}
    if args.gif:
        outputs["gif"] = str(save_gif(frames, run_dir / "episode.gif"))

    metrics = {n: results[n].metrics for n in names}
    write_manifest(run_dir, config, seed, label, metrics, outputs,
                   extra={"panel_order_left_to_right": names})

    source = f"benchmark_v1:{args.benchmark_tier}" if args.benchmark_tier else args.config.name
    print(f"config={source}  seed={seed}")
    for n in names:
        m = results[n].metrics
        print(
            f"  {n:9s} retained={str(m['retained']):5s} "
            f"min_margin={m['min_margin_px']:7.1f}px "
            f"mean_err={m['mean_center_err_norm'] if m['mean_center_err_norm'] is not None else float('nan'):.4f} "
            f"visible={m['visible_time_ratio']:.2f}"
        )
    print(f"outputs: {run_dir}")


if __name__ == "__main__":
    main()
