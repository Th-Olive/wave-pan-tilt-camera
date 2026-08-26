"""Run the frozen benchmark: controllers x tiers x seeds, with report + manifest.

Examples:
  python scripts/run_benchmark.py --controllers pgain,jacobian
  python scripts/run_benchmark.py --controllers pgain --split dev   # tuning runs
"""

import argparse
import json
from pathlib import Path

from wavept.evaluation.benchmark import load_benchmark, run_benchmark
from wavept.evaluation.reporting import save_report, summarize, to_markdown
from wavept.manifest import create_run_dir, git_info

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path, default=REPO_ROOT / "configs" / "benchmark_v1.yaml"
    )
    parser.add_argument("--controllers", default="pgain,jacobian")
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    parser.add_argument("--name", default=None)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    controllers = args.controllers.split(",")
    benchmark = load_benchmark(args.benchmark)
    rows = run_benchmark(benchmark, controllers, split=args.split, progress=args.progress)
    summary = summarize(rows)

    run_dir = create_run_dir(args.name or f"{benchmark.name}_{args.split}")
    outputs = save_report(rows, summary, run_dir)
    manifest = {
        "git": git_info(),
        "benchmark": str(args.benchmark),
        "split": args.split,
        "controllers": controllers,
        "outputs": outputs,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"benchmark={benchmark.name} split={args.split}")
    print(to_markdown(summary))
    print(f"outputs: {run_dir}")


if __name__ == "__main__":
    main()
