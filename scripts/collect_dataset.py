"""Collect the M5 transition dataset (mixture policies, episode/condition splits).

Example:
  python scripts/collect_dataset.py --name wavept_v1
"""

import argparse
from pathlib import Path

from wavept.models.collection import collect
from wavept.models.datasets import load_index

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="wavept_v1")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "outputs" / "datasets")
    parser.add_argument("--duration", type=float, default=None, help="override episode length (s)")
    args = parser.parse_args()

    out_dir = args.out_root / args.name
    collect(out_dir, duration=args.duration, progress=True)

    index = load_index(out_dir)
    print("\ntransitions per split:")
    print(index.groupby("split")["n_steps"].agg(["count", "sum"]).to_string())
    print(f"\ndataset: {out_dir}")


if __name__ == "__main__":
    main()
