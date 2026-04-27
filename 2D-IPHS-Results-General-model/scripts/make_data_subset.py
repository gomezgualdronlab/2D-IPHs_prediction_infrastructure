#!/usr/bin/env python3
"""Sample a fraction of each split (train / validation / test) into a new directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write stratified-by-file random subsets of train/val/test CSVs.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing the three CSVs (default: data/processed).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/subsets/subset_out"),
        help="Directory to create and write sampled CSVs into (use a distinct path per run).",
    )
    p.add_argument("--train-name", default="ml_ready_training.csv")
    p.add_argument("--validation-name", default="ml_ready_validation.csv")
    p.add_argument("--test-name", default="ml_ready_test.csv")
    p.add_argument("--frac", type=float, default=0.1, help="Fraction of rows to keep per file.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 < args.frac <= 1):
        raise SystemExit("--frac must be in (0, 1].")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = args.seed
    for name in (args.train_name, args.validation_name, args.test_name):
        src = args.input_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing input file: {src}")
        df = pd.read_csv(src)
        out = df.sample(frac=args.frac, random_state=rng)
        out.to_csv(args.output_dir / name, index=False)
        print(f"Wrote {len(out)} / {len(df)} rows -> {args.output_dir / name}")


if __name__ == "__main__":
    main()
