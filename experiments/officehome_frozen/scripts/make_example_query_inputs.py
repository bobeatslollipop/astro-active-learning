#!/usr/bin/env python3
"""Create a mechanical query/weight fixture to demonstrate the future selector contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from officehome.common import atomic_write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--num-queries", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.num_queries <= 0:
        raise ValueError("num-queries must be positive.")
    task_dir = Path(args.task_dir)
    source = pd.read_csv(task_dir / "source_labeled.csv")
    target_public = pd.read_csv(task_dir / "target_pool_public.csv").sort_values("row_id")
    selected = target_public.head(args.num_queries)[["row_id"]].copy()
    if len(selected) != args.num_queries:
        raise ValueError("Target pool has fewer rows than requested queries.")
    training_ids = pd.concat([source[["row_id"]], selected], ignore_index=True)
    weights = training_ids.copy()
    weights["weight"] = np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    output_dir = Path(args.output_dir)
    atomic_write_csv(output_dir / "query_ids.csv", selected)
    atomic_write_csv(output_dir / "sample_weights.csv", weights)
    print(
        "Created a deterministic interface fixture only; these IDs are not produced by "
        "an active-learning or Wasserstein selector."
    )


if __name__ == "__main__":
    main()
