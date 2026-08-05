#!/usr/bin/env python3
"""Train weighted 65-class softmax regression on cached features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from officehome.logreg import train_from_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--query-ids", default=None)
    parser.add_argument("--sample-weights", default=None)
    parser.add_argument("--l2", type=float, required=True)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metrics = train_from_task(
        args.features,
        args.feature_manifest,
        args.task_dir,
        args.output_dir,
        rho=args.l2,
        query_ids_path=args.query_ids,
        sample_weights_path=args.sample_weights,
        device_name=args.device,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        seed=args.seed,
    )
    optimization = metrics["optimization"]
    print(
        f"Training {optimization['status']}: iterations={optimization['accepted_iterations']} "
        f"objective={optimization['final_objective']:.8f}"
    )


if __name__ == "__main__":
    main()
