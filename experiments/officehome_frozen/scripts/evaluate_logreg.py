#!/usr/bin/env python3
"""Evaluate a saved softmax model without exposing target-test labels to training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from officehome.logreg import evaluate_saved_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--query-ids", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metrics = evaluate_saved_model(
        args.model,
        args.features,
        args.feature_manifest,
        args.task_dir,
        args.output_dir,
        query_ids_path=args.query_ids,
    )
    print(f"Evaluation: {metrics['evaluation']}")


if __name__ == "__main__":
    main()
