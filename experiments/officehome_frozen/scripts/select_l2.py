#!/usr/bin/env python3
"""Select rho using source-domain-only stratified cross-validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from officehome.logreg import L2_GRID, select_l2_source_cv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid", nargs="+", type=float, default=list(L2_GRID))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = select_l2_source_cv(
        args.features,
        args.feature_manifest,
        args.source_manifest,
        args.output_dir,
        grid=args.grid,
        folds=args.folds,
        device_name=args.device,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        seed=args.seed,
    )
    print(f"Selected rho: {result['selected_rho']:.12g}")


if __name__ == "__main__":
    main()
