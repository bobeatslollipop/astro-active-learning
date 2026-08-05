#!/usr/bin/env python3
"""Create row-wise L2-normalized features from cached raw features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from officehome.features import normalize_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = normalize_features(
        args.raw_features,
        args.output_dir,
        metadata_path=args.metadata,
        overwrite=args.overwrite,
    )
    print(f"L2 features: {metadata['feature_files']['l2']['path']}")


if __name__ == "__main__":
    main()
