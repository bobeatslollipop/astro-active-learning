#!/usr/bin/env python3
"""Extract deterministic raw ImageNet1K-V1 ResNet-50 features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from officehome.features import extract_resnet50_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Directory containing the four domain folders.")
    parser.add_argument("--manifest", required=True, help="Label-free feature manifest.csv.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-metadata", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = extract_resnet50_features(
        args.data_root,
        args.manifest,
        args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        overwrite=args.overwrite,
        dataset_metadata_path=args.dataset_metadata,
    )
    print(f"Raw features: {metadata['feature_files']['raw']['path']}")
    print(f"Shape: {metadata['feature_shape']}; seconds: {metadata['extraction_seconds']:.2f}")


if __name__ == "__main__":
    main()
