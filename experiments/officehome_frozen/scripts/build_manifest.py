#!/usr/bin/env python3
"""Build the deterministic global Office-Home manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from officehome.common import resolve_data_root
from officehome.manifest import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-source", choices=["official", "huggingface"], default="official")
    parser.add_argument("--skip-image-validation", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    metadata = build_manifest(
        resolve_data_root(args.data_root),
        args.output_dir,
        dataset_source=args.dataset_source,
        validate_images=not args.skip_image_validation,
    )
    print(f"Manifest images: {metadata['num_images']}")
    print(f"Domain counts: {metadata['domain_counts']}")
    if metadata["count_warning"]:
        print(f"WARNING: {metadata['count_warning']}")


if __name__ == "__main__":
    main()
