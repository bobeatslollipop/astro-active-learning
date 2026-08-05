#!/usr/bin/env python3
"""Create leakage-aware manifests for one directed Office-Home task."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from officehome.manifest import make_task_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-private", required=True)
    parser.add_argument("--source-domain", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--protocol", choices=["heldout", "transductive"], default="heldout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metadata = make_task_split(
        args.manifest_private,
        args.output_dir,
        source_domain=args.source_domain,
        target_domain=args.target_domain,
        protocol=args.protocol,
        seed=args.seed,
    )
    print(
        f"Task {metadata['source_domain']} -> {metadata['target_domain']} "
        f"({metadata['protocol']}): {metadata['counts']}"
    )


if __name__ == "__main__":
    main()
