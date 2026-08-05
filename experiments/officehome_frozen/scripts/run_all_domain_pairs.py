#!/usr/bin/env python3
"""Create all 12 directed Office-Home task manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from officehome.manifest import all_directed_pairs, make_task_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize leakage-aware manifests for all 12 directed pairs. Feature extraction "
            "is shared; use select_l2/train/evaluate for each task after inspecting these splits."
        )
    )
    parser.add_argument("--manifest-private", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol", choices=["heldout", "transductive"], default="heldout")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    for source, target in all_directed_pairs():
        task_dir = output_root / args.protocol / f"{source}_to_{target}_seed{args.seed}"
        metadata = make_task_split(
            args.manifest_private,
            task_dir,
            source_domain=source,
            target_domain=target,
            protocol=args.protocol,
            seed=args.seed,
        )
        print(f"{source} -> {target}: {metadata['counts']}")


if __name__ == "__main__":
    main()
