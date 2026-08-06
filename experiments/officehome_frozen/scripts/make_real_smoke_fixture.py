#!/usr/bin/env python3
"""Create a deterministic real-image smoke manifest and tiny Art-to-Clipart task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from officehome.common import CANONICAL_DOMAINS, atomic_write_csv, atomic_write_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-private", required=True)
    parser.add_argument("--rows-per-domain", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.rows_per_domain < 4:
        raise ValueError("rows-per-domain must be at least 4 for the smoke train/eval split.")
    source = pd.read_csv(args.manifest_private)
    pieces = []
    for domain in CANONICAL_DOMAINS:
        domain_rows = source.loc[source["domain"] == domain].sort_values(["class_id", "row_id"])
        part = domain_rows.groupby("class_id", sort=True, as_index=False).head(1).head(args.rows_per_domain)
        if len(part) != args.rows_per_domain:
            raise ValueError(f"Not enough rows for smoke domain {domain}.")
        pieces.append(part)
    private = pd.concat(pieces, ignore_index=True)
    private.insert(1, "source_row_id", private["row_id"].to_numpy(dtype=np.int64))
    private["row_id"] = np.arange(len(private), dtype=np.int64)
    output_dir = Path(args.output_dir)
    task_dir = output_dir / "task"
    atomic_write_csv(output_dir / "manifest_private.csv", private)
    atomic_write_csv(
        output_dir / "manifest.csv",
        private[["row_id", "relative_image_path", "domain"]],
    )

    source_rows = private.loc[private["domain"] == "art"].drop(columns=["source_row_id"])
    target_rows = private.loc[private["domain"] == "clipart"].drop(columns=["source_row_id"])
    target_pool = target_rows.iloc[:2].copy()
    target_test = target_rows.iloc[2:].copy()
    atomic_write_csv(task_dir / "source_labeled.csv", source_rows)
    atomic_write_csv(task_dir / "target_pool_oracle_private.csv", target_pool)
    atomic_write_csv(
        task_dir / "target_pool_public.csv",
        target_pool[["row_id", "relative_image_path", "domain"]],
    )
    atomic_write_csv(task_dir / "target_test_private.csv", target_test)
    atomic_write_json(task_dir / "task_metadata.json", {
        "schema_version": 1,
        "smoke_fixture": True,
        "protocol": "heldout",
        "source_domain": "art",
        "target_domain": "clipart",
        "counts": {
            "source": int(len(source_rows)),
            "target_pool": int(len(target_pool)),
            "target_test": int(len(target_test)),
        },
        "source_manifest_sha256": sha256_file(args.manifest_private),
    })
    print(f"Real-image smoke fixture: {len(private)} rows at {output_dir}")


if __name__ == "__main__":
    main()
