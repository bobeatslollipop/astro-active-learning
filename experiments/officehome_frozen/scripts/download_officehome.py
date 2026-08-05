#!/usr/bin/env python3
"""Acquire Office-Home from the official source or the explicit HF fallback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from officehome.common import resolve_data_root
from officehome.download import acquire_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--source", choices=["official", "huggingface"], default="official")
    args = parser.parse_args()
    metadata = acquire_dataset(resolve_data_root(args.data_root), args.source)
    print(f"Office-Home dataset ready: {metadata['dataset_root']}")
    print(f"Provider: {metadata['provider']}")


if __name__ == "__main__":
    main()
