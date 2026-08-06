#!/usr/bin/env python3
"""Run or aggregate the recoverable Office-Home round-1 campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from officehome.campaign import CampaignRunner, campaign_metadata
from officehome.common import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "full", "aggregate"], required=True)
    parser.add_argument(
        "--manifest-private",
        default=REPO_ROOT / "results/domain_adaptation/officehome_frozen/dataset/manifest_private.csv",
    )
    parser.add_argument(
        "--features",
        default=REPO_ROOT / "results/domain_adaptation/officehome_frozen/features/resnet50_imagenet1k_v1_l2.npy",
    )
    parser.add_argument(
        "--feature-manifest",
        default=REPO_ROOT / "results/domain_adaptation/officehome_frozen/features/manifest.csv",
    )
    parser.add_argument(
        "--campaign-root",
        default=REPO_ROOT / "results/domain_adaptation/officehome_frozen/campaigns/round1_150q",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query-budget", type=int, default=150)
    parser.add_argument("--classifier-max-iter", type=int, default=500)
    parser.add_argument("--classifier-tolerance", type=float, default=1e-6)
    parser.add_argument("--reweight-max-iter", type=int, default=1024)
    parser.add_argument("--l2-grid", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--l2-folds", type=int, default=3)
    args = parser.parse_args()

    runner = CampaignRunner(
        manifest_private=args.manifest_private,
        features=args.features,
        feature_manifest=args.feature_manifest,
        campaign_root=args.campaign_root,
        device=args.device,
        query_budget=args.query_budget,
        classifier_max_iter=args.classifier_max_iter,
        classifier_tolerance=args.classifier_tolerance,
        reweight_max_iter=args.reweight_max_iter,
        l2_grid=args.l2_grid,
        l2_folds=args.l2_folds,
    )
    metadata_path = Path(args.campaign_root).expanduser().resolve() / "campaign_metadata.json"
    if metadata_path.exists():
        import json

        with metadata_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        current = campaign_metadata(runner)
        immutable = (
            "manifest_private_sha256",
            "features_sha256",
            "feature_manifest_sha256",
            "query_budget",
            "lambda_grid",
            "l2_grid",
        )
        mismatches = [key for key in immutable if existing.get(key) != current.get(key)]
        if mismatches:
            raise RuntimeError(f"Campaign metadata mismatch for existing root: {mismatches}")
    else:
        atomic_write_json(metadata_path, campaign_metadata(runner))

    try:
        if args.stage == "smoke":
            runner.run_smoke()
        elif args.stage == "full":
            runner.run_full()
        else:
            selected = runner.aggregate(write_selection=True)
            print(selected)
    except Exception as exc:
        runner.update_state(
            status="failed",
            failure_type=type(exc).__name__,
            failure_message=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
