from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from officehome.campaign import BASELINE_METHODS, CampaignRunner, REGULARIZED_METHOD
from officehome.manifest import make_task_split
from officehome.selection import make_sample_weights, select_queries


def _synthetic_campaign_fixture(tmp_path: Path):
    rng = np.random.default_rng(123)
    domains = ("art", "clipart", "product", "real_world")
    rows = []
    features = []
    row_id = 0
    class_centers = rng.normal(size=(65, 8)).astype(np.float32)
    for domain_index, domain in enumerate(domains):
        per_class = 5 if domain == "clipart" else 2
        for class_id in range(65):
            for copy_index in range(per_class):
                rows.append({
                    "row_id": row_id,
                    "relative_image_path": f"{domain}/class_{class_id:02d}/{copy_index}.jpg",
                    "domain": domain,
                    "class_name": f"class_{class_id:02d}",
                    "class_id": class_id,
                })
                vector = class_centers[class_id] + 0.05 * rng.normal(size=8)
                vector += 0.02 * domain_index
                vector = vector.astype(np.float32)
                vector /= np.linalg.norm(vector)
                features.append(vector)
                row_id += 1
    private = pd.DataFrame(rows)
    manifest_private = tmp_path / "manifest_private.csv"
    private.to_csv(manifest_private, index=False)
    feature_manifest = tmp_path / "manifest.csv"
    private[["row_id", "relative_image_path", "domain"]].to_csv(feature_manifest, index=False)
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, np.asarray(features, dtype=np.float32))
    task_dir = tmp_path / "task"
    make_task_split(
        manifest_private,
        task_dir,
        source_domain="art",
        target_domain="clipart",
        protocol="heldout",
        seed=0,
    )
    return manifest_private, feature_path, feature_manifest, task_dir


def test_selectors_return_150_unique_public_reproducible_ids(tmp_path):
    _, features, feature_manifest, task_dir = _synthetic_campaign_fixture(tmp_path)
    public_ids = set(pd.read_csv(task_dir / "target_pool_public.csv")["row_id"].astype(int))
    for method in ("random", "wasserstein"):
        first_dir = tmp_path / f"{method}_first"
        second_dir = tmp_path / f"{method}_second"
        select_queries(
            task_dir, features, feature_manifest, first_dir, method=method, seed=7, budget=150
        )
        select_queries(
            task_dir, features, feature_manifest, second_dir, method=method, seed=7, budget=150
        )
        first = pd.read_csv(first_dir / "query_ids.csv")["row_id"].to_numpy()
        second = pd.read_csv(second_dir / "query_ids.csv")["row_id"].to_numpy()
        assert len(first) == len(np.unique(first)) == 150
        assert set(int(value) for value in first) <= public_ids
        np.testing.assert_array_equal(first, second)


def test_geometry_adapter_rejects_labels_and_nonpublic_query_ids(tmp_path):
    _, features, feature_manifest, task_dir = _synthetic_campaign_fixture(tmp_path)
    public_path = task_dir / "target_pool_public.csv"
    public = pd.read_csv(public_path)
    public["class_id"] = 0
    public.to_csv(public_path, index=False)
    with pytest.raises(ValueError, match="label-free"):
        select_queries(
            task_dir,
            features,
            feature_manifest,
            tmp_path / "bad_public",
            method="random",
            seed=0,
            budget=5,
        )

    public.drop(columns="class_id").to_csv(public_path, index=False)
    test_id = int(pd.read_csv(task_dir / "target_test_private.csv").iloc[0]["row_id"])
    leaked = tmp_path / "leaked_query.csv"
    pd.DataFrame({"row_id": [test_id]}).to_csv(leaked, index=False)
    with pytest.raises(ValueError, match="not members of the public target pool"):
        make_sample_weights(
            task_dir,
            features,
            feature_manifest,
            tmp_path / "leaked_weights",
            method="uniform",
            query_ids_path=leaked,
        )


def test_selector_does_not_read_private_target_manifests(tmp_path):
    _, features, feature_manifest, task_dir = _synthetic_campaign_fixture(tmp_path)
    (task_dir / "target_pool_oracle_private.csv").write_text("not,a,valid,oracle\n", encoding="utf-8")
    (task_dir / "target_test_private.csv").write_text("not,a,valid,test\n", encoding="utf-8")
    metadata = select_queries(
        task_dir,
        features,
        feature_manifest,
        tmp_path / "public_only",
        method="random",
        seed=0,
        budget=5,
    )
    assert metadata["target_labels_loaded"] is False


def test_synthetic_smoke_covers_five_method_families_and_reuses_queries(tmp_path):
    manifest_private, features, feature_manifest, _ = _synthetic_campaign_fixture(tmp_path)
    runner = CampaignRunner(
        manifest_private=manifest_private,
        features=features,
        feature_manifest=feature_manifest,
        campaign_root=tmp_path / "campaign",
        device="cpu",
        query_budget=5,
        classifier_max_iter=30,
        classifier_tolerance=1e-5,
        reweight_max_iter=128,
        l2_grid=[1e-2],
        l2_folds=2,
    )
    records = {}
    for method in BASELINE_METHODS:
        records[method] = runner.run_method("art", "clipart", 0, method)
    records[REGULARIZED_METHOD] = runner.run_method(
        "art", "clipart", 0, REGULARIZED_METHOD, reweight_lambda=100.0
    )
    assert all(record["status"] == "complete" for record in records.values())
    assert len(records) == 5

    random_hash = records["random_uniform"]["hashes"]["query_ids"]
    assert random_hash == records[REGULARIZED_METHOD]["hashes"]["query_ids"]
    wasserstein_hash = records["wasserstein_uniform"]["hashes"]["query_ids"]
    assert wasserstein_hash == records["wasserstein_hard_voronoi"]["hashes"]["query_ids"]

    for record in records.values():
        evaluation = record["metrics"]["evaluation"]
        assert "target_heldout" in evaluation
        assert "target_transductive_full" in evaluation
        assert "target_transductive_unqueried" in evaluation
        weight_frame = pd.read_csv(record["reweighting"]["sample_weights_path"])
        assert not weight_frame["row_id"].duplicated().any()
        assert np.isfinite(weight_frame["weight"]).all()
        assert (weight_frame["weight"] >= 0).all()
        assert np.isclose(weight_frame["weight"].sum(), 1.0)

    hard = records["wasserstein_hard_voronoi"]["reweighting"]["diagnostics"]
    regularized = records[REGULARIZED_METHOD]["reweighting"]["diagnostics"]
    assert hard["effective_sample_size"] > 0
    assert regularized["effective_sample_size"] > 0
    assert records[REGULARIZED_METHOD]["reweighting"]["solver"]["converged"] is True

    # A second call validates and skips every successful artifact.
    repeated = runner.run_method("art", "clipart", 0, "random_uniform")
    assert repeated["artifacts"]["metrics"]["sha256"] == records["random_uniform"]["artifacts"]["metrics"]["sha256"]
