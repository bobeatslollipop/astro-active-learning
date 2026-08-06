from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from officehome.logreg import (
    _logits_numpy,
    build_training_table,
    evaluate_saved_model,
    load_model,
    optimize_weighted_logreg,
    train_from_task,
)


def test_optimizer_loss_decreases_and_penalty_includes_bias():
    rng = np.random.default_rng(0)
    centers = np.eye(3, dtype=np.float32)
    y = np.repeat(np.arange(3), 20)
    X = centers[y] + 0.03 * rng.normal(size=(len(y), 3)).astype(np.float32)
    weights = np.full(len(y), 1.0 / len(y))
    rho = 0.01
    model, history, diagnostics = optimize_weighted_logreg(
        X, y, weights, rho=rho, num_classes=3, device_name="cpu", max_iter=50, tolerance=1e-7
    )
    assert history.iloc[-1]["objective"] < history.iloc[0]["objective"]
    expected_penalty = 0.5 * rho * (
        np.sum(model["W"].astype(np.float64) ** 2) + np.sum(model["b"].astype(np.float64) ** 2)
    )
    assert history.iloc[-1]["l2_penalty"] == pytest.approx(expected_penalty, rel=1e-5)
    assert diagnostics["theoretical_norm_bound_satisfied"]


def _experiment_fixture(tmp_path):
    rng = np.random.default_rng(1)
    dimension = 16
    records = []
    features = []
    row_id = 0
    source_records = []
    pool_records = []
    test_records = []
    prototypes = rng.normal(size=(65, dimension)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)

    def add(domain, class_id, count, destination):
        nonlocal row_id
        for item in range(count):
            feature = prototypes[class_id] + 0.02 * rng.normal(size=dimension).astype(np.float32)
            feature /= np.linalg.norm(feature)
            record = {
                "row_id": row_id,
                "relative_image_path": f"{domain}/Class_{class_id:02d}/{item}.jpg",
                "domain": domain,
                "class_name": f"Class_{class_id:02d}",
                "class_id": class_id,
            }
            records.append({key: record[key] for key in ("row_id", "relative_image_path", "domain")})
            destination.append(record)
            features.append(feature)
            row_id += 1

    for class_id in range(65):
        add("art", class_id, 3, source_records)
        add("clipart", class_id, 1, pool_records)
        add("clipart", class_id, 1, test_records)
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, np.asarray(features, dtype=np.float32))
    feature_manifest = tmp_path / "manifest.csv"
    pd.DataFrame(records).to_csv(feature_manifest, index=False)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    pd.DataFrame(source_records).to_csv(task_dir / "source_labeled.csv", index=False)
    pd.DataFrame(pool_records).to_csv(task_dir / "target_pool_oracle_private.csv", index=False)
    pd.DataFrame([{key: row[key] for key in ("row_id", "relative_image_path", "domain")} for row in pool_records]).to_csv(
        task_dir / "target_pool_public.csv", index=False
    )
    pd.DataFrame(test_records).to_csv(task_dir / "target_test_private.csv", index=False)
    (task_dir / "task_metadata.json").write_text(
        json.dumps({"protocol": "heldout", "source_domain": "art", "target_domain": "clipart"}),
        encoding="utf-8",
    )
    return feature_path, feature_manifest, task_dir, pool_records, test_records


def test_query_and_weight_validation_rejects_leakage_and_incomplete_weights(tmp_path):
    _, _, task_dir, pool, test = _experiment_fixture(tmp_path)
    leaked_query = tmp_path / "leaked.csv"
    pd.DataFrame({"row_id": [test[0]["row_id"]]}).to_csv(leaked_query, index=False)
    with pytest.raises(ValueError, match="target-test"):
        build_training_table(task_dir, query_ids_path=leaked_query)

    valid_query = tmp_path / "query.csv"
    pd.DataFrame({"row_id": [pool[0]["row_id"]]}).to_csv(valid_query, index=False)
    incomplete = tmp_path / "weights.csv"
    pd.DataFrame({"row_id": [pool[0]["row_id"]], "weight": [1.0]}).to_csv(incomplete, index=False)
    with pytest.raises(ValueError, match="cover every training row"):
        build_training_table(task_dir, query_ids_path=valid_query, sample_weights_path=incomplete)


def test_train_save_reload_and_evaluate_reproduce_predictions(tmp_path):
    feature_path, feature_manifest, task_dir, _, test = _experiment_fixture(tmp_path)
    output = tmp_path / "run"
    metrics = train_from_task(
        feature_path,
        feature_manifest,
        task_dir,
        output,
        rho=0.01,
        device_name="cpu",
        max_iter=40,
        tolerance=1e-6,
    )
    assert metrics["counts_and_weights"]["num_source"] == 195
    assert (output / "training_metrics.json").exists()
    assert not (output / "metrics.json").exists()
    model_before = load_model(output / "model.pt")
    model_after = load_model(output / "model.pt")
    features = np.load(feature_path)
    test_ids = np.array([row["row_id"] for row in test], dtype=np.int64)
    np.testing.assert_allclose(
        _logits_numpy(model_before, features[test_ids]),
        _logits_numpy(model_after, features[test_ids]),
        rtol=0,
        atol=0,
    )
    evaluated = evaluate_saved_model(
        output / "model.pt", feature_path, feature_manifest, task_dir, output
    )
    assert evaluated["evaluation"]["protocol"] == "heldout"
    assert evaluated["evaluation"]["target_test"]["num_examples"] == 65
    assert evaluated["evaluation"]["target_heldout"]["num_examples"] == 65
    assert evaluated["evaluation"]["target_transductive_full"]["num_examples"] == 130
    assert evaluated["evaluation"]["target_transductive_unqueried"]["num_examples"] == 130
    assert (output / "predictions.csv").exists()
    assert (output / "predictions_transductive.csv").exists()

    unknown_query = tmp_path / "unknown_query.csv"
    pd.DataFrame({"row_id": [999999]}).to_csv(unknown_query, index=False)
    with pytest.raises(ValueError, match="not members of the target pool"):
        evaluate_saved_model(
            output / "model.pt",
            feature_path,
            feature_manifest,
            task_dir,
            output,
            query_ids_path=unknown_query,
        )
