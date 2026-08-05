from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from officehome.features import (
    L2_FILENAME,
    RAW_FILENAME,
    extract_resnet50_features,
    normalize_features,
)
from officehome.logreg import _logits_numpy, evaluate_saved_model, load_model, train_from_task


@pytest.mark.skipif(
    os.environ.get("OFFICEHOME_REAL_SMOKE") != "1",
    reason="set OFFICEHOME_REAL_SMOKE=1 after downloading Office-Home",
)
def test_real_pretrained_resnet_and_micro_training(tmp_path):
    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = experiment_root.parents[1]
    data_root = Path(os.environ.get("OFFICEHOME_ROOT", experiment_root / ".data"))
    dataset_dir = Path(
        os.environ.get(
            "OFFICEHOME_DATASET_ARTIFACTS",
            repo_root / "results/domain_adaptation/officehome_frozen/dataset",
        )
    )
    private = pd.read_csv(dataset_dir / "manifest_private.csv")

    # Four deterministic, class-distinct rows from each real domain.
    selected = []
    for domain in ("art", "clipart", "product", "real_world"):
        rows = (
            private.loc[private["domain"] == domain]
            .sort_values(["class_name", "relative_image_path"])
            .drop_duplicates("class_name")
            .head(4)
        )
        assert len(rows) == 4
        selected.append(rows)
    mini = pd.concat(selected, ignore_index=True)
    mini["row_id"] = np.arange(len(mini), dtype=np.int64)

    feature_manifest = tmp_path / "manifest.csv"
    mini[["row_id", "relative_image_path", "domain"]].to_csv(feature_manifest, index=False)
    feature_dir = tmp_path / "features"
    metadata = extract_resnet50_features(
        data_root,
        feature_manifest,
        feature_dir,
        device_name=os.environ.get("OFFICEHOME_SMOKE_DEVICE", "auto"),
        batch_size=8,
        workers=0,
        reproducibility_rows=8,
    )
    assert metadata["feature_shape"] == [16, 2048]
    assert metadata["repeat_check"]["passed"] is True
    assert metadata["repeat_check"]["max_abs_error"] <= 1e-5
    raw = np.load(feature_dir / RAW_FILENAME)
    assert raw.shape == (16, 2048) and raw.dtype == np.float32

    normalize_features(feature_dir / RAW_FILENAME, feature_dir)
    normalized = np.load(feature_dir / L2_FILENAME)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-5, rtol=0)

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    labeled_columns = ["row_id", "relative_image_path", "domain", "class_name", "class_id"]
    source = mini.loc[mini["domain"] == "art", labeled_columns]
    target = mini.loc[mini["domain"] == "clipart", labeled_columns].reset_index(drop=True)
    target_pool = target.iloc[:2]
    target_test = target.iloc[2:]
    source.to_csv(task_dir / "source_labeled.csv", index=False)
    target_pool.to_csv(task_dir / "target_pool_oracle_private.csv", index=False)
    target_pool[["row_id", "relative_image_path", "domain"]].to_csv(
        task_dir / "target_pool_public.csv", index=False
    )
    target_test.to_csv(task_dir / "target_test_private.csv", index=False)
    (task_dir / "task_metadata.json").write_text(
        json.dumps({"protocol": "heldout", "num_classes": 65}), encoding="utf-8"
    )

    run_dir = tmp_path / "run"
    metrics = train_from_task(
        feature_dir / L2_FILENAME,
        feature_dir / "manifest.csv",
        task_dir,
        run_dir,
        rho=1e-3,
        device_name="cpu",
        max_iter=40,
        tolerance=1e-6,
        seed=0,
    )
    history = pd.read_csv(run_dir / "optimization_history.csv")
    assert history.iloc[-1]["objective"] < history.iloc[0]["objective"]
    assert metrics["parameter_norms"]["bias_norm"] > 0

    before = load_model(run_dir / "model.pt")
    evaluated = evaluate_saved_model(
        run_dir / "model.pt",
        feature_dir / L2_FILENAME,
        feature_dir / "manifest.csv",
        task_dir,
        run_dir,
    )
    after = load_model(run_dir / "model.pt")
    np.testing.assert_allclose(
        _logits_numpy(before, normalized),
        _logits_numpy(after, normalized),
        atol=1e-6,
        rtol=0,
    )
    assert evaluated["evaluation"]["target_test"]["num_examples"] == 2
