from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from officehome.features import (
    FEATURE_DIMENSION,
    L2_FILENAME,
    normalize_features,
    validate_feature_artifacts,
)


def _manifest(path, n):
    frame = pd.DataFrame({
        "row_id": np.arange(n),
        "relative_image_path": [f"Art/Class/image_{i}.jpg" for i in range(n)],
        "domain": ["art"] * n,
    })
    frame.to_csv(path, index=False)


def test_normalize_features_and_validate_artifacts(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(7, FEATURE_DIMENSION)).astype(np.float32)
    raw_path = tmp_path / "raw.npy"
    np.save(raw_path, raw)
    manifest_path = tmp_path / "manifest.csv"
    _manifest(manifest_path, len(raw))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"feature_files": {}}), encoding="utf-8")
    metadata = normalize_features(raw_path, tmp_path, metadata_path=metadata_path)
    normalized = np.load(tmp_path / L2_FILENAME)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-6)
    assert metadata["feature_files"]["l2"]["maximum_normalized_row_norm_error"] <= 1e-5
    audit = validate_feature_artifacts(raw_path, tmp_path / L2_FILENAME, manifest_path)
    assert audit["shape"] == [7, FEATURE_DIMENSION]


def test_normalize_rejects_zero_rows(tmp_path):
    raw = np.ones((3, FEATURE_DIMENSION), dtype=np.float32)
    raw[1] = 0
    raw_path = tmp_path / "raw.npy"
    np.save(raw_path, raw)
    with pytest.raises(FloatingPointError, match="Zero or near-zero"):
        normalize_features(raw_path, tmp_path)


def test_validate_rejects_nonfinite_features(tmp_path):
    raw = np.ones((2, FEATURE_DIMENSION), dtype=np.float32)
    normalized = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    raw[0, 0] = np.nan
    raw_path = tmp_path / "raw.npy"
    l2_path = tmp_path / "l2.npy"
    np.save(raw_path, raw)
    np.save(l2_path, normalized.astype(np.float32))
    manifest_path = tmp_path / "manifest.csv"
    _manifest(manifest_path, 2)
    with pytest.raises(FloatingPointError):
        validate_feature_artifacts(raw_path, l2_path, manifest_path)
