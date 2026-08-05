from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from officehome.common import CANONICAL_DOMAINS, NUM_CLASSES, normalize_domain_name
from officehome.manifest import build_manifest, make_task_split


DOMAIN_DIRS = {
    "art": "Art",
    "clipart": "Clipart",
    "product": "Product",
    "real_world": "Real World",
}


def _make_dataset(root: Path, *, images_per_class: int = 1) -> Path:
    dataset = root / "OfficeHomeDataset_10072016"
    for domain_index, domain in enumerate(CANONICAL_DOMAINS):
        for class_id in range(NUM_CLASSES):
            class_dir = dataset / DOMAIN_DIRS[domain] / f"Class_{class_id:02d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(images_per_class):
                value = (domain_index * 67 + class_id * 3 + image_index) % 255
                pixels = np.full((6, 7, 3), value, dtype=np.uint8)
                Image.fromarray(pixels).save(class_dir / f"image_{image_index:02d}.png")
    return dataset


def test_domain_aliases():
    assert normalize_domain_name("Real World") == "real_world"
    assert normalize_domain_name("RealWorld") == "real_world"
    assert normalize_domain_name("real_world") == "real_world"
    assert normalize_domain_name("Clip Art") == "clipart"
    with pytest.raises(ValueError):
        normalize_domain_name("amazon")


def test_manifest_is_stable_and_label_free_view_has_no_labels(tmp_path):
    _make_dataset(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    metadata = build_manifest(tmp_path, first, validate_images=True)
    build_manifest(tmp_path, second, validate_images=True)
    private_first = pd.read_csv(first / "manifest_private.csv")
    private_second = pd.read_csv(second / "manifest_private.csv")
    pd.testing.assert_frame_equal(private_first, private_second)
    assert len(private_first) == 4 * NUM_CLASSES
    assert private_first["row_id"].tolist() == list(range(len(private_first)))
    assert set(private_first["class_id"]) == set(range(NUM_CLASSES))
    public = pd.read_csv(first / "manifest.csv")
    assert list(public.columns) == ["row_id", "relative_image_path", "domain"]
    assert "class_id" not in public and "class_name" not in public
    assert metadata["validated_all_images"] is True
    assert metadata["count_warning"] is not None


def test_manifest_rejects_missing_class(tmp_path):
    dataset = _make_dataset(tmp_path)
    missing = dataset / "Clipart" / "Class_64"
    for path in missing.iterdir():
        path.unlink()
    missing.rmdir()
    with pytest.raises(ValueError, match="class sets differ"):
        build_manifest(tmp_path, tmp_path / "output")


def test_manifest_rejects_unreadable_image(tmp_path):
    dataset = _make_dataset(tmp_path)
    broken = dataset / "Art" / "Class_00" / "image_00.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unreadable"):
        build_manifest(tmp_path, tmp_path / "output")


def _private_frame(rows_per_class: int = 5) -> pd.DataFrame:
    records = []
    row_id = 0
    for domain in CANONICAL_DOMAINS:
        for class_id in range(NUM_CLASSES):
            for item in range(rows_per_class):
                records.append({
                    "row_id": row_id,
                    "relative_image_path": f"{domain}/Class_{class_id:02d}/{item}.jpg",
                    "domain": domain,
                    "class_name": f"Class_{class_id:02d}",
                    "class_id": class_id,
                })
                row_id += 1
    return pd.DataFrame(records)


def test_heldout_split_is_stratified_and_public_pool_is_label_free(tmp_path):
    private_path = tmp_path / "manifest_private.csv"
    _private_frame().to_csv(private_path, index=False)
    output = tmp_path / "task"
    metadata = make_task_split(
        private_path,
        output,
        source_domain="Art",
        target_domain="Clip Art",
        protocol="heldout",
        seed=0,
    )
    assert metadata["counts"] == {"source": 325, "target_pool": 260, "target_test": 65}
    public = pd.read_csv(output / "target_pool_public.csv")
    oracle = pd.read_csv(output / "target_pool_oracle_private.csv")
    test = pd.read_csv(output / "target_test_private.csv")
    assert list(public.columns) == ["row_id", "relative_image_path", "domain"]
    assert set(oracle["class_id"]) == set(range(NUM_CLASSES))
    assert set(test["class_id"]) == set(range(NUM_CLASSES))
    assert set(public["row_id"]).isdisjoint(set(test["row_id"]))


def test_transductive_split_uses_full_target(tmp_path):
    private_path = tmp_path / "manifest_private.csv"
    _private_frame(rows_per_class=2).to_csv(private_path, index=False)
    metadata = make_task_split(
        private_path,
        tmp_path / "task",
        source_domain="product",
        target_domain="real_world",
        protocol="transductive",
    )
    assert metadata["counts"]["target_pool"] == 130
    assert metadata["counts"]["target_test"] == 130
    assert metadata["evaluation_scope"] == "target_full_and_unqueried"
