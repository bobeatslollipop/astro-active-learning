"""Deterministic frozen ResNet-50 feature extraction and L2 normalization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from PIL import Image

from .common import (
    atomic_write_json,
    resolve_device,
    runtime_metadata,
    set_deterministic,
    sha256_file,
    utc_now,
)
from .manifest import locate_dataset_root


RAW_FILENAME = "resnet50_imagenet1k_v1_raw.npy"
L2_FILENAME = "resnet50_imagenet1k_v1_l2.npy"
METADATA_FILENAME = "resnet50_imagenet1k_v1_metadata.json"
FEATURE_DIMENSION = 2048
EXPECTED_MEAN = (0.485, 0.456, 0.406)
EXPECTED_STD = (0.229, 0.224, 0.225)


def validate_feature_manifest(frame: pd.DataFrame) -> None:
    required = ["row_id", "relative_image_path", "domain"]
    if list(frame.columns) != required:
        raise ValueError(f"Feature manifest columns must be exactly {required}, got {list(frame.columns)}")
    if frame["row_id"].duplicated().any():
        raise ValueError("Feature manifest contains duplicate row_id values.")
    expected = np.arange(len(frame), dtype=np.int64)
    actual = frame["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise ValueError("Feature manifest row_id values must be consecutive and match array row order.")
    if frame["relative_image_path"].duplicated().any():
        raise ValueError("Feature manifest contains duplicate relative image paths.")


class OfficeHomeImageDataset:
    def __init__(self, frame: pd.DataFrame, dataset_root: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        record = self.frame.iloc[index]
        path = self.dataset_root / record["relative_image_path"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return int(record["row_id"]), tensor


def _worker_seed(worker_id: int) -> None:
    import torch

    seed = int(torch.initial_seed() % (2**32))
    np.random.seed(seed)


def _resolved_transform(preprocess) -> dict:
    mean = tuple(float(value) for value in preprocess.mean)
    std = tuple(float(value) for value in preprocess.std)
    crop = tuple(int(value) for value in preprocess.crop_size)
    resize = tuple(int(value) for value in preprocess.resize_size)
    if mean != EXPECTED_MEAN or std != EXPECTED_STD:
        raise RuntimeError(f"Unexpected ResNet50 V1 normalization: mean={mean}, std={std}")
    if crop != (224,) or resize != (256,):
        raise RuntimeError(f"Unexpected ResNet50 V1 spatial transforms: resize={resize}, crop={crop}")
    return {
        "implementation": "torchvision ResNet50_Weights.IMAGENET1K_V1.transforms()",
        "convert_rgb": True,
        "resize_shorter_side": resize[0],
        "center_crop": [crop[0], crop[0]],
        "mean": mean,
        "std": std,
        "antialias": bool(getattr(preprocess, "antialias", True)),
        "resolved_repr": repr(preprocess),
        "augmentation": None,
    }


def _build_model_and_transform(device):
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    transform_metadata = _resolved_transform(preprocess)
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    return model, preprocess, transform_metadata


def _existing_raw_is_valid(raw_path: Path, metadata_path: Path, manifest_sha256: str) -> dict | None:
    if not raw_path.exists() or not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("manifest_sha256") != manifest_sha256:
        return None
    raw_info = metadata.get("feature_files", {}).get("raw", {})
    if raw_info.get("sha256") != sha256_file(raw_path):
        return None
    array = np.load(raw_path, mmap_mode="r")
    if array.ndim != 2 or array.shape[1] != FEATURE_DIMENSION or array.dtype != np.float32:
        return None
    return metadata


def extract_resnet50_features(
    data_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    batch_size: int = 64,
    workers: int = 4,
    seed: int = 0,
    overwrite: bool = False,
    reproducibility_rows: int = 8,
    dataset_metadata_path: str | Path | None = None,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and workers non-negative.")
    set_deterministic(seed)
    device = resolve_device(device_name)
    data_root = Path(data_root).expanduser().resolve()
    dataset_root, _ = locate_dataset_root(data_root)
    manifest_path = Path(manifest_path).expanduser().resolve()
    if dataset_metadata_path is None:
        inferred_dataset_metadata = manifest_path.parent / "dataset_metadata.json"
        dataset_metadata_path = inferred_dataset_metadata if inferred_dataset_metadata.exists() else None
    if dataset_metadata_path is not None:
        dataset_metadata_path = Path(dataset_metadata_path).expanduser().resolve()
        with dataset_metadata_path.open("r", encoding="utf-8") as handle:
            dataset_metadata = json.load(handle)
    else:
        dataset_metadata = None
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest_path)
    validate_feature_manifest(frame)
    manifest_sha256 = sha256_file(manifest_path)
    raw_path = output_dir / RAW_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    existing = _existing_raw_is_valid(raw_path, metadata_path, manifest_sha256)
    if existing is not None and not overwrite:
        print(f"Reusing verified raw features: {raw_path}")
        return existing
    if raw_path.exists() and not overwrite:
        raise FileExistsError(f"Raw feature output already exists but failed validation: {raw_path}")

    model, preprocess, transform_metadata = _build_model_and_transform(device)
    dataset = OfficeHomeImageDataset(frame, dataset_root, preprocess)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_seed,
        generator=generator,
        persistent_workers=workers > 0,
    )

    partial_path = output_dir / f".{RAW_FILENAME}.partial.npy"
    if partial_path.exists():
        partial_path.unlink()
    feature_map = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.float32, shape=(len(frame), FEATURE_DIMENSION)
    )
    start = time.perf_counter()
    written = 0
    try:
        with torch.inference_mode():
            for row_ids, images in loader:
                images = images.to(device, non_blocking=device.type == "cuda")
                features = model(images)
                if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSION:
                    raise RuntimeError(f"ResNet-50 returned feature shape {tuple(features.shape)}")
                features_np = features.detach().to(dtype=torch.float32).cpu().numpy()
                if not np.isfinite(features_np).all():
                    raise FloatingPointError("ResNet-50 produced non-finite raw features.")
                row_ids_np = row_ids.numpy().astype(np.int64, copy=False)
                expected_ids = np.arange(written, written + len(row_ids_np), dtype=np.int64)
                if not np.array_equal(row_ids_np, expected_ids):
                    raise RuntimeError("DataLoader order no longer matches manifest row_id order.")
                feature_map[written:written + len(features_np)] = features_np
                written += len(features_np)
        if written != len(frame):
            raise RuntimeError(f"Extracted {written} rows, expected {len(frame)}.")
        feature_map.flush()
        del feature_map

        check_rows = min(int(reproducibility_rows), len(dataset))
        # Recreate the complete first extraction batch so cuDNN sees the same
        # tensor shape and can use the same deterministic convolution kernels.
        # Compare only the fixed prefix requested by reproducibility_rows.
        repeat_batch_rows = min(int(batch_size), len(dataset))
        subset = torch.stack([dataset[index][1] for index in range(repeat_batch_rows)]).to(device)
        with torch.inference_mode():
            repeated_batch = model(subset).to(dtype=torch.float32).cpu().numpy()
        repeated = repeated_batch[:check_rows]
        stored = np.load(partial_path, mmap_mode="r")[:check_rows]
        if not np.allclose(repeated, stored, rtol=1e-5, atol=1e-5):
            max_error = float(np.max(np.abs(repeated - stored)))
            raise RuntimeError(f"Repeated fixed-batch extraction disagrees; max_abs_error={max_error:g}")
        max_repeat_error = float(np.max(np.abs(repeated - stored))) if check_rows else 0.0
        os.replace(partial_path, raw_path)
    finally:
        if partial_path.exists():
            partial_path.unlink()

    feature_manifest_path = output_dir / "manifest.csv"
    shutil.copyfile(manifest_path, feature_manifest_path)
    elapsed = time.perf_counter() - start
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "backbone": "torchvision.models.resnet50",
        "weights_enum": "ResNet50_Weights.IMAGENET1K_V1",
        "fine_tuned": False,
        "model_mode": "eval",
        "inference_context": "torch.inference_mode",
        "preprocessing": transform_metadata,
        "manifest_path": str(feature_manifest_path),
        "manifest_sha256": sha256_file(feature_manifest_path),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": manifest_sha256,
        "data_root": str(data_root),
        "dataset_root": str(dataset_root),
        "dataset_metadata": None if dataset_metadata is None else {
            "path": str(dataset_metadata_path),
            "sha256": sha256_file(dataset_metadata_path),
            "dataset_source": dataset_metadata.get("dataset_source"),
            "num_images": dataset_metadata.get("num_images"),
            "domain_counts": dataset_metadata.get("domain_counts"),
            "num_classes": dataset_metadata.get("num_classes"),
        },
        "feature_shape": [int(len(frame)), FEATURE_DIMENSION],
        "feature_dtype": "float32",
        "extraction_device": str(device),
        "extraction_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "extraction_seed": int(seed),
        "batch_size": int(batch_size),
        "workers": int(workers),
        "deterministic_order": True,
        "extraction_seconds": float(elapsed),
        "repeat_check": {
            "rows": int(check_rows),
            "batch_context_rows": int(repeat_batch_rows),
            "rtol": 1e-5,
            "atol": 1e-5,
            "max_abs_error": max_repeat_error,
            "passed": True,
        },
        "runtime": runtime_metadata(),
        "feature_files": {
            "raw": {
                "path": str(raw_path),
                "sha256": sha256_file(raw_path),
                "shape": [int(len(frame)), FEATURE_DIMENSION],
                "dtype": "float32",
            },
            "l2": None,
        },
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def normalize_features(
    raw_path: str | Path,
    output_dir: str | Path,
    *,
    metadata_path: str | Path | None = None,
    overwrite: bool = False,
    zero_epsilon: float = 1e-12,
    chunk_size: int = 4096,
) -> dict:
    raw_path = Path(raw_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata_path is None:
        metadata_path = output_dir / METADATA_FILENAME
    metadata_path = Path(metadata_path).expanduser().resolve()
    l2_path = output_dir / L2_FILENAME
    if l2_path.exists() and not overwrite:
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            info = metadata.get("feature_files", {}).get("l2")
            if info and info.get("sha256") == sha256_file(l2_path):
                print(f"Reusing verified L2 features: {l2_path}")
                return metadata
        raise FileExistsError(f"L2 feature output already exists but failed validation: {l2_path}")

    raw = np.load(raw_path, mmap_mode="r")
    if raw.ndim != 2 or raw.shape[1] != FEATURE_DIMENSION or raw.dtype != np.float32:
        raise ValueError(f"Raw features must have shape (n, {FEATURE_DIMENSION}) and dtype float32.")
    partial_path = output_dir / f".{L2_FILENAME}.partial.npy"
    if partial_path.exists():
        partial_path.unlink()
    normalized = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.float32, shape=raw.shape
    )
    min_norm = float("inf")
    max_norm_error = 0.0
    try:
        for start in range(0, len(raw), chunk_size):
            end = min(start + chunk_size, len(raw))
            chunk = np.asarray(raw[start:end], dtype=np.float32)
            if not np.isfinite(chunk).all():
                raise FloatingPointError(f"Raw features contain non-finite entries in rows {start}:{end}.")
            norms = np.linalg.norm(chunk.astype(np.float64), axis=1)
            min_norm = min(min_norm, float(norms.min(initial=np.inf)))
            if np.any(norms <= zero_epsilon):
                bad = np.flatnonzero(norms <= zero_epsilon) + start
                raise FloatingPointError(f"Zero or near-zero raw feature rows: {bad[:20].tolist()}")
            output = (chunk / norms[:, None]).astype(np.float32)
            if not np.isfinite(output).all():
                raise FloatingPointError("L2 normalization produced non-finite entries.")
            output_norms = np.linalg.norm(output.astype(np.float64), axis=1)
            max_norm_error = max(max_norm_error, float(np.max(np.abs(output_norms - 1.0))))
            normalized[start:end] = output
        normalized.flush()
        del normalized
        if max_norm_error > 1e-5:
            raise RuntimeError(f"Normalized row-norm error {max_norm_error:g} exceeds 1e-5.")
        os.replace(partial_path, l2_path)
    finally:
        if partial_path.exists():
            partial_path.unlink()

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    else:
        metadata = {"schema_version": 1, "created_at_utc": utc_now(), "feature_files": {}}
    metadata.setdefault("feature_files", {})["l2"] = {
        "path": str(l2_path),
        "sha256": sha256_file(l2_path),
        "shape": [int(value) for value in raw.shape],
        "dtype": "float32",
        "normalization": "row-wise L2 after frozen raw feature extraction",
        "zero_epsilon": float(zero_epsilon),
        "minimum_raw_row_norm": float(min_norm),
        "maximum_normalized_row_norm_error": float(max_norm_error),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def validate_feature_artifacts(
    raw_path: str | Path,
    l2_path: str | Path,
    manifest_path: str | Path,
) -> dict:
    frame = pd.read_csv(manifest_path)
    validate_feature_manifest(frame)
    raw = np.load(raw_path, mmap_mode="r")
    l2 = np.load(l2_path, mmap_mode="r")
    expected = (len(frame), FEATURE_DIMENSION)
    if raw.shape != expected or l2.shape != expected:
        raise ValueError(f"Feature shapes must both be {expected}; raw={raw.shape}, l2={l2.shape}")
    if raw.dtype != np.float32 or l2.dtype != np.float32:
        raise ValueError(f"Feature dtype must be float32; raw={raw.dtype}, l2={l2.dtype}")
    if not np.isfinite(raw).all() or not np.isfinite(l2).all():
        raise FloatingPointError("Feature arrays contain non-finite values.")
    norm_error = float(np.max(np.abs(np.linalg.norm(l2.astype(np.float64), axis=1) - 1.0)))
    if norm_error > 1e-5:
        raise ValueError(f"L2 row-norm error {norm_error:g} exceeds 1e-5.")
    return {
        "shape": list(expected),
        "dtype": "float32",
        "maximum_normalized_row_norm_error": norm_error,
        "raw_sha256": sha256_file(raw_path),
        "l2_sha256": sha256_file(l2_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
