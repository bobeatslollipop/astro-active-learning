"""Single-file run metadata and reproducibility helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import uuid

import h5py
import numpy as np

from al_data import CANONICAL_LABEL_ENCODING, MP_LABEL, MR_LABEL, _feature_cols


SCHEMA_VERSION = 2
ARTIFACT_LAYOUT_VERSION = 2


ACTIVE_INPUT_SECTIONS = {
    "data": (
        "warm_start_file",
        "full_data_file",
        "feh_threshold",
        "warm_start_max",
        "pool_max",
    ),
    "split": ("eval_size", "eval_source"),
    "query": (
        "strategy",
        "total_queries",
        "eval_every",
        "wass_pool_size",
        "wass_plan_size",
        "eot_temperature",
        "moment_ridge",
    ),
    "reweighting": (
        "reweighting",
        "reweight_lambda",
        "voronoi_l2_max_iter",
        "voronoi_l2_initial_max_iter",
        "temperature",
        "soft_topk",
        "reweight_pool_size",
        "reweight_source",
        "moment_weight_iters",
    ),
    "training": (
        "model",
        "lambda_MP",
        "class_balance_mode",
        "train_weight_sum_mode",
        "train_weight_sum",
        "C",
        "ridge_alpha",
        "xgb_n_estimators",
        "xgb_max_depth",
        "xgb_learning_rate",
        "xgb_subsample",
        "xgb_colsample_bytree",
        "xgb_min_child_weight",
        "xgb_gamma",
        "xgb_reg_lambda",
        "xgb_tree_method",
        "xgb_device",
        "xgb_n_jobs",
    ),
    "trials": ("n_trials", "n_snapshots", "seed", "include_zero_snapshot"),
}

ACTIVE_DERIVED_KEYS = (
    "initial_labeled_count",
    "original_warm_start_count",
    "eval_actual_size",
    "eval_original_warm_count",
    "eval_original_warm_fraction",
    "eval_final_warm_overlap",
    "eval_query_pool_overlap",
    "query_rng_mode",
    "initial_train_weight_target_sum",
    "reweight_source_total_count",
    "reweight_source_final_warm_count",
    "reweight_source_final_warm_fraction",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def canonical_hash(payload):
    encoded = json.dumps(
        _json_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path, payload):
    """Atomically replace a JSON file so interrupted writes cannot truncate it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            json.dump(_json_ready(payload), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _sample_dataset(hasher, dataset):
    if dataset.ndim == 0:
        hasher.update(np.asarray(dataset[()]).tobytes())
        return
    n_rows = int(dataset.shape[0])
    width = min(64, n_rows)
    starts = sorted({0, max(0, n_rows // 2 - width // 2), max(0, n_rows - width)})
    for start in starts:
        hasher.update(np.asarray(dataset[start:start + width]).tobytes())


def fast_hdf5_fingerprint(path):
    """Fingerprint HDF5 structure plus deterministic Fe/H/source-id samples."""
    path = Path(path)
    stat = path.stat()
    schema = []
    hasher = hashlib.sha256()
    hasher.update(str(stat.st_size).encode("ascii"))
    with h5py.File(path, "r") as handle:
        for name in sorted(handle.keys()):
            item = handle[name]
            if isinstance(item, h5py.Dataset):
                schema.append({
                    "name": name,
                    "shape": list(item.shape),
                    "dtype": str(item.dtype),
                })
        hasher.update(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        for name in ("feh", "source_id"):
            if name in handle and isinstance(handle[name], h5py.Dataset):
                hasher.update(name.encode("utf-8"))
                _sample_dataset(hasher, handle[name])
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "fingerprint_algorithm": "sha256-size-schema-feh-source-samples-v1",
        "fingerprint": "sha256:" + hasher.hexdigest(),
    }


def hdf5_feature_columns(path):
    with h5py.File(path, "r") as handle:
        return _feature_cols(list(handle.keys()))


def class_counts(labels):
    labels = np.asarray(labels)
    n_mp = int(np.sum(labels == MP_LABEL))
    n_mr = int(np.sum(labels == MR_LABEL))
    return {
        "total": int(len(labels)),
        "mp": n_mp,
        "mr": n_mr,
        "mp_fraction": float(n_mp / max(len(labels), 1)),
    }


def _package_versions():
    packages = (
        "numpy",
        "scipy",
        "scikit-learn",
        "h5py",
        "torch",
        "xgboost",
        "matplotlib",
        "umap-learn",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def environment_metadata():
    cuda_available = False
    cuda_devices = []
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": _package_versions(),
        "cuda_available": cuda_available,
        "cuda_devices": cuda_devices,
    }


def git_metadata(cwd=None):
    cwd = str(cwd or Path(__file__).resolve().parent)

    def run(*args):
        result = subprocess.run(
            ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def experiment_family(out_dir, experiment_type):
    parts = Path(out_dir).parts
    marker = "active_learning" if experiment_type == "active_learning" else "full_data"
    if marker in parts:
        index = parts.index(marker)
        if index + 1 < len(parts):
            return parts[index + 1]
    parent = Path(out_dir).parent.name
    return parent or "ad_hoc"


def _select(values, names):
    return {name: _json_ready(values[name]) for name in names if name in values}


def build_active_params(args, *, y_warm, y_pool, y_eval, data_load_seconds):
    """Build the complete schema-v2 active-learning params payload."""
    values = vars(args).copy()
    sections = {
        name: _select(values, keys) for name, keys in ACTIVE_INPUT_SECTIONS.items()
    }
    assigned = {key for keys in ACTIVE_INPUT_SECTIONS.values() for key in keys}
    extras = {
        key: _json_ready(value)
        for key, value in values.items()
        if key not in assigned and key not in ACTIVE_DERIVED_KEYS and key != "out_dir"
    }
    if extras:
        sections["other_inputs"] = extras

    derived = _select(values, ACTIVE_DERIVED_KEYS)
    feature_columns = hdf5_feature_columns(args.full_data_file)
    data = {
        "inputs": sections["data"],
        "warm_start": fast_hdf5_fingerprint(args.warm_start_file),
        "full_population": fast_hdf5_fingerprint(args.full_data_file),
        "feature_columns": feature_columns,
        "feature_count": int(len(feature_columns)),
        "feh_threshold": float(args.feh_threshold),
        "label_encoding": dict(CANONICAL_LABEL_ENCODING),
    }
    split = {
        **sections.pop("split"),
        "actual": {
            "warm_start": class_counts(y_warm),
            "query_pool": class_counts(y_pool),
            "eval": class_counts(y_eval),
        },
        "derived": derived,
    }

    scientific_config = {
        "data": {
            "warm_start_fingerprint": data["warm_start"]["fingerprint"],
            "full_population_fingerprint": data["full_population"]["fingerprint"],
            **sections["data"],
        },
        "split": {k: v for k, v in split.items() if k != "actual"},
        "query": sections["query"],
        "reweighting": sections["reweighting"],
        "training": sections["training"],
        "trials": sections["trials"],
        "other_inputs": sections.get("other_inputs", {}),
    }
    protocol_derived = {
        key: value
        for key, value in derived.items()
        if key != "query_rng_mode"
    }
    protocol_config = {
        "data": scientific_config["data"],
        "split": {
            **{key: value for key, value in split.items() if key not in {"actual", "derived"}},
            "actual": split["actual"],
            "derived": protocol_derived,
        },
        "query_budget": _select(
            sections["query"], ("total_queries", "eval_every")
        ),
        "reweight_target": _select(
            sections["reweighting"], ("reweight_source", "reweight_pool_size")
        ),
        "training": sections["training"],
        "trials": sections["trials"],
    }

    created_at = utc_now()
    run_id = created_at.replace(":", "").replace("+00:00", "Z") + "-" + uuid.uuid4().hex[:8]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_layout_version": ARTIFACT_LAYOUT_VERSION,
        "experiment_type": "active_learning",
        "run": {
            "run_id": run_id,
            "experiment_family": experiment_family(args.out_dir, "active_learning"),
            "output_dir": str(args.out_dir),
            "argv": list(sys.argv),
            "status": "running",
            "created_at_utc": created_at,
            "completed_at_utc": None,
            "git": git_metadata(),
            "config_hash": canonical_hash(scientific_config),
            "protocol_id": canonical_hash(protocol_config),
        },
        "data": data,
        "split": split,
        "query": sections["query"],
        "reweighting": sections["reweighting"],
        "training": sections["training"],
        "trials": sections["trials"],
        "other_inputs": sections.get("other_inputs", {}),
        "environment": environment_metadata(),
        "timing": {
            "data_load_seconds": float(data_load_seconds),
            "total_seconds": None,
        },
        "failure": None,
    }


def params_path(out_dir):
    return Path(out_dir) / "params.json"


def write_params(out_dir, payload):
    atomic_write_json(params_path(out_dir), payload)


def update_params_status(out_dir, status, *, total_seconds=None, error=None):
    path = params_path(out_dir)
    if not path.exists():
        return False
    with path.open() as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        return False
    payload["run"]["status"] = status
    if status in {"completed", "failed"}:
        payload["run"]["completed_at_utc"] = utc_now()
    if total_seconds is not None:
        payload.setdefault("timing", {})["total_seconds"] = float(total_seconds)
    if error is not None:
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    write_params(out_dir, payload)
    return True
