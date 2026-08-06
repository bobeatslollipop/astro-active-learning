"""Label-free Office-Home query selection and geometry-only reweighting adapters."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from .common import atomic_write_csv, atomic_write_json, sha256_file, utc_now
from .logreg import _feature_rows, _load_feature_index


PUBLIC_COLUMNS = ["row_id", "relative_image_path", "domain"]
SOURCE_COLUMNS = ["row_id", "relative_image_path", "domain", "class_name", "class_id"]
QUERY_BUDGET = 150


class PartialArtifactError(RuntimeError):
    """Raised when an existing selection/reweighting artifact is incomplete."""


class ReweightingConvergenceError(RuntimeError):
    """Raised after both regularized-Wasserstein convergence attempts fail."""


@dataclass(frozen=True)
class GeometryTask:
    source: pd.DataFrame
    target_public: pd.DataFrame
    source_features: np.ndarray
    target_features: np.ndarray
    feature_dimension: int


def _geometry_modules():
    """Import repository geometry code without coupling it to label/model logic."""
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from al_queries import query_random, query_wasserstein
    from al_reweighting import compute_voronoi_l2_weights, compute_voronoi_weights

    return query_random, query_wasserstein, compute_voronoi_weights, compute_voronoi_l2_weights


def _read_exact_columns(path: Path, columns: list[str], label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != columns:
        raise ValueError(
            f"{label} must contain exactly {columns}; got {list(frame.columns)}. "
            "Target geometry inputs must remain label-free."
        )
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise ValueError(f"{label} contains missing or duplicate row IDs.")
    frame["row_id"] = frame["row_id"].astype(np.int64)
    return frame.sort_values("row_id").reset_index(drop=True)


def load_geometry_task(
    task_dir: str | Path,
    features_path: str | Path,
    feature_manifest_path: str | Path,
) -> GeometryTask:
    """Load only source labels and the label-free public target pool."""
    task_dir = Path(task_dir).expanduser().resolve()
    source = _read_exact_columns(task_dir / "source_labeled.csv", SOURCE_COLUMNS, "source manifest")
    target_public = _read_exact_columns(
        task_dir / "target_pool_public.csv", PUBLIC_COLUMNS, "target public pool"
    )
    overlap = set(source["row_id"].astype(int)) & set(target_public["row_id"].astype(int))
    if overlap:
        raise ValueError(f"Source and public target pool overlap: {sorted(overlap)[:20]}")

    features, _, row_to_position = _load_feature_index(features_path, feature_manifest_path)
    source_features = _feature_rows(
        features, row_to_position, source["row_id"].to_numpy(dtype=np.int64)
    )
    target_features = _feature_rows(
        features, row_to_position, target_public["row_id"].to_numpy(dtype=np.int64)
    )
    return GeometryTask(
        source=source,
        target_public=target_public,
        source_features=source_features,
        target_features=target_features,
        feature_dimension=int(features.shape[1]),
    )


def _load_complete_query_artifact(
    query_path: Path,
    metadata_path: Path,
    *,
    method: str,
    seed: int,
    budget: int,
    public_ids: set[int],
) -> dict[str, Any] | None:
    existing = [query_path.exists(), metadata_path.exists()]
    if not any(existing):
        return None
    if not all(existing):
        raise PartialArtifactError(f"Partial query artifact at {query_path.parent}")
    frame = _read_exact_columns(query_path, ["row_id"], "query IDs")
    ids = set(frame["row_id"].astype(int))
    if len(frame) != budget or not ids <= public_ids:
        raise PartialArtifactError(f"Existing query IDs fail budget/public-pool validation: {query_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "selection_method": method,
        "seed": int(seed),
        "budget": int(budget),
        "query_ids_sha256": sha256_file(query_path),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise PartialArtifactError(
                f"Existing query metadata mismatch for {key}: {metadata.get(key)!r} != {value!r}"
            )
    return metadata


def select_queries(
    task_dir: str | Path,
    features_path: str | Path,
    feature_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    method: str,
    seed: int,
    budget: int = QUERY_BUDGET,
) -> dict[str, Any]:
    """Create a deterministic random or full-pool Wasserstein query plan."""
    if method not in {"random", "wasserstein"}:
        raise ValueError("selection method must be 'random' or 'wasserstein'.")
    geometry = load_geometry_task(task_dir, features_path, feature_manifest_path)
    if budget <= 0 or budget > len(geometry.target_public):
        raise ValueError(f"query budget must be in 1..{len(geometry.target_public)}")
    output_dir = Path(output_dir).expanduser().resolve()
    query_path = output_dir / "query_ids.csv"
    metadata_path = output_dir / "selection_metadata.json"
    public_ids = set(geometry.target_public["row_id"].astype(int))
    existing = _load_complete_query_artifact(
        query_path,
        metadata_path,
        method=method,
        seed=seed,
        budget=budget,
        public_ids=public_ids,
    )
    if existing is not None:
        return existing

    query_random, query_wasserstein, _, _ = _geometry_modules()
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    if method == "random":
        positions = query_random(geometry.target_features, None, budget, rng)
        objective = "uniform_without_replacement"
    else:
        positions = query_wasserstein(
            geometry.target_features,
            None,
            budget,
            rng,
            X_labeled=geometry.source_features,
            pool_size=len(geometry.target_features),
            plan_size=budget,
        )
        objective = "finite_pool_greedy_nearest_support_W1"
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) != budget or len(np.unique(positions)) != budget:
        raise RuntimeError(f"Selector returned invalid positions: shape={positions.shape}")
    if np.any(positions < 0) or np.any(positions >= len(geometry.target_public)):
        raise RuntimeError("Selector returned a position outside the public target pool.")
    query_ids = geometry.target_public.iloc[positions][["row_id"]].reset_index(drop=True)
    if not set(query_ids["row_id"].astype(int)) <= public_ids:
        raise RuntimeError("Selector returned non-public target IDs.")
    atomic_write_csv(query_path, query_ids)
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "selection_method": method,
        "selection_objective": objective,
        "seed": int(seed),
        "budget": int(budget),
        "source_support_rows": int(len(geometry.source)),
        "target_geometry_rows": int(len(geometry.target_public)),
        "target_geometry_scope": "complete_target_pool_public",
        "target_labels_loaded": False,
        "feature_dimension": geometry.feature_dimension,
        "features": str(Path(features_path).expanduser().resolve()),
        "features_sha256": sha256_file(features_path),
        "feature_manifest": str(Path(feature_manifest_path).expanduser().resolve()),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "source_manifest_sha256": sha256_file(Path(task_dir) / "source_labeled.csv"),
        "target_public_sha256": sha256_file(Path(task_dir) / "target_pool_public.csv"),
        "query_ids_path": str(query_path),
        "query_ids_sha256": sha256_file(query_path),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def _read_query_ids(query_ids_path: str | Path | None, public_ids: set[int]) -> np.ndarray:
    if query_ids_path is None:
        return np.empty(0, dtype=np.int64)
    frame = _read_exact_columns(Path(query_ids_path), ["row_id"], "query IDs")
    ids = frame["row_id"].to_numpy(dtype=np.int64)
    unknown = sorted(set(int(value) for value in ids) - public_ids)
    if unknown:
        raise ValueError(f"Query IDs are not members of the public target pool: {unknown[:20]}")
    return ids


def _weight_diagnostics(
    weights: np.ndarray,
    *,
    num_source: int,
    runtime_seconds: float,
) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=np.float64)
    squared_l2 = float(np.dot(weights, weights))
    return {
        "num_training_rows": int(len(weights)),
        "num_source_rows": int(num_source),
        "num_query_rows": int(len(weights) - num_source),
        "weight_sum": float(weights.sum()),
        "effective_sample_size": float(1.0 / squared_l2),
        "squared_l2_norm": squared_l2,
        "max_weight": float(weights.max()),
        "source_total_mass": float(weights[:num_source].sum()),
        "query_total_mass": float(weights[num_source:].sum()),
        "solver_runtime_seconds": float(runtime_seconds),
    }


def _compact_trace(trace: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    compact = {key: value for key, value in trace.items() if key != "records"}
    compact["trace_path"] = str(trace_path)
    compact["trace_sha256"] = sha256_file(trace_path)
    compact["trace_records"] = int(len(trace.get("records", [])))
    return compact


def _validate_weight_frame(
    frame: pd.DataFrame,
    support_ids: np.ndarray,
) -> np.ndarray:
    if list(frame.columns) != ["row_id", "weight"]:
        raise ValueError("Weight CSV must contain exactly row_id,weight.")
    if frame["row_id"].duplicated().any():
        raise ValueError("Weight CSV contains duplicate row IDs.")
    supplied = set(frame["row_id"].astype(int))
    expected = set(int(value) for value in support_ids)
    if supplied != expected:
        raise ValueError(
            f"Weight IDs do not match support; missing={sorted(expected - supplied)[:20]}, "
            f"extra={sorted(supplied - expected)[:20]}"
        )
    values = frame.set_index("row_id").loc[support_ids, "weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, and have positive total mass.")
    values /= values.sum()
    if not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Normalized weights do not sum to one.")
    return values


def make_sample_weights(
    task_dir: str | Path,
    features_path: str | Path,
    feature_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    method: str,
    query_ids_path: str | Path | None = None,
    reweight_lambda: float | None = None,
    max_iter: int = 1024,
) -> dict[str, Any]:
    """Create uniform, hard-Voronoi, or regularized-Wasserstein weights."""
    if method not in {"uniform", "hard_voronoi", "regularized_wasserstein"}:
        raise ValueError("Unknown weighting method.")
    if method == "regularized_wasserstein" and (
        reweight_lambda is None or reweight_lambda <= 0 or not np.isfinite(reweight_lambda)
    ):
        raise ValueError("regularized_wasserstein requires a positive finite lambda.")
    geometry = load_geometry_task(task_dir, features_path, feature_manifest_path)
    public_ids = set(geometry.target_public["row_id"].astype(int))
    query_ids = _read_query_ids(query_ids_path, public_ids)
    source_ids = geometry.source["row_id"].to_numpy(dtype=np.int64)
    support_ids = np.concatenate([source_ids, query_ids])
    if len(np.unique(support_ids)) != len(support_ids):
        raise ValueError("Source/query support contains duplicate row IDs.")

    output_dir = Path(output_dir).expanduser().resolve()
    weights_path = output_dir / "sample_weights.csv"
    metadata_path = output_dir / "reweighting_metadata.json"
    if weights_path.exists() or metadata_path.exists():
        if not (weights_path.exists() and metadata_path.exists()):
            raise PartialArtifactError(f"Partial reweighting artifact at {output_dir}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        frame = pd.read_csv(weights_path)
        _validate_weight_frame(frame, support_ids)
        expected_lambda = None if reweight_lambda is None else float(reweight_lambda)
        if (
            existing.get("status") != "complete"
            or existing.get("weighting_method") != method
            or existing.get("reweight_lambda") != expected_lambda
            or existing.get("sample_weights_sha256") != sha256_file(weights_path)
        ):
            raise PartialArtifactError(f"Existing reweighting metadata mismatch at {output_dir}")
        return existing

    _, _, compute_voronoi_weights, compute_voronoi_l2_weights = _geometry_modules()
    query_positions = pd.Series(
        np.arange(len(geometry.target_public), dtype=np.int64),
        index=geometry.target_public["row_id"].astype(np.int64),
    ).loc[query_ids].to_numpy(dtype=np.int64) if len(query_ids) else np.empty(0, dtype=np.int64)
    support_features = np.vstack([
        geometry.source_features,
        geometry.target_features[query_positions],
    ]).astype(np.float32, copy=False)
    started = time.perf_counter()
    solver_attempts: list[dict[str, Any]] = []
    status = "complete"

    if method == "uniform":
        weights = np.full(len(support_ids), 1.0 / len(support_ids), dtype=np.float64)
        solver = {"solver": "closed_form_uniform", "converged": True}
    elif method == "hard_voronoi":
        raw_weights, _ = compute_voronoi_weights(
            geometry.target_features, support_features, voronoi_state={}
        )
        weights = np.asarray(raw_weights, dtype=np.float64)
        weights /= weights.sum()
        solver = {"solver": "nearest_support_voronoi_assignment", "converged": True}
    else:
        state: dict[str, Any] = {}
        weights = None
        for attempt, ceiling in enumerate((int(max_iter), int(max_iter) * 2), start=1):
            trace_path = output_dir / f"optimizer_trace_attempt{attempt}.json"
            attempt_started = time.perf_counter()
            candidate_weights = compute_voronoi_l2_weights(
                geometry.target_features,
                support_features,
                float(reweight_lambda),
                state=state,
                max_iter=ceiling,
                trace_context={"attempt": attempt, "solve": "officehome_fixed_support"},
            )
            trace = copy.deepcopy(state["last_optimizer_trace"])
            trace["attempt_runtime_seconds"] = float(time.perf_counter() - attempt_started)
            atomic_write_json(trace_path, trace)
            solver_attempts.append(_compact_trace(trace, trace_path))
            if trace.get("converged"):
                weights = np.asarray(candidate_weights, dtype=np.float64)
                break
        if weights is None:
            status = "failed_convergence"
            failure = {
                "schema_version": 1,
                "created_at_utc": utc_now(),
                "status": status,
                "weighting_method": method,
                "reweight_lambda": float(reweight_lambda),
                "target_geometry_scope": "complete_target_pool_public",
                "target_labels_loaded": False,
                "solver_attempts": solver_attempts,
                "solver_runtime_seconds": float(time.perf_counter() - started),
            }
            atomic_write_json(metadata_path, failure)
            raise ReweightingConvergenceError(
                f"Regularized Wasserstein failed both {max_iter} and {2 * max_iter} iteration ceilings."
            )
        weights /= weights.sum()
        solver = {
            "solver": "existing_voronoi_l2_fixed_support_dual",
            "converged": True,
            "attempts": solver_attempts,
            "successful_attempt": int(len(solver_attempts)),
        }

    runtime_seconds = time.perf_counter() - started
    frame = pd.DataFrame({"row_id": support_ids, "weight": weights})
    normalized = _validate_weight_frame(frame, support_ids)
    frame["weight"] = normalized
    atomic_write_csv(weights_path, frame)
    diagnostics = _weight_diagnostics(
        normalized, num_source=len(source_ids), runtime_seconds=runtime_seconds
    )
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "status": status,
        "weighting_method": method,
        "reweight_lambda": None if reweight_lambda is None else float(reweight_lambda),
        "query_ids_path": None if query_ids_path is None else str(Path(query_ids_path).resolve()),
        "query_ids_sha256": None if query_ids_path is None else sha256_file(query_ids_path),
        "sample_weights_path": str(weights_path),
        "sample_weights_sha256": sha256_file(weights_path),
        "source_manifest_sha256": sha256_file(Path(task_dir) / "source_labeled.csv"),
        "target_public_sha256": sha256_file(Path(task_dir) / "target_pool_public.csv"),
        "features_sha256": sha256_file(features_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "target_geometry_rows": int(len(geometry.target_public)),
        "target_geometry_scope": "complete_target_pool_public",
        "target_labels_loaded": False,
        "diagnostics": diagnostics,
        "solver": solver,
    }
    atomic_write_json(metadata_path, metadata)
    return metadata
