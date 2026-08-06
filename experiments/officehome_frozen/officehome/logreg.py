"""Weighted multinomial logistic regression on cached Office-Home features."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold

from .common import (
    NUM_CLASSES,
    atomic_write_csv,
    atomic_write_json,
    load_single_column_ids,
    resolve_device,
    runtime_metadata,
    set_deterministic,
    sha256_file,
    utc_now,
)


L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def _load_feature_index(features_path: str | Path, feature_manifest_path: str | Path):
    features_path = Path(features_path).expanduser().resolve()
    feature_manifest_path = Path(feature_manifest_path).expanduser().resolve()
    features = np.load(features_path, mmap_mode="r")
    manifest = pd.read_csv(feature_manifest_path)
    required = {"row_id", "relative_image_path", "domain"}
    if set(manifest.columns) != required:
        raise ValueError(
            f"Feature manifest must contain exactly {sorted(required)}, got {list(manifest.columns)}"
        )
    if manifest["row_id"].duplicated().any():
        raise ValueError("Feature manifest contains duplicate row_id values.")
    if len(features) != len(manifest) or features.ndim != 2:
        raise ValueError(
            f"Feature/manifest length mismatch: features={features.shape}, manifest={len(manifest)}"
        )
    if features.dtype != np.float32:
        raise ValueError(f"Feature array dtype must be float32, got {features.dtype}.")
    if not np.isfinite(features).all():
        raise FloatingPointError("Feature array contains non-finite entries.")
    row_to_position = pd.Series(
        np.arange(len(manifest), dtype=np.int64), index=manifest["row_id"].astype(np.int64)
    )
    return features, manifest, row_to_position


def _feature_rows(features: np.ndarray, row_to_position: pd.Series, row_ids: np.ndarray) -> np.ndarray:
    missing = sorted(set(int(value) for value in row_ids) - set(int(value) for value in row_to_position.index))
    if missing:
        raise ValueError(f"Unknown row IDs in feature lookup: {missing[:20]}")
    positions = row_to_position.loc[row_ids].to_numpy(dtype=np.int64)
    return np.asarray(features[positions], dtype=np.float32)


def _read_labeled_manifest(path: str | Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["row_id", "relative_image_path", "domain", "class_name", "class_id"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} manifest is missing columns: {sorted(missing)}")
    frame = frame[required].copy()
    frame["row_id"] = frame["row_id"].astype(np.int64)
    frame["class_id"] = frame["class_id"].astype(np.int64)
    if frame["row_id"].duplicated().any():
        raise ValueError(f"{label} manifest contains duplicate row_id values.")
    if not frame["class_id"].between(0, NUM_CLASSES - 1).all():
        raise ValueError(f"{label} manifest has class IDs outside 0..{NUM_CLASSES - 1}.")
    return frame.sort_values("row_id").reset_index(drop=True)


def build_training_table(
    task_dir: str | Path,
    *,
    query_ids_path: str | Path | None = None,
    sample_weights_path: str | Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Return source plus explicitly queried target rows and normalized weights."""
    task_dir = Path(task_dir).expanduser().resolve()
    source = _read_labeled_manifest(task_dir / "source_labeled.csv", "source")
    source["training_origin"] = "source"
    query_ids = np.empty(0, dtype=np.int64)
    queried = source.iloc[0:0].copy()

    if query_ids_path is not None:
        query_ids = load_single_column_ids(query_ids_path)
        oracle = _read_labeled_manifest(
            task_dir / "target_pool_oracle_private.csv", "target-pool oracle"
        )
        oracle_by_id = oracle.set_index("row_id", drop=False)
        unknown = sorted(set(int(value) for value in query_ids) - set(oracle_by_id.index.astype(int)))
        if unknown:
            test = _read_labeled_manifest(task_dir / "target_test_private.csv", "target-test")
            test_ids = set(test["row_id"].astype(int))
            leaked = sorted(set(unknown) & test_ids)
            if leaked:
                raise ValueError(f"query IDs contain target-test rows: {leaked[:20]}")
            raise ValueError(f"query IDs are not members of the target pool: {unknown[:20]}")
        queried = oracle_by_id.loc[query_ids].reset_index(drop=True)
        queried["training_origin"] = "queried_target"

    training = pd.concat([source, queried], ignore_index=True)
    if training["row_id"].duplicated().any():
        raise ValueError("Training rows contain duplicate row IDs across source and queried target.")
    training_ids = training["row_id"].to_numpy(dtype=np.int64)

    if sample_weights_path is None:
        weights = np.full(len(training), 1.0 / len(training), dtype=np.float64)
        weight_source = "uniform"
    else:
        weight_frame = pd.read_csv(sample_weights_path)
        if set(weight_frame.columns) != {"row_id", "weight"}:
            raise ValueError("Sample-weight CSV must contain exactly row_id,weight columns.")
        if weight_frame["row_id"].duplicated().any():
            raise ValueError("Sample-weight CSV contains duplicate row_id values.")
        weight_frame["row_id"] = weight_frame["row_id"].astype(np.int64)
        values = weight_frame["weight"].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Sample weights must all be finite and nonnegative.")
        supplied_ids = set(weight_frame["row_id"].astype(int))
        expected_ids = set(int(value) for value in training_ids)
        if supplied_ids != expected_ids:
            raise ValueError(
                "Sample weights must cover every training row exactly once; "
                f"missing={sorted(expected_ids - supplied_ids)[:20]}, "
                f"extra={sorted(supplied_ids - expected_ids)[:20]}"
            )
        aligned = weight_frame.set_index("row_id").loc[training_ids, "weight"].to_numpy(dtype=np.float64)
        total = float(aligned.sum())
        if total <= 0:
            raise ValueError("Sample weights must have a positive sum.")
        weights = aligned / total
        weight_source = str(Path(sample_weights_path).expanduser().resolve())

    source_mask = training["training_origin"].to_numpy() == "source"
    summary = {
        "num_source": int(source_mask.sum()),
        "num_queried_target": int((~source_mask).sum()),
        "total_source_weight": float(weights[source_mask].sum()),
        "total_queried_target_weight": float(weights[~source_mask].sum()),
        "normalized_total_weight": float(weights.sum()),
        "weight_source": weight_source,
        "query_ids_path": None if query_ids_path is None else str(Path(query_ids_path).resolve()),
        "sample_weights_path": (
            None if sample_weights_path is None else str(Path(sample_weights_path).resolve())
        ),
    }
    return training, weights, summary


def _objective_parts(W, b, X, y, weights, rho: float):
    import torch

    logits = X @ W.T + b
    log_probabilities = torch.log_softmax(logits, dim=1)
    nll = -log_probabilities[torch.arange(len(y), device=y.device), y]
    weighted_ce = torch.sum(weights * nll)
    penalty = 0.5 * rho * (torch.sum(W * W) + torch.sum(b * b))
    return weighted_ce + penalty, weighted_ce, penalty


def optimize_weighted_logreg(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    rho: float,
    num_classes: int = NUM_CLASSES,
    device_name: str = "auto",
    max_iter: int = 500,
    tolerance: float = 1e-6,
    seed: int = 0,
    history_size: int = 100,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    """Fit full-batch softmax regression with explicit W and bias regularization."""
    import torch

    if rho <= 0 or not np.isfinite(rho):
        raise ValueError("rho must be positive and finite.")
    if max_iter <= 0 or tolerance < 0:
        raise ValueError("max_iter must be positive and tolerance nonnegative.")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if X.ndim != 2 or len(X) != len(y) or len(y) != len(weights):
        raise ValueError("X, y, and weights have incompatible shapes.")
    if len(X) == 0 or not np.isfinite(X).all():
        raise ValueError("Training features must be non-empty and finite.")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Training weights must be finite, nonnegative, and have positive sum.")
    weights = weights / weights.sum()
    if np.any(y < 0) or np.any(y >= num_classes):
        raise ValueError("Training labels are outside the configured class range.")

    set_deterministic(seed)
    device = resolve_device(device_name)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    W = torch.nn.Parameter(torch.zeros((num_classes, X.shape[1]), dtype=torch.float32, device=device))
    b = torch.nn.Parameter(torch.zeros(num_classes, dtype=torch.float32, device=device))
    optimizer = torch.optim.LBFGS(
        [W, b],
        lr=1.0,
        max_iter=1,
        max_eval=25,
        tolerance_grad=0.0,
        tolerance_change=0.0,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    closure_evaluations = 0

    def closure():
        nonlocal closure_evaluations
        optimizer.zero_grad(set_to_none=True)
        objective, _, _ = _objective_parts(W, b, X_t, y_t, weights_t, rho)
        objective.backward()
        closure_evaluations += 1
        return objective

    def snapshot(iteration: int, previous: float | None) -> tuple[dict[str, Any], float]:
        optimizer.zero_grad(set_to_none=True)
        objective, data_loss, penalty = _objective_parts(W, b, X_t, y_t, weights_t, rho)
        objective.backward()
        grad_inf = max(float(W.grad.abs().max().item()), float(b.grad.abs().max().item()))
        objective_value = float(objective.item())
        relative_change = None if previous is None else (
            abs(previous - objective_value) / max(1.0, abs(previous))
        )
        return ({
            "iteration": int(iteration),
            "closure_evaluations": int(closure_evaluations),
            "objective": objective_value,
            "weighted_cross_entropy": float(data_loss.item()),
            "l2_penalty": float(penalty.item()),
            "gradient_inf_norm": grad_inf,
            "relative_objective_change": relative_change,
        }, objective_value)

    history: list[dict[str, Any]] = []
    initial, previous_objective = snapshot(0, None)
    history.append(initial)
    status = "max_iter"
    stable_iterations = 0
    for iteration in range(1, max_iter + 1):
        optimizer.step(closure)
        record, current_objective = snapshot(iteration, previous_objective)
        history.append(record)
        if current_objective > previous_objective + 1e-6 * max(1.0, abs(previous_objective)):
            raise RuntimeError(
                f"L-BFGS accepted an objective increase at iteration {iteration}: "
                f"{previous_objective} -> {current_objective}"
            )
        if record["gradient_inf_norm"] <= tolerance:
            status = "converged_gradient"
            break
        if record["relative_objective_change"] is not None and record["relative_objective_change"] <= tolerance:
            stable_iterations += 1
        else:
            stable_iterations = 0
        if stable_iterations >= 3:
            status = "converged_objective"
            break
        previous_objective = current_objective

    W_np = W.detach().cpu().numpy().astype(np.float32, copy=True)
    b_np = b.detach().cpu().numpy().astype(np.float32, copy=True)
    history_frame = pd.DataFrame(history)
    final = history[-1]
    zero_objective = math.log(num_classes)
    norm_sq = float(np.sum(W_np.astype(np.float64) ** 2) + np.sum(b_np.astype(np.float64) ** 2))
    theoretical_bound = float(2.0 * zero_objective / rho)
    diagnostics = {
        "status": status,
        "accepted_iterations": int(history[-1]["iteration"]),
        "closure_evaluations": int(closure_evaluations),
        "max_iter": int(max_iter),
        "tolerance": float(tolerance),
        "initial_objective": float(history[0]["objective"]),
        "final_objective": float(final["objective"]),
        "final_weighted_cross_entropy": float(final["weighted_cross_entropy"]),
        "final_l2_penalty": float(final["l2_penalty"]),
        "final_gradient_inf_norm": float(final["gradient_inf_norm"]),
        "zero_classifier_objective": zero_objective,
        "objective_not_above_zero_classifier": bool(final["objective"] <= zero_objective + 1e-5),
        "parameter_norm_squared": norm_sq,
        "theoretical_norm_squared_bound": theoretical_bound,
        "theoretical_norm_bound_satisfied": bool(norm_sq <= theoretical_bound + 1e-3),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
    return {"W": W_np, "b": b_np}, history_frame, diagnostics


def _logits_numpy(model: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32) @ model["W"].T + model["b"]


def _probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    logits64 = np.asarray(logits, dtype=np.float64)
    logits64 -= logits64.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits64)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def classification_metrics(model: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    probabilities = _probabilities_from_logits(_logits_numpy(model, X))
    predictions = probabilities.argmax(axis=1)
    labels = list(range(model["W"].shape[0]))
    matrix = confusion_matrix(y, predictions, labels=labels)
    denominators = matrix.sum(axis=1)
    per_class = np.divide(
        np.diag(matrix),
        denominators,
        out=np.full(len(labels), np.nan, dtype=np.float64),
        where=denominators > 0,
    )
    return {
        "cross_entropy": float(log_loss(y, probabilities, labels=labels)),
        "top1_accuracy": float(accuracy_score(y, predictions)),
        "macro_accuracy": float(np.nanmean(per_class)),
        "classes_present": int(np.sum(denominators > 0)),
        "num_examples": int(len(y)),
    }


def _model_norms(model: dict[str, np.ndarray]) -> dict[str, float]:
    W64 = model["W"].astype(np.float64)
    b64 = model["b"].astype(np.float64)
    return {
        "W_frobenius_norm": float(np.linalg.norm(W64, ord="fro")),
        "W_operator_norm": float(np.linalg.svd(W64, compute_uv=False)[0]),
        "bias_norm": float(np.linalg.norm(b64)),
    }


def atomic_torch_save(path: str | Path, payload: dict[str, Any]) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    tmp_path = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def train_from_task(
    features_path: str | Path,
    feature_manifest_path: str | Path,
    task_dir: str | Path,
    output_dir: str | Path,
    *,
    rho: float,
    query_ids_path: str | Path | None = None,
    sample_weights_path: str | Path | None = None,
    device_name: str = "auto",
    max_iter: int = 500,
    tolerance: float = 1e-6,
    seed: int = 0,
) -> dict:
    train_started = time.perf_counter()
    features, _, row_to_position = _load_feature_index(features_path, feature_manifest_path)
    training, weights, weight_summary = build_training_table(
        task_dir,
        query_ids_path=query_ids_path,
        sample_weights_path=sample_weights_path,
    )
    row_ids = training["row_id"].to_numpy(dtype=np.int64)
    X = _feature_rows(features, row_to_position, row_ids)
    y = training["class_id"].to_numpy(dtype=np.int64)
    fit_started = time.perf_counter()
    model, history, optimization = optimize_weighted_logreg(
        X,
        y,
        weights,
        rho=rho,
        device_name=device_name,
        max_iter=max_iter,
        tolerance=tolerance,
        seed=seed,
    )
    optimization["fit_seconds"] = float(time.perf_counter() - fit_started)
    source_mask = training["training_origin"].to_numpy() == "source"
    source_metrics = classification_metrics(model, X[source_mask], y[source_mask])
    all_training_metrics = classification_metrics(model, X, y)
    norms = _model_norms(model)
    optimization["train_pipeline_seconds"] = float(time.perf_counter() - train_started)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "features": str(Path(features_path).resolve()),
        "features_sha256": sha256_file(features_path),
        "feature_manifest": str(Path(feature_manifest_path).resolve()),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "task_dir": str(Path(task_dir).resolve()),
        "rho": float(rho),
        "regularize_bias": True,
        "sample_weights_normalized_to_one": True,
        "class_balancing": False,
        "optimizer": "torch.optim.LBFGS",
        "line_search": "strong_wolfe",
        "max_iter": int(max_iter),
        "tolerance": float(tolerance),
        "seed": int(seed),
        "runtime": runtime_metadata(),
        "weight_summary": weight_summary,
    }
    metrics = {
        "schema_version": 1,
        "training": {
            "source": source_metrics,
            "source_plus_queried": all_training_metrics,
            "weighted_objective": float(optimization["final_objective"]),
            "weighted_cross_entropy": float(optimization["final_weighted_cross_entropy"]),
        },
        "counts_and_weights": weight_summary,
        "parameter_norms": norms,
        "optimization": optimization,
        "evaluation": None,
    }
    atomic_write_csv(output_dir / "optimization_history.csv", history)
    atomic_write_json(output_dir / "config.json", config)
    atomic_write_json(output_dir / "training_metrics.json", metrics)
    atomic_torch_save(output_dir / "model.pt", {
        "schema_version": 1,
        "W": model["W"],
        "b": model["b"],
        "num_classes": NUM_CLASSES,
        "feature_dimension": int(X.shape[1]),
        "rho": float(rho),
        "features_sha256": config["features_sha256"],
        "feature_manifest_sha256": config["feature_manifest_sha256"],
    })
    return metrics


def load_model(path: str | Path) -> dict[str, np.ndarray]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    W = np.asarray(payload["W"], dtype=np.float32)
    b = np.asarray(payload["b"], dtype=np.float32)
    if W.ndim != 2 or b.shape != (W.shape[0],):
        raise ValueError(f"Invalid saved model shapes: W={W.shape}, b={b.shape}")
    return {"W": W, "b": b}


def _prediction_frame(
    evaluation: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    query_ids: set[int],
) -> pd.DataFrame:
    result = evaluation[["row_id", "relative_image_path", "domain", "class_name", "class_id"]].copy()
    result.rename(columns={"class_id": "true_class_id"}, inplace=True)
    result["predicted_class_id"] = predictions
    result["correct"] = result["true_class_id"].to_numpy() == predictions
    result["is_queried"] = result["row_id"].isin(query_ids)
    true_ids = result["true_class_id"].to_numpy(dtype=np.int64)
    result["cross_entropy"] = -np.log(np.maximum(probabilities[np.arange(len(result)), true_ids], 1e-300))
    for class_id in range(probabilities.shape[1]):
        result[f"prob_class_{class_id:02d}"] = probabilities[:, class_id]
    return result


def evaluate_saved_model(
    model_path: str | Path,
    features_path: str | Path,
    feature_manifest_path: str | Path,
    task_dir: str | Path,
    output_dir: str | Path,
    *,
    query_ids_path: str | Path | None = None,
) -> dict:
    task_dir = Path(task_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    with (task_dir / "task_metadata.json").open("r", encoding="utf-8") as handle:
        task_metadata = json.load(handle)
    heldout = _read_labeled_manifest(task_dir / "target_test_private.csv", "target-test")
    target_pool_oracle = _read_labeled_manifest(
        task_dir / "target_pool_oracle_private.csv", "target-pool oracle"
    )
    target_pool_public = pd.read_csv(task_dir / "target_pool_public.csv")
    if list(target_pool_public.columns) != ["row_id", "relative_image_path", "domain"]:
        raise ValueError("Target-pool public manifest must be label-free.")
    if target_pool_public["row_id"].duplicated().any():
        raise ValueError("Target-pool public manifest contains duplicate row IDs.")
    public_ids = set(target_pool_public["row_id"].astype(int))
    oracle_ids = set(target_pool_oracle["row_id"].astype(int))
    if public_ids != oracle_ids:
        raise ValueError("Target-pool public and private-oracle row IDs do not match.")

    if task_metadata["protocol"] == "heldout":
        overlap = oracle_ids & set(heldout["row_id"].astype(int))
        if overlap:
            raise ValueError(f"Heldout and target-pool rows overlap: {sorted(overlap)[:20]}")
        target_full = pd.concat([target_pool_oracle, heldout], ignore_index=True)
        target_full.sort_values("row_id", inplace=True, ignore_index=True)
    else:
        if oracle_ids != set(heldout["row_id"].astype(int)):
            raise ValueError("Transductive target pool and target-test rows must match.")
        target_full = heldout.copy()

    features, _, row_to_position = _load_feature_index(features_path, feature_manifest_path)
    model = load_model(model_path)
    query_ids = set()
    if query_ids_path is not None:
        query_ids = set(int(value) for value in load_single_column_ids(query_ids_path))
        unknown = query_ids - public_ids
        if unknown:
            leaked = unknown & set(heldout["row_id"].astype(int))
            if leaked:
                raise ValueError(f"query IDs contain target-test rows: {sorted(leaked)[:20]}")
            raise ValueError(f"query IDs are not members of the target pool: {sorted(unknown)[:20]}")

    def evaluate_frame(frame: pd.DataFrame):
        X = _feature_rows(
            features, row_to_position, frame["row_id"].to_numpy(dtype=np.int64)
        )
        y = frame["class_id"].to_numpy(dtype=np.int64)
        probabilities = _probabilities_from_logits(_logits_numpy(model, X))
        predictions = probabilities.argmax(axis=1)
        metrics = classification_metrics(model, X, y)
        return X, y, probabilities, predictions, metrics

    _, _, heldout_probabilities, heldout_predictions, heldout_metrics = evaluate_frame(heldout)
    X_full, y_full, full_probabilities, full_predictions, full_metrics = evaluate_frame(target_full)
    unqueried_mask = ~target_full["row_id"].isin(query_ids).to_numpy()
    if not unqueried_mask.any():
        raise ValueError("No unqueried target examples remain for transductive evaluation.")
    unqueried_metrics = classification_metrics(
        model, X_full[unqueried_mask], y_full[unqueried_mask]
    )
    evaluation_metrics = {
        "protocol": task_metadata["protocol"],
        "target_transductive_full": full_metrics,
        "target_transductive_unqueried": unqueried_metrics,
    }
    if task_metadata["protocol"] == "heldout":
        evaluation_metrics["target_test"] = heldout_metrics
        evaluation_metrics["target_heldout"] = heldout_metrics
    else:
        evaluation_metrics["target_full"] = full_metrics
        evaluation_metrics["target_unqueried"] = unqueried_metrics

    heldout_predictions_frame = _prediction_frame(
        heldout, heldout_probabilities, heldout_predictions, query_ids
    )
    full_predictions_frame = _prediction_frame(
        target_full, full_probabilities, full_predictions, query_ids
    )
    atomic_write_csv(output_dir / "predictions.csv", heldout_predictions_frame)
    atomic_write_csv(output_dir / "predictions_transductive.csv", full_predictions_frame)
    training_metrics_path = output_dir / "training_metrics.json"
    metrics_path = output_dir / "metrics.json"
    if training_metrics_path.exists():
        with training_metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    elif metrics_path.exists():
        # Backward-compatible read for runs created before training metrics were
        # split from the evaluator's final output.
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    else:
        metrics = {"schema_version": 1}
    metrics["evaluation"] = evaluation_metrics
    metrics["evaluation"]["model_sha256"] = sha256_file(model_path)
    metrics["evaluation"]["predictions_sha256"] = sha256_file(output_dir / "predictions.csv")
    metrics["evaluation"]["predictions_transductive_sha256"] = sha256_file(
        output_dir / "predictions_transductive.csv"
    )
    atomic_write_json(metrics_path, metrics)
    return metrics


def select_l2_source_cv(
    features_path: str | Path,
    feature_manifest_path: str | Path,
    source_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    grid: Iterable[float] = L2_GRID,
    folds: int = 3,
    device_name: str = "auto",
    max_iter: int = 500,
    tolerance: float = 1e-6,
    seed: int = 0,
) -> dict:
    if folds < 2:
        raise ValueError("folds must be at least 2.")
    grid = sorted({float(value) for value in grid})
    if not grid or any(value <= 0 or not np.isfinite(value) for value in grid):
        raise ValueError("L2 grid values must be positive and finite.")
    features, _, row_to_position = _load_feature_index(features_path, feature_manifest_path)
    source = _read_labeled_manifest(source_manifest_path, "source")
    X = _feature_rows(features, row_to_position, source["row_id"].to_numpy(dtype=np.int64))
    y = source["class_id"].to_numpy(dtype=np.int64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    records: list[dict[str, Any]] = []
    for rho in grid:
        for fold, (train_idx, validation_idx) in enumerate(splitter.split(X, y)):
            print(
                f"[Source CV] rho={rho:g} fold={fold + 1}/{folds} "
                f"train={len(train_idx)} validation={len(validation_idx)}",
                flush=True,
            )
            fold_weights = np.full(len(train_idx), 1.0 / len(train_idx), dtype=np.float64)
            model, _, diagnostics = optimize_weighted_logreg(
                X[train_idx],
                y[train_idx],
                fold_weights,
                rho=rho,
                device_name=device_name,
                max_iter=max_iter,
                tolerance=tolerance,
                seed=seed + fold,
            )
            validation = classification_metrics(model, X[validation_idx], y[validation_idx])
            records.append({
                "rho": rho,
                "fold": int(fold),
                "train_examples": int(len(train_idx)),
                "validation_examples": int(len(validation_idx)),
                "validation_cross_entropy": validation["cross_entropy"],
                "validation_top1_accuracy": validation["top1_accuracy"],
                "validation_macro_accuracy": validation["macro_accuracy"],
                "optimization_status": diagnostics["status"],
                "accepted_iterations": diagnostics["accepted_iterations"],
                "closure_evaluations": diagnostics["closure_evaluations"],
                "final_gradient_inf_norm": diagnostics["final_gradient_inf_norm"],
            })
            print(
                f"[Source CV] rho={rho:g} fold={fold + 1}/{folds} "
                f"CE={validation['cross_entropy']:.8f} status={diagnostics['status']} "
                f"iterations={diagnostics['accepted_iterations']}",
                flush=True,
            )
    results = pd.DataFrame(records)
    summary = (
        results.groupby("rho", as_index=False)
        .agg(
            mean_validation_cross_entropy=("validation_cross_entropy", "mean"),
            std_validation_cross_entropy=("validation_cross_entropy", "std"),
            mean_validation_top1_accuracy=("validation_top1_accuracy", "mean"),
            mean_validation_macro_accuracy=("validation_macro_accuracy", "mean"),
        )
        .sort_values(["mean_validation_cross_entropy", "rho"], ascending=[True, False])
        .reset_index(drop=True)
    )
    best_loss = float(summary.iloc[0]["mean_validation_cross_entropy"])
    tied = summary.loc[np.isclose(summary["mean_validation_cross_entropy"], best_loss, rtol=0, atol=1e-8)]
    selected_rho = float(tied["rho"].max())
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "l2_cv_results.csv", results)
    atomic_write_csv(output_dir / "l2_cv_summary.csv", summary)
    payload = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "selection_scope": "source_domain_only",
        "selection_metric": "mean_validation_cross_entropy",
        "tie_break": "larger_rho_within_absolute_1e-8",
        "folds": int(folds),
        "seed": int(seed),
        "grid": grid,
        "selected_rho": selected_rho,
        "selected_mean_validation_cross_entropy": best_loss,
        "features": str(Path(features_path).resolve()),
        "features_sha256": sha256_file(features_path),
        "source_manifest": str(Path(source_manifest_path).resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "target_labels_loaded": False,
        "runtime": runtime_metadata(),
    }
    atomic_write_json(output_dir / "l2_selection.json", payload)
    return payload
