"""Recoverable Office-Home baseline and global-lambda campaign orchestration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import (
    CANONICAL_DOMAINS,
    atomic_write_csv,
    atomic_write_json,
    runtime_metadata,
    sha256_file,
    utc_now,
)
from .logreg import evaluate_saved_model, select_l2_source_cv, train_from_task
from .manifest import all_directed_pairs, make_task_split
from .selection import (
    PartialArtifactError,
    ReweightingConvergenceError,
    make_sample_weights,
    select_queries,
)


CAMPAIGN_ID = "officehome_round1_150q_lambda_calibration"
QUERY_BUDGET = 150
SEEDS = (0, 1, 2, 3, 4)
LAMBDA_GRID = (10.0, 100.0, 1000.0, 10000.0)
L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
BASELINE_METHODS = (
    "source_only_uniform",
    "random_uniform",
    "wasserstein_uniform",
    "wasserstein_hard_voronoi",
)
REGULARIZED_METHOD = "random_regularized_wasserstein"


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _pair_name(source: str, target: str) -> str:
    return f"{source}_to_{target}"


def _lambda_slug(value: float) -> str:
    return f"lambda_{float(value):g}".replace(".", "p")


def _method_configuration(method: str, reweight_lambda: float | None = None) -> dict[str, Any]:
    if method == "source_only_uniform":
        return {"selection": None, "weighting": "uniform", "lambda": None}
    if method == "random_uniform":
        return {"selection": "random", "weighting": "uniform", "lambda": None}
    if method == "wasserstein_uniform":
        return {"selection": "wasserstein", "weighting": "uniform", "lambda": None}
    if method == "wasserstein_hard_voronoi":
        return {"selection": "wasserstein", "weighting": "hard_voronoi", "lambda": None}
    if method == REGULARIZED_METHOD:
        if reweight_lambda is None:
            raise ValueError("Regularized-Wasserstein method requires lambda.")
        return {
            "selection": "random",
            "weighting": "regularized_wasserstein",
            "lambda": float(reweight_lambda),
        }
    raise ValueError(f"Unknown campaign method: {method}")


class CampaignRunner:
    def __init__(
        self,
        *,
        manifest_private: str | Path,
        features: str | Path,
        feature_manifest: str | Path,
        campaign_root: str | Path,
        device: str = "auto",
        query_budget: int = QUERY_BUDGET,
        classifier_max_iter: int = 500,
        classifier_tolerance: float = 1e-6,
        reweight_max_iter: int = 1024,
        l2_grid: Iterable[float] = L2_GRID,
        l2_folds: int = 3,
    ):
        self.repo_root = Path(__file__).resolve().parents[3]
        self.manifest_private = Path(manifest_private).expanduser().resolve()
        self.features = Path(features).expanduser().resolve()
        self.feature_manifest = Path(feature_manifest).expanduser().resolve()
        self.root = Path(campaign_root).expanduser().resolve()
        self.device = device
        self.query_budget = int(query_budget)
        self.classifier_max_iter = int(classifier_max_iter)
        self.classifier_tolerance = float(classifier_tolerance)
        self.reweight_max_iter = int(reweight_max_iter)
        self.l2_grid = tuple(float(value) for value in l2_grid)
        self.l2_folds = int(l2_folds)
        self.root.mkdir(parents=True, exist_ok=True)
        self.git_commit = _git_commit(self.repo_root)
        self._feature_sha256 = sha256_file(self.features)
        self._feature_manifest_sha256 = sha256_file(self.feature_manifest)
        self._manifest_private_sha256 = sha256_file(self.manifest_private)

    def task_dir(self, source: str, target: str, seed: int) -> Path:
        return self.root / "tasks" / _pair_name(source, target) / f"seed_{seed}"

    def query_dir(self, source: str, target: str, seed: int, selector: str) -> Path:
        return self.root / "queries" / _pair_name(source, target) / f"seed_{seed}" / selector

    def rho_dir(self, source: str, seed: int) -> Path:
        return self.root / "rho" / source / f"seed_{seed}"

    def run_dir(
        self,
        source: str,
        target: str,
        seed: int,
        method: str,
        reweight_lambda: float | None = None,
    ) -> Path:
        path = self.root / "runs" / _pair_name(source, target) / f"seed_{seed}" / method
        if reweight_lambda is not None:
            path = path / _lambda_slug(reweight_lambda)
        return path

    def update_state(self, **updates: Any) -> None:
        path = self.root / "campaign_state.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        else:
            state = {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "started_at_utc": utc_now(),
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "git_commit": self.git_commit,
                "command": " ".join(os.sys.argv),
            }
        state.update(updates)
        state["updated_at_utc"] = utc_now()
        atomic_write_json(path, state)

    def ensure_task(self, source: str, target: str, seed: int) -> Path:
        task_dir = self.task_dir(source, target, seed)
        metadata_path = task_dir / "task_metadata.json"
        required = {
            "source_labeled.csv",
            "target_pool_public.csv",
            "target_pool_oracle_private.csv",
            "target_test_private.csv",
            "task_metadata.json",
        }
        existing = {path.name for path in task_dir.iterdir()} if task_dir.exists() else set()
        if metadata_path.exists():
            if not required <= existing:
                raise PartialArtifactError(f"Incomplete task split at {task_dir}")
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            expected = {
                "source_domain": source,
                "target_domain": target,
                "protocol": "heldout",
                "seed": int(seed),
                "manifest_private_sha256": self._manifest_private_sha256,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise PartialArtifactError(
                        f"Task metadata mismatch at {task_dir}: {key}={metadata.get(key)!r}"
                    )
            return task_dir
        if existing:
            raise PartialArtifactError(f"Partial task split at {task_dir}")
        make_task_split(
            self.manifest_private,
            task_dir,
            source_domain=source,
            target_domain=target,
            protocol="heldout",
            seed=seed,
        )
        return task_dir

    def ensure_rho(self, source: str, target: str, seed: int, task_dir: Path) -> dict[str, Any]:
        output_dir = self.rho_dir(source, seed)
        selection_path = output_dir / "l2_selection.json"
        if selection_path.exists():
            with selection_path.open("r", encoding="utf-8") as handle:
                selection = json.load(handle)
            expected_source_sha = sha256_file(task_dir / "source_labeled.csv")
            if (
                selection.get("features_sha256") != self._feature_sha256
                or selection.get("source_manifest_sha256") != expected_source_sha
                or selection.get("seed") != int(seed)
                or selection.get("grid") != list(self.l2_grid)
            ):
                raise PartialArtifactError(f"Cached rho selection mismatch at {output_dir}")
            return selection
        if output_dir.exists() and any(output_dir.iterdir()):
            raise PartialArtifactError(f"Partial source-CV artifact at {output_dir}")
        return select_l2_source_cv(
            self.features,
            self.feature_manifest,
            task_dir / "source_labeled.csv",
            output_dir,
            grid=self.l2_grid,
            folds=self.l2_folds,
            device_name=self.device,
            max_iter=self.classifier_max_iter,
            tolerance=self.classifier_tolerance,
            seed=seed,
        )

    def ensure_query_plan(
        self,
        source: str,
        target: str,
        seed: int,
        task_dir: Path,
        selector: str | None,
    ) -> tuple[Path | None, dict[str, Any] | None]:
        if selector is None:
            return None, None
        output_dir = self.query_dir(source, target, seed, selector)
        metadata = select_queries(
            task_dir,
            self.features,
            self.feature_manifest,
            output_dir,
            method=selector,
            seed=seed,
            budget=self.query_budget,
        )
        return output_dir / "query_ids.csv", metadata

    def _complete_run_record(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("status") == "failed":
            return record
        if record.get("status") != "complete":
            raise PartialArtifactError(f"Unknown run status in {path}")
        metrics_path = Path(record["artifacts"]["metrics"]["path"])
        if (
            not metrics_path.exists()
            or record["artifacts"]["metrics"]["sha256"] != sha256_file(metrics_path)
        ):
            raise PartialArtifactError(f"Completed run record no longer validates: {path}")
        return record

    def run_method(
        self,
        source: str,
        target: str,
        seed: int,
        method: str,
        *,
        reweight_lambda: float | None = None,
    ) -> dict[str, Any]:
        configuration = _method_configuration(method, reweight_lambda)
        run_dir = self.run_dir(source, target, seed, method, reweight_lambda)
        record_path = run_dir / "run_record.json"
        existing = self._complete_run_record(record_path)
        if existing is not None:
            return existing

        run_started = time.perf_counter()
        task_dir = self.ensure_task(source, target, seed)
        rho_selection = self.ensure_rho(source, target, seed, task_dir)
        query_path, selection_metadata = self.ensure_query_plan(
            source, target, seed, task_dir, configuration["selection"]
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            reweighting_metadata = make_sample_weights(
                task_dir,
                self.features,
                self.feature_manifest,
                run_dir / "weights",
                method=configuration["weighting"],
                query_ids_path=query_path,
                reweight_lambda=configuration["lambda"],
                max_iter=self.reweight_max_iter,
            )
        except ReweightingConvergenceError as exc:
            failure = {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "created_at_utc": utc_now(),
                "status": "failed",
                "failure_stage": "reweighting",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "source_domain": source,
                "target_domain": target,
                "seed": int(seed),
                "method": method,
                "reweight_lambda": configuration["lambda"],
                "run_runtime_seconds": float(time.perf_counter() - run_started),
            }
            atomic_write_json(record_path, failure)
            return failure

        weights_path = run_dir / "weights" / "sample_weights.csv"
        training_required = [
            run_dir / "model.pt",
            run_dir / "training_metrics.json",
            run_dir / "config.json",
            run_dir / "optimization_history.csv",
        ]
        if not all(path.exists() for path in training_required):
            if (run_dir / "model.pt").exists():
                raise PartialArtifactError(f"Model exists without complete training artifacts: {run_dir}")
            train_from_task(
                self.features,
                self.feature_manifest,
                task_dir,
                run_dir,
                rho=float(rho_selection["selected_rho"]),
                query_ids_path=query_path,
                sample_weights_path=weights_path,
                device_name=self.device,
                max_iter=self.classifier_max_iter,
                tolerance=self.classifier_tolerance,
                seed=seed,
            )

        metrics_path = run_dir / "metrics.json"
        predictions_paths = [
            run_dir / "predictions.csv",
            run_dir / "predictions_transductive.csv",
        ]
        metrics = None
        if metrics_path.exists() and all(path.exists() for path in predictions_paths):
            with metrics_path.open("r", encoding="utf-8") as handle:
                candidate_metrics = json.load(handle)
            evaluation = candidate_metrics.get("evaluation") or {}
            if {
                "target_heldout",
                "target_transductive_full",
                "target_transductive_unqueried",
            } <= set(evaluation):
                metrics = candidate_metrics
        if metrics is None:
            metrics = evaluate_saved_model(
                run_dir / "model.pt",
                self.features,
                self.feature_manifest,
                task_dir,
                run_dir,
                query_ids_path=query_path,
            )

        record = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": utc_now(),
            "status": "complete",
            "source_domain": source,
            "target_domain": target,
            "pair": _pair_name(source, target),
            "protocol": "class_stratified_80_20_heldout_plus_full_target_transductive",
            "seed": int(seed),
            "query_budget": int(self.query_budget if query_path is not None else 0),
            "method": method,
            "selection_method": configuration["selection"],
            "weighting_method": configuration["weighting"],
            "rho": float(rho_selection["selected_rho"]),
            "reweight_lambda": configuration["lambda"],
            "class_balancing": False,
            "geometry_pool": "complete_target_pool_public",
            "git_commit": self.git_commit,
            "device_request": self.device,
            "runtime": runtime_metadata(),
            "run_runtime_seconds": float(time.perf_counter() - run_started),
            "hashes": {
                "manifest_private": self._manifest_private_sha256,
                "features": self._feature_sha256,
                "feature_manifest": self._feature_manifest_sha256,
                "task_metadata": sha256_file(task_dir / "task_metadata.json"),
                "rho_selection": sha256_file(self.rho_dir(source, seed) / "l2_selection.json"),
                "query_ids": None if query_path is None else sha256_file(query_path),
                "sample_weights": sha256_file(weights_path),
            },
            "selection": selection_metadata,
            "reweighting": reweighting_metadata,
            "metrics": metrics,
            "artifacts": {
                "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
                "training_metrics": {
                    "path": str(run_dir / "training_metrics.json"),
                    "sha256": sha256_file(run_dir / "training_metrics.json"),
                },
                "model": {
                    "path": str(run_dir / "model.pt"),
                    "sha256": sha256_file(run_dir / "model.pt"),
                    "local_only": True,
                },
            },
        }
        atomic_write_json(record_path, record)
        return record

    def run_smoke(self, *, source: str = "art", target: str = "clipart", seed: int = 0) -> None:
        methods = list(BASELINE_METHODS) + [REGULARIZED_METHOD]
        for method in methods:
            lam = 100.0 if method == REGULARIZED_METHOD else None
            self.update_state(
                status="running_smoke",
                current_pair=_pair_name(source, target),
                current_seed=seed,
                current_method=method,
                current_lambda=lam,
            )
            record = self.run_method(source, target, seed, method, reweight_lambda=lam)
            if record.get("status") != "complete":
                raise RuntimeError(f"Smoke method failed: {method}: {record}")
        self.aggregate(write_selection=False)
        self.update_state(status="smoke_complete")

    def run_full(self) -> None:
        pairs = all_directed_pairs()
        completed = 0
        total = len(pairs) * (len(BASELINE_METHODS) + len(LAMBDA_GRID))
        total += len(pairs) * (len(SEEDS) - 1) * (len(BASELINE_METHODS) + 1)

        # Seed 0 establishes all baselines and the global lambda calibration.
        for source, target in pairs:
            for method in BASELINE_METHODS:
                self.update_state(
                    status="running_seed0_calibration",
                    completed_runs=completed,
                    total_runs=total,
                    current_pair=_pair_name(source, target),
                    current_seed=0,
                    current_method=method,
                    current_lambda=None,
                )
                self.run_method(source, target, 0, method)
                completed += 1
            for lam in LAMBDA_GRID:
                self.update_state(
                    status="running_seed0_calibration",
                    completed_runs=completed,
                    total_runs=total,
                    current_pair=_pair_name(source, target),
                    current_seed=0,
                    current_method=REGULARIZED_METHOD,
                    current_lambda=lam,
                )
                self.run_method(
                    source, target, 0, REGULARIZED_METHOD, reweight_lambda=lam
                )
                completed += 1

        selected = self.aggregate(write_selection=True)
        if selected.get("status") != "selected":
            self.update_state(status="failed_lambda_selection", completed_runs=completed, total_runs=total)
            raise RuntimeError(f"No reliable global lambda selected: {selected}")
        selected_lambda = float(selected["selected_lambda"])

        # Seeds 1-4 use the frozen global lambda without recalibration.
        for seed in SEEDS[1:]:
            for source, target in pairs:
                for method in BASELINE_METHODS:
                    self.update_state(
                        status="running_evaluation_seeds_1_4",
                        completed_runs=completed,
                        total_runs=total,
                        selected_lambda=selected_lambda,
                        current_pair=_pair_name(source, target),
                        current_seed=seed,
                        current_method=method,
                        current_lambda=None,
                    )
                    self.run_method(source, target, seed, method)
                    completed += 1
                self.update_state(
                    status="running_evaluation_seeds_1_4",
                    completed_runs=completed,
                    total_runs=total,
                    selected_lambda=selected_lambda,
                    current_pair=_pair_name(source, target),
                    current_seed=seed,
                    current_method=REGULARIZED_METHOD,
                    current_lambda=selected_lambda,
                )
                self.run_method(
                    source,
                    target,
                    seed,
                    REGULARIZED_METHOD,
                    reweight_lambda=selected_lambda,
                )
                completed += 1

        self.aggregate(write_selection=True)
        self.write_report()
        self.update_state(
            status="complete",
            completed_runs=completed,
            total_runs=total,
            selected_lambda=selected_lambda,
            completed_at_utc=utc_now(),
        )

    def collect_records(self) -> list[dict[str, Any]]:
        records = []
        for path in sorted((self.root / "runs").glob("**/run_record.json")):
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            record["run_record_path"] = str(path)
            records.append(record)
        return records

    @staticmethod
    def _record_row(record: dict[str, Any]) -> dict[str, Any]:
        row = {
            "status": record.get("status"),
            "pair": record.get("pair") or _pair_name(
                record.get("source_domain", "unknown"), record.get("target_domain", "unknown")
            ),
            "source_domain": record.get("source_domain"),
            "target_domain": record.get("target_domain"),
            "seed": record.get("seed"),
            "method": record.get("method"),
            "rho": record.get("rho"),
            "lambda": record.get("reweight_lambda"),
            "run_runtime_seconds": record.get("run_runtime_seconds"),
            "failure_stage": record.get("failure_stage"),
            "failure_message": record.get("failure_message"),
            "run_record_path": record.get("run_record_path"),
        }
        if record.get("status") != "complete":
            return row
        evaluation = record["metrics"]["evaluation"]
        heldout = evaluation.get("target_heldout", evaluation.get("target_test", {}))
        transductive = evaluation.get("target_transductive_full", evaluation.get("target_full", {}))
        unqueried = evaluation.get(
            "target_transductive_unqueried", evaluation.get("target_unqueried", {})
        )
        for prefix, values in (
            ("heldout", heldout),
            ("transductive_full", transductive),
            ("transductive_unqueried", unqueried),
        ):
            row[f"{prefix}_cross_entropy"] = values.get("cross_entropy")
            row[f"{prefix}_top1_accuracy"] = values.get("top1_accuracy")
            row[f"{prefix}_macro_accuracy"] = values.get("macro_accuracy")
            row[f"{prefix}_num_examples"] = values.get("num_examples")
        diagnostics = record["reweighting"]["diagnostics"]
        for key in (
            "effective_sample_size",
            "squared_l2_norm",
            "max_weight",
            "source_total_mass",
            "query_total_mass",
            "solver_runtime_seconds",
        ):
            row[key] = diagnostics.get(key)
        optimization = record["metrics"]["optimization"]
        row["classifier_status"] = optimization.get("status")
        row["classifier_fit_seconds"] = optimization.get("fit_seconds")
        row["classifier_iterations"] = optimization.get("accepted_iterations")
        row["classifier_closure_evaluations"] = optimization.get("closure_evaluations")
        return row

    def aggregate(self, *, write_selection: bool) -> dict[str, Any]:
        records = self.collect_records()
        rows = [self._record_row(record) for record in records]
        summary = pd.DataFrame(rows)
        aggregate_dir = self.root / "aggregates"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(aggregate_dir / "per_run_results.csv", summary)

        complete = summary.loc[summary["status"] == "complete"].copy() if len(summary) else summary
        evaluation = complete.loc[complete["seed"].isin(SEEDS[1:])].copy() if len(complete) else complete
        metric_columns = [
            column
            for column in complete.columns
            if column.endswith(("cross_entropy", "top1_accuracy", "macro_accuracy"))
        ]
        grouped_rows = []
        if len(evaluation):
            for keys, group in evaluation.groupby(
                ["source_domain", "target_domain", "method", "lambda"], dropna=False
            ):
                row = dict(zip(["source_domain", "target_domain", "method", "lambda"], keys))
                row["coverage_seeds"] = int(group["seed"].nunique())
                for column in metric_columns:
                    row[f"{column}_mean"] = float(group[column].mean())
                    row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
                grouped_rows.append(row)
        atomic_write_csv(
            aggregate_dir / "pair_method_seeds1_4_mean_std.csv", pd.DataFrame(grouped_rows)
        )

        method_rows = []
        if len(evaluation):
            for keys, group in evaluation.groupby(["method", "lambda"], dropna=False):
                method, lam = keys
                row = {
                    "method": method,
                    "lambda": lam,
                    "coverage_runs": int(len(group)),
                    "coverage_pairs": int(group["pair"].nunique()),
                    "coverage_seeds": int(group["seed"].nunique()),
                }
                for column in metric_columns:
                    row[f"{column}_mean"] = float(group[column].mean())
                    row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
                method_rows.append(row)
        atomic_write_csv(
            aggregate_dir / "method_seeds1_4_mean_std.csv", pd.DataFrame(method_rows)
        )

        selected = {"status": "not_requested"}
        if write_selection:
            selected = self._select_global_lambda(complete, aggregate_dir)
        return selected

    def _select_global_lambda(self, complete: pd.DataFrame, aggregate_dir: Path) -> dict[str, Any]:
        calibration = complete.loc[
            (complete["method"] == REGULARIZED_METHOD)
            & (complete["seed"] == 0)
            & (complete["lambda"].isin(LAMBDA_GRID))
        ].copy()
        pairs_by_lambda = {
            float(lam): set(group["pair"])
            for lam, group in calibration.groupby("lambda")
        }
        common_pairs = set.intersection(
            *(pairs_by_lambda.get(float(lam), set()) for lam in LAMBDA_GRID)
        ) if LAMBDA_GRID else set()
        calibration = calibration.loc[calibration["pair"].isin(common_pairs)].copy()
        output_rows = []
        for _, row in calibration.sort_values(["pair", "lambda"]).iterrows():
            output_rows.append({
                "scope": "pair",
                "pair": row["pair"],
                "lambda": row["lambda"],
                "heldout_cross_entropy": row["heldout_cross_entropy"],
                "heldout_top1_accuracy": row["heldout_top1_accuracy"],
                "heldout_macro_accuracy": row["heldout_macro_accuracy"],
                "common_pair_coverage": len(common_pairs),
            })
        global_rows = []
        for lam in LAMBDA_GRID:
            group = calibration.loc[calibration["lambda"] == float(lam)]
            row = {
                "scope": "global_equal_pair_mean",
                "pair": "ALL_COMMON_PAIRS",
                "lambda": float(lam),
                "heldout_cross_entropy": float(group["heldout_cross_entropy"].mean()) if len(group) else np.nan,
                "heldout_top1_accuracy": float(group["heldout_top1_accuracy"].mean()) if len(group) else np.nan,
                "heldout_macro_accuracy": float(group["heldout_macro_accuracy"].mean()) if len(group) else np.nan,
                "common_pair_coverage": len(common_pairs),
            }
            global_rows.append(row)
            output_rows.append(row)
        atomic_write_csv(
            aggregate_dir / "lambda_calibration_seed0.csv", pd.DataFrame(output_rows)
        )

        payload = {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "selection_scope": "seed_0_equal_weighted_directed_pairs",
            "selection_metric": "mean_heldout_target_cross_entropy",
            "lambda_grid": list(LAMBDA_GRID),
            "common_valid_pairs": sorted(common_pairs),
            "common_pair_coverage": len(common_pairs),
            "required_pair_coverage": len(all_directed_pairs()),
            "lambda_0_computed": False,
            "extra_scale_ot_computed": False,
            "git_commit": self.git_commit,
        }
        finite = [row for row in global_rows if np.isfinite(row["heldout_cross_entropy"])]
        if len(common_pairs) != len(all_directed_pairs()) or len(finite) != len(LAMBDA_GRID):
            payload.update({
                "status": "insufficient_common_pair_coverage",
                "selected_lambda": None,
            })
        else:
            best_value = min(row["heldout_cross_entropy"] for row in finite)
            winners = [
                row for row in finite
                if np.isclose(row["heldout_cross_entropy"], best_value, rtol=0, atol=1e-12)
            ]
            if len(winners) != 1:
                payload.update({"status": "ambiguous_numerical_tie", "selected_lambda": None})
            else:
                payload.update({
                    "status": "selected",
                    "selected_lambda": float(winners[0]["lambda"]),
                    "selected_mean_heldout_cross_entropy": float(best_value),
                })
        atomic_write_json(aggregate_dir / "selected_lambda.json", payload)
        return payload

    def write_report(self) -> None:
        aggregate_dir = self.root / "aggregates"
        selected_path = aggregate_dir / "selected_lambda.json"
        summary_path = aggregate_dir / "per_run_results.csv"
        with selected_path.open("r", encoding="utf-8") as handle:
            selected = json.load(handle)
        summary = pd.read_csv(summary_path)
        completed = int((summary["status"] == "complete").sum())
        failed = int((summary["status"] != "complete").sum())
        expected = 336
        report_dir = Path(__file__).resolve().parents[1] / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "officehome_round1_campaign.md"
        lines = [
            "# Office-Home round-1 baseline and lambda campaign",
            "",
            "## Question",
            "",
            "Compare source-only, random, finite-pool Wasserstein, hard-Voronoi, and "
            "regularized-Wasserstein weighting across all 12 directed Office-Home tasks, "
            "and select one global lambda using seed-0 heldout cross-entropy.",
            "",
            "## Configuration",
            "",
            f"- Git commit used for execution: `{self.git_commit}`",
            "- Frozen row-wise L2-normalized ResNet-50 V1 features; 65-class weighted softmax.",
            "- Class-stratified 80/20 heldout split plus full-target transductive evaluation.",
            f"- Query budget {self.query_budget}; seeds {list(SEEDS)}; no class balancing.",
            f"- Lambda grid {list(LAMBDA_GRID)}; lambda_0 and scale-setting OT were not computed.",
            "- Selection and reweighting geometry used the complete label-free target public pool.",
            "",
            "## Results",
            "",
            f"- Completed runs: {completed}/{expected}",
            f"- Failed runs: {failed}",
            f"- Global lambda status: `{selected.get('status')}`",
            f"- Selected lambda: `{selected.get('selected_lambda')}`",
            "- Full numerical results: `results/.../aggregates/per_run_results.csv` and "
            "`lambda_calibration_seed0.csv`.",
            "",
            "## Failures",
            "",
            "See failed rows in `per_run_results.csv` and retained optimizer traces."
            if failed else "No failed run records.",
            "",
            "## Verdict",
            "",
            "The global lambda is accepted only when all four lambdas share all 12 valid "
            "directed pairs and the heldout-CE minimum is unique.",
            "",
            "## Next step",
            "",
            "Compare heldout and full-target transductive rankings, then decide whether the "
            "next campaign should expand seeds, classifier regularization, or query budgets.",
            "",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")

        index_path = report_dir / "campaign_index.csv"
        row = pd.DataFrame([{
            "campaign_id": CAMPAIGN_ID,
            "report": str(report_path.relative_to(self.repo_root)),
            "git_commit": self.git_commit,
            "status": "complete" if completed == expected and failed == 0 else "incomplete",
            "completed_runs": completed,
            "expected_runs": expected,
            "selected_lambda": selected.get("selected_lambda"),
            "updated_at_utc": utc_now(),
        }])
        if index_path.exists():
            index = pd.read_csv(index_path)
            index = index.loc[index["campaign_id"] != CAMPAIGN_ID]
            row = pd.concat([index, row], ignore_index=True)
        atomic_write_csv(index_path, row)


def campaign_metadata(runner: CampaignRunner) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": utc_now(),
        "git_commit": runner.git_commit,
        "manifest_private": str(runner.manifest_private),
        "manifest_private_sha256": runner._manifest_private_sha256,
        "features": str(runner.features),
        "features_sha256": runner._feature_sha256,
        "feature_manifest": str(runner.feature_manifest),
        "feature_manifest_sha256": runner._feature_manifest_sha256,
        "query_budget": runner.query_budget,
        "seeds": list(SEEDS),
        "directed_pairs": [_pair_name(*pair) for pair in all_directed_pairs()],
        "lambda_grid": list(LAMBDA_GRID),
        "l2_grid": list(runner.l2_grid),
        "device_request": runner.device,
        "classifier_max_iter": runner.classifier_max_iter,
        "classifier_tolerance": runner.classifier_tolerance,
        "reweight_max_iter_initial": runner.reweight_max_iter,
        "reweight_max_iter_retry": runner.reweight_max_iter * 2,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "runtime": runtime_metadata(),
    }
