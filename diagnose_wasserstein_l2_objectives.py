#!/usr/bin/env python3
"""Small-scale audit for Wasserstein-L2 query objectives.

This diagnostic compares the historical captured-mass score, the production
full-Voronoi-L2 score, and candidate-wise solves of the complete regularized
transport objective.  It is deliberately not registered as an active-learning
strategy: full candidate enumeration is only intended as a numerical oracle on
small problems.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.spatial.distance import cdist

from al_data import load_features_and_labels
from al_metadata import atomic_write_json, fast_hdf5_fingerprint
from al_queries import (
    WASSERSTEIN_L2_IMPLEMENTATION_VERSION,
    WASSERSTEIN_L2_QUERY_OBJECTIVE,
    _wasserstein_initial_wwds_numpy,
    _wasserstein_l2_base_cells_numpy,
    _wasserstein_l2_full_penalties_numpy,
    _wasserstein_l2_initial_capture_counts_numpy,
)


ORACLE_SCHEMA_VERSION = 1
LEGACY_OBJECTIVE = "wwd_plus_candidate_captured_mass_l2"


def _feasible_primal(pi_flat, n_target, n_support):
    """Project tiny numerical constraint violations to a feasible transport."""
    pi = np.maximum(
        np.asarray(pi_flat, dtype=np.float64).reshape(n_target, n_support), 0.0
    )
    row_sums = pi.sum(axis=1)
    missing = row_sums <= 0
    if np.any(missing):
        pi[missing] = 1.0 / n_support
        row_sums = pi.sum(axis=1)
    pi /= row_sums[:, np.newaxis]
    pi /= float(n_target)
    return pi


def _primal_objective(distances, pi, reweight_lambda):
    weights = pi.sum(axis=0)
    return float(
        np.sum(distances * pi) + reweight_lambda * np.dot(weights, weights)
    )


def _dual_lower_bound(distances, z, reweight_lambda):
    z = np.asarray(z, dtype=np.float64)
    z_pos = np.maximum(z, 0.0)
    return float(
        np.min(distances + z[np.newaxis, :], axis=1).mean()
        - np.dot(z_pos, z_pos) / (4.0 * reweight_lambda)
    )


def solve_regularized_ot_objective(
    target,
    support,
    reweight_lambda,
    *,
    max_iter=2000,
    tolerance=1e-12,
):
    """Return certified bounds for the full fixed-support regularized objective.

    The primal SLSQP solution supplies a feasible upper bound.  An independent
    L-BFGS solve of the convex dual supplies a lower bound.  Neither solver's
    success flag is treated as a certificate by itself; callers must use the
    recorded interval.
    """
    target = np.asarray(target, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    lam = float(reweight_lambda)
    if target.ndim != 2 or support.ndim != 2:
        raise ValueError("target and support must be two-dimensional arrays")
    if len(target) == 0 or len(support) == 0:
        raise ValueError("target and support must both be non-empty")
    if target.shape[1] != support.shape[1]:
        raise ValueError("target and support feature dimensions must match")
    if lam <= 0:
        raise ValueError("reweight_lambda must be positive")
    if max_iter <= 0 or tolerance <= 0:
        raise ValueError("max_iter and tolerance must be positive")

    distances = cdist(target, support, metric="euclidean")
    n_target, n_support = distances.shape
    row_mass = 1.0 / n_target

    def primal_fun(pi_flat):
        pi = pi_flat.reshape(n_target, n_support)
        weights = pi.sum(axis=0)
        objective = (
            np.sum(distances * pi) + lam * np.dot(weights, weights)
        )
        gradient = distances + 2.0 * lam * weights[np.newaxis, :]
        return float(objective), gradient.reshape(-1)

    constraint_matrix = np.zeros(
        (n_target, n_target * n_support), dtype=np.float64
    )
    for row in range(n_target):
        start = row * n_support
        constraint_matrix[row, start:start + n_support] = 1.0
    row_constraint = LinearConstraint(
        constraint_matrix,
        np.full(n_target, row_mass),
        np.full(n_target, row_mass),
    )
    primal_x0 = np.full(
        n_target * n_support, 1.0 / (n_target * n_support), dtype=np.float64
    )
    primal_result = minimize(
        primal_fun,
        primal_x0,
        jac=True,
        method="SLSQP",
        bounds=Bounds(0.0, np.inf),
        constraints=[row_constraint],
        options={"maxiter": int(max_iter), "ftol": float(tolerance), "disp": False},
    )
    primal_pi = _feasible_primal(primal_result.x, n_target, n_support)
    primal_upper = _primal_objective(distances, primal_pi, lam)
    primal_weights = primal_pi.sum(axis=0)

    def dual_fun(z):
        z_pos = np.maximum(z, 0.0)
        adjusted = distances + z[np.newaxis, :]
        assignments = adjusted.argmin(axis=1)
        min_values = adjusted[np.arange(n_target), assignments]
        counts = np.bincount(assignments, minlength=n_support).astype(np.float64)
        objective = np.dot(z_pos, z_pos) / (4.0 * lam) - min_values.mean()
        gradient = z_pos / (2.0 * lam) - counts / n_target
        return float(objective), gradient

    z0 = 2.0 * lam * primal_weights
    dual_result = minimize(
        dual_fun,
        z0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": int(max_iter),
            "ftol": float(tolerance),
            "gtol": float(tolerance),
            "maxls": 50,
        },
    )
    dual_candidates = [z0, np.asarray(dual_result.x, dtype=np.float64)]
    dual_lowers = [_dual_lower_bound(distances, z, lam) for z in dual_candidates]
    best_dual_index = int(np.argmax(dual_lowers))
    dual_lower = float(dual_lowers[best_dual_index])

    best_z = dual_candidates[best_dual_index]
    hard_assignments = (distances + best_z[np.newaxis, :]).argmin(axis=1)
    hard_pi = np.zeros((n_target, n_support), dtype=np.float64)
    hard_pi[np.arange(n_target), hard_assignments] = row_mass
    hard_upper = _primal_objective(distances, hard_pi, lam)
    primal_upper = min(primal_upper, hard_upper)

    gap = float(primal_upper - dual_lower)
    certificate_tolerance = max(1e-10, 100.0 * float(tolerance))
    return {
        "primal_upper_bound": float(primal_upper),
        "dual_lower_bound": float(dual_lower),
        "primal_dual_gap": gap,
        "certificate_valid": bool(gap >= -certificate_tolerance),
        "primal_solver": {
            "success": bool(primal_result.success),
            "status": int(primal_result.status),
            "message": str(primal_result.message),
            "iterations": int(getattr(primal_result, "nit", 0)),
        },
        "dual_solver": {
            "success": bool(dual_result.success),
            "status": int(dual_result.status),
            "message": str(dual_result.message),
            "iterations": int(getattr(dual_result, "nit", 0)),
        },
    }


def certify_exact_winner(candidate_records):
    """Certify the minimum objective when one upper bound beats all other lowers."""
    if not candidate_records:
        return {
            "status": "unresolved",
            "candidate_index": None,
            "reason": "no_candidates",
        }
    valid = [record for record in candidate_records if record["certificate_valid"]]
    if len(valid) != len(candidate_records):
        return {
            "status": "unresolved",
            "candidate_index": None,
            "reason": "invalid_candidate_certificate",
        }

    best = min(
        candidate_records,
        key=lambda record: (record["primal_upper_bound"], record["candidate_index"]),
    )
    competitor_lowers = [
        record["dual_lower_bound"]
        for record in candidate_records
        if record["candidate_index"] != best["candidate_index"]
    ]
    if not competitor_lowers:
        return {
            "status": "certified",
            "candidate_index": int(best["candidate_index"]),
            "winner_upper_bound": float(best["primal_upper_bound"]),
            "best_competitor_lower_bound": None,
            "separation": None,
        }
    competitor_lower = min(competitor_lowers, default=float("inf"))
    if best["primal_upper_bound"] < competitor_lower:
        return {
            "status": "certified",
            "candidate_index": int(best["candidate_index"]),
            "winner_upper_bound": float(best["primal_upper_bound"]),
            "best_competitor_lower_bound": float(competitor_lower),
            "separation": float(competitor_lower - best["primal_upper_bound"]),
        }
    return {
        "status": "unresolved",
        "candidate_index": None,
        "reason": "overlapping_objective_intervals",
        "lowest_upper_candidate": int(best["candidate_index"]),
        "lowest_upper_bound": float(best["primal_upper_bound"]),
        "best_competitor_lower_bound": float(competitor_lower),
    }


def _heuristic_scores(target, support, reweight_lambda):
    intra_dists = cdist(target, target, metric="euclidean").astype(np.float32)
    base_min, cell_ids, cell_counts = _wasserstein_l2_base_cells_numpy(
        target, support
    )
    transport = _wasserstein_initial_wwds_numpy(base_min, intra_dists).astype(
        np.float64
    )
    captured = _wasserstein_l2_initial_capture_counts_numpy(
        base_min, intra_dists
    ).astype(np.float64)
    legacy = transport + float(reweight_lambda) * np.square(captured / len(target))
    full_penalties = _wasserstein_l2_full_penalties_numpy(
        base_min, intra_dists, cell_ids, cell_counts
    )
    full_voronoi = transport + float(reweight_lambda) * full_penalties
    return transport, legacy, full_voronoi, full_penalties


def _regret_interval(records_by_index, selected, winner):
    if selected == winner:
        return {"lower": 0.0, "upper": 0.0}
    selected_record = records_by_index[selected]
    winner_record = records_by_index[winner]
    return {
        "lower": float(max(
            0.0,
            selected_record["dual_lower_bound"]
            - winner_record["primal_upper_bound"],
        )),
        "upper": float(max(
            0.0,
            selected_record["primal_upper_bound"]
            - winner_record["dual_lower_bound"],
        )),
    }


def compare_objectives(
    target,
    initial_support,
    reweight_lambda,
    *,
    n_pick=5,
    max_iter=2000,
    tolerance=1e-12,
):
    """Compare v1, v2, and the exact oracle along a certified exact path."""
    target = np.asarray(target, dtype=np.float64)
    support = np.asarray(initial_support, dtype=np.float64).copy()
    if len(target) == 0 or len(support) == 0:
        raise ValueError("target and initial_support must be non-empty")
    available = np.ones(len(target), dtype=bool)
    steps = []
    candidate_rows = []
    total_t0 = time.perf_counter()

    for step_index in range(min(int(n_pick), len(target))):
        transport, legacy_scores, full_scores, full_penalties = _heuristic_scores(
            target, support, reweight_lambda
        )
        legacy_masked = legacy_scores.copy()
        full_masked = full_scores.copy()
        legacy_masked[~available] = np.inf
        full_masked[~available] = np.inf
        legacy_choice = int(np.argmin(legacy_masked))
        full_choice = int(np.argmin(full_masked))

        exact_records = []
        for candidate_index in np.flatnonzero(available):
            solve_t0 = time.perf_counter()
            candidate_support = np.vstack([support, target[candidate_index]])
            record = solve_regularized_ot_objective(
                target,
                candidate_support,
                reweight_lambda,
                max_iter=max_iter,
                tolerance=tolerance,
            )
            record.update({
                "step": int(step_index + 1),
                "candidate_index": int(candidate_index),
                "transport_score": float(transport[candidate_index]),
                "legacy_score": float(legacy_scores[candidate_index]),
                "full_voronoi_score": float(full_scores[candidate_index]),
                "full_voronoi_mass_penalty": float(full_penalties[candidate_index]),
                "solve_seconds": float(time.perf_counter() - solve_t0),
            })
            exact_records.append(record)
            candidate_rows.append(record)

        exact_decision = certify_exact_winner(exact_records)
        step_record = {
            "step": int(step_index + 1),
            "support_size_before": int(len(support)),
            "available_candidates": int(available.sum()),
            "legacy_choice": legacy_choice,
            "full_voronoi_choice": full_choice,
            "exact_decision": exact_decision,
        }
        if exact_decision["status"] != "certified":
            steps.append(step_record)
            break

        exact_choice = int(exact_decision["candidate_index"])
        records_by_index = {
            record["candidate_index"]: record for record in exact_records
        }
        step_record.update({
            "exact_choice": exact_choice,
            "legacy_matches_exact": bool(legacy_choice == exact_choice),
            "full_voronoi_matches_exact": bool(full_choice == exact_choice),
            "legacy_regret_interval": _regret_interval(
                records_by_index, legacy_choice, exact_choice
            ),
            "full_voronoi_regret_interval": _regret_interval(
                records_by_index, full_choice, exact_choice
            ),
        })
        steps.append(step_record)
        available[exact_choice] = False
        support = np.vstack([support, target[exact_choice]])

    return {
        "steps": steps,
        "candidate_records": candidate_rows,
        "certified_steps": int(sum(
            step["exact_decision"]["status"] == "certified" for step in steps
        )),
        "requested_steps": int(n_pick),
        "completed_all_requested_steps": bool(
            len(steps) == min(int(n_pick), len(target))
            and all(step["exact_decision"]["status"] == "certified" for step in steps)
        ),
        "elapsed_seconds": float(time.perf_counter() - total_t0),
    }


def _write_csv(path, rows):
    columns = [
        "step", "candidate_index", "transport_score", "legacy_score",
        "full_voronoi_score", "full_voronoi_mass_penalty",
        "primal_upper_bound", "dual_lower_bound", "primal_dual_gap",
        "certificate_valid", "solve_seconds",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path, payload):
    comparison = payload["comparison"]
    lines = [
        "Wasserstein-L2 objective audit",
        f"lambda: {payload['config']['reweight_lambda']}",
        f"target/candidate size: {payload['config']['pool_size']}",
        f"initial support size: {payload['config']['support_size']}",
        f"certified steps: {comparison['certified_steps']}/{comparison['requested_steps']}",
        f"elapsed seconds: {comparison['elapsed_seconds']:.6f}",
        "",
    ]
    for step in comparison["steps"]:
        decision = step["exact_decision"]
        lines.append(
            f"step {step['step']}: legacy={step['legacy_choice']}, "
            f"full_voronoi={step['full_voronoi_choice']}, "
            f"exact={decision.get('candidate_index')}, status={decision['status']}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warm-start-file", default="bp_rp_lamost_normalized_low_teff.h5"
    )
    parser.add_argument("--full-data-file", default="bp_rp_lamost_normalized.h5")
    parser.add_argument("--feh-threshold", type=float, default=-2.0)
    parser.add_argument("--support-size", type=int, default=6)
    parser.add_argument("--pool-size", type=int, default=12)
    parser.add_argument("--n-pick", type=int, default=3)
    parser.add_argument("--reweight-lambda", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--out-dir",
        default="results/diagnostics/wasserstein_l2_exact_oracle",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in ("support_size", "pool_size", "n_pick", "max_iter"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.reweight_lambda <= 0 or args.tolerance <= 0:
        parser.error("--reweight-lambda and --tolerance must be positive")
    if args.n_pick > args.pool_size:
        parser.error("--n-pick cannot exceed --pool-size")
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    output_paths = [
        out_dir / "diagnostics.json",
        out_dir / "diagnostics.csv",
        out_dir / "summary.txt",
    ]
    if not args.overwrite and any(path.exists() for path in output_paths):
        raise FileExistsError(
            f"diagnostic output already exists in {out_dir}; use --overwrite"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    support, _, support_ids = load_features_and_labels(
        args.warm_start_file,
        args.feh_threshold,
        max_samples=args.support_size,
        seed=args.seed,
    )
    full_sample_size = max(args.pool_size * 3, args.pool_size + args.support_size)
    target_candidates, _, target_ids = load_features_and_labels(
        args.full_data_file,
        args.feh_threshold,
        max_samples=full_sample_size,
        seed=args.seed + 1,
    )
    non_overlap = ~np.isin(target_ids, support_ids)
    target_candidates = target_candidates[non_overlap][:args.pool_size]
    if len(target_candidates) < args.pool_size:
        raise RuntimeError("not enough non-overlapping target/candidate rows")

    comparison = compare_objectives(
        target_candidates,
        support,
        args.reweight_lambda,
        n_pick=args.n_pick,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
    )
    payload = {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "diagnostic": "wasserstein_l2_objective_oracle",
        "production_query_objective": WASSERSTEIN_L2_QUERY_OBJECTIVE,
        "production_query_implementation_version": WASSERSTEIN_L2_IMPLEMENTATION_VERSION,
        "legacy_query_objective": LEGACY_OBJECTIVE,
        "exact_objective": "regularized_w1_with_optimized_transport_and_weights",
        "config": {
            "warm_start_file": args.warm_start_file,
            "full_data_file": args.full_data_file,
            "feh_threshold": args.feh_threshold,
            "support_size": args.support_size,
            "pool_size": args.pool_size,
            "n_pick": args.n_pick,
            "reweight_lambda": args.reweight_lambda,
            "seed": args.seed,
            "max_iter": args.max_iter,
            "tolerance": args.tolerance,
        },
        "data": {
            "warm_start": fast_hdf5_fingerprint(args.warm_start_file),
            "full_population": fast_hdf5_fingerprint(args.full_data_file),
        },
        "comparison": comparison,
    }
    atomic_write_json(output_paths[0], payload)
    _write_csv(output_paths[1], comparison["candidate_records"])
    _write_summary(output_paths[2], payload)
    print(json.dumps({
        "out_dir": str(out_dir),
        "certified_steps": comparison["certified_steps"],
        "requested_steps": comparison["requested_steps"],
        "elapsed_seconds": comparison["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
