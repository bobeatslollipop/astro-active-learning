#!/usr/bin/env python3
"""Calibrate Voronoi-L2 gap/stability stopping on the 100K warm-start split.

The diagnostic reuses saved trial-1 query plans, does not train XGBoost, and
never edits the completed active-learning family that supplies the plans.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from active_learning import _configure_torch_runtime, compute_voronoi_l2_weights
from al_metadata import atomic_write_json
from al_reweighting import _relative_primal_dual_gap
from diagnose_voronoi_l2_convergence import (
    build_full_heldout_split,
    cleanup_state_gpu_cache,
    make_reweight_pool,
    normalized_weight_stats,
)


DEFAULT_FAMILY = Path(
    "results/active_learning/"
    "xgb_noclassbalance_100k_warm_fixed10k_fullheldout_reweightfull_"
    "150q_5seeds_eval30_v2"
)


def load_trial_plan(run_dir, trial=0):
    path = Path(run_dir) / "query_plan_trials.json"
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return np.asarray(
        payload["trial_query_plans"][trial]["pool_indices"], dtype=np.intp
    ), path


def summarize_trace(trace):
    return {
        key: trace.get(key)
        for key in (
            "backend",
            "max_iter",
            "accepted_updates_completed",
            "function_evaluations",
            "converged",
            "certified",
            "stable_not_certified",
            "termination_class",
            "stop_reason",
            "initial_dual_objective",
            "final_dual_objective",
            "total_dual_improvement",
            "final_primal_dual_gap",
            "final_relative_primal_dual_gap",
            "final_grad_inf",
            "final_dual_relative_improvement_window",
            "final_normalized_weight_l1_change_window",
            "elapsed_seconds",
        )
    }


def historical_trace_summary(run_dir, n_queries, trial=0):
    path = Path(run_dir) / "optimizer_trace_trials.json"
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    solves = payload["trials"][trial]["solves"]
    solve = next(item for item in solves if int(item["n_queries"]) == n_queries)
    final_record = solve["records"][-1]
    relative_gap = solve.get("final_relative_primal_dual_gap")
    if relative_gap is None:
        relative_gap, _ = _relative_primal_dual_gap(
            final_record["primal_upper_bound"], final_record["dual_lower_bound"]
        )
    return {
        "trace_schema_version": payload.get("schema_version"),
        "accepted_updates_or_legacy_iterations": solve.get(
            "accepted_updates_completed", solve.get("iterations_completed")
        ),
        "function_evaluations": solve.get("function_evaluations"),
        "stop_reason": solve.get("stop_reason"),
        "termination_class": solve.get("termination_class", "legacy_converged"),
        "final_dual_objective": solve.get("final_dual_objective"),
        "final_primal_dual_gap": solve.get("final_primal_dual_gap"),
        "final_relative_primal_dual_gap": float(relative_gap),
        "final_grad_inf": solve.get("final_grad_inf"),
        "source": str(path),
    }


def write_tabular_outputs(out_dir, payload):
    rows = []
    for result in payload["results"]:
        trace = result["new_trace_summary"]
        old = result["historical_trace_summary"]
        rows.append({
            "strategy": result["strategy"],
            "reweight_lambda": result["reweight_lambda"],
            "n_queries": result["n_queries"],
            "termination_class": trace["termination_class"],
            "stop_reason": trace["stop_reason"],
            "accepted_updates": trace["accepted_updates_completed"],
            "function_evaluations": trace["function_evaluations"],
            "final_dual_objective": trace["final_dual_objective"],
            "final_primal_dual_gap": trace["final_primal_dual_gap"],
            "final_relative_primal_dual_gap": trace[
                "final_relative_primal_dual_gap"
            ],
            "final_grad_inf": trace["final_grad_inf"],
            "final_dual_relative_improvement_window": trace[
                "final_dual_relative_improvement_window"
            ],
            "final_normalized_weight_l1_change_window": trace[
                "final_normalized_weight_l1_change_window"
            ],
            "runtime_seconds": result["runtime_seconds"],
            "historical_stop_reason": old["stop_reason"],
            "historical_gap": old["final_primal_dual_gap"],
            "historical_relative_gap": old["final_relative_primal_dual_gap"],
            "historical_grad_inf": old["final_grad_inf"],
        })

    with open(out_dir / "diagnostics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as handle:
        handle.write("Voronoi-L2 100K gap/stability calibration\n")
        handle.write(json.dumps(payload["config"], indent=2))
        handle.write("\n\n")
        for row in rows:
            handle.write(
                f"{row['strategy']} lambda={row['reweight_lambda']:g} "
                f"q={row['n_queries']}: class={row['termination_class']} "
                f"updates={row['accepted_updates']} "
                f"rel_gap={row['final_relative_primal_dual_gap']:.6g} "
                f"grad_inf={row['final_grad_inf']:.6g} "
                f"weight_l1_window={row['final_normalized_weight_l1_change_window']} "
                f"runtime={row['runtime_seconds']:.1f}s\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warm-start-file",
        default="bp_rp_lamost_normalized_low_teff_100k_seed42.h5",
    )
    parser.add_argument("--full-data-file", default="bp_rp_lamost_normalized.h5")
    parser.add_argument("--family-root", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--out-dir", type=Path, default=Path(
        "results/diagnostics/voronoi_l2_gap_stability_100k_seed42"
    ))
    parser.add_argument("--feh-threshold", type=float, default=-2.0)
    parser.add_argument("--eval-size", type=int, default=500000)
    parser.add_argument("--reweight-pool-size", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=512)
    parser.add_argument("--relative-gap-tol", type=float, default=1e-2)
    parser.add_argument("--gradient-tol", type=float, default=1e-4)
    parser.add_argument("--stability-window", type=int, default=10)
    parser.add_argument("--dual-relative-tol", type=float, default=1e-4)
    parser.add_argument("--weight-l1-tol", type=float, default=5e-3)
    parser.add_argument("--stability-patience", type=int, default=2)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[10, 100, 1000, 10000])
    parser.add_argument("--skip-kmedian", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; launch with CUDA_VISIBLE_DEVICES=1.")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    _configure_torch_runtime()

    split_args = SimpleNamespace(
        warm_start_file=args.warm_start_file,
        full_data_file=args.full_data_file,
        feh_threshold=args.feh_threshold,
        seed=args.seed,
        trial=args.trial,
        eval_size=args.eval_size,
        reweight_pool_size=args.reweight_pool_size,
    )
    started = time.perf_counter()
    X_warm, _, X_pool, _, data_counts = build_full_heldout_split(split_args)
    X_reweight, reweight_counts = make_reweight_pool(X_warm, X_pool, split_args)

    solver_kwargs = {
        "max_iter": args.max_iter,
        "relative_gap_tol": args.relative_gap_tol,
        "gradient_tol": args.gradient_tol,
        "stability_window": args.stability_window,
        "dual_relative_tol": args.dual_relative_tol,
        "weight_l1_tol": args.weight_l1_tol,
        "stability_patience": args.stability_patience,
    }
    payload = {
        "schema_version": 1,
        "status": "running",
        "config": {
            **solver_kwargs,
            "seed": args.seed,
            "trial": args.trial,
            "warm_start_file": args.warm_start_file,
            "full_data_file": args.full_data_file,
            "family_root": str(args.family_root),
            "lambdas": args.lambdas,
            "snapshots": [0, 150],
            "kmedian_q150": not args.skip_kmedian,
        },
        "data_counts": data_counts,
        "reweight_counts": reweight_counts,
        "results": [],
    }
    atomic_write_json(args.out_dir / "diagnostics.json", payload)

    q0_z_by_lambda = {}
    for reweight_lambda in args.lambdas:
        run_name = f"al_xgb_wasserstein_l2_v2_150_lambda_{reweight_lambda:g}"
        run_dir = args.family_root / run_name
        plan, plan_path = load_trial_plan(run_dir, args.trial)
        if len(plan) < 150:
            raise RuntimeError(f"Query plan has only {len(plan)} rows: {plan_path}")

        state = {}
        for n_queries in (0, 150):
            X_labeled = (
                X_warm
                if n_queries == 0
                else np.vstack([X_warm, X_pool[plan[:n_queries]]]).astype(
                    np.float32, copy=False
                )
            )
            solve_started = time.perf_counter()
            weights = compute_voronoi_l2_weights(
                X_reweight,
                X_labeled,
                reweight_lambda,
                state=state,
                trace_context={
                    "trial_index": args.trial,
                    "trial": args.trial + 1,
                    "seed": args.seed + args.trial,
                    "n_queries": n_queries,
                    "solve": n_queries // 150 + 1,
                    "diagnostic_role": "gap_stability_calibration",
                },
                **solver_kwargs,
            )
            if n_queries == 0:
                q0_z_by_lambda[float(reweight_lambda)] = state["z"].copy()
            result = {
                "strategy": "wasserstein_l2",
                "reweight_lambda": float(reweight_lambda),
                "n_queries": n_queries,
                "query_plan_source": str(plan_path),
                "runtime_seconds": time.perf_counter() - solve_started,
                "weight_stats": normalized_weight_stats(weights),
                "historical_trace_summary": historical_trace_summary(
                    run_dir, n_queries, args.trial
                ),
                "new_trace_summary": summarize_trace(state["last_optimizer_trace"]),
                "trace": state["last_optimizer_trace"],
            }
            payload["results"].append(result)
            atomic_write_json(args.out_dir / "diagnostics.json", payload)
        cleanup_state_gpu_cache(state)

    if not args.skip_kmedian:
        run_dir = args.family_root / "al_xgb_kmedianpp_l2_150_lambda_100"
        plan, plan_path = load_trial_plan(run_dir, args.trial)
        X_labeled = np.vstack([X_warm, X_pool[plan[:150]]]).astype(
            np.float32, copy=False
        )
        state = {"z": q0_z_by_lambda[100.0].copy()}
        solve_started = time.perf_counter()
        weights = compute_voronoi_l2_weights(
            X_reweight,
            X_labeled,
            100.0,
            state=state,
            trace_context={
                "trial_index": args.trial,
                "trial": args.trial + 1,
                "seed": args.seed + args.trial,
                "n_queries": 150,
                "solve": 2,
                "diagnostic_role": "gap_stability_calibration_kmedianpp",
            },
            **solver_kwargs,
        )
        payload["results"].append({
            "strategy": "kmedianpp",
            "reweight_lambda": 100.0,
            "n_queries": 150,
            "query_plan_source": str(plan_path),
            "runtime_seconds": time.perf_counter() - solve_started,
            "weight_stats": normalized_weight_stats(weights),
            "historical_trace_summary": historical_trace_summary(
                run_dir, 150, args.trial
            ),
            "new_trace_summary": summarize_trace(state["last_optimizer_trace"]),
            "trace": state["last_optimizer_trace"],
        })
        cleanup_state_gpu_cache(state)

    payload["status"] = "completed"
    payload["total_runtime_seconds"] = time.perf_counter() - started
    atomic_write_json(args.out_dir / "diagnostics.json", payload)
    write_tabular_outputs(args.out_dir, payload)
    print(f"Saved calibration to {args.out_dir}")
    print(f"Total runtime: {payload['total_runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
