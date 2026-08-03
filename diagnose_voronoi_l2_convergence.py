#!/usr/bin/env python3
"""Lightweight convergence diagnostics for Voronoi-L2 reweighting.

This script reuses the matched no-class-balancing active-learning split and a
saved kmedian++ query plan.  It does not train XGBoost; it only solves the
Voronoi-L2 reweighting subproblem at a few query counts and compares the
unified production solve against additional, stricter L-BFGS iterations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from active_learning import (
    _configure_torch_runtime,
    _distance_chunk_size_torch,
    compute_voronoi_l2_weights,
    load_features_and_labels,
    query_kmedianpp,
)


def _json_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def build_full_heldout_split(args):
    X_warm, y_warm, sid_warm = load_features_and_labels(
        args.warm_start_file, args.feh_threshold, None, args.seed
    )
    X_full, y_full, sid_full = load_features_and_labels(
        args.full_data_file, args.feh_threshold, None, args.seed + 1
    )
    if sid_warm is None or sid_full is None:
        raise RuntimeError("This diagnostic expects source_id in both H5 files.")

    rng = np.random.RandomState(args.seed)
    eval_n = min(args.eval_size, len(X_full))
    eval_idx = rng.choice(len(X_full), eval_n, replace=False)
    eval_sids = sid_full[eval_idx]

    sid_warm_original = sid_warm.copy()
    warm_train_mask = ~np.isin(sid_warm, eval_sids)
    pool_mask = (~np.isin(sid_full, sid_warm_original)) & (~np.isin(sid_full, eval_sids))

    X_warm_train = X_warm[warm_train_mask].copy()
    y_warm_train = y_warm[warm_train_mask].copy()
    X_pool = X_full[pool_mask].copy()
    y_pool = y_full[pool_mask].copy()

    eval_warm_overlap = int(np.intersect1d(eval_sids, sid_warm[warm_train_mask]).size)
    eval_pool_overlap = int(np.intersect1d(eval_sids, sid_full[pool_mask]).size)
    if eval_warm_overlap != 0 or eval_pool_overlap != 0:
        raise RuntimeError(
            "full_heldout split overlap check failed: "
            f"warm={eval_warm_overlap}, pool={eval_pool_overlap}"
        )

    counts = {
        "warm_start_original": int(len(X_warm)),
        "warm_start_train": int(len(X_warm_train)),
        "warm_start_train_mp": int(np.sum(y_warm_train == 0)),
        "pool": int(len(X_pool)),
        "pool_mp": int(np.sum(y_pool == 0)),
        "eval": int(eval_n),
        "eval_mp": int(np.sum(y_full[eval_idx] == 0)),
        "eval_original_warm": int(np.isin(eval_sids, sid_warm_original).sum()),
        "eval_final_warm_overlap": eval_warm_overlap,
        "eval_query_pool_overlap": eval_pool_overlap,
    }
    return X_warm_train, y_warm_train, X_pool, y_pool, counts


def make_reweight_pool(X_warm, X_pool, args):
    total_n = len(X_warm) + len(X_pool)
    rng = np.random.RandomState(args.seed + args.trial)
    idx = rng.choice(total_n, args.reweight_pool_size, replace=False)
    warm_mask = idx < len(X_warm)
    warm_idx = idx[warm_mask]
    pool_idx = idx[~warm_mask] - len(X_warm)
    parts = []
    if len(warm_idx):
        parts.append(X_warm[warm_idx])
    if len(pool_idx):
        parts.append(X_pool[pool_idx])
    if len(parts) == 1:
        X_rw = parts[0].copy()
    else:
        X_rw = np.vstack(parts).astype(np.float32, copy=False)
    return X_rw, {
        "reweight_source_total": int(total_n),
        "reweight_pool_size": int(len(X_rw)),
        "reweight_pool_warm_rows": int(len(warm_idx)),
    }


def load_or_make_query_plan(X_pool, X_warm, args):
    plan_path = Path(args.query_plan_json)
    if plan_path.exists():
        with open(plan_path) as f:
            data = json.load(f)
        trial_plans = data["trial_query_plans"]
        pool_indices = trial_plans[args.trial]["pool_indices"]
        return np.asarray(pool_indices[: args.diagnostic_queries], dtype=np.intp), str(plan_path)

    rng = np.random.RandomState(args.seed + args.trial)
    state = {}
    plan = query_kmedianpp(X_pool, None, args.diagnostic_queries, rng, X_labeled=X_warm, state=state)
    return np.asarray(plan, dtype=np.intp), "generated"


def assignment_diagnostics_torch(X_target, X_labeled, z_np, reweight_lambda):
    import torch

    device = torch.device("cuda")
    X_t = torch.as_tensor(X_target, dtype=torch.float32, device=device).contiguous()
    X_l = torch.as_tensor(X_labeled, dtype=torch.float32, device=device).contiguous()
    z = torch.as_tensor(z_np, dtype=torch.float32, device=device).contiguous()
    target_sq = torch.sum(X_t ** 2, dim=1)
    labeled_sq = torch.sum(X_l ** 2, dim=1)

    n_target = len(X_t)
    n_labeled = len(X_l)
    chunk_size = _distance_chunk_size_torch(n_target, n_labeled, device)

    counts = torch.zeros(n_labeled, dtype=torch.float32, device=device)
    total_min = torch.zeros((), dtype=torch.float64, device=device)

    with torch.no_grad():
        for start in range(0, n_target, chunk_size):
            end = min(start + chunk_size, n_target)
            chunk = X_t[start:end]
            dists = target_sq[start:end].unsqueeze(1) + labeled_sq.unsqueeze(0)
            dists.addmm_(chunk, X_l.T, beta=1.0, alpha=-2.0)
            dists.clamp_(min=0.0).sqrt_()
            dists.add_(z.unsqueeze(0))
            min_vals, argmin_idx = torch.min(dists, dim=1)
            total_min += min_vals.double().sum()
            counts.scatter_add_(0, argmin_idx, torch.ones_like(argmin_idx, dtype=torch.float32))
            del dists

        z_pos = torch.clamp(z, min=0.0)
        u = z_pos / (2.0 * float(reweight_lambda))
        assignment_mass = counts / float(n_target)
        grad = u - assignment_mass
        objective = (
            (torch.sum(z_pos ** 2).double() / (4.0 * float(reweight_lambda)))
            - total_min / float(n_target)
        )

        u_np = u.cpu().numpy().astype(np.float64)
        a_np = assignment_mass.cpu().numpy().astype(np.float64)
        grad_np = grad.cpu().numpy().astype(np.float64)

    del X_t, X_l, z, target_sq, labeled_sq, counts
    torch.cuda.empty_cache()

    raw_sum = float(u_np.sum())
    p = u_np / raw_sum if raw_sum > 0.0 else np.ones_like(u_np) / max(1, len(u_np))
    l2_sq = float(np.dot(p, p))
    ess = float(1.0 / l2_sq) if l2_sq > 0.0 else float("inf")
    active = u_np > 0.0
    assigned = a_np > 0.0
    return {
        "dual_objective": float(objective.cpu().item()),
        "grad_inf": float(np.max(np.abs(grad_np))) if len(grad_np) else 0.0,
        "grad_l1": float(np.sum(np.abs(grad_np))),
        "grad_l2": float(np.linalg.norm(grad_np)),
        "raw_weight_sum": raw_sum,
        "raw_weight_sum_error": float(raw_sum - 1.0),
        "positive_raw_weight_count": int(np.count_nonzero(active)),
        "assigned_count": int(np.count_nonzero(assigned)),
        "assigned_mass_on_zero_weight": float(a_np[~active].sum()) if len(a_np) else 0.0,
        "effective_sample_size": ess,
        "objective_l2_norm": float(np.sqrt(l2_sq)),
        "max_mass": float(np.max(p)) if len(p) else 0.0,
        "top100_mass": float(np.partition(p, -min(100, len(p)))[-min(100, len(p)):].sum()) if len(p) else 0.0,
    }


def normalized_weight_stats(w):
    w = np.asarray(w, dtype=np.float64)
    p = w / w.sum()
    l2_sq = float(np.dot(p, p))
    return {
        "ess": float(1.0 / l2_sq),
        "l2_norm": float(np.sqrt(l2_sq)),
        "max_mass": float(np.max(p)),
    }


def cleanup_state_gpu_cache(state):
    for key in ("X_pool_t", "X_pool_sq"):
        state.pop(key, None)
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def plot_results(rows, out_dir):
    figure_dir = out_dir / "figures" / "generated"
    figure_dir.mkdir(parents=True, exist_ok=True)
    q = np.array([r["n_queries"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(q, [r["prod_grad_inf"] for r in rows], "o-", label="production")
    ax.plot(q, [r["ref_grad_inf"] for r in rows], "o-", label="production + extra")
    ax.set_xlabel("New queries")
    ax.set_ylabel("KKT/gradient infinity norm")
    ax.set_title("Voronoi-L2 KKT Residual")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "kkt_grad_inf.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(q, [r["objective_gap_to_ref"] for r in rows], "o-")
    ax.set_xlabel("New queries")
    ax.set_ylabel("Production objective - reference objective")
    ax.set_title("Extra-Iteration Objective Gap")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "objective_gap_to_reference.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(q, [r["weight_distribution_l1_diff"] for r in rows], "o-", label="L1 diff")
    ax.plot(q, [r["weight_distribution_linf_diff"] for r in rows], "o-", label="Linf diff")
    ax.set_xlabel("New queries")
    ax.set_ylabel("Difference in normalized weight distribution")
    ax.set_title("Production vs Extra-Iteration Weights")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "weight_distribution_diff.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start-file", default="bp_rp_lamost_normalized_low_teff.h5")
    parser.add_argument("--full-data-file", default="bp_rp_lamost_normalized.h5")
    parser.add_argument("--feh-threshold", type=float, default=-2.0)
    parser.add_argument("--eval-size", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--diagnostic-queries", type=int, default=60)
    parser.add_argument("--snapshot-every", type=int, default=15)
    parser.add_argument("--reweight-pool-size", type=int, default=100000)
    parser.add_argument("--reweight-lambda", type=float, default=100.0)
    parser.add_argument("--production-max-iter", type=int, default=128)
    parser.add_argument("--objective-tol", type=float, default=1e-4)
    parser.add_argument("--objective-patience", type=int, default=2)
    parser.add_argument("--gradient-tol", type=float, default=1e-5)
    parser.add_argument("--reference-extra-iter", type=int, default=32)
    parser.add_argument(
        "--query-plan-json",
        default=(
            "results/active_learning/"
            "xgb_wasserstein_l2_noclassbalance_fixed10k_fullheldout_reweightfull_"
            "150q_5seeds_eval15/al_xgb_kmedianpp_l2_150_lambda_100/query_plan_trials.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "results/diagnostics/"
            "voronoi_l2_convergence_kmedianpp_l2_lambda100_60q"
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _configure_torch_runtime()
    t0 = time.perf_counter()

    print("Loading matched full-heldout no-class-balancing split...")
    X_warm, y_warm, X_pool, y_pool, counts = build_full_heldout_split(args)
    print(f"  Warm train: {len(X_warm)} rows; pool: {len(X_pool)} rows")

    X_reweight, rw_counts = make_reweight_pool(X_warm, X_pool, args)
    print(
        "  Reweight target: "
        f"{len(X_reweight)} rows from {rw_counts['reweight_source_total']} "
        f"(warm rows={rw_counts['reweight_pool_warm_rows']})"
    )

    query_indices, query_plan_source = load_or_make_query_plan(X_pool, X_warm, args)
    query_indices = query_indices[: args.diagnostic_queries]
    print(f"  Query plan source: {query_plan_source}; using {len(query_indices)} queries")

    max_labeled = len(X_warm) + len(query_indices)
    X_labeled = np.empty((max_labeled, X_warm.shape[1]), dtype=np.float32)
    X_labeled[: len(X_warm)] = X_warm

    snapshots = list(range(0, len(query_indices) + 1, args.snapshot_every))
    if snapshots[-1] != len(query_indices):
        snapshots.append(len(query_indices))

    prod_state = {}
    rows = []
    detailed = []

    for snap_i, n_queries in enumerate(snapshots):
        if n_queries > 0:
            X_labeled[len(X_warm) : len(X_warm) + n_queries] = X_pool[query_indices[:n_queries]]
        Xl = X_labeled[: len(X_warm) + n_queries]
        prod_iter = args.production_max_iter

        print(f"\n[Snapshot q={n_queries}] production max_iter={prod_iter}; n_labeled={len(Xl)}")
        t_prod = time.perf_counter()
        w_prod = compute_voronoi_l2_weights(
            X_reweight,
            Xl,
            args.reweight_lambda,
            state=prod_state,
            max_iter=prod_iter,
            objective_tol=args.objective_tol,
            objective_patience=args.objective_patience,
            gradient_tol=args.gradient_tol,
            trace_context={
                "trial": int(args.trial + 1),
                "seed": int(args.seed + args.trial),
                "n_queries": int(n_queries),
                "solve": int(snap_i + 1),
                "diagnostic_role": "production",
            },
        )
        prod_time = time.perf_counter() - t_prod
        z_prod = prod_state["z"].copy()
        prod_diag = assignment_diagnostics_torch(X_reweight, Xl, z_prod, args.reweight_lambda)
        prod_w_stats = normalized_weight_stats(w_prod)

        print(
            f"  production: time={prod_time:.1f}s "
            f"grad_inf={prod_diag['grad_inf']:.3e} "
            f"raw_sum={prod_diag['raw_weight_sum']:.6f} "
            f"ESS={prod_diag['effective_sample_size']:.1f}"
        )

        ref_state = {"z": z_prod.copy()}
        t_ref = time.perf_counter()
        w_ref = compute_voronoi_l2_weights(
            X_reweight,
            Xl,
            args.reweight_lambda,
            state=ref_state,
            max_iter=args.reference_extra_iter,
            objective_tol=1e-12,
            objective_patience=2,
            gradient_tol=0.0,
            trace_context={
                "trial": int(args.trial + 1),
                "seed": int(args.seed + args.trial),
                "n_queries": int(n_queries),
                "solve": int(snap_i + 1),
                "diagnostic_role": "strict_reference",
            },
        )
        ref_time = time.perf_counter() - t_ref
        z_ref = ref_state["z"].copy()
        ref_diag = assignment_diagnostics_torch(X_reweight, Xl, z_ref, args.reweight_lambda)
        ref_w_stats = normalized_weight_stats(w_ref)
        cleanup_state_gpu_cache(ref_state)

        p_prod = np.asarray(w_prod, dtype=np.float64)
        p_prod = p_prod / p_prod.sum()
        p_ref = np.asarray(w_ref, dtype=np.float64)
        p_ref = p_ref / p_ref.sum()
        diff = p_prod - p_ref
        row = {
            "n_queries": int(n_queries),
            "n_labeled": int(len(Xl)),
            "production_max_iter": int(prod_iter),
            "reference_extra_iter": int(args.reference_extra_iter),
            "production_iterations_completed": int(
                prod_state["last_optimizer_trace"]["iterations_completed"]
            ),
            "production_stop_reason": prod_state["last_optimizer_trace"]["stop_reason"],
            "production_primal_dual_gap": prod_state["last_optimizer_trace"]["final_primal_dual_gap"],
            "reference_iterations_completed": int(
                ref_state["last_optimizer_trace"]["iterations_completed"]
            ),
            "reference_stop_reason": ref_state["last_optimizer_trace"]["stop_reason"],
            "reference_primal_dual_gap": ref_state["last_optimizer_trace"]["final_primal_dual_gap"],
            "production_time_s": float(prod_time),
            "reference_time_s": float(ref_time),
            "prod_dual_objective": prod_diag["dual_objective"],
            "ref_dual_objective": ref_diag["dual_objective"],
            "objective_gap_to_ref": float(prod_diag["dual_objective"] - ref_diag["dual_objective"]),
            "prod_grad_inf": prod_diag["grad_inf"],
            "ref_grad_inf": ref_diag["grad_inf"],
            "prod_grad_l1": prod_diag["grad_l1"],
            "ref_grad_l1": ref_diag["grad_l1"],
            "prod_raw_weight_sum": prod_diag["raw_weight_sum"],
            "ref_raw_weight_sum": ref_diag["raw_weight_sum"],
            "prod_ess": prod_diag["effective_sample_size"],
            "ref_ess": ref_diag["effective_sample_size"],
            "prod_returned_ess": prod_w_stats["ess"],
            "ref_returned_ess": ref_w_stats["ess"],
            "weight_distribution_l1_diff": float(np.sum(np.abs(diff))),
            "weight_distribution_l2_diff": float(np.linalg.norm(diff)),
            "weight_distribution_linf_diff": float(np.max(np.abs(diff))),
        }
        rows.append(row)
        detailed.append({"row": row, "production": prod_diag, "reference": ref_diag})

        print(
            f"  reference:  time={ref_time:.1f}s "
            f"grad_inf={ref_diag['grad_inf']:.3e} "
            f"raw_sum={ref_diag['raw_weight_sum']:.6f} "
            f"ESS={ref_diag['effective_sample_size']:.1f}"
        )
        print(
            f"  delta: objective_gap={row['objective_gap_to_ref']:.3e} "
            f"p_l1={row['weight_distribution_l1_diff']:.3e} "
            f"p_linf={row['weight_distribution_linf_diff']:.3e}"
        )

    total_time = time.perf_counter() - t0
    config = {
        "seed": args.seed,
        "trial": args.trial,
        "diagnostic_queries": args.diagnostic_queries,
        "snapshot_every": args.snapshot_every,
        "reweight_lambda": args.reweight_lambda,
        "reweight_pool_size": args.reweight_pool_size,
        "production_max_iter": args.production_max_iter,
        "objective_tolerance": args.objective_tol,
        "objective_patience": args.objective_patience,
        "gradient_tolerance": args.gradient_tol,
        "reference_extra_iter": args.reference_extra_iter,
        "reference_objective_tolerance": 1e-12,
        "reference_objective_patience": 2,
        "reference_gradient_tolerance": 0.0,
        "query_plan_source": query_plan_source,
        "total_runtime_s": total_time,
        "data_counts": counts,
        "reweight_counts": rw_counts,
    }

    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump({"config": config, "snapshots": detailed}, f, indent=2)

    with open(out_dir / "diagnostics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "summary.txt", "w") as f:
        f.write("Voronoi-L2 optimizer convergence diagnostic\n")
        f.write(json.dumps(config, indent=2))
        f.write("\n\n")
        for row in rows:
            f.write(
                f"q={row['n_queries']}: prod_grad_inf={row['prod_grad_inf']:.6g}, "
                f"ref_grad_inf={row['ref_grad_inf']:.6g}, "
                f"objective_gap={row['objective_gap_to_ref']:.6g}, "
                f"p_l1={row['weight_distribution_l1_diff']:.6g}, "
                f"p_linf={row['weight_distribution_linf_diff']:.6g}\n"
            )

    plot_results(rows, out_dir)
    print(f"\nSaved diagnostics to {out_dir}")
    print(f"Total runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
