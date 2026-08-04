#!/usr/bin/env python3
"""Active-learning CLI and compatibility import surface.

The implementation is split across al_data, al_queries, al_reweighting,
al_models, al_reporting, and al_runner. Existing commands can continue to
invoke this file, and the public helpers historically imported from
active_learning remain re-exported here.
"""

import argparse

from al_data import *
from al_data import _configure_torch_runtime, _feature_cols, _json_scalar, _nsort, _timing
from al_models import *
from al_queries import *
from al_reporting import *
from al_reweighting import *
from al_reweighting import _distance_chunk_size_torch
from al_runner import run_active_learning
from al_metadata import update_params_status


def main():
    p = argparse.ArgumentParser(description="Active Learning with warm-start for stellar Fe/H classification.")
    a = p.add_argument

    # Data
    a("--warm-start-file", default="bp_rp_lamost_normalized_low_teff.h5", help="H5 file for biased warm-start set.")
    a("--full-data-file",  default="bp_rp_lamost_normalized.h5",          help="H5 file for full population.")
    a("--feh-threshold",   type=float, default=-2.0, help="Fe/H threshold: <thr → MP(0), >=thr → MR(1).")

    # Strategy
    a("--strategy",       default="uncertainty", choices=list(STRATEGIES.keys()), help="Query strategy.")
    a("--total-queries",  type=int, default=3000, help="Total points to query.")
    a("--eval-every",     type=int, default=200,  help="Retrain & evaluate every k queries.")

    # Model
    a("--model", default="logistic", choices=["logistic", "ridge", "xgboost"],
      help="Final classifier: logistic regression, ridge-regression classifier, or XGBoost boosted trees.")
    a("--lambda-MP", type=float, default=1.0, help="Desired total-weight ratio MP/MR. Per-sample weights are auto-scaled so n_MP*w_MP / n_MR*w_MR = lambda_MP.")
    a("--class-balance-mode", default="ratio", choices=["ratio", "none"],
      help="How final training weights treat MP/MR totals. ratio preserves the "
           "historical behavior by forcing total MP/MR weight ratio --lambda-MP. "
           "none preserves raw sample_weight ratios and only normalizes the total "
           "weight sum.")
    a("--train-weight-sum-mode", default="fixed",
      choices=["fixed", "initial_labeled", "current_labeled"],
      help="How to set the total sum of final training sample weights. "
           "fixed uses --train-weight-sum; initial_labeled uses the post-heldout "
           "warm-start labeled count for all snapshots; current_labeled uses the "
           "current labeled count at each snapshot.")
    a("--train-weight-sum", type=float, default=DEFAULT_TRAIN_WEIGHT_SUM,
      help="Target total final training sample-weight sum when --train-weight-sum-mode=fixed.")
    a("--C",         type=float, default=1.0, help="Inverse regularisation strength.")
    a("--ridge-alpha", type=float, default=1.0,
      help="L2 regularisation strength for --model=ridge. Larger values mean stronger ridge regularization.")
    a("--xgb-n-estimators", type=int, default=400,
      help="Number of boosted trees for --model=xgboost. The paper searched 100..1200.")
    a("--xgb-max-depth", type=int, default=6,
      help="Maximum tree depth for --model=xgboost. The paper searched 2..15.")
    a("--xgb-learning-rate", type=float, default=0.1,
      help="Learning rate eta for --model=xgboost. The paper searched 0.05..1.")
    a("--xgb-subsample", type=float, default=0.8,
      help="Row subsample fraction for --model=xgboost. The paper searched 0.5..1.")
    a("--xgb-colsample-bytree", type=float, default=0.8,
      help="Column subsample fraction per tree for --model=xgboost. The paper searched 0.3..0.9.")
    a("--xgb-min-child-weight", type=float, default=1.0,
      help="Minimum child weight for --model=xgboost. The paper searched 1..20.")
    a("--xgb-gamma", type=float, default=0.0,
      help="Minimum loss reduction required for a split for --model=xgboost. The paper searched 0..0.7.")
    a("--xgb-reg-lambda", type=float, default=1.0,
      help="XGBoost L2 regularization on leaf weights for --model=xgboost.")
    a("--xgb-tree-method", default="hist",
      help="XGBoost tree_method, e.g. hist. For XGBoost >=2 use --xgb-device cuda for GPU.")
    a("--xgb-device", default="auto",
      help="XGBoost device, e.g. auto, cpu, cuda, or cuda:0. Use cuda with tree_method=hist for XGBoost >=2.")
    a("--xgb-n-jobs", type=int, default=-1,
      help="Parallel workers for --model=xgboost.")
    a("--reweighting", default="none", choices=["none", "hard", "soft", "voronoi_l2", "kl", "moment_l2"],
       help="Covariate-shift correction: none=uniform, hard=Voronoi assignment, soft=temperature softmin, voronoi_l2/kl=regularized Wasserstein final weights, moment_l2=linear second-moment weights.")
    a("--reweight-lambda", type=float, default=1.0,
       help="Regularisation strength lambda for voronoi_l2, kl, or moment_l2 reweighting.")
    a("--voronoi-l2-max-iter", type=int, default=512,
      help="Maximum accepted L-BFGS updates for each voronoi_l2 reweighting solve.")
    a("--voronoi-l2-relative-gap-tol", type=float, default=1e-2,
       help="Certified relative primal-dual gap tolerance for voronoi_l2 convergence.")
    a("--voronoi-l2-gradient-tol", type=float, default=1e-4,
       help="Certified gradient infinity-norm tolerance for voronoi_l2 convergence.")
    a("--voronoi-l2-stability-window", type=int, default=10,
       help="Accepted-update window used for non-certified voronoi_l2 stability checks.")
    a("--voronoi-l2-dual-relative-tol", type=float, default=1e-4,
       help="Maximum relative dual improvement across the stability window.")
    a("--voronoi-l2-weight-l1-tol", type=float, default=5e-3,
       help="Maximum normalized-weight L1 change across the stability window.")
    a("--voronoi-l2-stability-patience", type=int, default=2,
       help="Consecutive stable accepted updates required for stable_not_certified.")
    a("--temperature", type=float, default=1.0,
       help="Temperature τ for soft reweighting. τ→0 = hard, τ→∞ = uniform. Only used when --reweighting=soft.")
    a("--soft-topk", type=int, default=0,
       help="Top-K for soft reweighting. 0=auto (calibrate K per snapshot). Only used when --reweighting=soft.")
    a("--reweight-pool-size", dest="reweight_pool_size", type=int, default=None,
       help="Subsample pool to this size for soft/voronoi_l2/kl/moment_l2 reweighting. By default (None) uses the full pool. "
            "Setting e.g. 500000 computes weights on a 500k subsample instead of the full pool. "
            "Hard reweighting uses the full selected --reweight-source target and is not subsampled by this option.")
    a("--softmax-pool-size", dest="reweight_pool_size", type=int, default=None,
       help=argparse.SUPPRESS)
    a("--reweight-source", default="query_pool", choices=["query_pool", "full_non_eval"],
      help="Target distribution used for reweighting. query_pool preserves the historical "
           "behavior. full_non_eval, valid with --eval-source full_heldout, uses final "
           "warm-start training rows plus the query pool so the warm-start region also "
           "contributes target mass.")
    a("--include-zero-snapshot", action="store_true",
      help="Include the warm-start-only 0-query classifier in AUC/AP trial curves.")

    # Practical
    a("--eval-size",       type=int, default=100_000, help="Eval subsample size.")
    a("--eval-source", default="pool", choices=["pool", "full_heldout"],
      help="Evaluation construction. pool preserves the legacy behavior by sampling "
           "from the query pool. full_heldout samples eval rows from the full dataset "
           "first and removes them from warm-start training and query candidates.")
    a("--warm-start-max",  type=int, default=None,    help="Cap warm-start size.")
    a("--pool-max",        type=int, default=None,    help="Cap pool size.")
    a("--wass-pool-size",  type=int, default=50000,    help="Subpool size for Wasserstein / Wasserstein-L2 / entropicOT strategy. Brute-force search is O(n × pool_size²).")
    a("--wass-plan-size", type=int, default=None,
      help="Number of Wasserstein-greedy points to plan before rebuilding the random subpool. Defaults to eval_every; set to total_queries to reproduce old one-shot planning.")
    a("--eot-temperature", type=float, default=1.0,
       help="Temperature τ for entropicOT query strategy. τ→0 = hard Wasserstein, τ→∞ = uniform. Only used when --strategy=entropicOT.")
    a("--moment-ridge", type=float, default=1.0,
      help="Ridge regularization used by the moment_matching query strategy.")
    a("--moment-weight-iters", type=int, default=200,
      help="Projected subgradient iterations for --reweighting=moment_l2.")
    a("--n-trials",        type=int, default=1,       help="Number of independent trials.  When > 1, a mean±std PR-AUC plot is generated.")
    a("--n-snapshots",     type=int, default=3,       help="Number of evenly-spaced AUC measurement points.  total_queries must be divisible by n_snapshots × eval_every (default: 3×200=600 divides 3000).")
    a("--seed",            type=int, default=42)
    a("--out-dir", default=None,
      help="Output directory (default: results/active_learning/ad_hoc/al_{strategy}).")

    args = p.parse_args()
    if args.wass_plan_size is None:
        args.wass_plan_size = args.eval_every
    if args.wass_plan_size <= 0:
        p.error("--wass-plan-size must be positive.")
    if args.voronoi_l2_max_iter <= 0:
        p.error("--voronoi-l2-max-iter must be positive.")
    if args.voronoi_l2_relative_gap_tol < 0:
        p.error("--voronoi-l2-relative-gap-tol must be non-negative.")
    if args.voronoi_l2_gradient_tol < 0:
        p.error("--voronoi-l2-gradient-tol must be non-negative.")
    if args.voronoi_l2_stability_window <= 0:
        p.error("--voronoi-l2-stability-window must be positive.")
    if args.voronoi_l2_dual_relative_tol < 0:
        p.error("--voronoi-l2-dual-relative-tol must be non-negative.")
    if args.voronoi_l2_weight_l1_tol < 0:
        p.error("--voronoi-l2-weight-l1-tol must be non-negative.")
    if args.voronoi_l2_stability_patience <= 0:
        p.error("--voronoi-l2-stability-patience must be positive.")
    if args.train_weight_sum <= 0:
        p.error("--train-weight-sum must be positive.")
    if args.out_dir is None:
        args.out_dir = f"results/active_learning/ad_hoc/al_{args.strategy}"
    try:
        run_active_learning(args)
    except Exception as exc:
        update_params_status(args.out_dir, "failed", error=exc)
        raise


if __name__ == "__main__":
    main()
