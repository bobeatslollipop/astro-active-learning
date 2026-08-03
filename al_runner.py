"""Experiment orchestration for the active-learning command-line interface."""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from al_data import (
    MP_LABEL,
    _configure_torch_runtime,
    _json_scalar,
    _timing,
    load_features_and_labels,
)
from al_models import (
    _class_ratio_sample_weights,
    _final_weight_summary,
    _resolve_train_weight_target_sum,
    evaluate,
    train_logistic,
    train_ridge_classifier,
    train_xgboost_classifier,
)
from al_metadata import build_active_params, update_params_status, write_params
from al_queries import STRATEGIES
from al_reporting import (
    _compute_reweight_stats,
    _generated_figure_path,
    _log,
    _record,
    _save_auc_trials_plot,
    _save_average_precision_trials_plot,
    _save_mp_trials_plot,
    _save_test_loss_trials_plot,
    _save_weight_stats_trials_plot,
    compute_average_precision,
    compute_pr_auc,
    generate_pr_curve,
    save_final_model_summary,
)
from al_reweighting import (
    compute_kl_weights,
    compute_moment_l2_weights,
    compute_soft_voronoi_weights,
    compute_voronoi_l2_weights,
    compute_voronoi_weights,
)


def run_active_learning(args):
    # ── Validate n_snapshots constraint ──
    if args.total_queries % (args.n_snapshots * args.eval_every) != 0:
        raise ValueError(
            f"total_queries ({args.total_queries}) must be divisible by "
            f"n_snapshots × eval_every ({args.n_snapshots} × {args.eval_every} = "
            f"{args.n_snapshots * args.eval_every}).  "
            f"Adjust parameters so that AUC snapshot points align with eval boundaries."
        )

    snap_interval = args.total_queries // args.n_snapshots
    positive_auc_query_points = [snap_interval * (i + 1) for i in range(args.n_snapshots)]
    auc_query_points = ([0] if args.include_zero_snapshot else []) + positive_auc_query_points
    auc_query_set = set(positive_auc_query_points)

    strategy_fn = STRATEGIES[args.strategy]
    if args.strategy == "wasserstein_l2":
        if args.reweighting != "voronoi_l2":
            raise ValueError(
                "--strategy wasserstein_l2 is only valid with "
                "--reweighting voronoi_l2."
            )
        if args.reweight_lambda <= 0:
            raise ValueError("--strategy wasserstein_l2 requires --reweight-lambda > 0.")

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.perf_counter()
    _configure_torch_runtime()

    # 1. Load data (shared across all trials)
    print(f"Loading warm-start data from {args.warm_start_file} ...")
    X_warm, y_warm, sid_warm = load_features_and_labels(
        args.warm_start_file, args.feh_threshold, args.warm_start_max, args.seed)

    print(f"Loading full population from {args.full_data_file} ...")
    X_full, y_full, sid_full = load_features_and_labels(
        args.full_data_file, args.feh_threshold, args.pool_max, args.seed + 1)

    t_load = time.perf_counter() - t0
    print(f"  Data loaded in {t_load:.1f}s")

    # 2-3. Build evaluation set and candidate pool.
    # Default mode preserves the historical behavior: build full-minus-warm
    # pool first, then sample eval rows from that pool without removing them
    # from the query candidates.  full_heldout mode creates a true held-out
    # eval set from the full population first, then removes those rows from
    # both warm-start training and the query pool.
    eval_rng = np.random.RandomState(args.seed)
    original_warm_count = len(X_warm)
    eval_original_warm_count = 0
    eval_warm_overlap = None
    eval_pool_overlap = None
    eval_source_id_mode = sid_warm is not None and sid_full is not None
    sid_pool = None

    if args.eval_source == "full_heldout":
        eval_n = min(args.eval_size, len(X_full))
        eval_idx = eval_rng.choice(len(X_full), eval_n, replace=False)
        eval_mask_full = np.zeros(len(X_full), dtype=bool)
        eval_mask_full[eval_idx] = True
        X_eval, y_eval = X_full[eval_idx].copy(), y_full[eval_idx].copy()

        if eval_source_id_mode:
            sid_warm_original = sid_warm.copy()
            eval_sids = sid_full[eval_idx]
            eval_original_warm_count = int(np.isin(eval_sids, sid_warm_original).sum())

            warm_train_mask = ~np.isin(sid_warm, eval_sids)
            X_warm, y_warm = X_warm[warm_train_mask].copy(), y_warm[warm_train_mask].copy()
            sid_warm_train = sid_warm[warm_train_mask]

            pool_mask = (~np.isin(sid_full, sid_warm_original)) & (~np.isin(sid_full, eval_sids))
            eval_warm_overlap = int(np.intersect1d(eval_sids, sid_warm_train).size)
            eval_pool_overlap = int(np.intersect1d(eval_sids, sid_full[pool_mask]).size)
        else:
            print("  Warning: source_id unavailable; using approximate full-heldout dedup.")
            sid_warm_original = None
            eval_keys = {tuple(np.round(x, 6)) for x in X_eval}
            original_warm_keys = {tuple(np.round(x, 6)) for x in X_warm}

            eval_original_warm_count = sum(1 for k in eval_keys if k in original_warm_keys)
            warm_train_mask = np.array(
                [tuple(np.round(x, 6)) not in eval_keys for x in X_warm],
                dtype=bool,
            )
            X_warm, y_warm = X_warm[warm_train_mask].copy(), y_warm[warm_train_mask].copy()

            pool_not_warm = np.array(
                [tuple(np.round(x, 6)) not in original_warm_keys for x in X_full],
                dtype=bool,
            )
            pool_not_eval = np.array(
                [tuple(np.round(x, 6)) not in eval_keys for x in X_full],
                dtype=bool,
            )
            pool_mask = pool_not_warm & pool_not_eval
            eval_warm_overlap = sum(
                1 for x in X_warm if tuple(np.round(x, 6)) in eval_keys
            )
            eval_pool_overlap = sum(
                1 for x in X_full[pool_mask] if tuple(np.round(x, 6)) in eval_keys
            )

        if eval_warm_overlap != 0 or eval_pool_overlap != 0:
            raise RuntimeError(
                "full_heldout eval overlap check failed: "
                f"eval/final-warm={eval_warm_overlap}, eval/query-pool={eval_pool_overlap}."
            )

        X_pool, y_pool = X_full[pool_mask].copy(), y_full[pool_mask].copy()
        if eval_source_id_mode:
            sid_pool = sid_full[pool_mask].copy()
    else:
        if eval_source_id_mode:
            pool_mask = ~np.isin(sid_full, sid_warm)
        else:
            print("  Warning: source_id unavailable; using approximate dedup.")
            warm_set = {tuple(np.round(x, 6)) for x in X_warm}
            pool_mask = np.array([tuple(np.round(x, 6)) not in warm_set for x in X_full])

        X_pool, y_pool = X_full[pool_mask].copy(), y_full[pool_mask].copy()
        if eval_source_id_mode:
            sid_pool = sid_full[pool_mask].copy()
        eval_n = min(args.eval_size, len(X_pool))
        eval_idx = eval_rng.choice(len(X_pool), eval_n, replace=False)
        X_eval, y_eval = X_pool[eval_idx], y_pool[eval_idx]

    # Free the full arrays (only pool & eval are needed hereafter)
    del X_full, y_full, sid_full, sid_warm

    for tag, n, mp in [
        ("Warm-start", len(X_warm), (y_warm == MP_LABEL).sum()),
        ("Pool", len(X_pool), (y_pool == MP_LABEL).sum()),
        ("Eval set", eval_n, (y_eval == MP_LABEL).sum()),
    ]:
        print(f"  {tag}: {n} (MP={mp}, MR={n - mp})")

    if args.eval_source == "full_heldout":
        warm_frac = eval_original_warm_count / max(1, eval_n)
        print(f"  Eval source: full_heldout")
        print(f"  Eval original warm-start membership: "
              f"{eval_original_warm_count}/{eval_n} ({warm_frac:.4%})")
        print(f"  Eval/final warm-start overlap: {eval_warm_overlap}")
        print(f"  Eval/query pool overlap: {eval_pool_overlap}")
    else:
        print("  Eval source: pool (legacy behavior; eval rows are sampled from the query pool)")

    args.initial_labeled_count = int(len(X_warm) if args.strategy != "purely_random" else 0)
    args.original_warm_start_count = int(original_warm_count)
    args.eval_actual_size = int(eval_n)
    args.eval_original_warm_count = int(eval_original_warm_count)
    args.eval_original_warm_fraction = float(eval_original_warm_count / max(1, eval_n))
    args.eval_final_warm_overlap = None if eval_warm_overlap is None else int(eval_warm_overlap)
    args.eval_query_pool_overlap = None if eval_pool_overlap is None else int(eval_pool_overlap)
    args.query_rng_mode = "dedicated_for_kmedianpp" if args.strategy == "kmedianpp" else "shared"

    initial_target_sum = _resolve_train_weight_target_sum(
        args.train_weight_sum_mode,
        args.train_weight_sum,
        args.initial_labeled_count,
        args.initial_labeled_count,
    )
    if initial_target_sum <= 0:
        raise ValueError(
            f"Resolved initial train weight sum must be positive; got {initial_target_sum}."
        )
    args.initial_train_weight_target_sum = float(initial_target_sum)
    print(f"  Train weight-sum mode: {args.train_weight_sum_mode}; "
          f"initial labeled count={args.initial_labeled_count}; "
          f"initial target sum={initial_target_sum:.6g}")

    if args.reweight_source == "full_non_eval" and args.eval_source != "full_heldout":
        raise ValueError("--reweight-source full_non_eval requires --eval-source full_heldout.")

    if args.reweight_source == "full_non_eval":
        reweight_source_n = len(X_warm) + len(X_pool)
        reweight_source_warm_n = len(X_warm)
        print("  Reweight target source: full_non_eval "
              f"(final warm-start + query pool); total={reweight_source_n}, "
              f"final warm-start contribution={reweight_source_warm_n} "
              f"({reweight_source_warm_n / max(1, reweight_source_n):.4%})")
    else:
        reweight_source_n = len(X_pool)
        reweight_source_warm_n = 0
        print(f"  Reweight target source: query_pool; total={reweight_source_n}")
    args.reweight_source_total_count = int(reweight_source_n)
    args.reweight_source_final_warm_count = int(reweight_source_warm_n)
    args.reweight_source_final_warm_fraction = float(reweight_source_warm_n / max(1, reweight_source_n))

    run_params = build_active_params(
        args,
        y_warm=y_warm,
        y_pool=y_pool,
        y_eval=y_eval,
        data_load_seconds=t_load,
    )
    write_params(args.out_dir, run_params)

    # 4. Pre-allocate labeled arrays (reused across trials)
    max_labeled = len(X_warm) + args.total_queries
    n_features = X_warm.shape[1]
    X_labeled = np.empty((max_labeled, n_features), dtype=np.float32)
    y_labeled = np.empty(max_labeled, dtype=np.int32)

    subsampled_reweighting = args.reweighting in ("soft", "voronoi_l2", "kl", "moment_l2")
    uses_reweight_subsample = (
        subsampled_reweighting
        and args.reweight_pool_size
        and args.reweight_pool_size < reweight_source_n
    )
    X_reweight_full_non_eval = None
    if args.reweighting != "none" and args.reweight_source == "full_non_eval" and not uses_reweight_subsample:
        X_reweight_full_non_eval = np.vstack([X_warm, X_pool]).astype(np.float32, copy=False)

    def make_reweight_pool(trial_rng):
        """Build the fixed target pool used for reweighting in one trial."""
        if args.reweight_source == "full_non_eval":
            total_n = reweight_source_n
            warm_n = len(X_warm)
            if uses_reweight_subsample:
                idx = trial_rng.choice(total_n, args.reweight_pool_size, replace=False)
                warm_mask = idx < warm_n
                warm_idx = idx[warm_mask]
                pool_idx = idx[~warm_mask] - warm_n
                parts = []
                if len(warm_idx):
                    parts.append(X_warm[warm_idx])
                if len(pool_idx):
                    parts.append(X_pool[pool_idx])
                if len(parts) == 1:
                    X_rw = parts[0].copy()
                else:
                    X_rw = np.vstack(parts).astype(np.float32, copy=False)
                return X_rw, total_n, int(len(warm_idx))
            if X_reweight_full_non_eval is not None:
                return X_reweight_full_non_eval, total_n, warm_n
            return np.vstack([X_warm, X_pool]).astype(np.float32, copy=False), total_n, warm_n

        if uses_reweight_subsample:
            idx = trial_rng.choice(len(X_pool), args.reweight_pool_size, replace=False)
            return X_pool[idx], len(X_pool), 0
        return X_pool, len(X_pool), 0

    # ── Multi-trial loop ──
    all_trial_aucs = []          # legacy trapezoidal PR-AUC values
    all_trial_average_precisions = []  # sklearn average_precision_score values
    all_trial_mp_counts = []     # list of lists, cumulative MP counts at each snapshot
    all_trial_test_losses = []   # list of lists of dicts, test losses at each eval point
    all_trial_weight_stats = []  # list of lists of reweighting concentration stats
    all_trial_query_plans = []   # exact queried pool indices/source_ids per trial
    first_trial_results = None

    for trial in range(args.n_trials):
        if args.n_trials > 1:
            print(f"\n{'=' * 60}")
            print(f"Trial {trial + 1} / {args.n_trials}  (seed={args.seed + trial})")
            print(f"{'=' * 60}")

        rng = np.random.RandomState(args.seed + trial)
        query_rng = (
            np.random.RandomState(args.seed + trial)
            if args.strategy == "kmedianpp"
            else rng
        )
        if args.strategy == "kmedianpp":
            print("  [Query RNG] kmedianpp uses a dedicated RNG stream; "
                  "reweighting/training RNG draws cannot change the query plan.")

        # Reset labeled set for this trial
        if args.strategy == "purely_random":
            n_labeled = 0
        else:
            n_labeled = len(X_warm)
            X_labeled[:n_labeled] = X_warm
            y_labeled[:n_labeled] = y_warm

        warm_n = n_labeled  # 0 for purely_random, len(X_warm) otherwise
        available = np.ones(len(X_pool), dtype=bool)
        results = []
        strategy_state = {}
        voronoi_state = {}
        soft_voronoi_state = {}
        voronoi_l2_state = {}
        kl_state = {}
        moment_weight_state = {}
        final_sw = None

        # Pre-compute fixed reweight target for this trial (reused across snapshots
        # so the incremental top-K/cache state stays valid)
        X_reweight_pool, reweight_source_full_n, reweight_pool_warm_n = make_reweight_pool(rng)
        if args.reweighting != "none":
            reweight_note = (
                f" (subsampled from {reweight_source_full_n})"
                if len(X_reweight_pool) < reweight_source_full_n
                else ""
            )
            if args.reweight_source == "full_non_eval":
                print(f"  [ReweightTarget] source=full_non_eval; "
                      f"target_rows={len(X_reweight_pool)}{reweight_note}; "
                      f"warm_rows_in_target={reweight_pool_warm_n}")
            else:
                print(f"  [ReweightTarget] source=query_pool; "
                      f"target_rows={len(X_reweight_pool)}{reweight_note}")

        trial_aucs = []
        trial_average_precisions = []
        trial_mp_counts = []
        trial_test_losses = []   # test losses at each eval point for this trial
        trial_weight_stats = []
        trial_query_pool_indices = []
        trial_query_source_ids = []
        trial_query_labels = []
        trial_query_batches = []
        clf_snapshots = []       # (queries, clf) pairs for PR curve — first trial only

        # Helper: train → evaluate → record → log
        def snapshot(n_queries, prev_clf=None):
            nonlocal voronoi_state, voronoi_l2_state, kl_state, moment_weight_state, final_sw
            Xl, yl = X_labeled[:n_labeled], y_labeled[:n_labeled]
            if len(np.unique(yl)) < 2:
                # Both classes required; skip this checkpoint and keep previous clf.
                print(f"[Query {n_queries:4d}] Skipped — only one class in labeled set so far.")
                return prev_clf

            # Reweighting: compute per-sample weights to correct covariate shift
            sw = None
            reweight_t0 = None
            reweight_label = None
            if args.reweighting == "hard":
                reweight_t0 = time.perf_counter()
                reweight_label = "Voronoi-hard weights"
                print(f"  [Voronoi-Hard] Computing sample weights ({n_labeled} labeled vs {len(X_reweight_pool)} target rows)...")
                sw, voronoi_state = compute_voronoi_weights(X_reweight_pool, Xl, voronoi_state)
            elif args.reweighting == "soft":
                reweight_t0 = time.perf_counter()
                reweight_label = "Voronoi-soft weights"
                print(f"  [Voronoi-Soft] Computing sample weights (τ={args.temperature}, "
                      f"{n_labeled} labeled vs {len(X_reweight_pool)} target rows"
                      f"{f' (subsampled from {reweight_source_full_n})' if len(X_reweight_pool) < reweight_source_full_n else ''})...")
                sw = compute_soft_voronoi_weights(X_reweight_pool, Xl, args.temperature,
                                                   soft_state=soft_voronoi_state,
                                                   topk=args.soft_topk)
            elif args.reweighting == "voronoi_l2":
                reweight_t0 = time.perf_counter()
                reweight_label = "Voronoi-L2 weights"
                l2_max_iter = args.voronoi_l2_max_iter
                if "z" not in voronoi_l2_state:
                    l2_max_iter = args.voronoi_l2_initial_max_iter
                print(f"  [Voronoi-L2] Computing sample weights (λ={args.reweight_lambda}, "
                      f"max_iter={l2_max_iter}, "
                      f"{n_labeled} labeled vs {len(X_reweight_pool)} target rows"
                      f"{f' (subsampled from {reweight_source_full_n})' if len(X_reweight_pool) < reweight_source_full_n else ''})...")
                sw = compute_voronoi_l2_weights(X_reweight_pool, Xl, args.reweight_lambda,
                                                state=voronoi_l2_state,
                                                max_iter=l2_max_iter)
            elif args.reweighting == "kl":
                reweight_t0 = time.perf_counter()
                reweight_label = "KL weights"
                print(f"  [KL] Computing sample weights (λ={args.reweight_lambda}, "
                      f"{n_labeled} labeled vs {len(X_reweight_pool)} target rows"
                      f"{f' (subsampled from {reweight_source_full_n})' if len(X_reweight_pool) < reweight_source_full_n else ''})...")
                sw = compute_kl_weights(X_reweight_pool, Xl, args.reweight_lambda,
                                        state=kl_state,
                                        max_iter=args.voronoi_l2_max_iter)
            elif args.reweighting == "moment_l2":
                reweight_t0 = time.perf_counter()
                reweight_label = "Moment-L2 weights"
                print(f"  [Moment-L2] Computing sample weights (λ={args.reweight_lambda}, "
                      f"{n_labeled} labeled vs {len(X_reweight_pool)} target rows"
                      f"{f' (subsampled from {reweight_source_full_n})' if len(X_reweight_pool) < reweight_source_full_n else ''})...")
                sw = compute_moment_l2_weights(X_reweight_pool, Xl, args.reweight_lambda,
                                               state=moment_weight_state,
                                               max_iter=args.moment_weight_iters)

            if reweight_t0 is not None:
                _timing(reweight_label, reweight_t0)

            final_sw = sw
            stats_sw = sw
            if args.reweighting == "none":
                stats_sw = np.ones(len(yl), dtype=np.float64)
            weight_stats = _compute_reweight_stats(stats_sw, yl, n_queries)
            if weight_stats is not None:
                trial_weight_stats.append(weight_stats)

            target_weight_sum = _resolve_train_weight_target_sum(
                args.train_weight_sum_mode,
                args.train_weight_sum,
                args.initial_labeled_count,
                len(yl),
            )
            final_train_w = _class_ratio_sample_weights(
                yl, args.lambda_MP, sw,
                target_sum=target_weight_sum,
                class_balance_mode=args.class_balance_mode,
            )
            weight_summary = _final_weight_summary(
                yl, final_train_w, target_weight_sum, args.lambda_MP,
                class_balance_mode=args.class_balance_mode,
            )
            print(
                "  [TrainWeights] "
                f"class_balance={args.class_balance_mode} "
                f"mode={args.train_weight_sum_mode} "
                f"target={weight_summary['train_weight_target_sum']:.6g} "
                f"total={weight_summary['train_weight_actual_sum']:.6g} "
                f"MP={weight_summary['train_weight_MP_sum']:.6g} "
                f"MR={weight_summary['train_weight_MR_sum']:.6g}"
            )

            train_t0 = time.perf_counter()
            if args.model == "logistic":
                clf = train_logistic(Xl, yl, args.lambda_MP, args.C,
                                     prev_clf=prev_clf, sample_weight=sw,
                                     target_sum=target_weight_sum,
                                     class_balance_mode=args.class_balance_mode)
            elif args.model == "ridge":
                clf = train_ridge_classifier(Xl, yl, args.lambda_MP,
                                             alpha=args.ridge_alpha,
                                             sample_weight=sw,
                                             target_sum=target_weight_sum,
                                             class_balance_mode=args.class_balance_mode)
            elif args.model == "xgboost":
                clf = train_xgboost_classifier(
                    Xl, yl, args.lambda_MP, sample_weight=sw,
                    n_estimators=args.xgb_n_estimators,
                    max_depth=args.xgb_max_depth,
                    learning_rate=args.xgb_learning_rate,
                    subsample=args.xgb_subsample,
                    colsample_bytree=args.xgb_colsample_bytree,
                    min_child_weight=args.xgb_min_child_weight,
                    gamma=args.xgb_gamma,
                    reg_lambda=args.xgb_reg_lambda,
                    tree_method=args.xgb_tree_method,
                    device=args.xgb_device,
                    n_jobs=args.xgb_n_jobs,
                    random_state=args.seed + trial,
                    target_sum=target_weight_sum,
                    class_balance_mode=args.class_balance_mode,
                )
            else:
                raise ValueError(f"Unknown model: {args.model}")
            _timing("Classifier train", train_t0)

            eval_t0 = time.perf_counter()
            m = _record(evaluate(clf, X_eval, y_eval), n_queries, yl)
            _timing("Evaluation", eval_t0)

            # Average test loss across both classes
            m["avg_test_loss"] = (m["loss_MP"] + m["loss_MR"]) / 2.0
            m.update(weight_summary)
            m["train_weight_sum_mode"] = args.train_weight_sum_mode
            m["class_balance_mode"] = args.class_balance_mode

            # Track test losses for cross-trial plotting
            trial_test_losses.append({
                "loss_MP": m["loss_MP"],
                "loss_MR": m["loss_MR"],
                "avg_test_loss": m["avg_test_loss"],
                "n_queries": n_queries,
            })

            results.append(m)
            _log(m)
            return clf

        # 5. Initial evaluation
        # For purely_random the labeled set starts empty, so skip the initial fit.
        if args.strategy != "purely_random":
            clf = snapshot(0)
            if args.include_zero_snapshot:
                if clf is not None:
                    auc_val = compute_pr_auc(clf, X_eval, y_eval)
                    ap_val = compute_average_precision(clf, X_eval, y_eval)
                    print(f"  >> AUC snapshot at 0 queries: "
                          f"PR-AUC(trapz) = {auc_val:.4f}; AP = {ap_val:.4f}")
                    if trial == 0:
                        import copy
                        clf_snapshots.append((0, copy.deepcopy(clf)))
                else:
                    auc_val = float('nan')
                    ap_val = float('nan')
                    print("  >> AUC snapshot at 0 queries: skipped (clf not ready)")
                trial_aucs.append(auc_val)
                trial_average_precisions.append(ap_val)
                trial_mp_counts.append(0)
        else:
            clf = None

        # 6. Active learning loop
        queried = 0

        while queried < args.total_queries and available.any():
            batch = min(args.eval_every, args.total_queries - queried, int(available.sum()))
            query_start = queried
            query_t0 = time.perf_counter()

            if args.strategy in ("wasserstein", "wasserstein_l2"):
                pool_idx = strategy_fn(X_pool, clf, batch, query_rng,
                                       X_labeled=X_labeled[:n_labeled],
                                       state=strategy_state,
                                       pool_size=args.wass_pool_size,
                                       plan_size=args.wass_plan_size,
                                       available_mask=available,
                                       reweight_lambda=args.reweight_lambda,
                                       temperature=args.eot_temperature)
            else:
                avail_idx = np.where(available)[0]
                sel = strategy_fn(X_pool[avail_idx], clf, batch, query_rng,
                                  X_labeled=X_labeled[:n_labeled], state=strategy_state,
                                  pool_size=args.wass_pool_size,
                                  temperature=args.eot_temperature,
                                  moment_ridge=args.moment_ridge,
                                  sample_weight=final_sw if args.reweighting == "moment_l2" else None)
                pool_idx = avail_idx[sel]

            _timing(f"{args.strategy} query selection", query_t0)

            if len(pool_idx) == 0:
                print("  [Query] No available points selected; stopping trial early.")
                break

            # Append to pre-allocated arrays (no vstack/concatenate)
            n_new = len(pool_idx)
            pool_idx_list = [int(i) for i in pool_idx]
            label_list = [int(v) for v in y_pool[pool_idx]]
            if sid_pool is not None:
                source_id_list = [_json_scalar(v) for v in sid_pool[pool_idx]]
            else:
                source_id_list = None

            trial_query_pool_indices.extend(pool_idx_list)
            trial_query_labels.extend(label_list)
            if source_id_list is not None:
                trial_query_source_ids.extend(source_id_list)
            trial_query_batches.append({
                "n_queries_before": int(query_start),
                "n_queries_after": int(query_start + n_new),
                "pool_indices": pool_idx_list,
                "source_ids": source_id_list,
                "labels": label_list,
            })

            X_labeled[n_labeled:n_labeled + n_new] = X_pool[pool_idx]
            y_labeled[n_labeled:n_labeled + n_new] = y_pool[pool_idx]
            n_labeled += n_new
            available[pool_idx] = False
            queried += n_new

            clf = snapshot(queried, prev_clf=clf)

            # Record AUC at snapshot query points
            if queried in auc_query_set:
                if clf is not None:
                    auc_val = compute_pr_auc(clf, X_eval, y_eval)
                    ap_val = compute_average_precision(clf, X_eval, y_eval)
                    print(f"  >> AUC snapshot at {queried} queries: "
                          f"PR-AUC(trapz) = {auc_val:.4f}; AP = {ap_val:.4f}")
                    if trial == 0:
                        import copy
                        clf_snapshots.append((queried, copy.deepcopy(clf)))
                else:
                    auc_val = float('nan')
                    ap_val = float('nan')
                    print(f"  >> AUC snapshot at {queried} queries: skipped (clf not ready)")
                trial_aucs.append(auc_val)
                trial_average_precisions.append(ap_val)

                # Track cumulative MP count among queried samples
                n_queried_mp = int(
                    np.sum(y_labeled[warm_n:n_labeled] == MP_LABEL)
                )
                trial_mp_counts.append(n_queried_mp)

        all_trial_aucs.append(trial_aucs)
        all_trial_average_precisions.append(trial_average_precisions)
        all_trial_mp_counts.append(trial_mp_counts)
        all_trial_test_losses.append(trial_test_losses)
        all_trial_weight_stats.append(trial_weight_stats)
        all_trial_query_plans.append({
            "trial": int(trial),
            "seed": int(args.seed + trial),
            "query_rng_mode": args.query_rng_mode,
            "pool_indices": trial_query_pool_indices,
            "source_ids": trial_query_source_ids if sid_pool is not None else None,
            "labels": trial_query_labels,
            "batches": trial_query_batches,
        })

        # 7. Save detailed outputs (first trial only)
        if trial == 0:
            first_trial_results = results
            t_trial = time.perf_counter() - t0
            print(f"\nTrial 1 runtime: {t_trial:.1f}s  (data loading: {t_load:.1f}s)")

            with open(os.path.join(args.out_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)

            # Linear models write coefficients; tree models write feature importances.
            save_final_model_summary(clf, args.full_data_file, args.out_dir)

            if args.reweighting != "none" and final_sw is not None:
                fig, ax = plt.subplots(figsize=(8, 5))
                sw_pos = final_sw[final_sw > 0]
                if len(sw_pos) > 0:
                    min_w, max_w = np.min(sw_pos), np.max(sw_pos)
                    if min_w < max_w:
                        bins = np.logspace(np.log10(min_w), np.log10(max_w), 50)
                    else:
                        bins = 50
                    ax.hist(sw_pos, bins=bins, color="#4A90D9", edgecolor="white", alpha=0.8, log=True)
                    ax.set_xscale("log")
                else:
                    ax.hist(final_sw, bins=50, color="#4A90D9", edgecolor="white", alpha=0.8, log=True)

                ax.set_xlabel("Sample Weight (log scale)", fontsize=12)
                ax.set_ylabel("Frequency (log scale)", fontsize=12)
                ax.set_title("Distribution of Sample Weights (Last Iteration)", fontsize=14, fontweight="bold")
                ax.grid(True, alpha=0.3, which="both", ls="--")
                fig.tight_layout()
                wt_plot_path = _generated_figure_path(
                    args.out_dir, "weight_distribution.png"
                )
                fig.savefig(wt_plot_path, dpi=200)
                plt.close(fig)
                print(f"\nSaved weight distribution plot to {wt_plot_path}")

            # PR curve from snapshot classifiers
            pr_curves = [(f"{q} queries", c) for q, c in clf_snapshots]
            generate_pr_curve(pr_curves, X_eval, y_eval, args.out_dir)

    # 8. Summary & multi-trial AUC plot
    t_total = time.perf_counter() - t0
    print(f"\nTotal runtime ({args.n_trials} trial(s)): {t_total:.1f}s")

    query_plan_data = {
        "strategy": args.strategy,
        "query_rng_mode": args.query_rng_mode,
        "seed": int(args.seed),
        "n_trials": int(args.n_trials),
        "total_queries": int(args.total_queries),
        "eval_every": int(args.eval_every),
        "trial_query_plans": all_trial_query_plans,
    }
    with open(os.path.join(args.out_dir, "query_plan_trials.json"), "w") as f:
        json.dump(query_plan_data, f, indent=2)

    if args.n_trials > 1:
        # Derive eval query points from the first trial's test loss records
        eval_query_points = [d["n_queries"] for d in all_trial_test_losses[0]]

        auc_data = {
            "auc_query_points": auc_query_points,
            "trial_aucs": all_trial_aucs,
            "average_precision_query_points": auc_query_points,
            "trial_average_precisions": all_trial_average_precisions,
            "trial_mp_counts": all_trial_mp_counts,
            "eval_query_points": eval_query_points,
            "trial_test_losses": all_trial_test_losses,
            "trial_weight_stats": all_trial_weight_stats,
        }
        with open(os.path.join(args.out_dir, "auc_trials.json"), "w") as f:
            json.dump(auc_data, f, indent=2)

        weight_stats_data = {
            "weight_query_points": sorted({
                int(d["n_queries"])
                for trial_data in all_trial_weight_stats
                for d in trial_data
                if "n_queries" in d
            }),
            "trial_weight_stats": all_trial_weight_stats,
        }
        with open(os.path.join(args.out_dir, "weight_stats_trials.json"), "w") as f:
            json.dump(weight_stats_data, f, indent=2)

        _save_auc_trials_plot(auc_query_points, all_trial_aucs, args.out_dir, args.n_trials)
        _save_average_precision_trials_plot(
            auc_query_points, all_trial_average_precisions, args.out_dir, args.n_trials
        )
        _save_mp_trials_plot(auc_query_points, all_trial_mp_counts, args.out_dir, args.n_trials)
        _save_test_loss_trials_plot(eval_query_points, all_trial_test_losses, args.out_dir, args.n_trials)
        _save_weight_stats_trials_plot(all_trial_weight_stats, args.out_dir, args.n_trials)

    update_params_status(args.out_dir, "completed", total_seconds=t_total)
    print(f"\nAll outputs saved to {args.out_dir}/")
    return first_trial_results
