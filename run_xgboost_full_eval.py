#!/usr/bin/env python3
"""Full-dataset XGBoost benchmark for MP/MR classification.

This script tests whether boosted decision trees generalize better than the
linear/logistic baseline on a natural heldout split of the full labeled H5
dataset.  It intentionally does not use active-learning reweighting because
the train/validation/eval splits are random i.i.d. splits from the same H5.
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
import sys

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from al_data import (
    CANONICAL_LABEL_ENCODING,
    MP_LABEL,
    MR_LABEL,
    labels_from_feh,
    mp_probability,
    mp_target,
)
from al_metadata import (
    ARTIFACT_LAYOUT_VERSION,
    SCHEMA_VERSION,
    atomic_write_json,
    canonical_hash,
    environment_metadata,
    experiment_family,
    fast_hdf5_fingerprint,
    git_metadata,
    update_params_status,
    utc_now,
)


XGB_CONFIGS = {
    "xgb_shallow": {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 20,
        "gamma": 0.1,
        "reg_lambda": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "xgb_medium": {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 10,
        "gamma": 0.05,
        "reg_lambda": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "xgb_deeper": {
        "n_estimators": 700,
        "max_depth": 6,
        "learning_rate": 0.03,
        "min_child_weight": 10,
        "gamma": 0.0,
        "reg_lambda": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
}

PRECISION_KS = (100, 300, 1000, 3000, 10000)
_ACTIVE_OUT_DIR = None


def nsort(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)]


def feature_cols(h5_keys):
    bp = sorted([k for k in h5_keys if k.startswith("bp_")], key=nsort)
    rp = sorted([k for k in h5_keys if k.startswith("rp_")], key=nsort)
    cols = bp + rp
    if "ebv" in h5_keys:
        cols.append("ebv")
    return cols


def load_dataset(path, feh_threshold):
    t0 = time.perf_counter()
    with h5py.File(path, "r") as f:
        cols = feature_cols(list(f.keys()))
        n = f[cols[0]].shape[0]
        X = np.empty((n, len(cols)), dtype=np.float32)
        for j, col in enumerate(cols):
            X[:, j] = np.nan_to_num(f[col][()], nan=0.0).astype(np.float32, copy=False)

        feh = f["feh"][()].astype(np.float32, copy=False)
        source_id = f["source_id"][()] if "source_id" in f else np.arange(n, dtype=np.int64)

    valid = np.isfinite(feh)
    if not valid.all():
        X = X[valid]
        feh = feh[valid]
        source_id = source_id[valid]

    end = -1 if cols[-1] == "ebv" else X.shape[1]
    norms = np.linalg.norm(X[:, :end], axis=1, keepdims=True) + 1e-8
    X[:, :end] /= norms

    y = labels_from_feh(feh, feh_threshold)
    elapsed = time.perf_counter() - t0
    return X, y, feh, source_id, cols, elapsed


def stratified_split(y, seed, train_frac=0.70, val_frac=0.10):
    rng = np.random.RandomState(seed)
    train_parts, val_parts, eval_parts = [], [], []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_train = int(round(train_frac * len(idx)))
        n_val = int(round(val_frac * len(idx)))
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train:n_train + n_val])
        eval_parts.append(idx[n_train + n_val:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    eval_idx = np.concatenate(eval_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(eval_idx)
    return train_idx, val_idx, eval_idx


def split_counts(y, indices):
    yy = y[indices]
    n_mp = int((yy == MP_LABEL).sum())
    n_mr = int((yy == MR_LABEL).sum())
    return {
        "total": int(len(indices)),
        "mp": n_mp,
        "mr": n_mr,
        "mp_fraction": float(n_mp / max(len(indices), 1)),
    }


def balanced_sample_weight(y):
    y = np.asarray(y, dtype=np.int32)
    weights = np.ones(len(y), dtype=np.float32)
    n = float(len(y))
    for cls in (MP_LABEL, MR_LABEL):
        mask = y == cls
        count = int(mask.sum())
        if count > 0:
            weights[mask] = n / (2.0 * count)
    return weights


def xgb_tree_params(tree_method):
    if tree_method == "gpu_hist":
        return {"tree_method": "gpu_hist"}
    if tree_method == "cuda_hist":
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": "hist"}


def import_xgboost():
    try:
        import xgboost as xgb
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit(
            "xgboost is not installed. Install it with `python -m pip install xgboost` "
            "and rerun this script."
        ) from exc
    return xgb, XGBClassifier


def smoke_test_xgboost(XGBClassifier, requested_tree_method, seed):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(512, 8)).astype(np.float32)
    y = np.where(rng.rand(512) < 0.08, MP_LABEL, MR_LABEL).astype(np.int32)
    weights = balanced_sample_weight(y)

    attempts = [requested_tree_method]
    if requested_tree_method == "gpu_hist":
        attempts.append("cuda_hist")
    if "hist" not in attempts:
        attempts.append("hist")

    errors = {}
    for method in attempts:
        try:
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=4,
                max_depth=2,
                learning_rate=0.1,
                random_state=seed,
                n_jobs=1,
                **xgb_tree_params(method),
            )
            model.fit(X, y, sample_weight=weights, verbose=False)
            _ = model.predict_proba(X[:8])
            config = json.loads(model.get_booster().save_config())
            device = config.get("learner", {}).get("generic_param", {}).get("device", "")
            if method in ("gpu_hist", "cuda_hist") and device == "cpu":
                raise RuntimeError("XGBoost accepted CUDA parameters but actually used CPU.")
            return method, errors
        except Exception as exc:
            errors[method] = repr(exc)
    raise RuntimeError(f"All XGBoost smoke-test attempts failed: {errors}")


def model_metrics(y_true, p_mp):
    y_true_mp = mp_target(y_true)
    y_pred_mp = (p_mp >= 0.5).astype(np.int32)
    cm = confusion_matrix(y_true_mp, y_pred_mp, labels=[1, 0])
    return {
        "pr_auc": float(average_precision_score(y_true_mp, p_mp)),
        "roc_auc": float(roc_auc_score(y_true_mp, p_mp)),
        "log_loss": float(log_loss(
            y_true_mp, np.column_stack([1.0 - p_mp, p_mp]), labels=[0, 1]
        )),
        "precision_at_0_5": float(precision_score(
            y_true_mp, y_pred_mp, zero_division=0
        )),
        "recall_at_0_5": float(recall_score(
            y_true_mp, y_pred_mp, zero_division=0
        )),
        "f1_at_0_5": float(f1_score(y_true_mp, y_pred_mp, zero_division=0)),
        "confusion_matrix_labels_mp_mr": cm.tolist(),
        "n": int(len(y_true)),
        "n_mp": int(y_true_mp.sum()),
        "predicted_mp_at_0_5": int(y_pred_mp.sum()),
    }


def precision_at_k_rows(model_name, y_true, p_mp, ks=PRECISION_KS):
    order = np.argsort(-p_mp)
    y_true_mp = mp_target(y_true)
    total_mp = int(y_true_mp.sum())
    rows = []
    for k in ks:
        kk = min(int(k), len(order))
        top = order[:kk]
        tp = int(y_true_mp[top].sum())
        rows.append({
            "model": model_name,
            "k": kk,
            "true_mp_in_top_k": tp,
            "precision_at_k": float(tp / max(kk, 1)),
            "recall_at_k": float(tp / max(total_mp, 1)),
        })
    return rows


def save_top_candidates(out_dir, model_name, y_true, p_mp, feh, source_id, max_k=10000):
    order = np.argsort(-p_mp)[:min(max_k, len(p_mp))]
    path = out_dir / f"top_candidates_{model_name}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank", "source_id", "feh", "true_class_label", "is_mp", "p_mp"
            ],
        )
        writer.writeheader()
        for rank, idx in enumerate(order, start=1):
            writer.writerow({
                "rank": rank,
                "source_id": int(source_id[idx]),
                "feh": float(feh[idx]),
                "true_class_label": int(y_true[idx]),
                "is_mp": int(y_true[idx] == MP_LABEL),
                "p_mp": float(p_mp[idx]),
            })


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(out_dir, curve_data):
    figure_dir = out_dir / "figures" / "final"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for item in curve_data:
        precision, recall, _ = precision_recall_curve(
            mp_target(item["y_true"]), item["p_mp"]
        )
        ax.plot(recall, precision, lw=2, label=f"{item['name']} (AP={item['eval_pr_auc']:.4f})")
    ax.set_xlabel("Recall (MP)")
    ax.set_ylabel("Precision (MP)")
    ax.set_title("Natural Heldout PR Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / "pr_curves.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for item in curve_data:
        fpr, tpr, _ = roc_curve(mp_target(item["y_true"]), item["p_mp"])
        ax.plot(fpr, tpr, lw=2, label=f"{item['name']} (AUC={item['eval_roc_auc']:.4f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Natural Heldout ROC Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / "roc_curves.png", dpi=200)
    plt.close(fig)


def train_logistic(args, X_train, y_train, X_val, X_eval):
    t0 = time.perf_counter()
    clf = LogisticRegression(
        C=args.logistic_c,
        solver=args.logistic_solver,
        max_iter=args.logistic_max_iter,
        tol=args.logistic_tol,
        n_jobs=args.n_jobs,
    )
    clf.fit(X_train, y_train, sample_weight=balanced_sample_weight(y_train))
    train_seconds = time.perf_counter() - t0
    return {
        "model": clf,
        "train_seconds": train_seconds,
        "p_val": mp_probability(clf, X_val),
        "p_eval": mp_probability(clf, X_eval),
        "params": {
            "C": args.logistic_c,
            "solver": args.logistic_solver,
            "max_iter": args.logistic_max_iter,
            "tol": args.logistic_tol,
        },
    }


def train_xgb(args, XGBClassifier, name, config, tree_method, X_train, y_train, X_val, y_val, X_eval):
    t0 = time.perf_counter()
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
        **config,
        **xgb_tree_params(tree_method),
    }
    model = XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=balanced_sample_weight(y_train),
        eval_set=[(X_val, y_val)],
        verbose=args.xgb_verbose,
    )
    train_seconds = time.perf_counter() - t0
    return {
        "model": model,
        "train_seconds": train_seconds,
        "p_val": mp_probability(model, X_val),
        "p_eval": mp_probability(model, X_eval),
        "params": params,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="bp_rp_lamost_normalized.h5")
    parser.add_argument("--out-dir", default="results/full_data/natural_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feh-threshold", type=float, default=-2.0)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--tree-method", default="gpu_hist", choices=["gpu_hist", "cuda_hist", "hist"])
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--logistic-solver", default="saga", choices=["saga", "lbfgs"])
    parser.add_argument("--logistic-max-iter", type=int, default=100)
    parser.add_argument("--logistic-tol", type=float, default=1e-3)
    parser.add_argument("--xgb-verbose", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=max(PRECISION_KS))
    parser.add_argument("--skip-logistic", action="store_true")
    parser.add_argument("--smoke-test-only", action="store_true")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(XGB_CONFIGS.keys()),
        choices=list(XGB_CONFIGS.keys()),
    )
    return parser.parse_args()


def _run_main():
    global _ACTIVE_OUT_DIR
    total_t0 = time.perf_counter()
    args = parse_args()
    out_dir = Path(args.out_dir)
    _ACTIVE_OUT_DIR = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    xgb, XGBClassifier = import_xgboost()
    actual_tree_method, smoke_errors = smoke_test_xgboost(XGBClassifier, args.tree_method, args.seed)
    print(f"[XGBoost] version={xgb.__version__} tree_method={actual_tree_method}")
    if smoke_errors:
        print(f"[XGBoost] smoke-test fallback errors: {smoke_errors}")

    if args.smoke_test_only:
        return

    print(f"[Data] Loading {args.data_file} ...")
    X, y, feh, source_id, cols, load_seconds = load_dataset(args.data_file, args.feh_threshold)
    train_idx, val_idx, eval_idx = stratified_split(
        y, args.seed, train_frac=args.train_frac, val_frac=args.val_frac
    )
    print(f"[Data] loaded {len(y):,} rows x {X.shape[1]} features in {load_seconds:.1f}s")
    print(f"[Split] train={split_counts(y, train_idx)}")
    print(f"[Split] val={split_counts(y, val_idx)}")
    print(f"[Split] eval={split_counts(y, eval_idx)}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_eval, y_eval = X[eval_idx], y[eval_idx]
    feh_eval = feh[eval_idx]
    source_eval = source_id[eval_idx]

    data_metadata = {
        "full_population": fast_hdf5_fingerprint(args.data_file),
        "feature_count": len(cols),
        "feature_columns": cols,
        "feh_threshold": args.feh_threshold,
        "label_encoding": dict(CANONICAL_LABEL_ENCODING),
    }
    split_metadata = {
        "train_fraction": args.train_frac,
        "validation_fraction": args.val_frac,
        "actual": {
            "train": split_counts(y, train_idx),
            "validation": split_counts(y, val_idx),
            "eval": split_counts(y, eval_idx),
        },
    }
    training_metadata = {
        "requested_tree_method": args.tree_method,
        "actual_tree_method": actual_tree_method,
        "xgboost_version": xgb.__version__,
        "xgb_n_jobs": args.n_jobs,
        "xgb_verbose": args.xgb_verbose,
        "xgb_configs": {name: XGB_CONFIGS[name] for name in args.configs},
        "smoke_test_errors": smoke_errors,
        "top_k": args.top_k,
        "logistic": {
            "enabled": not args.skip_logistic,
            "C": args.logistic_c,
            "solver": args.logistic_solver,
            "max_iter": args.logistic_max_iter,
            "tol": args.logistic_tol,
        },
    }
    scientific_config = {
        "data": {
            "fingerprint": data_metadata["full_population"]["fingerprint"],
            "feh_threshold": args.feh_threshold,
            "label_encoding": dict(CANONICAL_LABEL_ENCODING),
        },
        "split": {
            "train_fraction": args.train_frac,
            "validation_fraction": args.val_frac,
            "seed": args.seed,
        },
        "training": {
            key: value
            for key, value in training_metadata.items()
            if key not in {"actual_tree_method", "smoke_test_errors", "xgboost_version"}
        },
    }
    created_at = utc_now()
    config_hash = canonical_hash(scientific_config)
    params = {
        "schema_version": SCHEMA_VERSION,
        "artifact_layout_version": ARTIFACT_LAYOUT_VERSION,
        "experiment_type": "full_data_benchmark",
        "run": {
            "run_id": created_at.replace(":", "").replace("+00:00", "Z")
            + "-" + config_hash.split(":", 1)[1][:8],
            "experiment_family": experiment_family(out_dir, "full_data_benchmark"),
            "output_dir": str(out_dir),
            "argv": list(sys.argv),
            "status": "running",
            "created_at_utc": created_at,
            "completed_at_utc": None,
            "git": git_metadata(),
            "config_hash": config_hash,
            "protocol_id": canonical_hash({
                "data": scientific_config["data"],
                "split": scientific_config["split"],
                "training_family": "balanced_full_data_model_selection",
            }),
        },
        "data": data_metadata,
        "split": split_metadata,
        "query": None,
        "reweighting": None,
        "training": training_metadata,
        "trials": {"seed": args.seed},
        "environment": environment_metadata(),
        "timing": {
            "data_load_seconds": float(load_seconds),
            "total_seconds": None,
        },
        "failure": None,
    }
    atomic_write_json(out_dir / "params.json", params)

    results = {}
    comparison_rows = []
    precision_rows = []
    curve_data = []

    def write_outputs():
        completed_xgb = [name for name in args.configs if name in results]
        results_out = dict(results)
        if completed_xgb:
            results_out["best_xgboost_by_validation_pr_auc"] = max(
                completed_xgb, key=lambda n: results[n]["validation"]["pr_auc"]
            )

        comparison_rows.sort(key=lambda r: r["eval_pr_auc"], reverse=True)
        write_csv(out_dir / "model_comparison.csv", comparison_rows)
        write_csv(out_dir / "precision_at_k.csv", precision_rows)
        with (out_dir / "results.json").open("w") as f:
            json.dump(results_out, f, indent=2)
        if curve_data:
            plot_curves(out_dir, curve_data)

    def record_run(name, run):
        val_metrics = model_metrics(y_val, run["p_val"])
        eval_metrics = model_metrics(y_eval, run["p_eval"])
        print(
            f"[Eval] {name}: val AP={val_metrics['pr_auc']:.4f} "
            f"eval AP={eval_metrics['pr_auc']:.4f} "
            f"eval ROC={eval_metrics['roc_auc']:.4f}"
        )

        results[name] = {
            "params": run["params"],
            "train_seconds": float(run["train_seconds"]),
            "validation": val_metrics,
            "eval": eval_metrics,
        }

        row = {
            "model": name,
            "validation_pr_auc": val_metrics["pr_auc"],
            "eval_pr_auc": eval_metrics["pr_auc"],
            "eval_roc_auc": eval_metrics["roc_auc"],
            "eval_precision_at_0_5": eval_metrics["precision_at_0_5"],
            "eval_recall_at_0_5": eval_metrics["recall_at_0_5"],
            "eval_f1_at_0_5": eval_metrics["f1_at_0_5"],
            "eval_predicted_mp_at_0_5": eval_metrics["predicted_mp_at_0_5"],
            "train_seconds": float(run["train_seconds"]),
        }
        comparison_rows.append(row)
        precision_rows.extend(precision_at_k_rows(name, y_eval, run["p_eval"]))
        save_top_candidates(out_dir, name, y_eval, run["p_eval"], feh_eval, source_eval, args.top_k)
        curve_data.append({
            "name": name,
            "y_true": y_eval,
            "p_mp": run["p_eval"],
            "eval_pr_auc": eval_metrics["pr_auc"],
            "eval_roc_auc": eval_metrics["roc_auc"],
        })
        write_outputs()

    for name in args.configs:
        print(f"[Train] {name} ...")
        run = train_xgb(
            args,
            XGBClassifier,
            name,
            XGB_CONFIGS[name],
            actual_tree_method,
            X_train,
            y_train,
            X_val,
            y_val,
            X_eval,
        )
        record_run(name, run)

    if not args.skip_logistic:
        print("[Train] logistic ...")
        record_run("logistic", train_logistic(args, X_train, y_train, X_val, X_eval))

    xgb_names = [name for name in args.configs if name in results]
    best_xgb = max(xgb_names, key=lambda n: results[n]["validation"]["pr_auc"])
    write_outputs()

    update_params_status(
        out_dir, "completed", total_seconds=time.perf_counter() - total_t0
    )

    print(f"[Done] best_xgboost_by_validation_pr_auc={best_xgb}")
    print(f"[Done] outputs saved to {out_dir}")


def main():
    try:
        return _run_main()
    except Exception as exc:
        if _ACTIVE_OUT_DIR is not None:
            update_params_status(_ACTIVE_OUT_DIR, "failed", error=exc)
        raise


if __name__ == "__main__":
    main()
