"""
Active Learning with Warm Start for Stellar Classification.

Workflow:
  1. Load a biased warm-start dataset (default: low T_eff stars).
  2. Load the full population dataset.
  3. Build a candidate pool = full population minus warm-start set.
  4. Iteratively query points from the pool using a chosen strategy.
  5. Every k queries, retrain logistic regression on all labeled data
     and evaluate on a random subsample of the full population.

Usage:
  python active_learning.py --strategy random --total-queries 500 --eval-every 50
  python active_learning.py --strategy uncertainty --total-queries 500 --eval-every 50
  python active_learning.py --strategy wasserstein --total-queries 200 --eval-every 20
"""

import argparse
import json
import os
import re

import h5py
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Helpers ──────────────────────────────────────────────

def _nsort(s):
    """Natural sort key for strings like 'bp_2', 'bp_10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)]


def _feature_cols(h5_keys):
    """Return ordered feature column names from an h5 key list."""
    bp = sorted([k for k in h5_keys if k.startswith("bp_")], key=_nsort)
    rp = sorted([k for k in h5_keys if k.startswith("rp_")], key=_nsort)
    cols = bp + rp
    if "ebv" in h5_keys:
        cols.append("ebv")
    return cols


def load_features_and_labels(h5_path, feh_threshold=-2.0, max_samples=None, seed=42):
    """Load BP/RP + ebv features and binary Fe/H label from an h5 file.

    Returns (X, y, source_ids) where X is L2-normalised, y ∈ {0=MP, 1=MR}.
    """
    with h5py.File(h5_path, "r") as f:
        cols = _feature_cols(list(f.keys()))
        n = f[cols[0]].shape[0]

        # Optional subsample
        idx = np.arange(n)
        if max_samples is not None and max_samples < n:
            idx = np.sort(np.random.RandomState(seed).choice(n, max_samples, replace=False))

        X = np.column_stack([np.nan_to_num(f[c][:][idx], nan=0.0).astype(np.float64) for c in cols])
        feh = f["feh"][:][idx].astype(np.float64)
        valid = np.isfinite(feh)
        X, feh = X[valid], feh[valid]
        sids = f["source_id"][:][idx][valid] if "source_id" in f else None

    # L2-normalise spectral coefficients (everything except ebv)
    end = -1 if cols[-1] == "ebv" else X.shape[1]
    norms = np.linalg.norm(X[:, :end], axis=1, keepdims=True) + 1e-8
    X[:, :end] /= norms

    y = (feh >= feh_threshold).astype(np.int32)
    return X, y, sids


# ── Query Strategies ─────────────────────────────────────
# All strategies share the signature (X_pool, clf, n, rng, **kw) → index array.

def query_random(X_pool, clf, n, rng, **kw):
    """Uniform random sampling."""
    return rng.choice(len(X_pool), min(n, len(X_pool)), replace=False)


def query_uncertainty(X_pool, clf, n, rng, **kw):
    """Pick points whose predicted probability is closest to 0.5."""
    probs = clf.predict_proba(X_pool)[:, 1]
    return np.argsort(np.abs(probs - 0.5))[:n]


def query_margin(X_pool, clf, n, rng, **kw):
    """KWIK-style: pick points with smallest |decision_function| (closest to boundary)."""
    return np.argsort(np.abs(clf.decision_function(X_pool)))[:n]


def query_wasserstein(X_pool, clf, n, rng, *, X_labeled=None, **kw):
    """Greedy core-set: iteratively pick the pool point farthest from current labeled set."""
    MAX_POOL = 5000
    sub_idx = rng.choice(len(X_pool), min(MAX_POOL, len(X_pool)), replace=False) if len(X_pool) > MAX_POOL else np.arange(len(X_pool))
    X_sub = X_pool[sub_idx]

    S = X_labeled.copy()
    chosen = []
    for _ in range(min(n, len(X_sub))):
        min_dists = cdist(X_sub, S).min(axis=1)          # (|sub|,)
        best = np.argmax(min_dists)
        chosen.append(best)
        S = np.vstack([S, X_sub[best:best + 1]])
        min_dists[best] = -1  # prevent re-selection
    return sub_idx[chosen]


STRATEGIES = {
    "random": query_random,
    "uncertainty": query_uncertainty,
    "margin": query_margin,
    "wasserstein": query_wasserstein,
}


# ── Training & Evaluation ────────────────────────────────

def train_logistic(X, y, lambda_MP=1.0, C=1.0):
    """Train logistic regression with class reweighting."""
    clf = LogisticRegression(C=C, class_weight={0: lambda_MP, 1: 1.0},
                             solver="lbfgs", max_iter=2000)
    clf.fit(X, y)
    return clf


def evaluate(clf, X, y):
    """Return a flat dict of metrics."""
    yp = clf.predict(X)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yp, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, yp)),
        "precision_MP": float(prec[0]), "recall_MP": float(rec[0]), "f1_MP": float(f1[0]),
        "precision_MR": float(prec[1]), "recall_MR": float(rec[1]), "f1_MR": float(f1[1]),
        "confusion_matrix": confusion_matrix(y, yp, labels=[0, 1]).tolist(),
    }


def _record(metrics, n_queries, y_labeled):
    """Augment a metrics dict with bookkeeping fields."""
    metrics["n_queries"] = n_queries
    metrics["n_labeled"] = len(y_labeled)
    metrics["n_labeled_MP"] = int(np.sum(y_labeled == 0))
    metrics["n_labeled_MR"] = int(np.sum(y_labeled == 1))
    return metrics


def _log(m):
    """One-line summary of a metrics snapshot."""
    print(f"[Query {m['n_queries']:4d}] Acc={m['accuracy']:.4f}  "
          f"MP(P={m['precision_MP']:.4f} R={m['recall_MP']:.4f})  "
          f"MR(P={m['precision_MR']:.4f} R={m['recall_MR']:.4f})  "
          f"labeled={m['n_labeled']} (MP={m['n_labeled_MP']}, MR={m['n_labeled_MR']})")


# ── Plotting ─────────────────────────────────────────────

def _save_plots(results, out_dir):
    """Generate learning-curve and class-distribution plots."""
    qs = [r["n_queries"] for r in results]

    # --- Learning curve ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label, color, marker in [
        ("accuracy", "Accuracy", "#4A90D9", "o"),
        ("recall_MP", "MP Recall", "#E07070", "s"),
        ("f1_MP", "MP F1", "#5A9E7A", "^"),
    ]:
        ax.plot(qs, [r[key] for r in results], f"{marker}-", label=label, color=color, lw=2)
    ax.set(xlabel="Number of Queries", ylabel="Score", ylim=(0, 1.05))
    ax.set_title("Active Learning Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "learning_curve.png"), dpi=200); plt.close(fig)

    # --- Class distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(qs, [r["n_labeled_MP"] for r in results], "o-", label="Labeled MP", color="#E07070", lw=2)
    ax.plot(qs, [r["n_labeled_MR"] for r in results], "s-", label="Labeled MR", color="#4A90D9", lw=2)
    ax.set(xlabel="Number of Queries", ylabel="Count")
    ax.set_title("Labeled Set Composition", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=200); plt.close(fig)


# ── Main Loop ────────────────────────────────────────────

def run_active_learning(args):
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load data
    print(f"Loading warm-start data from {args.warm_start_file} ...")
    X_warm, y_warm, sid_warm = load_features_and_labels(
        args.warm_start_file, args.feh_threshold, args.warm_start_max, args.seed)

    print(f"Loading full population from {args.full_data_file} ...")
    X_full, y_full, sid_full = load_features_and_labels(
        args.full_data_file, args.feh_threshold, args.pool_max, args.seed + 1)

    # 2. Build pool = full minus warm-start
    if sid_warm is not None and sid_full is not None:
        pool_mask = ~np.isin(sid_full, sid_warm)
    else:
        print("  Warning: source_id unavailable; using approximate dedup.")
        warm_set = {tuple(np.round(x, 6)) for x in X_warm}
        pool_mask = np.array([tuple(np.round(x, 6)) not in warm_set for x in X_full])

    X_pool, y_pool = X_full[pool_mask].copy(), y_full[pool_mask].copy()

    # 3. Evaluation set
    eval_n = min(args.eval_size, len(X_full))
    eval_idx = rng.choice(len(X_full), eval_n, replace=False)
    X_eval, y_eval = X_full[eval_idx], y_full[eval_idx]

    for tag, n, mp in [("Warm-start", len(X_warm), (y_warm == 0).sum()),
                       ("Full pop.", len(X_full), (y_full == 0).sum()),
                       ("Pool", len(X_pool), (y_pool == 0).sum()),
                       ("Eval set", eval_n, (y_eval == 0).sum())]:
        print(f"  {tag}: {n} (MP={mp}, MR={n - mp})")

    # 4. Initialise
    X_labeled, y_labeled = X_warm.copy(), y_warm.copy()
    available = np.ones(len(X_pool), dtype=bool)
    strategy_fn = STRATEGIES[args.strategy]
    results = []

    # Helper: train → evaluate → record → log
    def snapshot(n_queries):
        clf = train_logistic(X_labeled, y_labeled, args.lambda_MP, args.C)
        m = _record(evaluate(clf, X_eval, y_eval), n_queries, y_labeled)
        results.append(m)
        _log(m)
        return clf

    # 5. Initial evaluation
    clf = snapshot(0)

    # 6. Active learning loop
    queried = 0
    while queried < args.total_queries and available.any():
        batch = min(args.eval_every, args.total_queries - queried, available.sum())
        avail_idx = np.where(available)[0]

        sel = strategy_fn(X_pool[avail_idx], clf, batch, rng, X_labeled=X_labeled)
        pool_idx = avail_idx[sel]

        X_labeled = np.vstack([X_labeled, X_pool[pool_idx]])
        y_labeled = np.concatenate([y_labeled, y_pool[pool_idx]])
        available[pool_idx] = False
        queried += len(pool_idx)

        clf = snapshot(queried)

    # 7. Save outputs
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Final weights
    cols = _feature_cols(list(h5py.File(args.full_data_file, "r").keys()))
    w, b = clf.coef_.flatten(), clf.intercept_[0]
    with open(os.path.join(args.out_dir, "final_weights.csv"), "w") as f:
        f.write("feature,weight\n" + f"BIAS,{b}\n")
        f.writelines(f"{name},{wv}\n" for name, wv in zip(cols, w))

    with open(os.path.join(args.out_dir, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    _save_plots(results, args.out_dir)
    print(f"\nAll outputs saved to {args.out_dir}/")
    return results


# ── CLI ──────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Active Learning with warm-start for stellar Fe/H classification.")
    a = p.add_argument

    # Data
    a("--warm-start-file", default="bp_rp_lamost_normalized_low_teff.h5", help="H5 file for biased warm-start set.")
    a("--full-data-file",  default="bp_rp_lamost_normalized.h5",          help="H5 file for full population.")
    a("--feh-threshold",   type=float, default=-2.0, help="Fe/H threshold: <thr → MP(0), >=thr → MR(1).")

    # Strategy
    a("--strategy",       default="uncertainty", choices=list(STRATEGIES.keys()), help="Query strategy.")
    a("--total-queries",  type=int, default=500,  help="Total points to query.")
    a("--eval-every",     type=int, default=50,   help="Retrain & evaluate every k queries.")

    # Model
    a("--lambda-MP", type=float, default=1.0, help="Class reweight for MP in logistic regression.")
    a("--C",         type=float, default=1.0, help="Inverse regularisation strength.")

    # Practical
    a("--eval-size",       type=int, default=100_000, help="Eval subsample size.")
    a("--warm-start-max",  type=int, default=None,    help="Cap warm-start size.")
    a("--pool-max",        type=int, default=None,    help="Cap pool size.")
    a("--seed",            type=int, default=42)
    a("--out-dir",         default=None, help="Output directory (default: al_{strategy}).")

    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = f"al_{args.strategy}"
    run_active_learning(args)


if __name__ == "__main__":
    main()
