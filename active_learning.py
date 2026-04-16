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
import time

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

    Optimisations over the original:
      - Uses float32 instead of float64 (halves memory, speeds up compute).
      - Reads HDF5 slices directly via sorted fancy indexing instead of
        loading entire columns then sub-indexing.
    """
    with h5py.File(h5_path, "r") as f:
        cols = _feature_cols(list(f.keys()))
        n = f[cols[0]].shape[0]

        # Optional subsample
        if max_samples is not None and max_samples < n:
            idx = np.sort(np.random.RandomState(seed).choice(n, max_samples, replace=False))
        else:
            idx = None  # read everything – use slice for speed

        # --- Fast column read ---------------------------------------------------
        # When idx is None we read the whole dataset at once (contiguous I/O).
        # When idx is a sorted int array HDF5 can still do a relatively fast
        # fancy-index read without materialising the full column first.
        if idx is None:
            parts = [np.nan_to_num(f[c][()], nan=0.0).astype(np.float32) for c in cols]
            feh = f["feh"][()].astype(np.float32)
            sids = f["source_id"][()] if "source_id" in f else None
        else:
            parts = [np.nan_to_num(f[c][idx], nan=0.0).astype(np.float32) for c in cols]
            feh = f["feh"][idx].astype(np.float32)
            sids = f["source_id"][idx] if "source_id" in f else None

        X = np.column_stack(parts)
        del parts  # free intermediate list

    valid = np.isfinite(feh)
    if not valid.all():
        X, feh = X[valid], feh[valid]
        if sids is not None:
            sids = sids[valid]

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
    """Soft uncertainty sampling: sample proportional to proximity to decision boundary.

    Instead of deterministically picking the top-n most uncertain points
    (which can cluster around the same boundary region), we treat uncertainty
    scores as unnormalised sampling weights.  Points near p=0.5 get higher
    probability but are not guaranteed to be selected, adding diversity.
    """
    probs = clf.predict_proba(X_pool)[:, 1]
    n = min(n, len(probs))
    # score ∈ [0, 0.5]: higher means closer to boundary (more uncertain)
    scores = 0.5 - np.abs(probs - 0.5)
    scores += 1e-8  # ensure no zero weights
    weights = scores / scores.sum()
    return rng.choice(len(probs), n, replace=False, p=weights)


def query_margin(X_pool, clf, n, rng, **kw):
    """KWIK-style: pick points with smallest |decision_function| (closest to boundary)."""
    dvals = np.abs(clf.decision_function(X_pool))
    if n >= len(dvals):
        return np.arange(len(dvals))
    top_n = np.argpartition(dvals, n)[:n]
    return top_n[np.argsort(dvals[top_n])]


def query_purely_random(X_pool, clf, n, rng, **kw):
    """Uniform random sampling starting from an empty labeled set (no warm-start bias)."""
    return rng.choice(len(X_pool), min(n, len(X_pool)), replace=False)


def query_wasserstein(X_pool, clf, n, rng, *, X_labeled=None, state=None, **kw):
    """
    Greedy core-set (Wasserstein approx).
    Uses PyTorch (GPU) if available for instantaneous global search over millions of points.
    Falls back to a batched numpy/scipy implementation if PyTorch is unavailable.
    """
    if state is None:
        state = {}

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch and torch.cuda.is_available():
        return _query_wasserstein_torch(X_pool, n, rng, X_labeled, state)
    else:
        return _query_wasserstein_numpy(X_pool, n, rng, X_labeled, state)


def _query_wasserstein_torch(X_pool, n, rng, X_labeled, state):
    import torch
    device = torch.device('cuda')

    # Load everything to GPU float32 once. 1M stars * 100 features * 4 bytes ≈ 400MB.
    # 24GB is plenty for even 10M-50M stars.
    X_sub = torch.tensor(X_pool, dtype=torch.float32, device=device)
    n_pick = min(n, len(X_sub))
    
    # Precompute squared norms for GEMM-based distance: ||x-y|| = sqrt(||x||^2 + ||y||^2 - 2*x.y)
    X_sub_sq_norms = (X_sub**2).sum(dim=1)

    if 'min_dists' not in state:
        min_dists = torch.full((len(X_sub),), float('inf'), device=device)
        if X_labeled is not None and len(X_labeled) > 0:
            X_lab = torch.tensor(X_labeled, dtype=torch.float32, device=device)
            X_lab_sq_norms = (X_lab**2).sum(dim=1)
            
            # --- Conservative VRAM Optimization: Safe Dynamic Chunking + In-place GEMM ---
            try:
                props = torch.cuda.get_device_properties(device)
                free_vram = props.total_memory - torch.cuda.memory_allocated(device)
                # Compute operations create intermediate matrices. To be extremely safe,
                # we target only ~15% of the FREE VRAM for the base distance matrix chunk.
                target_elements = int(free_vram * 0.15 / 4) 
                CHUNK_L = 10000 
                CHUNK_P = max(10000, target_elements // CHUNK_L)
            except:
                CHUNK_P, CHUNK_L = 50000, 10000

            print(f"  [GPU] Initializing Wasserstein distances (Dynamic Chunks: {CHUNK_P}x{CHUNK_L})...")
            
            for start_p in range(0, len(X_sub), CHUNK_P):
                end_p = start_p + CHUNK_P
                X_p_chunk = X_sub[start_p:end_p]
                P_norms = X_sub_sq_norms[start_p:end_p].unsqueeze(1)
                
                chunk_min = torch.full((len(X_p_chunk),), float('inf'), device=device)
                for start_l in range(0, len(X_lab), CHUNK_L):
                    end_l = start_l + CHUNK_L
                    X_l_chunk = X_lab[start_l:end_l]
                    L_norms = X_lab_sq_norms[start_l:end_l].unsqueeze(0)
                    
                    # Compute Euclidean dists via GEMM: ||p-l|| = sqrt(||p||^2 + ||l||^2 - 2*p.l)
                    dists = P_norms + L_norms
                    dists.addmm_(X_p_chunk, X_l_chunk.T, beta=1.0, alpha=-2.0)
                    dists.clamp_(min=0.0).sqrt_()  # clamp for numerical safety, then L2
                    
                    torch.minimum(chunk_min, dists.min(dim=1)[0], out=chunk_min)
                    del dists # Free the large chunk immediately
                
                min_dists[start_p:end_p] = chunk_min
                if (start_p // CHUNK_P) % 5 == 0:
                    print(f"    [GPU] Initialized {min(end_p, len(X_sub))} / {len(X_sub)} stars...")
            
            del X_lab, X_lab_sq_norms
            torch.cuda.empty_cache()
    else:
        min_dists = state['min_dists'].to(device)

    chosen = []
    # Vectorized Greedy Loop
    for _ in range(n_pick):
        if torch.isinf(min_dists[0]):
            best_idx = int(rng.choice(len(X_sub)))
        else:
            best_idx = torch.argmax(min_dists).item()
            
        chosen.append(best_idx)
        
        # Speedup: ||pool - new|| = sqrt(||pool||^2 + ||new||^2 - 2 * pool @ new)
        # Using torch.mv (matrix-vector) is much faster than explicit subtraction on GPU
        new_pt = X_sub[best_idx]
        new_pt_sq_norm = X_sub_sq_norms[best_idx]
        new_dists = (X_sub_sq_norms + new_pt_sq_norm - 2 * torch.mv(X_sub, new_pt)).clamp(min=0.0).sqrt()
        
        torch.minimum(min_dists, new_dists, out=min_dists)
        min_dists[best_idx] = -1.0

    mask = torch.ones(len(min_dists), dtype=torch.bool, device=device)
    mask[chosen] = False
    state['min_dists'] = min_dists[mask]
    
    return np.array(chosen, dtype=np.intp)


def _query_wasserstein_numpy(X_pool, n, rng, X_labeled, state):
    """Fallback numpy implementation that uses the same global logic but limits initial search chunking on CPU."""
    X_sub = X_pool
    n_pick = min(n, len(X_sub))

    if 'min_dists' not in state:
        min_dists = np.full(len(X_sub), np.inf, dtype=np.float32)
        if X_labeled is not None and len(X_labeled) > 0:
            print("  [CPU] Initializing global Wasserstein distances (this might take a while on CPU)...")
            # Limit chunk size strictly to avoid RAM explosion
            CHUNK_P, CHUNK_L = 20000, 5000
            for start_p in range(0, len(X_sub), CHUNK_P):
                end_p = min(start_p + CHUNK_P, len(X_sub))
                X_p_chunk = X_sub[start_p:end_p]
                for start_l in range(0, len(X_labeled), CHUNK_L):
                    end_l = min(start_l + CHUNK_L, len(X_labeled))
                    # Euclidean distance
                    dists = cdist(X_p_chunk, X_labeled[start_l:end_l], metric="euclidean")
                    np.minimum(min_dists[start_p:end_p], dists.min(axis=1), out=min_dists[start_p:end_p])
    else:
        min_dists = state['min_dists']

    chosen = []
    for _ in range(n_pick):
        if np.isinf(min_dists[0]):
            best_idx = int(rng.choice(len(X_sub)))
        else:
            best_idx = int(np.argmax(min_dists))
            
        chosen.append(best_idx)
        new_dists = np.sqrt(((X_sub - X_sub[best_idx]) ** 2).sum(axis=1))
        np.minimum(min_dists, new_dists, out=min_dists)
        min_dists[best_idx] = -1.0

    mask = np.ones(len(min_dists), dtype=bool)
    mask[chosen] = False
    state['min_dists'] = min_dists[mask]
    
    return np.array(chosen, dtype=np.intp)


# ── Voronoi Reweighting (for wasserstein_weighted) ───────

def compute_voronoi_weights(X_pool, X_labeled, voronoi_state=None):
    """Compute optimal sample weights for labeled points that minimise
    W_2(Uniform(pool), Weighted(labeled)).

    Solution: assign each pool point to its nearest labeled point
    (Voronoi partition).  Weight of labeled point i equals the fraction
    of pool points assigned to it.

    Supports **incremental updates**: pass a `voronoi_state` dict that
    persists across calls.  On the first call the full assignment is
    computed; on subsequent calls only distances to *newly added* labeled
    points are evaluated and the cached assignments are patched in-place.
    This reduces per-snapshot cost from O(pool × labeled) to
    O(pool × n_new) — typically a ~100× speedup in steady state.

    Returns (weights, voronoi_state) where weights has shape (n_labeled,)
    scaled so that sum(weights) == n_labeled.
    """
    if voronoi_state is None:
        voronoi_state = {}

    try:
        import torch
        if torch.cuda.is_available():
            w = _voronoi_weights_torch(X_pool, X_labeled, voronoi_state)
            return w, voronoi_state
    except ImportError:
        pass
    w = _voronoi_weights_numpy(X_pool, X_labeled, voronoi_state)
    return w, voronoi_state


def _voronoi_weights_torch(X_pool, X_labeled, state):
    import torch
    device = torch.device('cuda')

    n_pool = len(X_pool)
    n_labeled = len(X_labeled)

    # --- Incremental path: only check newly added labeled points ---
    prev_n = state.get('n_labeled', 0)
    if prev_n > 0 and prev_n < n_labeled and 'nearest_idx' in state:
        # Only compute distances to the NEW labeled points [prev_n : n_labeled]
        X_new = torch.tensor(X_labeled[prev_n:], dtype=torch.float32, device=device)
        X_new_sq = (X_new ** 2).sum(dim=1)         # (n_new,)
        n_new = n_labeled - prev_n

        nearest_idx = state['nearest_idx']         # (n_pool,) int64 on CPU
        nearest_dist = state['nearest_dist']       # (n_pool,) float32 on CPU

        try:
            props = torch.cuda.get_device_properties(device)
            free_vram = props.total_memory - torch.cuda.memory_allocated(device)
            target_elements = int(free_vram * 0.4 / 4)
            CHUNK_P = max(5000, target_elements // max(n_new, 1))
        except:
            CHUNK_P = 100000

        X_p = torch.tensor(X_pool, dtype=torch.float32, device=device)

        for start_p in range(0, n_pool, CHUNK_P):
            end_p = min(start_p + CHUNK_P, n_pool)
            chunk_p = X_p[start_p:end_p]
            chunk_p_sq = (chunk_p ** 2).sum(dim=1)

            # Full euclidean distance to new points: sqrt(||p||^2 + ||l||^2 - 2*p@l)
            dists = chunk_p_sq.unsqueeze(1) + X_new_sq.unsqueeze(0)
            dists.addmm_(chunk_p, X_new.T, beta=1.0, alpha=-2.0)
            dists.clamp_(min=0.0).sqrt_()

            chunk_min_dists, chunk_argmin = dists.min(dim=1)
            # Shift indices to global labeled index space
            chunk_argmin += prev_n

            # Compare with cached nearest distances (on CPU, then update)
            chunk_min_np = chunk_min_dists.cpu().numpy()
            chunk_arg_np = chunk_argmin.cpu().numpy()
            cached_slice = nearest_dist[start_p:end_p]

            improved = chunk_min_np < cached_slice
            nearest_dist[start_p:end_p][improved] = chunk_min_np[improved]
            nearest_idx[start_p:end_p][improved] = chunk_arg_np[improved]

            del dists, chunk_min_dists, chunk_argmin

        del X_p, X_new
        torch.cuda.empty_cache()

        state['n_labeled'] = n_labeled
        # state['nearest_idx'] and state['nearest_dist'] are updated in-place

        counts = np.bincount(nearest_idx, minlength=n_labeled)
        weights = counts.astype(np.float64) / counts.sum()
        return weights * n_labeled

    # --- Full computation (first call) ---
    X_p = torch.tensor(X_pool, dtype=torch.float32, device=device)
    X_l = torch.tensor(X_labeled, dtype=torch.float32, device=device)
    X_p_sq = (X_p ** 2).sum(dim=1)
    X_l_sq = (X_l ** 2).sum(dim=1)

    try:
        props = torch.cuda.get_device_properties(device)
        free_vram = props.total_memory - torch.cuda.memory_allocated(device)
        target_elements = int(free_vram * 0.4 / 4)
        CHUNK_P = max(5000, target_elements // max(n_labeled, 1))
    except:
        CHUNK_P = 50000

    nearest_idx = np.empty(n_pool, dtype=np.int64)
    nearest_dist = np.full(n_pool, np.inf, dtype=np.float32)

    for start_p in range(0, n_pool, CHUNK_P):
        end_p = min(start_p + CHUNK_P, n_pool)
        chunk_p = X_p[start_p:end_p]

        # Full euclidean dist for caching: sqrt(||p||^2 + ||l||^2 - 2*p@l)
        dists = X_p_sq[start_p:end_p].unsqueeze(1) + X_l_sq.unsqueeze(0)
        dists.addmm_(chunk_p, X_l.T, beta=1.0, alpha=-2.0)
        dists.clamp_(min=0.0).sqrt_()

        chunk_min_dists, chunk_argmin = dists.min(dim=1)
        nearest_dist[start_p:end_p] = chunk_min_dists.cpu().numpy()
        nearest_idx[start_p:end_p] = chunk_argmin.cpu().numpy()
        del dists, chunk_min_dists, chunk_argmin

    del X_p, X_l
    torch.cuda.empty_cache()

    # Cache for next incremental call
    state['nearest_idx'] = nearest_idx
    state['nearest_dist'] = nearest_dist
    state['n_labeled'] = n_labeled

    counts = np.bincount(nearest_idx, minlength=n_labeled)
    weights = counts.astype(np.float64) / counts.sum()
    return weights * n_labeled


def _voronoi_weights_numpy(X_pool, X_labeled, state=None):
    """CPU fallback for Voronoi weight computation (no incremental support)."""
    n_pool = len(X_pool)
    n_labeled = len(X_labeled)
    counts = np.zeros(n_labeled, dtype=np.int64)

    CHUNK_P, CHUNK_L = 20000, 5000
    for start_p in range(0, n_pool, CHUNK_P):
        end_p = min(start_p + CHUNK_P, n_pool)
        best_dists = np.full(end_p - start_p, np.inf, dtype=np.float32)
        best_indices = np.zeros(end_p - start_p, dtype=np.int64)

        for start_l in range(0, n_labeled, CHUNK_L):
            end_l = min(start_l + CHUNK_L, n_labeled)
            dists = cdist(X_pool[start_p:end_p], X_labeled[start_l:end_l],
                          metric="euclidean").astype(np.float32)
            chunk_min = dists.min(axis=1)
            chunk_argmin = dists.argmin(axis=1) + start_l
            improved = chunk_min < best_dists
            best_dists[improved] = chunk_min[improved]
            best_indices[improved] = chunk_argmin[improved]

        np.add.at(counts, best_indices, 1)

    weights = counts.astype(np.float64) / counts.sum()
    weights = weights * n_labeled
    return weights


STRATEGIES = {
    "random": query_random,
    "uncertainty": query_uncertainty,
    "margin": query_margin,
    "wasserstein": query_wasserstein,
    "wasserstein_weighted": query_wasserstein,  # same query; different training
    "purely_random": query_purely_random,
}


# ── Training & Evaluation ────────────────────────────────

def train_logistic(X, y, lambda_MP=1.0, C=1.0, prev_clf=None, sample_weight=None):
    """Train logistic regression with guaranteed class weight totals.

    Regardless of the per-sample weights provided (e.g. Voronoi weights),
    the final training weights are rescaled so that:
        sum of weights on MP (class 0) == lambda_MP
        sum of weights on MR (class 1) == 1.0

    Within each class, the relative weights from sample_weight are preserved
    (i.e. the Voronoi density correction is respected), but the class-level
    totals are locked to the lambda_MP ratio.

    If prev_clf is given, its coefficients are used to warm-start LBFGS
    so that convergence takes only a few iterations.
    """
    n_MP, n_MR = int(np.sum(y == 0)), int(np.sum(y == 1))

    # Start from per-sample base weights
    if sample_weight is not None:
        sw = np.array(sample_weight, dtype=np.float64)
    else:
        sw = np.ones(len(y), dtype=np.float64)

    # Rescale each class so totals are exactly lambda_MP (MP) and 1.0 (MR)
    final_w = np.empty_like(sw)
    mp_mask = (y == 0)
    mr_mask = (y == 1)

    if n_MP > 0:
        sum_mp = sw[mp_mask].sum()
        final_w[mp_mask] = sw[mp_mask] * (lambda_MP / sum_mp) if sum_mp > 0 else lambda_MP / n_MP

    if n_MR > 0:
        sum_mr = sw[mr_mask].sum()
        final_w[mr_mask] = sw[mr_mask] * (1.0 / sum_mr) if sum_mr > 0 else 1.0 / n_MR

    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                             warm_start=True)
    # Seed from previous solution so LBFGS starts near the optimum
    if prev_clf is not None:
        clf.coef_ = prev_clf.coef_.copy()
        clf.intercept_ = prev_clf.intercept_.copy()
        clf.classes_ = prev_clf.classes_.copy()
    clf.fit(X, y, sample_weight=final_w)
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
        ("precision_MP", "MP Precision", "#5A9E7A", "^"),
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

    t0 = time.perf_counter()

    # 1. Load data
    print(f"Loading warm-start data from {args.warm_start_file} ...")
    X_warm, y_warm, sid_warm = load_features_and_labels(
        args.warm_start_file, args.feh_threshold, args.warm_start_max, args.seed)

    print(f"Loading full population from {args.full_data_file} ...")
    X_full, y_full, sid_full = load_features_and_labels(
        args.full_data_file, args.feh_threshold, args.pool_max, args.seed + 1)

    t_load = time.perf_counter() - t0
    print(f"  Data loaded in {t_load:.1f}s")

    # 2. Build pool = full minus warm-start
    if sid_warm is not None and sid_full is not None:
        pool_mask = ~np.isin(sid_full, sid_warm)
    else:
        print("  Warning: source_id unavailable; using approximate dedup.")
        warm_set = {tuple(np.round(x, 6)) for x in X_warm}
        pool_mask = np.array([tuple(np.round(x, 6)) not in warm_set for x in X_full])

    X_pool, y_pool = X_full[pool_mask].copy(), y_full[pool_mask].copy()

    # Free the full arrays (only pool & eval are needed hereafter)
    del X_full, y_full, sid_full, sid_warm

    # 3. Evaluation set
    eval_n = min(args.eval_size, len(X_pool))
    eval_idx = rng.choice(len(X_pool), eval_n, replace=False)
    X_eval, y_eval = X_pool[eval_idx], y_pool[eval_idx]

    for tag, n, mp in [("Warm-start", len(X_warm), (y_warm == 0).sum()),
                       ("Pool", len(X_pool), (y_pool == 0).sum()),
                       ("Eval set", eval_n, (y_eval == 0).sum())]:
        print(f"  {tag}: {n} (MP={mp}, MR={n - mp})")

    # 4. Initialise — pre-allocate labeled arrays to avoid repeated vstack
    max_labeled = len(X_warm) + args.total_queries
    n_features = X_warm.shape[1]
    X_labeled = np.empty((max_labeled, n_features), dtype=np.float32)
    y_labeled = np.empty(max_labeled, dtype=np.int32)
    if args.strategy == "purely_random":
        # Start from an empty labeled set — ignore the biased warm-start data.
        n_labeled = 0
        del X_warm, y_warm
    else:
        n_labeled = len(X_warm)
        X_labeled[:n_labeled] = X_warm
        y_labeled[:n_labeled] = y_warm
        del X_warm, y_warm  # free

    available = np.ones(len(X_pool), dtype=bool)
    strategy_fn = STRATEGIES[args.strategy]
    results = []

    # Helper: train → evaluate → record → log
    voronoi_state = {}  # persisted across snapshots for incremental Voronoi updates

    def snapshot(n_queries, prev_clf=None):
        nonlocal voronoi_state
        Xl, yl = X_labeled[:n_labeled], y_labeled[:n_labeled]
        if len(np.unique(yl)) < 2:
            # Both classes required; skip this checkpoint and keep previous clf.
            print(f"[Query {n_queries:4d}] Skipped — only one class in labeled set so far.")
            return prev_clf

        # Voronoi reweighting for wasserstein_weighted strategy
        sw = None
        if args.strategy == "wasserstein_weighted":
            print(f"  [Voronoi] Computing optimal sample weights ({n_labeled} labeled vs {len(X_pool)} pool)...")
            sw, voronoi_state = compute_voronoi_weights(X_pool, Xl, voronoi_state)

        clf = train_logistic(Xl, yl, args.lambda_MP, args.C, prev_clf=prev_clf,
                             sample_weight=sw)
        m = _record(evaluate(clf, X_eval, y_eval), n_queries, yl)
        results.append(m)
        _log(m)
        return clf

    # 5. Initial evaluation
    # For purely_random the labeled set starts empty, so skip the initial fit.
    if args.strategy != "purely_random":
        clf = snapshot(0)
    else:
        clf = None

    # 6. Active learning loop
    queried = 0
    strategy_state = {}
    
    while queried < args.total_queries and available.any():
        batch = min(args.eval_every, args.total_queries - queried, int(available.sum()))
        avail_idx = np.where(available)[0]

        sel = strategy_fn(X_pool[avail_idx], clf, batch, rng,
                          X_labeled=X_labeled[:n_labeled], state=strategy_state)
        pool_idx = avail_idx[sel]

        # Append to pre-allocated arrays (no vstack/concatenate)
        n_new = len(pool_idx)
        X_labeled[n_labeled:n_labeled + n_new] = X_pool[pool_idx]
        y_labeled[n_labeled:n_labeled + n_new] = y_pool[pool_idx]
        n_labeled += n_new
        available[pool_idx] = False
        queried += n_new

        clf = snapshot(queried, prev_clf=clf)

    t_total = time.perf_counter() - t0
    print(f"\nTotal runtime: {t_total:.1f}s  (data loading: {t_load:.1f}s)")

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
    a("--lambda-MP", type=float, default=1.0, help="Desired total-weight ratio MP/MR. Per-sample weights are auto-scaled so n_MP*w_MP / n_MR*w_MR = lambda_MP.")
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
