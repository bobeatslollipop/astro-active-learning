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


def query_entropy(X_pool, clf, n, rng, **kw):
    """Entropy sampling: sample proportional to Shannon entropy.

    Treats the predicted probability's entropy H(p) = -p*log(p) - (1-p)*log(1-p)
    as unnormalised sampling weights. The difference from 'uncertainty' is that
    Entropy has fatter tails for high/low confident predictions.
    """
    probs = clf.predict_proba(X_pool)[:, 1]
    n = min(n, len(probs))
    
    # Calculate binary entropy: -p log2(p) - (1-p) log2(1-p)
    eps = 1e-15
    p_clipped = np.clip(probs, eps, 1.0 - eps)
    scores = -p_clipped * np.log2(p_clipped) - (1 - p_clipped) * np.log2(1 - p_clipped)
    
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


def query_wasserstein(X_pool, clf, n, rng, *, X_labeled=None, state=None, pool_size=5000, **kw):
    """
    Approximate Wasserstein sampling via optimal coupling (skAI-style).

    Randomly samples a subpool of ``pool_size`` points from X_pool to serve
    as both the empirical target distribution and the candidate search set.
    Then greedily selects n points from the subpool that minimise the
    Weighted Wasserstein Distance:

        WWD(S, T) = (1/|T|) * Σ_{t∈T} min_{s∈S} ||t − s||

    where  S = labeled ∪ already-selected  and  T = subpool.
    At each greedy step the candidate whose addition yields the lowest WWD
    is chosen — equivalent to the skAI ``find_Set`` algorithm but restricted
    to a random subpool for tractability.

    Parameters
    ----------
    pool_size : int
        Number of candidate points subsampled from X_pool.  Controls the
        approximation quality vs. compute trade-off.  The brute-force
        search is O(n × pool_size²), so keep this manageable (1 000–10 000).
    """
    n_pool = len(X_pool)
    n_pick = min(n, n_pool)
    effective_ps = min(pool_size, n_pool)

    # Random subpool — serves as both target distribution and candidate set
    subpool_idx = rng.choice(n_pool, effective_ps, replace=False)
    T = X_pool[subpool_idx]  # (effective_ps, d)

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch and torch.cuda.is_available():
        chosen = _wasserstein_coupling_torch(T, X_labeled, n_pick, rng)
    else:
        chosen = _wasserstein_coupling_numpy(T, X_labeled, n_pick, rng)

    return subpool_idx[np.array(chosen, dtype=np.intp)]


def _init_min_dists_torch(X_sub, X_sub_sq_norms, X_labeled, state, label="Wasserstein"):
    """Shared GPU initialisation of min-distance vector from labeled points.

    Returns min_dists tensor on GPU.  Reads/writes state['min_dists'].
    """
    import torch
    device = X_sub.device

    if 'min_dists' not in state:
        min_dists = torch.full((len(X_sub),), float('inf'), device=device)
        if X_labeled is not None and len(X_labeled) > 0:
            X_lab = torch.tensor(X_labeled, dtype=torch.float32, device=device)
            X_lab_sq_norms = (X_lab**2).sum(dim=1)

            try:
                props = torch.cuda.get_device_properties(device)
                free_vram = props.total_memory - torch.cuda.memory_allocated(device)
                target_elements = int(free_vram * 0.15 / 4)
                CHUNK_L = 10000
                CHUNK_P = max(10000, target_elements // CHUNK_L)
            except:
                CHUNK_P, CHUNK_L = 50000, 10000

            print(f"  [GPU] Initializing {label} distances (Dynamic Chunks: {CHUNK_P}x{CHUNK_L})...")

            for start_p in range(0, len(X_sub), CHUNK_P):
                end_p = start_p + CHUNK_P
                X_p_chunk = X_sub[start_p:end_p]
                P_norms = X_sub_sq_norms[start_p:end_p].unsqueeze(1)

                chunk_min = torch.full((len(X_p_chunk),), float('inf'), device=device)
                for start_l in range(0, len(X_lab), CHUNK_L):
                    end_l = start_l + CHUNK_L
                    X_l_chunk = X_lab[start_l:end_l]
                    L_norms = X_lab_sq_norms[start_l:end_l].unsqueeze(0)

                    dists = P_norms + L_norms
                    dists.addmm_(X_p_chunk, X_l_chunk.T, beta=1.0, alpha=-2.0)
                    dists.clamp_(min=0.0).sqrt_()

                    torch.minimum(chunk_min, dists.min(dim=1)[0], out=chunk_min)
                    del dists

                min_dists[start_p:end_p] = chunk_min
                if (start_p // CHUNK_P) % 5 == 0:
                    print(f"    [GPU] Initialized {min(end_p, len(X_sub))} / {len(X_sub)} stars...")

            del X_lab, X_lab_sq_norms
            torch.cuda.empty_cache()
    else:
        min_dists = state['min_dists'].to(device)

    return min_dists


def _init_min_dists_numpy(X_sub, X_labeled, state, label="Wasserstein"):
    """Shared CPU initialisation of min-distance vector from labeled points.

    Returns min_dists numpy array.  Reads/writes state['min_dists'].
    """
    if 'min_dists' not in state:
        min_dists = np.full(len(X_sub), np.inf, dtype=np.float32)
        if X_labeled is not None and len(X_labeled) > 0:
            print(f"  [CPU] Initializing global {label} distances (this might take a while on CPU)...")
            CHUNK_P, CHUNK_L = 20000, 5000
            for start_p in range(0, len(X_sub), CHUNK_P):
                end_p = min(start_p + CHUNK_P, len(X_sub))
                X_p_chunk = X_sub[start_p:end_p]
                for start_l in range(0, len(X_labeled), CHUNK_L):
                    end_l = min(start_l + CHUNK_L, len(X_labeled))
                    dists = cdist(X_p_chunk, X_labeled[start_l:end_l], metric="euclidean")
                    np.minimum(min_dists[start_p:end_p], dists.min(axis=1), out=min_dists[start_p:end_p])
    else:
        min_dists = state['min_dists']

    return min_dists


def _update_min_dists_torch(min_dists, X_sub, X_sub_sq_norms, best_idx):
    """Update min_dists after selecting best_idx (GPU)."""
    import torch
    new_pt = X_sub[best_idx]
    new_pt_sq_norm = X_sub_sq_norms[best_idx]
    new_dists = (X_sub_sq_norms + new_pt_sq_norm - 2 * torch.mv(X_sub, new_pt)).clamp(min=0.0).sqrt()
    torch.minimum(min_dists, new_dists, out=min_dists)
    min_dists[best_idx] = -1.0


def _update_min_dists_numpy(min_dists, X_sub, best_idx):
    """Update min_dists after selecting best_idx (CPU)."""
    new_dists = np.sqrt(((X_sub - X_sub[best_idx]) ** 2).sum(axis=1))
    np.minimum(min_dists, new_dists, out=min_dists)
    min_dists[best_idx] = -1.0


def _finalize_state(min_dists, chosen, state, is_torch=False):
    """Remove chosen indices from min_dists and cache in state."""
    if is_torch:
        import torch
        mask = torch.ones(len(min_dists), dtype=torch.bool, device=min_dists.device)
        mask[chosen] = False
        state['min_dists'] = min_dists[mask]
    else:
        mask = np.ones(len(min_dists), dtype=bool)
        mask[chosen] = False
        state['min_dists'] = min_dists[mask]


def _wasserstein_coupling_numpy(T, X_labeled, n_pick, rng):
    """CPU brute-force Wasserstein coupling (skAI-style find_Set).

    For each greedy step, evaluates WWD(S ∪ {u}, T) for every remaining
    candidate u in T and picks the one that minimises it.  Uses an
    incremental update: after selecting u*, only the columns j where
    base_min actually decreased are re-evaluated, reducing per-step
    cost from O(ps²) to O(ps × |changed|).
    """
    ps = len(T)

    # Pairwise distances within the subpool: (ps, ps)
    intra_dists = cdist(T, T, metric='euclidean').astype(np.float32)

    # Base min distances: for each subpool point j, min ||t_j − s|| over labeled set S
    base_min = np.full(ps, np.inf, dtype=np.float32)
    if X_labeled is not None and len(X_labeled) > 0:
        CHUNK = 5000
        for start in range(0, len(X_labeled), CHUNK):
            end = min(start + CHUNK, len(X_labeled))
            dists = cdist(T, X_labeled[start:end], metric='euclidean').astype(np.float32)
            np.minimum(base_min, dists.min(axis=1), out=base_min)
            del dists

    print(f"  [CPU] Wasserstein coupling: pool_size={ps}, selecting {n_pick}")

    # Initial full WWD computation for all candidates
    wwds = np.minimum(base_min[np.newaxis, :], intra_dists).mean(axis=1)  # (ps,)

    chosen = []
    available = np.ones(ps, dtype=bool)

    for k in range(n_pick):
        wwds_masked = wwds.copy()
        wwds_masked[~available] = np.inf

        best = int(np.argmin(wwds_masked))
        chosen.append(best)
        available[best] = False

        # Update base_min and find which columns j changed
        old_base = base_min.copy()
        np.minimum(base_min, intra_dists[best], out=base_min)
        changed = np.where(base_min < old_base)[0]

        if len(changed) > 0:
            # Incremental update: only recompute the contribution of changed columns
            old_contribs = np.minimum(old_base[changed], intra_dists[:, changed])  # (ps, |changed|)
            new_contribs = np.minimum(base_min[changed], intra_dists[:, changed])  # (ps, |changed|)
            wwds += (new_contribs - old_contribs).sum(axis=1) / ps

    return chosen


def _wasserstein_coupling_torch(T, X_labeled, n_pick, rng):
    """GPU brute-force Wasserstein coupling (skAI-style find_Set).

    Uses incremental WWD updates: after each selection, only columns
    where base_min decreased are re-evaluated.
    """
    import torch
    device = torch.device('cuda')
    ps = len(T)

    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    T_sq = (T_t ** 2).sum(dim=1)

    # Pairwise distances within subpool
    intra_dists = T_sq.unsqueeze(1) + T_sq.unsqueeze(0)
    intra_dists.addmm_(T_t, T_t.T, beta=1.0, alpha=-2.0)
    intra_dists.clamp_(min=0.0).sqrt_()

    # Base min distances to labeled set
    base_min = torch.full((ps,), float('inf'), device=device)
    if X_labeled is not None and len(X_labeled) > 0:
        X_l = torch.tensor(X_labeled, dtype=torch.float32, device=device)
        X_l_sq = (X_l ** 2).sum(dim=1)
        CHUNK = 10000
        for start in range(0, len(X_l), CHUNK):
            end = min(start + CHUNK, len(X_l))
            dists = T_sq.unsqueeze(1) + X_l_sq[start:end].unsqueeze(0)
            dists.addmm_(T_t, X_l[start:end].T, beta=1.0, alpha=-2.0)
            dists.clamp_(min=0.0).sqrt_()
            torch.minimum(base_min, dists.min(dim=1)[0], out=base_min)
            del dists
        del X_l, X_l_sq
        torch.cuda.empty_cache()

    print(f"  [GPU] Wasserstein coupling: pool_size={ps}, selecting {n_pick}")

    # Initial full WWD computation
    wwds = torch.minimum(base_min.unsqueeze(0).expand(ps, ps), intra_dists).mean(dim=1)

    chosen = []
    available = torch.ones(ps, dtype=torch.bool, device=device)

    for k in range(n_pick):
        wwds_masked = wwds.clone()
        wwds_masked[~available] = float('inf')

        best = torch.argmin(wwds_masked).item()
        chosen.append(best)
        available[best] = False

        # Update base_min and find changed columns
        old_base = base_min.clone()
        torch.minimum(base_min, intra_dists[best], out=base_min)
        changed = torch.where(base_min < old_base)[0]

        if len(changed) > 0:
            old_contribs = torch.minimum(old_base[changed], intra_dists[:, changed])
            new_contribs = torch.minimum(base_min[changed], intra_dists[:, changed])
            wwds += (new_contribs - old_contribs).sum(dim=1) / ps

    del T_t, intra_dists
    torch.cuda.empty_cache()
    return chosen


# ── k-Median++ Sampling ──────────────────────────────────
# Core-set / farthest-first distance maintenance, but replaces
# the greedy argmax with D(x) sampling ∝ min_dist (k-median++ init).

def query_kmedianpp(X_pool, clf, n, rng, *, X_labeled=None, state=None, **kw):
    """
    k-Median++ style sampling.

    Uses farthest-first / core-set distance bookkeeping, but instead
    of deterministically picking argmax(min_dist), each new point
    is sampled with probability proportional to its min distance to the
    already-labeled set.  This is the classical D(x) sampling from
    Arthur & Vassilvitskii (2007) generalised from k-means to the
    active-learning core-set setting.

    Advantages over greedy argmax:
      • Introduces controlled randomness → less susceptible to outlier
        attraction and boundary artifacts.
      • Still biases selection toward under-represented regions.
    """
    if state is None:
        state = {}

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch and torch.cuda.is_available():
        return _query_kmedianpp_torch(X_pool, n, rng, X_labeled, state)
    else:
        return _query_kmedianpp_numpy(X_pool, n, rng, X_labeled, state)


def _query_kmedianpp_torch(X_pool, n, rng, X_labeled, state):
    import torch
    device = torch.device('cuda')

    X_sub = torch.tensor(X_pool, dtype=torch.float32, device=device)
    n_pick = min(n, len(X_sub))
    X_sub_sq_norms = (X_sub**2).sum(dim=1)

    min_dists = _init_min_dists_torch(X_sub, X_sub_sq_norms, X_labeled, state, label="k-Median++")

    chosen = []
    for _ in range(n_pick):
        if torch.isinf(min_dists[0]):
            best_idx = int(rng.choice(len(X_sub)))
        else:
            # D(x) sampling: probability ∝ min_dist
            # Clamp negatives (already-chosen sentinels) to 0
            weights = min_dists.clamp(min=0.0)
            total = weights.sum().item()
            if total <= 0:
                # All distances are zero/negative — fall back to uniform
                best_idx = int(rng.choice(len(X_sub)))
            else:
                probs = (weights / total).cpu().numpy()
                best_idx = int(rng.choice(len(X_sub), p=probs))

        chosen.append(best_idx)
        _update_min_dists_torch(min_dists, X_sub, X_sub_sq_norms, best_idx)

    _finalize_state(min_dists, chosen, state, is_torch=True)
    return np.array(chosen, dtype=np.intp)


def _query_kmedianpp_numpy(X_pool, n, rng, X_labeled, state):
    """CPU fallback for k-Median++ sampling."""
    X_sub = X_pool
    n_pick = min(n, len(X_sub))

    min_dists = _init_min_dists_numpy(X_sub, X_labeled, state, label="k-Median++")

    chosen = []
    for _ in range(n_pick):
        if np.isinf(min_dists[0]):
            best_idx = int(rng.choice(len(X_sub)))
        else:
            # D(x) sampling: probability ∝ min_dist
            weights = np.maximum(min_dists, 0.0)
            total = weights.sum()
            if total <= 0:
                best_idx = int(rng.choice(len(X_sub)))
            else:
                probs = weights / total
                best_idx = int(rng.choice(len(X_sub), p=probs))

        chosen.append(best_idx)
        _update_min_dists_numpy(min_dists, X_sub, best_idx)

    _finalize_state(min_dists, chosen, state, is_torch=False)
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


# ── Soft Voronoi Reweighting (temperature softmin) ───────

def compute_soft_voronoi_weights(X_pool, X_labeled, temperature=1.0):
    """Compute soft Voronoi weights via temperature-scaled softmin.

    For each pool point p_i, a soft assignment over labeled points is:
        w_{ij} = exp(-||p_i - l_j|| / τ)  /  Σ_k exp(-||p_i - l_k|| / τ)
    The weight of labeled point j is the average assignment:
        weight_j = (1/n_pool) Σ_i w_{ij}

    τ → 0  converges to hard Voronoi (argmin).
    τ → ∞  converges to uniform weights.

    Returns weights array of shape (n_labeled,), scaled so that
    sum(weights) == n_labeled.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return _soft_voronoi_torch(X_pool, X_labeled, temperature)
    except ImportError:
        pass
    return _soft_voronoi_numpy(X_pool, X_labeled, temperature)


def _soft_voronoi_torch(X_pool, X_labeled, temperature):
    import torch
    device = torch.device('cuda')

    n_pool = len(X_pool)
    n_labeled = len(X_labeled)

    X_p = torch.tensor(X_pool, dtype=torch.float32, device=device)
    X_l = torch.tensor(X_labeled, dtype=torch.float32, device=device)
    X_p_sq = (X_p ** 2).sum(dim=1)
    X_l_sq = (X_l ** 2).sum(dim=1)

    # Dynamic chunk sizing — softmax needs the dist matrix + exp buffer,
    # so be slightly more conservative than hard Voronoi
    try:
        props = torch.cuda.get_device_properties(device)
        free_vram = props.total_memory - torch.cuda.memory_allocated(device)
        target_elements = int(free_vram * 0.25 / 4)
        CHUNK_P = max(5000, target_elements // max(n_labeled, 1))
    except:
        CHUNK_P = 30000

    # Accumulate soft assignment sums for each labeled point
    weight_accum = torch.zeros(n_labeled, dtype=torch.float64, device=device)

    for start_p in range(0, n_pool, CHUNK_P):
        end_p = min(start_p + CHUNK_P, n_pool)
        chunk_p = X_p[start_p:end_p]

        # Euclidean distances: sqrt(||p||^2 + ||l||^2 - 2*p@l)
        dists = X_p_sq[start_p:end_p].unsqueeze(1) + X_l_sq.unsqueeze(0)
        dists.addmm_(chunk_p, X_l.T, beta=1.0, alpha=-2.0)
        dists.clamp_(min=0.0).sqrt_()

        # Softmin: softmax(-dist / τ) along labeled dimension
        # torch.softmax handles numerical stability (log-sum-exp) internally
        logits = dists.neg_().div_(temperature)       # -dist / τ, in-place
        soft_assign = torch.softmax(logits, dim=1)    # (CHUNK_P, n_labeled)

        weight_accum += soft_assign.sum(dim=0).to(torch.float64)
        del dists, logits, soft_assign

    weights = weight_accum / n_pool
    weights = (weights * n_labeled).cpu().numpy().astype(np.float64)

    del X_p, X_l
    torch.cuda.empty_cache()
    return weights


def _soft_voronoi_numpy(X_pool, X_labeled, temperature):
    """CPU fallback for soft Voronoi weight computation."""
    n_pool = len(X_pool)
    n_labeled = len(X_labeled)
    weight_accum = np.zeros(n_labeled, dtype=np.float64)

    CHUNK_P = 10000
    for start_p in range(0, n_pool, CHUNK_P):
        end_p = min(start_p + CHUNK_P, n_pool)
        dists = cdist(X_pool[start_p:end_p], X_labeled, metric='euclidean')
        logits = -dists / temperature
        # Numerically stable softmax
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        soft_assign = exp_l / exp_l.sum(axis=1, keepdims=True)
        weight_accum += soft_assign.sum(axis=0)

    weights = weight_accum / n_pool
    weights = weights * n_labeled
    return weights

STRATEGIES = {
    "random": query_random,
    "uncertainty": query_uncertainty,
    "entropy": query_entropy,
    "margin": query_margin,
    "wasserstein": query_wasserstein,
    "kmedianpp": query_kmedianpp,
    "purely_random": query_purely_random,
}


# ── Training & Evaluation ────────────────────────────────

def train_logistic(X, y, lambda_MP=1.0, C=1.0, prev_clf=None, sample_weight=None):
    """Train logistic regression with guaranteed class weight totals.

    Regardless of the per-sample weights provided (e.g. Voronoi weights),
    the final training weights are rescaled in two steps:

    1. **Class-ratio lock**: MP weights are scaled to sum to lambda_MP,
       MR weights to 1.0.  Within each class, relative Voronoi weights
       are preserved.
    2. **Global normalisation**: all weights are uniformly rescaled so
       that sum(weights) == 1.0.  This makes the data-fit term O(1)
       regardless of dataset size, so C has a stable, consistent
       meaning throughout the active-learning loop (n_labeled grows
       from warm-start size to warm-start + all queries).

    If prev_clf is given, its coefficients are used to warm-start LBFGS
    so that convergence takes only a few iterations.
    """
    n_MP, n_MR = int(np.sum(y == 0)), int(np.sum(y == 1))

    # Start from per-sample base weights
    if sample_weight is not None:
        sw = np.array(sample_weight, dtype=np.float64)
    else:
        sw = np.ones(len(y), dtype=np.float64)

    # Step 1: Rescale each class so totals are exactly lambda_MP (MP) and 1.0 (MR)
    final_w = np.empty_like(sw)
    mp_mask = (y == 0)
    mr_mask = (y == 1)

    if n_MP > 0:
        sum_mp = sw[mp_mask].sum()
        final_w[mp_mask] = sw[mp_mask] * (lambda_MP / sum_mp) if sum_mp > 0 else lambda_MP / n_MP

    if n_MR > 0:
        sum_mr = sw[mr_mask].sum()
        final_w[mr_mask] = sw[mr_mask] * (1.0 / sum_mr) if sum_mr > 0 else 1.0 / n_MR

    # Step 2: Normalise so sum(final_w) == 1.0.
    # The sklearn objective is: sum_i(w_i * loss_i) + (1/2C)*||coef||^2
    # sklearn does NOT normalise sample_weight internally, so sum(w_i) sets
    # the scale of the data-fit term.  In active learning n_labeled grows
    # over time; if we normalised to n_labeled the fit term would grow with
    # every snapshot, making C effectively weaker and weaker.  Normalising
    # to 1.0 keeps the data-fit term O(1) throughout — C then has a fixed,
    # dataset-size-independent meaning.  The class ratio and within-class
    # Voronoi corrections are unaffected (we only multiply by a scalar).
    total_w = final_w.sum()
    if total_w > 0:
        final_w *= (10_000.0 / total_w)  # sum → 10_000 (fixed constant, dataset-size-independent)

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
    """Return a flat dict of metrics including per-class average log-loss."""
    yp = clf.predict(X)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yp, labels=[0, 1], zero_division=0)

    # Per-class average log-loss:  -mean[ y*log(p) + (1-y)*log(1-p) ] for each class
    probs = clf.predict_proba(X)  # columns: [P(class=0), P(class=1)]
    eps = 1e-15
    # For each sample: log-loss = -[y==0]*log(P(0)) - [y==1]*log(P(1))
    log_loss_per_sample = -np.log(np.clip(probs[np.arange(len(y)), y], eps, 1.0))
    mp_mask = (y == 0)
    mr_mask = (y == 1)
    loss_MP = float(log_loss_per_sample[mp_mask].mean()) if mp_mask.any() else 0.0
    loss_MR = float(log_loss_per_sample[mr_mask].mean()) if mr_mask.any() else 0.0

    return {
        "accuracy": float(accuracy_score(y, yp)),
        "precision_MP": float(prec[0]), "recall_MP": float(rec[0]), "f1_MP": float(f1[0]),
        "precision_MR": float(prec[1]), "recall_MR": float(rec[1]), "f1_MR": float(f1[1]),
        "loss_MP": loss_MP, "loss_MR": loss_MR,
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
    train_mp = m.get('train_loss_MP', float('nan'))
    train_mr = m.get('train_loss_MR', float('nan'))
    print(f"[Query {m['n_queries']:4d}] Acc={m['accuracy']:.4f}  "
          f"Loss(test MP={m['loss_MP']:.4f} MR={m['loss_MR']:.4f} | "
          f"train MP={train_mp:.4f} MR={train_mr:.4f})  "
          f"labeled={m['n_labeled']} (MP={m['n_labeled_MP']}, MR={m['n_labeled_MR']})")


# ── Plotting ─────────────────────────────────────────────

def _save_plots(results, out_dir):
    """Generate learning-curve plots (per-class average log-loss)."""
    qs = [r["n_queries"] for r in results]

    # --- Loss learning curve ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label, color, marker in [
        ("loss_MP",       "MP Loss (test)",  "#E07070", "o"),
        ("loss_MR",       "MR Loss (test)",  "#4A90D9", "o"),
        ("train_loss_MP", "MP Loss (train)", "#E07070", "^"),
        ("train_loss_MR", "MR Loss (train)", "#4A90D9", "^"),
    ]:
        vals = [r.get(key, float('nan')) for r in results]
        ax.plot(qs, vals, marker=marker, ls="-", label=label, color=color, lw=2, markersize=5)
    ax.set(xlabel="Number of Queries", ylabel="Average Log-Loss")
    ax.set_yscale("log")
    ax.set_title("Per-Class Average Log-Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "learning_curve.png"), dpi=200); plt.close(fig)


def generate_confusion_matrix(clf, X_full, y_full, out_dir):
    from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_recall_fscore_support
    from matplotlib.colors import LogNorm

    y_pred = clf.predict(X_full)
    
    acc = accuracy_score(y_full, y_pred)
    cm = confusion_matrix(y_full, y_pred, labels=[0, 1])
    
    precision, recall, _, _ = precision_recall_fscore_support(y_full, y_pred, labels=[0, 1], zero_division=0)
    
    print(f"\nOverall Accuracy on all data: {acc:.4%}")
    print(f"Class MP (0): Precision = {precision[0]:.4f}, Recall = {recall[0]:.4f}")
    print(f"Class MR (1): Precision = {precision[1]:.4f}, Recall = {recall[1]:.4f}\n")
    
    print("Confusion Matrix:")
    print("                 | Pred MP (0) | Pred MR (1)")
    print("--------------------------------------------")
    print(f"True MP (0)     | {cm[0, 0]:11d} | {cm[0, 1]:11d}")
    print(f"True MR (1)     | {cm[1, 0]:11d} | {cm[1, 1]:11d}")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['MP (Fe/H < -2)', 'MR (Fe/H >= -2)'])
    disp.plot(cmap='Blues', ax=ax, values_format=',') 
    try:
        disp.im_.set_norm(LogNorm(vmin=max(cm.min(), 1), vmax=cm.max()))
    except Exception:
        pass

    title_str = f'Overall Evaluation\nAcc: {acc:.2%}  MP(P:{precision[0]:.3f}, R:{recall[0]:.3f}) MR(P:{precision[1]:.3f}, R:{recall[1]:.3f})'
    plt.title(title_str, fontsize=11)
    
    plt.tight_layout()
    out_file = os.path.join(out_dir, 'confusion_matrix_all_data.png')
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix plot to {out_file}.")


def generate_pr_curve(clf_list, X_full, y_full, out_dir):
    """Plot one or more Precision-Recall curves on the same figure.

    Parameters
    ----------
    clf_list : list of (label, clf) tuples
        Each entry is a (human-readable label, trained classifier) pair.
        E.g. [("Halfway (2500 queries)", clf_half), ("Final (5000 queries)", clf_final)].
    """
    from sklearn.metrics import precision_recall_curve, auc

    colors = ['#E07070', '#4A90D9', '#5A9E7A', '#D4A24E', '#9B59B6']
    y_true_mp = (y_full == 0).astype(int)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (label, clf) in enumerate(clf_list):
        if hasattr(clf, "predict_proba"):
            y_scores = clf.predict_proba(X_full)[:, 0]
        else:
            y_scores = -clf.decision_function(X_full)

        precision, recall, _ = precision_recall_curve(y_true_mp, y_scores)
        # Drop the sklearn sentinel point (recall=0, precision=1) at the end
        precision, recall = precision[:-1], recall[:-1]
        pr_auc = auc(recall, precision)

        color = colors[i % len(colors)]
        ax.plot(recall, precision, color=color, lw=2,
                label=f'{label} (AUC = {pr_auc:.3f})')

    ax.set_xlabel('Recall (MP Class)', fontsize=12)
    ax.set_ylabel('Precision (MP Class)', fontsize=12)
    ax.set_title('Precision-Recall Curve for MP Class', fontsize=14, fontweight='bold')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)

    fig.tight_layout()
    out_file = os.path.join(out_dir, 'pr_curve_mp.png')
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved PR curve plot to {out_file}.")


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
    final_sw = None

    def snapshot(n_queries, prev_clf=None):
        nonlocal voronoi_state, final_sw
        Xl, yl = X_labeled[:n_labeled], y_labeled[:n_labeled]
        if len(np.unique(yl)) < 2:
            # Both classes required; skip this checkpoint and keep previous clf.
            print(f"[Query {n_queries:4d}] Skipped — only one class in labeled set so far.")
            return prev_clf

        # Reweighting: compute per-sample weights to correct covariate shift
        sw = None
        if args.reweighting == "hard":
            print(f"  [Voronoi-Hard] Computing sample weights ({n_labeled} labeled vs {len(X_pool)} pool)...")
            sw, voronoi_state = compute_voronoi_weights(X_pool, Xl, voronoi_state)
        elif args.reweighting == "soft":
            print(f"  [Voronoi-Soft] Computing sample weights (τ={args.temperature}, "
                  f"{n_labeled} labeled vs {len(X_pool)} pool)...")
            sw = compute_soft_voronoi_weights(X_pool, Xl, args.temperature)

        final_sw = sw

        clf = train_logistic(Xl, yl, args.lambda_MP, args.C, prev_clf=prev_clf,
                             sample_weight=sw)
        m = _record(evaluate(clf, X_eval, y_eval), n_queries, yl)

        # Training loss (per-class average log-loss on labeled set)
        train_m = evaluate(clf, Xl, yl)
        m["train_loss_MP"] = train_m["loss_MP"]
        m["train_loss_MR"] = train_m["loss_MR"]

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
    third_point = args.total_queries // 3
    two_third_point = 2 * args.total_queries // 3
    clf_one_third = None
    clf_two_third = None
    queries_one_third = None
    queries_two_third = None
    
    while queried < args.total_queries and available.any():
        batch = min(args.eval_every, args.total_queries - queried, int(available.sum()))
        avail_idx = np.where(available)[0]

        sel = strategy_fn(X_pool[avail_idx], clf, batch, rng,
                          X_labeled=X_labeled[:n_labeled], state=strategy_state,
                          pool_size=args.wass_pool_size)
        pool_idx = avail_idx[sel]

        # Append to pre-allocated arrays (no vstack/concatenate)
        n_new = len(pool_idx)
        X_labeled[n_labeled:n_labeled + n_new] = X_pool[pool_idx]
        y_labeled[n_labeled:n_labeled + n_new] = y_pool[pool_idx]
        n_labeled += n_new
        available[pool_idx] = False
        queried += n_new

        clf = snapshot(queried, prev_clf=clf)

        # Save deep copies at 1/3 and 2/3 marks for PR curve
        if clf_one_third is None and queried >= third_point and clf is not None:
            import copy
            clf_one_third = copy.deepcopy(clf)
            queries_one_third = queried
            print(f"  >> Saved 1/3 checkpoint at {queried} queries")

        if clf_two_third is None and queried >= two_third_point and clf is not None:
            import copy
            clf_two_third = copy.deepcopy(clf)
            queries_two_third = queried
            print(f"  >> Saved 2/3 checkpoint at {queried} queries")

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
        wt_plot_path = os.path.join(args.out_dir, "weight_distribution.png")
        fig.savefig(wt_plot_path, dpi=200)
        plt.close(fig)
        print(f"\nSaved weight distribution plot to {wt_plot_path}")
    
    # Build list of (label, clf) pairs for multi-curve PR plot (using eval set)
    pr_curves = []
    if clf_one_third is not None:
        pr_curves.append((f"1/3 ({queries_one_third} queries)", clf_one_third))
    if clf_two_third is not None:
        pr_curves.append((f"2/3 ({queries_two_third} queries)", clf_two_third))
    pr_curves.append((f"Final ({queried} queries)", clf))
    generate_pr_curve(pr_curves, X_eval, y_eval, args.out_dir)
    
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
    a("--reweighting", default="none", choices=["none", "hard", "soft"],
       help="Covariate-shift correction: none=uniform, hard=Voronoi assignment, soft=temperature softmin.")
    a("--temperature", type=float, default=1.0,
       help="Temperature τ for soft reweighting. τ→0 = hard, τ→∞ = uniform. Only used when --reweighting=soft.")

    # Practical
    a("--eval-size",       type=int, default=100_000, help="Eval subsample size.")
    a("--warm-start-max",  type=int, default=None,    help="Cap warm-start size.")
    a("--pool-max",        type=int, default=None,    help="Cap pool size.")
    a("--wass-pool-size",  type=int, default=5000,    help="Subpool size for Wasserstein strategy. Brute-force search is O(n × pool_size²).")
    a("--seed",            type=int, default=42)
    a("--out-dir",         default=None, help="Output directory (default: al_{strategy}).")

    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = f"al_{args.strategy}"
    run_active_learning(args)


if __name__ == "__main__":
    main()
