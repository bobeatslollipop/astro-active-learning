"""
Active Learning with Warm Start for Stellar Classification.

Workflow:
  1. Load a biased warm-start dataset (default: low T_eff stars).
  2. Load the full population dataset.
  3. Build a candidate pool = full population minus warm-start set.
  4. Iteratively query points from the pool using a chosen strategy.
  5. Every k queries, retrain the selected classifier on all labeled data
     and evaluate on a random subsample of the full population.

Usage:
  python active_learning.py --strategy random --total-queries 3000 --eval-every 200
  python active_learning.py --strategy uncertainty --total-queries 3000 --eval-every 200
  python active_learning.py --strategy wasserstein --total-queries 3000 --eval-every 200
  python active_learning.py --strategy wasserstein_l2 --reweighting voronoi_l2 --reweight-lambda 3000
  python active_learning.py --strategy uncertainty --n-trials 5 --n-snapshots 3
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

def _configure_torch_runtime():
    """Enable fast CUDA matmul settings when PyTorch/CUDA are available."""
    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        print("  [Torch] CUDA detected; TF32 matmul enabled where supported.")
    except Exception as exc:
        print(f"  [Torch] Warning: CUDA runtime tuning skipped: {exc}")


def _timing(label, start_time):
    """Print elapsed wall time for a pipeline stage."""
    print(f"  [Timing] {label}: {time.perf_counter() - start_time:.2f}s")


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


def query_wasserstein(X_pool, clf, n, rng, *, X_labeled=None, state=None,
                      pool_size=50000, plan_size=None, available_mask=None, **kw):
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
    if state is None:
        state = {}

    n_pool = len(X_pool)
    n_pick = min(n, n_pool)
    if n_pick <= 0:
        return np.empty(0, dtype=np.intp)

    if available_mask is None:
        available_idx = np.arange(n_pool, dtype=np.intp)
    else:
        available_idx = np.flatnonzero(available_mask).astype(np.intp, copy=False)

    if len(available_idx) == 0:
        return np.empty(0, dtype=np.intp)

    plan_n = int(plan_size) if plan_size is not None else n_pick
    plan_n = max(plan_n, n_pick)
    plan_is_valid = (
        state.get("pool_n") == n_pool
        and state.get("pool_array_id") == id(X_pool)
        and state.get("pool_size") == pool_size
        and state.get("plan_size") == plan_n
        and "planned_indices" in state
        and "plan_cursor" in state
    )

    if not plan_is_valid or state["plan_cursor"] >= len(state["planned_indices"]):
        t_plan = time.perf_counter()
        state["planned_indices"] = _build_wasserstein_plan(
            X_pool, X_labeled, plan_n, rng, pool_size, available_idx
        )
        state["plan_cursor"] = 0
        state["pool_n"] = n_pool
        state["pool_array_id"] = id(X_pool)
        state["pool_size"] = pool_size
        state["plan_size"] = plan_n
        _timing("Wasserstein plan build", t_plan)

    selected = []
    while len(selected) < n_pick:
        planned = state["planned_indices"]
        cursor = state["plan_cursor"]
        while len(selected) < n_pick and cursor < len(planned):
            idx = int(planned[cursor])
            cursor += 1
            if available_mask is None or available_mask[idx]:
                selected.append(idx)
        state["plan_cursor"] = cursor

        if len(selected) >= n_pick or cursor < len(planned):
            break

        # Edge case: the requested batch is larger than the cached plan tail.
        # Extend with the same greedy objective on the still-available pool.
        selected_arr = np.array(selected, dtype=np.intp)
        if len(selected_arr) > 0:
            keep = ~np.isin(available_idx, selected_arr)
            extra_available = available_idx[keep]
            X_seed = X_pool[selected_arr]
            if X_labeled is not None and len(X_labeled) > 0:
                X_seed = np.vstack([X_labeled, X_seed])
        else:
            extra_available = available_idx
            X_seed = X_labeled

        missing = n_pick - len(selected)
        if len(extra_available) == 0:
            break

        t_plan = time.perf_counter()
        state["planned_indices"] = _build_wasserstein_plan(
            X_pool, X_seed, missing, rng, pool_size, extra_available
        )
        state["plan_cursor"] = 0
        _timing("Wasserstein plan extension", t_plan)

    return np.array(selected, dtype=np.intp)


def query_wasserstein_l2(X_pool, clf, n, rng, *, X_labeled=None, state=None,
                         pool_size=50000, plan_size=None, available_mask=None,
                         reweight_lambda=1.0, **kw):
    """Greedy Wasserstein query with a Voronoi-L2 mass penalty.

    This follows the same subpool planning structure as ``query_wasserstein``,
    but scores each candidate by

        WWD(S union {u}, T) + lambda * w_u^2

    where ``w_u`` is the fraction of target-subpool points whose nearest
    representative would become the candidate.  The old support's L2 term
    is shared by all candidates at a greedy step, so the candidate-specific
    ranking term is exactly this induced mass penalty.  The regularizer is
    the same lambda used by ``--reweighting voronoi_l2``.
    """
    if state is None:
        state = {}

    lam = float(reweight_lambda)
    if lam <= 0:
        raise ValueError("wasserstein_l2 query requires a positive reweight_lambda.")

    n_pool = len(X_pool)
    n_pick = min(n, n_pool)
    if n_pick <= 0:
        return np.empty(0, dtype=np.intp)

    if available_mask is None:
        available_idx = np.arange(n_pool, dtype=np.intp)
    else:
        available_idx = np.flatnonzero(available_mask).astype(np.intp, copy=False)

    if len(available_idx) == 0:
        return np.empty(0, dtype=np.intp)

    plan_n = int(plan_size) if plan_size is not None else n_pick
    plan_n = max(plan_n, n_pick)
    plan_is_valid = (
        state.get("pool_n") == n_pool
        and state.get("pool_array_id") == id(X_pool)
        and state.get("pool_size") == pool_size
        and state.get("plan_size") == plan_n
        and state.get("reweight_lambda") == lam
        and "planned_indices" in state
        and "plan_cursor" in state
    )

    if not plan_is_valid or state["plan_cursor"] >= len(state["planned_indices"]):
        t_plan = time.perf_counter()
        state["planned_indices"] = _build_wasserstein_l2_plan(
            X_pool, X_labeled, plan_n, rng, pool_size, available_idx, lam
        )
        state["plan_cursor"] = 0
        state["pool_n"] = n_pool
        state["pool_array_id"] = id(X_pool)
        state["pool_size"] = pool_size
        state["plan_size"] = plan_n
        state["reweight_lambda"] = lam
        _timing("Wasserstein-L2 plan build", t_plan)

    selected = []
    while len(selected) < n_pick:
        planned = state["planned_indices"]
        cursor = state["plan_cursor"]
        while len(selected) < n_pick and cursor < len(planned):
            idx = int(planned[cursor])
            cursor += 1
            if available_mask is None or available_mask[idx]:
                selected.append(idx)
        state["plan_cursor"] = cursor

        if len(selected) >= n_pick or cursor < len(planned):
            break

        selected_arr = np.array(selected, dtype=np.intp)
        if len(selected_arr) > 0:
            keep = ~np.isin(available_idx, selected_arr)
            extra_available = available_idx[keep]
            X_seed = X_pool[selected_arr]
            if X_labeled is not None and len(X_labeled) > 0:
                X_seed = np.vstack([X_labeled, X_seed])
        else:
            extra_available = available_idx
            X_seed = X_labeled

        missing = n_pick - len(selected)
        if len(extra_available) == 0:
            break

        t_plan = time.perf_counter()
        state["planned_indices"] = _build_wasserstein_l2_plan(
            X_pool, X_seed, missing, rng, pool_size, extra_available, lam
        )
        state["plan_cursor"] = 0
        _timing("Wasserstein-L2 plan extension", t_plan)

    return np.array(selected, dtype=np.intp)


def _build_wasserstein_plan(X_pool, X_labeled, n_plan, rng, pool_size, available_idx):
    """Build a greedy Wasserstein query plan over a fixed available pool."""
    effective_ps = min(pool_size, len(available_idx))
    if effective_ps <= 0:
        return np.empty(0, dtype=np.intp)

    n_plan = min(n_plan, effective_ps)

    # Random subpool — serves as both target distribution and candidate set.
    subpool_idx = rng.choice(available_idx, effective_ps, replace=False)
    T = X_pool[subpool_idx]  # (effective_ps, d)

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch and torch.cuda.is_available():
        chosen = _wasserstein_coupling_torch(T, X_labeled, n_plan, rng)
    else:
        chosen = _wasserstein_coupling_numpy(T, X_labeled, n_plan, rng)

    return subpool_idx[np.array(chosen, dtype=np.intp)]


def _build_wasserstein_l2_plan(X_pool, X_labeled, n_plan, rng, pool_size,
                               available_idx, reweight_lambda):
    """Build a greedy regularized-Wasserstein query plan over a fixed pool."""
    effective_ps = min(pool_size, len(available_idx))
    if effective_ps <= 0:
        return np.empty(0, dtype=np.intp)

    n_plan = min(n_plan, effective_ps)
    subpool_idx = rng.choice(available_idx, effective_ps, replace=False)
    T = X_pool[subpool_idx]

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch and torch.cuda.is_available():
        chosen = _wasserstein_l2_coupling_torch(T, X_labeled, n_plan, reweight_lambda)
    else:
        chosen = _wasserstein_l2_coupling_numpy(T, X_labeled, n_plan, reweight_lambda)

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


def _wasserstein_initial_wwds_numpy(base_min, intra_dists):
    """Compute initial WWD scores without materialising a second ps x ps array."""
    ps = len(base_min)
    wwds = np.empty(ps, dtype=np.float32)
    target_bytes = 256 * 1024 * 1024
    row_chunk = max(1, min(ps, target_bytes // max(ps * 4, 1)))
    for start in range(0, ps, row_chunk):
        end = min(start + row_chunk, ps)
        wwds[start:end] = np.minimum(base_min[np.newaxis, :],
                                     intra_dists[start:end]).mean(axis=1)
    return wwds


def _wasserstein_delta_update_numpy(wwds, intra_dists, old_base, base_min, changed):
    """Apply an incremental WWD update in column chunks."""
    ps = len(base_min)
    target_bytes = 256 * 1024 * 1024
    col_chunk = max(1, min(len(changed), target_bytes // max(ps * 12, 1)))
    for start in range(0, len(changed), col_chunk):
        cols = changed[start:start + col_chunk]
        old_contribs = np.minimum(old_base[cols][np.newaxis, :], intra_dists[:, cols])
        new_contribs = np.minimum(base_min[cols][np.newaxis, :], intra_dists[:, cols])
        wwds += (new_contribs - old_contribs).sum(axis=1) / ps


def _wasserstein_l2_initial_capture_counts_numpy(base_min, intra_dists):
    """Count target points each candidate would capture under current base_min."""
    ps = len(base_min)
    capture_counts = np.empty(ps, dtype=np.float32)
    target_bytes = 256 * 1024 * 1024
    row_chunk = max(1, min(ps, target_bytes // max(ps, 1)))
    for start in range(0, ps, row_chunk):
        end = min(start + row_chunk, ps)
        capture_counts[start:end] = (
            intra_dists[start:end] < base_min[np.newaxis, :]
        ).sum(axis=1)
    return capture_counts


def _wasserstein_l2_delta_update_numpy(wwds, capture_counts, intra_dists,
                                       old_base, base_min, changed):
    """Apply an incremental WWD and captured-mass update in column chunks."""
    ps = len(base_min)
    target_bytes = 256 * 1024 * 1024
    col_chunk = max(1, min(len(changed), target_bytes // max(ps * 14, 1)))
    for start in range(0, len(changed), col_chunk):
        cols = changed[start:start + col_chunk]
        d_cols = intra_dists[:, cols]

        old_contribs = np.minimum(old_base[cols][np.newaxis, :], d_cols)
        new_contribs = np.minimum(base_min[cols][np.newaxis, :], d_cols)
        wwds += (new_contribs - old_contribs).sum(axis=1) / ps

        old_captures = d_cols < old_base[cols][np.newaxis, :]
        new_captures = d_cols < base_min[cols][np.newaxis, :]
        capture_counts += (
            new_captures.sum(axis=1) - old_captures.sum(axis=1)
        ).astype(np.float32)


def _torch_matrix_chunk(length, width, device, *, n_matrices=1, fraction=0.20):
    """Choose a chunk length for a temporary CUDA matrix budget."""
    import torch
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        target_bytes = int(free_bytes * fraction)
        denom = max(width * n_matrices * 4, 1)
        return max(1, min(length, target_bytes // denom))
    except Exception:
        return max(1, min(length, 2048))


def _wasserstein_initial_wwds_torch(base_min, intra_dists):
    """Compute initial WWD scores without materialising a second ps x ps tensor."""
    import torch
    ps = len(base_min)
    device = base_min.device
    wwds = torch.empty(ps, dtype=intra_dists.dtype, device=device)
    row_chunk = _torch_matrix_chunk(ps, ps, device, n_matrices=1, fraction=0.20)
    for start in range(0, ps, row_chunk):
        end = min(start + row_chunk, ps)
        wwds[start:end] = torch.minimum(
            base_min.unsqueeze(0), intra_dists[start:end]
        ).mean(dim=1)
    return wwds


def _wasserstein_delta_update_torch(wwds, intra_dists, old_base, base_min, changed):
    """Apply an incremental WWD update in column chunks."""
    import torch
    ps = len(base_min)
    device = base_min.device
    col_chunk = _torch_matrix_chunk(len(changed), ps, device, n_matrices=3, fraction=0.12)
    for start in range(0, len(changed), col_chunk):
        cols = changed[start:start + col_chunk]
        old_contribs = torch.minimum(old_base[cols].unsqueeze(0), intra_dists[:, cols])
        new_contribs = torch.minimum(base_min[cols].unsqueeze(0), intra_dists[:, cols])
        wwds += (new_contribs - old_contribs).sum(dim=1) / ps


def _wasserstein_l2_initial_capture_counts_torch(base_min, intra_dists):
    """Count target points each candidate would capture under current base_min."""
    import torch
    ps = len(base_min)
    device = base_min.device
    capture_counts = torch.empty(ps, dtype=intra_dists.dtype, device=device)
    row_chunk = _torch_matrix_chunk(ps, ps, device, n_matrices=1, fraction=0.16)
    for start in range(0, ps, row_chunk):
        end = min(start + row_chunk, ps)
        capture_counts[start:end] = (
            intra_dists[start:end] < base_min.unsqueeze(0)
        ).sum(dim=1).to(intra_dists.dtype)
    return capture_counts


def _wasserstein_l2_delta_update_torch(wwds, capture_counts, intra_dists,
                                      old_base, base_min, changed):
    """Apply an incremental WWD and captured-mass update in column chunks."""
    import torch
    ps = len(base_min)
    device = base_min.device
    col_chunk = _torch_matrix_chunk(len(changed), ps, device,
                                    n_matrices=4, fraction=0.10)
    for start in range(0, len(changed), col_chunk):
        cols = changed[start:start + col_chunk]
        d_cols = intra_dists[:, cols]

        old_contribs = torch.minimum(old_base[cols].unsqueeze(0), d_cols)
        new_contribs = torch.minimum(base_min[cols].unsqueeze(0), d_cols)
        wwds += (new_contribs - old_contribs).sum(dim=1) / ps

        old_captures = d_cols < old_base[cols].unsqueeze(0)
        new_captures = d_cols < base_min[cols].unsqueeze(0)
        capture_counts += (
            new_captures.sum(dim=1) - old_captures.sum(dim=1)
        ).to(capture_counts.dtype)


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

    # Initial WWD computation for all candidates.
    wwds = _wasserstein_initial_wwds_numpy(base_min, intra_dists)

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
            _wasserstein_delta_update_numpy(wwds, intra_dists, old_base, base_min, changed)

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

    # Initial WWD computation.
    wwds = _wasserstein_initial_wwds_torch(base_min, intra_dists)

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
            _wasserstein_delta_update_torch(wwds, intra_dists, old_base, base_min, changed)

    del T_t, intra_dists
    torch.cuda.empty_cache()
    return chosen


def _wasserstein_l2_coupling_numpy(T, X_labeled, n_pick, reweight_lambda):
    """CPU greedy coupling for WWD + lambda * captured_mass^2."""
    ps = len(T)
    lam = float(reweight_lambda)

    intra_dists = cdist(T, T, metric='euclidean').astype(np.float32)

    base_min = np.full(ps, np.inf, dtype=np.float32)
    if X_labeled is not None and len(X_labeled) > 0:
        CHUNK = 5000
        for start in range(0, len(X_labeled), CHUNK):
            end = min(start + CHUNK, len(X_labeled))
            dists = cdist(T, X_labeled[start:end], metric='euclidean').astype(np.float32)
            np.minimum(base_min, dists.min(axis=1), out=base_min)
            del dists

    print(f"  [CPU] Wasserstein-L2 coupling: pool_size={ps}, "
          f"lambda={lam:g}, selecting {n_pick}")

    wwds = _wasserstein_initial_wwds_numpy(base_min, intra_dists)
    capture_counts = _wasserstein_l2_initial_capture_counts_numpy(base_min, intra_dists)

    chosen = []
    available = np.ones(ps, dtype=bool)
    denom = float(max(ps, 1))

    for _ in range(n_pick):
        masses = np.maximum(capture_counts, 0.0) / denom
        scores = wwds + lam * (masses ** 2)
        scores[~available] = np.inf

        best = int(np.argmin(scores))
        if not np.isfinite(scores[best]):
            break

        chosen.append(best)
        available[best] = False

        old_base = base_min.copy()
        np.minimum(base_min, intra_dists[best], out=base_min)
        changed = np.where(base_min < old_base)[0]

        if len(changed) > 0:
            _wasserstein_l2_delta_update_numpy(
                wwds, capture_counts, intra_dists, old_base, base_min, changed
            )

    return chosen


def _wasserstein_l2_coupling_torch(T, X_labeled, n_pick, reweight_lambda):
    """GPU greedy coupling for WWD + lambda * captured_mass^2."""
    import torch
    device = torch.device('cuda')
    ps = len(T)
    lam = float(reweight_lambda)

    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    T_sq = (T_t ** 2).sum(dim=1)

    intra_dists = T_sq.unsqueeze(1) + T_sq.unsqueeze(0)
    intra_dists.addmm_(T_t, T_t.T, beta=1.0, alpha=-2.0)
    intra_dists.clamp_(min=0.0).sqrt_()

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

    print(f"  [GPU] Wasserstein-L2 coupling: pool_size={ps}, "
          f"lambda={lam:g}, selecting {n_pick}")

    wwds = _wasserstein_initial_wwds_torch(base_min, intra_dists)
    capture_counts = _wasserstein_l2_initial_capture_counts_torch(base_min, intra_dists)

    chosen = []
    available = torch.ones(ps, dtype=torch.bool, device=device)
    denom = float(max(ps, 1))

    for _ in range(n_pick):
        masses = torch.clamp(capture_counts, min=0.0) / denom
        scores = wwds + lam * (masses ** 2)
        scores = scores.clone()
        scores[~available] = float('inf')

        best = torch.argmin(scores).item()
        if not torch.isfinite(scores[best]).item():
            break

        chosen.append(best)
        available[best] = False

        old_base = base_min.clone()
        torch.minimum(base_min, intra_dists[best], out=base_min)
        changed = torch.where(base_min < old_base)[0]

        if len(changed) > 0:
            _wasserstein_l2_delta_update_torch(
                wwds, capture_counts, intra_dists, old_base, base_min, changed
            )

    del T_t, intra_dists, wwds, capture_counts
    torch.cuda.empty_cache()
    return chosen


# ── Entropic OT Sampling ────────────────────────────────
# Same greedy framework as Wasserstein, but replaces the hard min
# (nearest-neighbor assignment) with a soft min (entropic / log-sum-exp),
# yielding an entropic OT cost.

def query_entropicOT(X_pool, clf, n, rng, *, X_labeled=None, state=None,
                     pool_size=50000, temperature=1.0, **kw):
    """
    Approximate entropic OT sampling via soft greedy coupling.

    Like ``query_wasserstein``, this subsamples a pool and greedily selects
    n points.  The difference is the objective being minimised at each step:

        Wasserstein:   WWD = (1/|T|) Σ_j  min_i  d(t_j, s_i)
        Entropic OT:   EOT = (1/|T|) Σ_j  -τ · log Σ_i exp(-d(t_j, s_i)/τ)

    The entropic OT cost is a smooth (temperature-controlled) relaxation of
    the Wasserstein cost.  τ → 0 recovers hard Wasserstein; τ → ∞ makes all
    candidates equally good.

    The greedy step picks the candidate u that yields the lowest EOT when u
    is added to the current labeled/selected set S.  We maintain a running
    sum-of-exponentials for each target point j, which enables O(ps) updates
    per step (vs. O(ps × |changed|) for Wasserstein).

    Parameters
    ----------
    pool_size : int
        Number of candidate points subsampled from X_pool.
    temperature : float
        Regularisation parameter τ for the soft-min.  Smaller values
        approach the hard Wasserstein solution.
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
        chosen = _entropicOT_coupling_torch(T, X_labeled, n_pick, rng, temperature)
    else:
        chosen = _entropicOT_coupling_numpy(T, X_labeled, n_pick, rng, temperature)

    return subpool_idx[np.array(chosen, dtype=np.intp)]


def _entropicOT_coupling_numpy(T, X_labeled, n_pick, rng, temperature):
    """CPU greedy entropic OT coupling — memory-efficient.

    Avoids storing the full ps×ps distance matrix.  Instead, at each greedy
    step, distances from remaining candidates to all targets are computed
    on-the-fly in chunks.

    Memory: O(ps × CHUNK_C) instead of O(ps²).
    """
    ps = len(T)
    tau = float(temperature)

    # Initialise log_exp_sum from labeled points (log-space for stability)
    # log_exp_sum[j] = log Σ_{i ∈ labeled} exp(-d(t_j, l_i) / τ)
    log_exp_sum = np.full(ps, -np.inf, dtype=np.float64)
    if X_labeled is not None and len(X_labeled) > 0:
        CHUNK = 5000
        for start in range(0, len(X_labeled), CHUNK):
            end = min(start + CHUNK, len(X_labeled))
            dists = cdist(T, X_labeled[start:end], metric='euclidean').astype(np.float32)
            chunk_logits = (-dists / tau).astype(np.float64)
            # logsumexp over the chunk dimension
            max_cl = chunk_logits.max(axis=1)
            chunk_lse = max_cl + np.log(np.exp(chunk_logits - max_cl[:, None]).sum(axis=1))
            log_exp_sum = np.logaddexp(log_exp_sum, chunk_lse)
            del dists, chunk_logits

    print(f"  [CPU] Entropic OT coupling: pool_size={ps}, τ={tau}, selecting {n_pick}")

    CHUNK_C = min(2000, ps)
    chosen = []
    available = np.ones(ps, dtype=bool)

    for k in range(n_pick):
        avail_idx = np.where(available)[0]
        n_avail = len(avail_idx)

        best_cost = np.inf
        best_u = -1

        # Evaluate all available candidates in chunks
        for start_c in range(0, n_avail, CHUNK_C):
            end_c = min(start_c + CHUNK_C, n_avail)
            chunk_idx = avail_idx[start_c:end_c]

            # Distances from chunk candidates to all targets: (chunk_size, ps)
            dists = cdist(T[chunk_idx], T, metric='euclidean').astype(np.float32)
            logits = (-dists / tau).astype(np.float64)
            del dists

            # cost(u) = -(τ/ps) * Σ_j logaddexp(log_exp_sum[j], logits[u, j])
            costs = -(tau / ps) * np.logaddexp(
                log_exp_sum[np.newaxis, :], logits
            ).sum(axis=1)
            del logits

            chunk_best = np.argmin(costs)
            if costs[chunk_best] < best_cost:
                best_cost = costs[chunk_best]
                best_u = int(chunk_idx[chunk_best])

        chosen.append(best_u)
        available[best_u] = False

        # Update log_exp_sum with the selected point's contribution
        best_dists = cdist(T[best_u:best_u + 1], T, metric='euclidean').astype(np.float32).ravel()
        best_logits = (-best_dists / tau).astype(np.float64)
        log_exp_sum = np.logaddexp(log_exp_sum, best_logits)

    return chosen


def _entropicOT_coupling_torch(T, X_labeled, n_pick, rng, temperature):
    """GPU greedy entropic OT coupling — memory-efficient.

    Avoids storing the full ps×ps distance/logits matrices by computing
    candidate distances on-the-fly in chunks during each greedy step.

    Memory: O(ps × d + ps × CHUNK_C) instead of O(ps²).
    For ps=50k the savings are ~50 GB → ~2 GB peak GPU usage.
    """
    import torch
    device = torch.device('cuda')
    ps = len(T)
    tau = float(temperature)

    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    T_sq = (T_t ** 2).sum(dim=1)  # (ps,)

    # Initialise log_exp_sum from labeled points (in log-space for stability)
    # log_exp_sum[j] = log Σ_{i ∈ labeled} exp(-d(t_j, l_i)/τ)
    log_exp_sum = torch.full((ps,), -float('inf'), dtype=torch.float64, device=device)
    if X_labeled is not None and len(X_labeled) > 0:
        X_l = torch.tensor(X_labeled, dtype=torch.float32, device=device)
        X_l_sq = (X_l ** 2).sum(dim=1)
        CHUNK = 10000
        for start in range(0, len(X_l), CHUNK):
            end = min(start + CHUNK, len(X_l))
            dists = T_sq.unsqueeze(1) + X_l_sq[start:end].unsqueeze(0)
            dists.addmm_(T_t, X_l[start:end].T, beta=1.0, alpha=-2.0)
            dists.clamp_(min=0.0).sqrt_()
            chunk_logits = (-dists / tau).to(torch.float64)
            chunk_lse = torch.logsumexp(chunk_logits, dim=1)  # (ps,)
            log_exp_sum = torch.logaddexp(log_exp_sum, chunk_lse)
            del dists, chunk_logits, chunk_lse
        del X_l, X_l_sq
        torch.cuda.empty_cache()

    print(f"  [GPU] Entropic OT coupling: pool_size={ps}, τ={tau}, selecting {n_pick}")

    # Determine chunk size for candidate evaluation based on available VRAM
    # Each chunk needs: (CHUNK_C, ps) float32 dists + (CHUNK_C, ps) float64 logaddexp
    # ≈ ps × CHUNK_C × 12 bytes
    try:
        props = torch.cuda.get_device_properties(device)
        free_vram = props.total_memory - torch.cuda.memory_allocated(device)
        CHUNK_C = max(500, int(free_vram * 0.25 / (ps * 12)))
        CHUNK_C = min(CHUNK_C, ps)
    except Exception:
        CHUNK_C = min(2000, ps)

    chosen = []
    available = torch.ones(ps, dtype=torch.bool, device=device)
    eot_costs = torch.full((ps,), float('inf'), dtype=torch.float64, device=device)

    for k in range(n_pick):
        # Compute EOT cost for all available candidates (chunked)
        avail_idx = torch.where(available)[0]
        n_avail = len(avail_idx)

        for start_c in range(0, n_avail, CHUNK_C):
            end_c = min(start_c + CHUNK_C, n_avail)
            chunk_idx = avail_idx[start_c:end_c]

            # Distances from chunk candidates to all targets: (chunk_size, ps)
            chunk_T = T_t[chunk_idx]
            dists = T_sq[chunk_idx].unsqueeze(1) + T_sq.unsqueeze(0)
            dists.addmm_(chunk_T, T_t.T, beta=1.0, alpha=-2.0)
            dists.clamp_(min=0.0).sqrt_()

            # logits[c, j] = -d(t_j, candidate_c) / τ
            logits = (-dists / tau).to(torch.float64)
            del dists

            # cost(c) = -(τ/ps) * Σ_j logaddexp(log_exp_sum[j], logits[c, j])
            costs = -(tau / ps) * torch.logaddexp(
                log_exp_sum.unsqueeze(0), logits
            ).sum(dim=1)
            eot_costs[chunk_idx] = costs
            del logits, costs, chunk_T

        best = torch.argmin(eot_costs).item()
        chosen.append(best)
        available[best] = False
        eot_costs[best] = float('inf')

        # Update log_exp_sum with the selected point's contribution
        best_dists = (T_sq + T_sq[best] - 2.0 * T_t @ T_t[best]).clamp_(min=0.0).sqrt_()
        best_logits = (-best_dists / tau).to(torch.float64)
        log_exp_sum = torch.logaddexp(log_exp_sum, best_logits)
        del best_dists, best_logits

    del T_t
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


def query_moment_matching(X_pool, clf, n, rng, *, X_labeled=None, state=None,
                          pool_size=50000, moment_ridge=1.0,
                          sample_weight=None, **kw):
    """Greedy ridge moment/design matching for linear-regression experiments.

    On a random target subpool T, approximate the linear prediction-design
    objective tr(M_T G^{-1}), where M_T is the target second moment and
    G is the ridge-regularized labeled design matrix. Each greedy step
    selects the candidate with the largest Sherman-Morrison reduction in
    this objective.
    """
    n_pool = len(X_pool)
    n_pick = min(n, n_pool)
    if n_pick <= 0:
        return np.empty(0, dtype=np.intp)

    if pool_size is None:
        effective_ps = n_pool
    else:
        effective_ps = min(n_pool, max(n_pick, int(pool_size)))

    subpool_idx = rng.choice(n_pool, effective_ps, replace=False)
    T = X_pool[subpool_idx].astype(np.float64, copy=False)

    chosen = _moment_matching_greedy(T, X_labeled, n_pick, moment_ridge,
                                     labeled_weight=sample_weight)
    return subpool_idx[np.array(chosen, dtype=np.intp)]


def _moment_matching_greedy(T, X_labeled, n_pick, moment_ridge, labeled_weight=None):
    ps, d = T.shape
    ridge = max(float(moment_ridge), 1e-12)
    print(f"  [Moment] pool_size={ps}, ridge={ridge:g}, selecting {n_pick}")

    M_target = (T.T @ T) / max(ps, 1)
    G = ridge * np.eye(d, dtype=np.float64)
    if X_labeled is not None and len(X_labeled) > 0:
        X_l = np.asarray(X_labeled, dtype=np.float64)
        if labeled_weight is None:
            G += X_l.T @ X_l
        else:
            w = np.asarray(labeled_weight, dtype=np.float64)
            if len(w) != len(X_l):
                raise ValueError("labeled_weight must have one entry per labeled point.")
            w_sum = float(w.sum())
            if w_sum <= 0:
                w = np.ones(len(X_l), dtype=np.float64)
                w_sum = float(len(X_l))
            # Keep the weighted source design on the same scale as an
            # unweighted design with len(X_l) labeled observations.
            w = np.maximum(w, 0.0) / w_sum * len(X_l)
            G += X_l.T @ (w[:, None] * X_l)

    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.pinv(G)

    chosen = []
    available = np.ones(ps, dtype=bool)

    for _ in range(min(n_pick, ps)):
        # Reduction in tr(M G^{-1}) from adding x x^T:
        #   (x^T G^{-1} M G^{-1} x) / (1 + x^T G^{-1} x)
        design_gain = G_inv @ M_target @ G_inv
        T_gain = T @ design_gain
        T_inv = T @ G_inv
        numer = np.einsum("ij,ij->i", T_gain, T)
        denom = 1.0 + np.einsum("ij,ij->i", T_inv, T)
        scores = numer / np.maximum(denom, 1e-12)
        scores[~available] = -np.inf

        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            break

        chosen.append(best)
        available[best] = False

        x = T[best]
        G_inv_x = G_inv @ x
        update_denom = 1.0 + float(x @ G_inv_x)
        if update_denom > 1e-12:
            G_inv -= np.outer(G_inv_x, G_inv_x) / update_denom

    return chosen


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

def _auto_topk(X_pool, X_labeled, temperature, device, coverage=0.999, n_probe=500):
    """Determine minimum K so that top-K softmax covers ≥ coverage of full softmax.

    Probes a random subset of pool points, computes the full softmax,
    then sorts the softmax values (descending) and finds cumulative coverage.
    Tensors are freed eagerly to minimise peak GPU memory.
    Cost: O(n_probe × n_labeled) — negligible compared to the main loop.
    """
    import torch
    rng = np.random.RandomState(0)
    n_labeled = len(X_labeled)
    n_probe = min(n_probe, len(X_pool))
    probe_idx = rng.choice(len(X_pool), n_probe, replace=False)

    X_p = torch.tensor(X_pool[probe_idx], dtype=torch.float32, device=device)
    X_l = torch.tensor(X_labeled, dtype=torch.float32, device=device)

    # Distance matrix (n_probe × n_labeled)
    X_p_sq = (X_p ** 2).sum(dim=1)
    X_l_sq = (X_l ** 2).sum(dim=1)
    dists = X_p_sq.unsqueeze(1) + X_l_sq.unsqueeze(0)
    dists.addmm_(X_p, X_l.T, beta=1.0, alpha=-2.0)
    dists.clamp_(min=0.0).sqrt_()
    del X_p, X_l, X_p_sq, X_l_sq          # free source tensors early

    # Softmax (in-place neg + div reuses the dists buffer for logits)
    dists.neg_().div_(temperature)
    soft_sm = torch.softmax(dists, dim=1)  # (n_probe, n_labeled)
    del dists

    # Sort softmax values descending — largest contributor first.
    # This is equivalent to sorting by distance ascending then gathering,
    # but avoids allocating the int64 index tensor.
    sorted_sm, _ = soft_sm.sort(dim=1, descending=True)
    del soft_sm
    cumsum = sorted_sm.cumsum(dim=1)
    del sorted_sm

    # For each probe, find minimum K where cumulative coverage >= target
    sufficient = (cumsum >= coverage)
    k_per_probe = sufficient.int().argmax(dim=1) + 1  # (n_probe,)
    del cumsum, sufficient

    # Use 95th percentile as K, clamped to [10, n_labeled]
    k95 = int(torch.quantile(k_per_probe.float(), 0.95).item())
    k95 = max(k95, 10)
    k95 = min(k95, n_labeled)

    del k_per_probe
    torch.cuda.empty_cache()
    return k95


def compute_soft_voronoi_weights(X_pool, X_labeled, temperature=1.0,
                                  soft_state=None, topk=0):
    """Compute soft Voronoi weights via temperature-scaled softmin.

    For each pool point p_i, a soft assignment over labeled points is:
        w_{ij} = exp(-||p_i - l_j|| / τ)  /  Σ_k exp(-||p_i - l_k|| / τ)
    The weight of labeled point j is the average assignment:
        weight_j = (1/n_pool) Σ_i w_{ij}

    τ → 0  converges to hard Voronoi (argmin).
    τ → ∞  converges to uniform weights.

    When topk > 0, only the K nearest labeled points contribute to each
    pool point's softmax, reducing cost from O(pool×labeled) to O(pool×K).
    When topk == 0, K is auto-calibrated per snapshot for >= 99.9% coverage.

    Returns weights array of shape (n_labeled,), scaled so that
    sum(weights) == n_labeled.
    """
    if soft_state is None:
        soft_state = {}
    try:
        import torch
        if torch.cuda.is_available():
            return _soft_voronoi_torch(X_pool, X_labeled, temperature,
                                       state=soft_state, topk=topk)
    except ImportError:
        pass
    return _soft_voronoi_numpy(X_pool, X_labeled, temperature, topk=topk)


def _soft_voronoi_torch(X_pool, X_labeled, temperature, state=None, topk=0):
    import torch
    device = torch.device('cuda')

    n_pool = len(X_pool)
    n_labeled = len(X_labeled)

    # ── Determine effective K for top-K truncated softmax ──
    if topk <= 0:
        if n_labeled <= 50:
            K = n_labeled
        else:
            # Reuse cached K (auto-topk is expensive and changes slowly)
            if state and 'K' in state:
                K = min(state['K'], n_labeled)
                print(f"    [Top-K] Reusing cached K={K} / {n_labeled} labeled")
            else:
                try:
                    K = _auto_topk(X_pool, X_labeled, temperature, device)
                    print(f"    [Auto top-K] K={K} / {n_labeled} labeled "
                          f"(>= 99.9% softmax coverage)")
                except RuntimeError:
                    K = min(50, n_labeled)
                    print(f"    [Auto top-K] OOM during calibration, falling back to K={K}")
                    torch.cuda.empty_cache()
    else:
        K = min(topk, n_labeled)
        if K < n_labeled:
            print(f"    [Top-K] Using user-specified K={K} / {n_labeled} labeled")

    use_topk = K < n_labeled
    prev_n = state.get('n_labeled', 0) if state else 0

    # ── Incremental path: only compute distances to NEW labeled points ──
    if (state and use_topk and prev_n > 0 and prev_n < n_labeled
            and 'topk_dists' in state
            and state['topk_dists'].shape[1] >= K):
        n_new = n_labeled - prev_n
        K_old = state['topk_dists'].shape[1]
        cached_dists = state['topk_dists']   # (n_pool, K_old) float32 CPU
        cached_idx = state['topk_idx']       # (n_pool, K_old) int64 CPU
        print(f"    [Incremental soft] {n_new} new labeled, merging with cached K={K_old}")

        X_p = torch.tensor(X_pool, dtype=torch.float32, device=device)
        X_new = torch.tensor(X_labeled[prev_n:], dtype=torch.float32, device=device)
        X_p_sq = (X_p ** 2).sum(dim=1)
        X_new_sq = (X_new ** 2).sum(dim=1)

        try:
            props = torch.cuda.get_device_properties(device)
            free_vram = props.total_memory - torch.cuda.memory_allocated(device)
            target_elements = int(free_vram * 0.25 / 4)
            CHUNK_P = max(5000, target_elements // max(n_new + K_old, 1))
        except:
            CHUNK_P = 30000

        weight_accum = torch.zeros(n_labeled, dtype=torch.float64, device=device)
        new_topk_d = torch.empty(n_pool, K, dtype=torch.float32)
        new_topk_i = torch.empty(n_pool, K, dtype=torch.int64)

        for start_p in range(0, n_pool, CHUNK_P):
            end_p = min(start_p + CHUNK_P, n_pool)
            cs = end_p - start_p
            chunk_p = X_p[start_p:end_p]

            # Distances to new labeled points only
            nd = X_p_sq[start_p:end_p].unsqueeze(1) + X_new_sq.unsqueeze(0)
            nd.addmm_(chunk_p, X_new.T, beta=1.0, alpha=-2.0)
            nd.clamp_(min=0.0).sqrt_()  # (cs, n_new)

            new_idx = torch.arange(prev_n, n_labeled, device=device).unsqueeze(0).expand(cs, -1)
            old_d = cached_dists[start_p:end_p].to(device)
            old_i = cached_idx[start_p:end_p].to(device)

            # Merge cached top-K with new distances, re-select top-K
            mg_d = torch.cat([old_d, nd], dim=1)
            mg_i = torch.cat([old_i, new_idx], dim=1)
            tk_d, tk_pos = mg_d.topk(K, dim=1, largest=False)
            tk_i = torch.gather(mg_i, 1, tk_pos)

            new_topk_d[start_p:end_p] = tk_d.cpu()
            new_topk_i[start_p:end_p] = tk_i.cpu()

            logits = tk_d.neg_().div_(temperature)
            sa = torch.softmax(logits, dim=1)
            weight_accum.scatter_add_(0, tk_i.reshape(-1).long(),
                                      sa.reshape(-1).to(torch.float64))
            del nd, old_d, old_i, mg_d, mg_i, tk_d, tk_pos, tk_i, logits, sa

        state['topk_dists'] = new_topk_d
        state['topk_idx'] = new_topk_i
        state['n_labeled'] = n_labeled
        state['K'] = K

        weights = weight_accum / n_pool
        weights = (weights * n_labeled).cpu().numpy().astype(np.float64)
        del X_p, X_new
        torch.cuda.empty_cache()
        return weights

    # ── Full computation (first call or non-topk) ──
    X_p = torch.tensor(X_pool, dtype=torch.float32, device=device)
    X_l = torch.tensor(X_labeled, dtype=torch.float32, device=device)
    X_p_sq = (X_p ** 2).sum(dim=1)
    X_l_sq = (X_l ** 2).sum(dim=1)

    try:
        props = torch.cuda.get_device_properties(device)
        free_vram = props.total_memory - torch.cuda.memory_allocated(device)
        target_elements = int(free_vram * 0.25 / 4)
        CHUNK_P = max(5000, target_elements // max(n_labeled, 1))
    except:
        CHUNK_P = 30000

    weight_accum = torch.zeros(n_labeled, dtype=torch.float64, device=device)

    # Allocate cache for top-K if applicable
    cache_topk = use_topk and state is not None
    if cache_topk:
        all_topk_d = torch.empty(n_pool, K, dtype=torch.float32)
        all_topk_i = torch.empty(n_pool, K, dtype=torch.int64)

    for start_p in range(0, n_pool, CHUNK_P):
        end_p = min(start_p + CHUNK_P, n_pool)
        chunk_p = X_p[start_p:end_p]

        dists = X_p_sq[start_p:end_p].unsqueeze(1) + X_l_sq.unsqueeze(0)
        dists.addmm_(chunk_p, X_l.T, beta=1.0, alpha=-2.0)
        dists.clamp_(min=0.0).sqrt_()

        if use_topk:
            topk_dists, topk_idx = dists.topk(K, dim=1, largest=False)
            del dists

            if cache_topk:
                all_topk_d[start_p:end_p] = topk_dists.cpu()
                all_topk_i[start_p:end_p] = topk_idx.cpu()

            logits = topk_dists.neg_().div_(temperature)
            soft_assign = torch.softmax(logits, dim=1)
            flat_idx = topk_idx.reshape(-1).long()
            flat_vals = soft_assign.reshape(-1).to(torch.float64)
            weight_accum.scatter_add_(0, flat_idx, flat_vals)
            del topk_dists, topk_idx, logits, soft_assign
        else:
            logits = dists.neg_().div_(temperature)
            soft_assign = torch.softmax(logits, dim=1)
            weight_accum += soft_assign.sum(dim=0).to(torch.float64)
            del dists, logits, soft_assign

    if state is not None:
        if cache_topk:
            state['topk_dists'] = all_topk_d
            state['topk_idx'] = all_topk_i
        state['n_labeled'] = n_labeled
        state['K'] = K

    weights = weight_accum / n_pool
    weights = (weights * n_labeled).cpu().numpy().astype(np.float64)

    del X_p, X_l
    torch.cuda.empty_cache()
    return weights


def _soft_voronoi_numpy(X_pool, X_labeled, temperature, topk=0):
    """CPU fallback for soft Voronoi weight computation."""
    n_pool = len(X_pool)
    n_labeled = len(X_labeled)
    weight_accum = np.zeros(n_labeled, dtype=np.float64)

    K = min(topk, n_labeled) if topk > 0 else n_labeled
    use_topk = K < n_labeled

    CHUNK_P = 10000
    for start_p in range(0, n_pool, CHUNK_P):
        end_p = min(start_p + CHUNK_P, n_pool)
        dists = cdist(X_pool[start_p:end_p], X_labeled, metric='euclidean')

        if use_topk:
            # Partial sort: O(n_labeled) per row instead of O(n_labeled log n_labeled)
            idx = np.argpartition(dists, K, axis=1)[:, :K]
            topk_dists = np.take_along_axis(dists, idx, axis=1)
            logits = -topk_dists / temperature
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            soft_assign = exp_l / exp_l.sum(axis=1, keepdims=True)
            np.add.at(weight_accum, idx.ravel(), soft_assign.ravel())
        else:
            logits = -dists / temperature
            # Numerically stable softmax
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            soft_assign = exp_l / exp_l.sum(axis=1, keepdims=True)
            weight_accum += soft_assign.sum(axis=0)

    weights = weight_accum / n_pool
    weights = weights * n_labeled
    return weights


def _distance_chunk_size_torch(n_pool, n_labeled, device):
    """Choose a CUDA chunk size for distance-based reweighting blocks."""
    import torch
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        target_elements = int(free_bytes * 0.20 / 4)
        chunk_size = target_elements // max(n_labeled, 1)
        return max(1, min(n_pool, chunk_size))
    except Exception:
        max_elements = 150_000_000
        return max(1, min(n_pool, max_elements // max(n_labeled, 1)))


def compute_voronoi_l2_weights(X_pool, X_labeled, reweight_lambda=1.0, state=None, max_iter=15):
    """Compute optimal sample weights for labeled points that minimize
    W_1(Uniform(pool), Weighted(labeled)) + lambda * ||w||_2^2.

    We solve the dual convex problem over z in R^n:
    min_z 1/(4*lambda) * ||max(0, z)||_2^2 - 1/N_p * sum_j min_i (D_ji + z_i)

    The optimal weights are w_i = max(0, z_i) / (2 * lambda).

    Supports warm-starting: initializes z from state['z'] if available.
    """
    if state is None:
        state = {}

    try:
        import torch
        if torch.cuda.is_available():
            return _voronoi_l2_weights_torch(X_pool, X_labeled, reweight_lambda, state, max_iter)
    except ImportError:
        pass

    return _voronoi_l2_weights_numpy(X_pool, X_labeled, reweight_lambda, state, max_iter)


def _voronoi_l2_weights_torch(X_pool, X_labeled, reweight_lambda, state, max_iter=15):
    import torch
    
    class VoronoiL2ReweightFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z, X_pool_t, X_labeled_t, X_pool_sq, X_labeled_sq, reweight_lambda_val, chunk_size):
            n_pool = len(X_pool_t)
            n_labeled = len(X_labeled_t)
            device = z.device
            
            z_pos = torch.clamp(z, min=0.0)
            term1 = (1.0 / (4.0 * reweight_lambda_val)) * torch.sum(z_pos ** 2)
            
            total_min_sum = 0.0
            counts = torch.zeros(n_labeled, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                for start_p in range(0, n_pool, chunk_size):
                    end_p = min(start_p + chunk_size, n_pool)
                    chunk_p = X_pool_t[start_p:end_p]
                    
                    # Compute distance matrix chunk
                    dists = X_pool_sq[start_p:end_p].unsqueeze(1) + X_labeled_sq.unsqueeze(0)
                    dists.addmm_(chunk_p, X_labeled_t.T, beta=1.0, alpha=-2.0)
                    dists.clamp_(min=0.0).sqrt_()
                    
                    dists.add_(z.unsqueeze(0))
                    min_vals, argmin_idx = torch.min(dists, dim=1)
                    
                    total_min_sum += min_vals.sum()
                    
                    ones = torch.ones_like(argmin_idx, dtype=torch.float32)
                    counts.scatter_add_(0, argmin_idx, ones)
            
            loss = term1 - total_min_sum / n_pool
            ctx.save_for_backward(z_pos, counts)
            ctx.reweight_lambda = reweight_lambda_val
            ctx.n_pool = n_pool
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            z_pos, counts = ctx.saved_tensors
            reweight_lambda_val = ctx.reweight_lambda
            n_pool = ctx.n_pool
            
            grad_z = (1.0 / (2.0 * reweight_lambda_val)) * z_pos - counts / n_pool
            return grad_output * grad_z, None, None, None, None, None, None

    device = torch.device('cuda')

    n_pool = len(X_pool)
    n_labeled = len(X_labeled)

    uniform_z = 2.0 * float(reweight_lambda) / max(n_labeled, 1)
    if 'z' in state:
        z_prev = state['z']
        if len(z_prev) < n_labeled:
            z_init = np.full(n_labeled, uniform_z, dtype=np.float32)
            z_init[:len(z_prev)] = z_prev
        else:
            z_init = z_prev[:n_labeled]
    else:
        z_init = np.full(n_labeled, uniform_z, dtype=np.float32)

    z = torch.tensor(z_init, dtype=torch.float32, device=device, requires_grad=True)

    pool_cache_id = (id(X_pool), X_pool.shape, str(X_pool.dtype))
    if state.get('pool_cache_id') == pool_cache_id and 'X_pool_t' in state:
        X_p = state['X_pool_t']
        X_pool_sq = state['X_pool_sq']
    else:
        if 'X_pool_t' in state:
            del state['X_pool_t'], state['X_pool_sq']
            torch.cuda.empty_cache()
        X_p = torch.as_tensor(X_pool, dtype=torch.float32, device=device).contiguous()
        X_pool_sq = torch.sum(X_p ** 2, dim=1)
        state['X_pool_t'] = X_p
        state['X_pool_sq'] = X_pool_sq
        state['pool_cache_id'] = pool_cache_id
        print(f"    [Voronoi-L2] Cached pool tensor on GPU: {n_pool} x {X_p.shape[1]}")

    X_l = torch.as_tensor(X_labeled, dtype=torch.float32, device=device).contiguous()
    X_labeled_sq = torch.sum(X_l ** 2, dim=1)

    chunk_size = _distance_chunk_size_torch(n_pool, n_labeled, device)
    print(f"    [Voronoi-L2] CUDA chunk size: {chunk_size} pool rows x {n_labeled} labeled")

    optimizer = torch.optim.LBFGS([z], lr=1.0, max_iter=max_iter,
                                  history_size=10, tolerance_grad=1e-5,
                                  tolerance_change=1e-9)

    def closure():
        optimizer.zero_grad()
        loss = VoronoiL2ReweightFunction.apply(z, X_p, X_l, X_pool_sq, X_labeled_sq, reweight_lambda, chunk_size)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        z_final = z.cpu().numpy()
        w = np.maximum(z_final, 0.0) / (2.0 * reweight_lambda)
        
    state['z'] = z_final

    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum * n_labeled
    else:
        w = np.ones(n_labeled, dtype=np.float64)

    del X_l, X_labeled_sq, z
    return w


def _voronoi_l2_weights_numpy(X_pool, X_labeled, reweight_lambda, state, max_iter=15):
    from scipy.optimize import minimize

    n_pool = len(X_pool)
    n_labeled = len(X_labeled)

    uniform_z = 2.0 * float(reweight_lambda) / max(n_labeled, 1)
    if 'z' in state:
        z_prev = state['z']
        if len(z_prev) < n_labeled:
            z_init = np.full(n_labeled, uniform_z, dtype=np.float64)
            z_init[:len(z_prev)] = z_prev
        else:
            z_init = z_prev[:n_labeled].astype(np.float64)
    else:
        z_init = np.full(n_labeled, uniform_z, dtype=np.float64)

    # Dynamic chunk size based on n_labeled to bound memory
    max_elements = 50_000_000
    chunk_size = max(1, max_elements // n_labeled)

    def objective_and_grad(z_val):
        z_pos = np.maximum(z_val, 0.0)
        term1 = (1.0 / (4.0 * reweight_lambda)) * np.sum(z_pos ** 2)
        grad1 = (1.0 / (2.0 * reweight_lambda)) * z_pos
        
        total_min_sum = 0.0
        counts = np.zeros(n_labeled, dtype=np.float64)
        
        for start_p in range(0, n_pool, chunk_size):
            end_p = min(start_p + chunk_size, n_pool)
            D_chunk = cdist(X_pool[start_p:end_p], X_labeled, metric='euclidean')
            D_chunk += z_val[np.newaxis, :]

            min_vals = D_chunk.min(axis=1)
            argmin_idx = D_chunk.argmin(axis=1)
            
            total_min_sum += min_vals.sum()
            counts += np.bincount(argmin_idx, minlength=n_labeled)
            
        term2 = -total_min_sum / n_pool
        grad2 = -counts / n_pool
        
        return term1 + term2, grad1 + grad2

    res = minimize(
        fun=objective_and_grad,
        x0=z_init,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': 1e-5}
    )

    z_final = res.x
    state['z'] = z_final

    w = np.maximum(z_final, 0.0) / (2.0 * reweight_lambda)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum * n_labeled
    else:
        w = np.ones(n_labeled, dtype=np.float64)

    return w


def _project_simplex(v):
    """Project a vector onto {w >= 0, sum(w) = 1}."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.ones_like(v) / len(v)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def compute_moment_l2_weights(X_pool, X_labeled, reweight_lambda=1.0, state=None,
                              max_iter=200):
    """Feature-only weights for the linear-regression moment objective.

    This is the Section-3/linear-moment analogue of source reweighting:

        min_{a >= 0, sum a = 1}
            ||M_pool - sum_i a_i x_i x_i^T||_op
            + lambda * ||a||_2.

    The returned sample weights are rescaled to sum to n_labeled, matching the
    convention used by the training code.
    """
    if state is None:
        state = {}

    pool_cache_id = (id(X_pool), X_pool.shape, str(X_pool.dtype))
    X_pool = np.asarray(X_pool, dtype=np.float64)
    X_labeled = np.asarray(X_labeled, dtype=np.float64)
    n_labeled = len(X_labeled)
    if n_labeled == 0:
        return np.empty(0, dtype=np.float64)

    if state.get("moment_pool_cache_id") == pool_cache_id and "moment_M_pool" in state:
        M_pool = state["moment_M_pool"]
    else:
        M_pool = (X_pool.T @ X_pool) / max(len(X_pool), 1)
        state["moment_M_pool"] = M_pool
        state["moment_pool_cache_id"] = pool_cache_id

    prev = state.get("moment_weights_sum1")
    if prev is not None and len(prev) > 0:
        a = np.full(n_labeled, 1.0 / n_labeled, dtype=np.float64)
        n_prev = min(len(prev), n_labeled)
        a[:n_prev] = np.asarray(prev[:n_prev], dtype=np.float64)
        a = _project_simplex(a)
    else:
        a = np.full(n_labeled, 1.0 / n_labeled, dtype=np.float64)

    lam = max(float(reweight_lambda), 0.0)
    best_a = a.copy()
    best_obj = np.inf

    for it in range(max(1, int(max_iter))):
        M_weighted = X_labeled.T @ (a[:, None] * X_labeled)
        residual = M_pool - M_weighted
        eigvals, eigvecs = np.linalg.eigh(residual)
        j = int(np.argmax(np.abs(eigvals)))
        eig = float(eigvals[j])
        spectral = abs(eig)
        l2 = float(np.linalg.norm(a))
        obj = spectral + lam * l2
        if obj < best_obj:
            best_obj = obj
            best_a = a.copy()

        sign = 1.0 if eig >= 0 else -1.0
        v = eigvecs[:, j]
        xv = X_labeled @ v
        grad = -sign * (xv ** 2)
        if lam > 0 and l2 > 1e-12:
            grad += lam * a / l2

        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= 1e-12:
            break
        step = 0.5 / (np.sqrt(it + 1.0) * grad_norm)
        a_next = _project_simplex(a - step * grad)
        if np.linalg.norm(a_next - a, ord=1) < 1e-7:
            a = a_next
            break
        a = a_next

    state["moment_weights_sum1"] = best_a
    print(f"    [Moment-L2] op_mismatch={best_obj - lam * np.linalg.norm(best_a):.6g}, "
          f"l2={np.linalg.norm(best_a):.6g}, nonzero={(best_a > 1e-10).sum()}/{n_labeled}")
    return best_a * n_labeled


def compute_kl_weights(X_pool, X_labeled, reweight_lambda=1.0, state=None, max_iter=15):
    """Compute sample weights with final-weight KL regularization.

    This solves the KL-regularized analogue of the L2 reweighting dual:

        min_z lambda * log(mean_i exp(z_i / lambda))
              - 1/N_p * sum_j min_i (D_ji + z_i)

    The induced final weights are softmax(z / lambda), scaled so that
    sum(weights) == n_labeled.
    """
    if reweight_lambda <= 0:
        raise ValueError("reweight_lambda must be positive for KL reweighting.")

    if state is None:
        state = {}

    try:
        import torch
        if torch.cuda.is_available():
            return _kl_weights_torch(X_pool, X_labeled, reweight_lambda, state, max_iter)
    except ImportError:
        pass

    return _kl_weights_numpy(X_pool, X_labeled, reweight_lambda, state, max_iter)


def _kl_weights_torch(X_pool, X_labeled, reweight_lambda, state, max_iter=15):
    import torch

    class KLReweightFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z, X_pool_t, X_labeled_t, X_pool_sq, X_labeled_sq,
                    reweight_lambda_val, chunk_size):
            n_pool = len(X_pool_t)
            n_labeled = len(X_labeled_t)
            device = z.device

            logits = z / reweight_lambda_val
            probs = torch.softmax(logits, dim=0)
            term1 = reweight_lambda_val * (
                torch.logsumexp(logits, dim=0) - np.log(n_labeled)
            )

            total_min_sum = 0.0
            counts = torch.zeros(n_labeled, dtype=torch.float32, device=device)

            with torch.no_grad():
                for start_p in range(0, n_pool, chunk_size):
                    end_p = min(start_p + chunk_size, n_pool)
                    chunk_p = X_pool_t[start_p:end_p]

                    dists = X_pool_sq[start_p:end_p].unsqueeze(1) + X_labeled_sq.unsqueeze(0)
                    dists.addmm_(chunk_p, X_labeled_t.T, beta=1.0, alpha=-2.0)
                    dists.clamp_(min=0.0).sqrt_()
                    dists.add_(z.unsqueeze(0))

                    min_vals, argmin_idx = torch.min(dists, dim=1)
                    total_min_sum += min_vals.sum()

                    ones = torch.ones_like(argmin_idx, dtype=torch.float32)
                    counts.scatter_add_(0, argmin_idx, ones)

            loss = term1 - total_min_sum / n_pool
            ctx.save_for_backward(probs, counts)
            ctx.n_pool = n_pool
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            probs, counts = ctx.saved_tensors
            grad_z = probs - counts / ctx.n_pool
            return grad_output * grad_z, None, None, None, None, None, None

    device = torch.device('cuda')
    n_pool = len(X_pool)
    n_labeled = len(X_labeled)

    if 'z' in state:
        z_prev = state['z']
        if len(z_prev) < n_labeled:
            z_init = np.zeros(n_labeled, dtype=np.float32)
            z_init[:len(z_prev)] = z_prev
        else:
            z_init = z_prev[:n_labeled]
    else:
        z_init = np.zeros(n_labeled, dtype=np.float32)

    z = torch.tensor(z_init, dtype=torch.float32, device=device, requires_grad=True)

    pool_cache_id = (id(X_pool), X_pool.shape, str(X_pool.dtype))
    if state.get('pool_cache_id') == pool_cache_id and 'X_pool_t' in state:
        X_p = state['X_pool_t']
        X_pool_sq = state['X_pool_sq']
    else:
        if 'X_pool_t' in state:
            del state['X_pool_t'], state['X_pool_sq']
            torch.cuda.empty_cache()
        X_p = torch.as_tensor(X_pool, dtype=torch.float32, device=device).contiguous()
        X_pool_sq = torch.sum(X_p ** 2, dim=1)
        state['X_pool_t'] = X_p
        state['X_pool_sq'] = X_pool_sq
        state['pool_cache_id'] = pool_cache_id
        print(f"    [KL] Cached pool tensor on GPU: {n_pool} x {X_p.shape[1]}")

    X_l = torch.as_tensor(X_labeled, dtype=torch.float32, device=device).contiguous()
    X_labeled_sq = torch.sum(X_l ** 2, dim=1)

    chunk_size = _distance_chunk_size_torch(n_pool, n_labeled, device)
    print(f"    [KL] CUDA chunk size: {chunk_size} pool rows x {n_labeled} labeled")

    optimizer = torch.optim.LBFGS([z], lr=1.0, max_iter=max_iter,
                                  history_size=10, tolerance_grad=1e-5,
                                  tolerance_change=1e-9)

    def closure():
        optimizer.zero_grad()
        loss = KLReweightFunction.apply(z, X_p, X_l, X_pool_sq, X_labeled_sq,
                                        reweight_lambda, chunk_size)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        z_centered = z - z.mean()
        w = torch.softmax(z_centered / reweight_lambda, dim=0)
        z_final = z_centered.cpu().numpy()
        w = (w * n_labeled).cpu().numpy().astype(np.float64)

    state['z'] = z_final
    del X_l, X_labeled_sq, z
    return w


def _kl_weights_numpy(X_pool, X_labeled, reweight_lambda, state, max_iter=15):
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    n_pool = len(X_pool)
    n_labeled = len(X_labeled)

    if 'z' in state:
        z_prev = state['z']
        if len(z_prev) < n_labeled:
            z_init = np.zeros(n_labeled, dtype=np.float64)
            z_init[:len(z_prev)] = z_prev
        else:
            z_init = z_prev[:n_labeled].astype(np.float64)
    else:
        z_init = np.zeros(n_labeled, dtype=np.float64)

    max_elements = 50_000_000
    chunk_size = max(1, max_elements // n_labeled)

    def objective_and_grad(z_val):
        logits = z_val / reweight_lambda
        lse = logsumexp(logits)
        probs = np.exp(logits - lse)
        term1 = reweight_lambda * (lse - np.log(n_labeled))

        total_min_sum = 0.0
        counts = np.zeros(n_labeled, dtype=np.float64)

        for start_p in range(0, n_pool, chunk_size):
            end_p = min(start_p + chunk_size, n_pool)
            D_chunk = cdist(X_pool[start_p:end_p], X_labeled, metric='euclidean')
            D_chunk += z_val[np.newaxis, :]

            min_vals = D_chunk.min(axis=1)
            argmin_idx = D_chunk.argmin(axis=1)

            total_min_sum += min_vals.sum()
            counts += np.bincount(argmin_idx, minlength=n_labeled)

        term2 = -total_min_sum / n_pool
        grad = probs - counts / n_pool
        return term1 + term2, grad

    res = minimize(
        fun=objective_and_grad,
        x0=z_init,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': 1e-5}
    )

    z_final = res.x - res.x.mean()
    state['z'] = z_final

    logits = z_final / reweight_lambda
    probs = np.exp(logits - logsumexp(logits))
    return (probs * n_labeled).astype(np.float64)


STRATEGIES = {
    "random": query_random,
    "uncertainty": query_uncertainty,
    "entropy": query_entropy,
    "margin": query_margin,
    "wasserstein": query_wasserstein,
    "wasserstein_l2": query_wasserstein_l2,
    "entropicOT": query_entropicOT,
    "kmedianpp": query_kmedianpp,
    "moment_matching": query_moment_matching,
    "purely_random": query_purely_random,
}


# ── Training & Evaluation ────────────────────────────────

DEFAULT_TRAIN_WEIGHT_SUM = 10_000.0


def _class_ratio_sample_weights(y, lambda_MP=1.0, sample_weight=None,
                                target_sum=None):
    """Apply MP/MR total-weight locking while preserving within-class weights."""
    if target_sum is None:
        raise ValueError("target_sum must be resolved explicitly before training.")
    target_sum = float(target_sum)
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError(f"target_sum must be positive and finite, got {target_sum!r}.")

    n_MP, n_MR = int(np.sum(y == 0)), int(np.sum(y == 1))

    if sample_weight is not None:
        sw = np.array(sample_weight, dtype=np.float64)
    else:
        sw = np.ones(len(y), dtype=np.float64)

    final_w = np.zeros_like(sw)
    mp_mask = (y == 0)
    mr_mask = (y == 1)

    if n_MP > 0:
        sum_mp = sw[mp_mask].sum()
        final_w[mp_mask] = sw[mp_mask] * (lambda_MP / sum_mp) if sum_mp > 0 else lambda_MP / n_MP

    if n_MR > 0:
        sum_mr = sw[mr_mask].sum()
        final_w[mr_mask] = sw[mr_mask] * (1.0 / sum_mr) if sum_mr > 0 else 1.0 / n_MR

    total_w = final_w.sum()
    if total_w > 0:
        final_w *= (target_sum / total_w)
    return final_w


def _resolve_train_weight_target_sum(mode, fixed_sum, initial_labeled_count, current_labeled_count):
    """Resolve the training sample-weight total for the current snapshot."""
    if mode == "fixed":
        return float(fixed_sum)
    if mode == "initial_labeled":
        return float(initial_labeled_count)
    if mode == "current_labeled":
        return float(current_labeled_count)
    raise ValueError(f"Unknown train weight sum mode: {mode!r}")


def _final_weight_summary(y, final_w, target_sum, lambda_MP, *, rtol=1e-6):
    """Validate and summarize final class-balanced training weights."""
    y = np.asarray(y)
    final_w = np.asarray(final_w, dtype=np.float64)
    mp_sum = float(final_w[y == 0].sum())
    mr_sum = float(final_w[y == 1].sum())
    total = float(final_w.sum())
    target_sum = float(target_sum)
    atol = max(1e-6, abs(target_sum) * 1e-8)
    if not np.isclose(total, target_sum, rtol=rtol, atol=atol):
        raise RuntimeError(
            f"Final training weight total {total:.12g} does not match target "
            f"{target_sum:.12g}."
        )
    if np.isclose(float(lambda_MP), 1.0, rtol=rtol, atol=1e-12):
        half = target_sum / 2.0
        if not np.isclose(mp_sum, half, rtol=rtol, atol=atol):
            raise RuntimeError(
                f"MP final training weight total {mp_sum:.12g} does not match "
                f"target/2 {half:.12g}."
            )
        if not np.isclose(mr_sum, half, rtol=rtol, atol=atol):
            raise RuntimeError(
                f"MR final training weight total {mr_sum:.12g} does not match "
                f"target/2 {half:.12g}."
            )
    return {
        "train_weight_target_sum": target_sum,
        "train_weight_actual_sum": total,
        "train_weight_MP_sum": mp_sum,
        "train_weight_MR_sum": mr_sum,
    }


def train_logistic(X, y, lambda_MP=1.0, C=1.0, prev_clf=None, sample_weight=None,
                   target_sum=None):
    """Train logistic regression with guaranteed class weight totals.

    Regardless of the per-sample weights provided (e.g. Voronoi weights),
    the final training weights are rescaled in two steps:

    1. **Class-ratio lock**: MP weights are scaled to sum to lambda_MP,
       MR weights to 1.0.  Within each class, relative Voronoi weights
       are preserved.
    2. **Global normalisation**: all weights are uniformly rescaled to a
       fixed total. This makes the data-fit term comparable across snapshots,
       so C has a stable, consistent
       meaning throughout the active-learning loop (n_labeled grows
       from warm-start size to warm-start + all queries).

    If prev_clf is given, its coefficients are used to warm-start LBFGS
    so that convergence takes only a few iterations.
    """
    # Step 2: Normalise to a fixed total.
    # The sklearn objective is: sum_i(w_i * loss_i) + (1/2C)*||coef||^2
    # sklearn does NOT normalise sample_weight internally, so sum(w_i) sets
    # the scale of the data-fit term.  In active learning n_labeled grows
    # over time; if we normalised to n_labeled the fit term would grow with
    # every snapshot, making C effectively weaker and weaker.  Normalising
    # to a fixed total keeps the data-fit term comparable throughout — C then has a fixed,
    # dataset-size-independent meaning.  The class ratio and within-class
    # Voronoi corrections are unaffected (we only multiply by a scalar).
    final_w = _class_ratio_sample_weights(y, lambda_MP, sample_weight,
                                          target_sum=target_sum)

    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                             warm_start=True)
    # Seed from previous solution so LBFGS starts near the optimum
    if prev_clf is not None:
        clf.coef_ = prev_clf.coef_.copy()
        clf.intercept_ = prev_clf.intercept_.copy()
        clf.classes_ = prev_clf.classes_.copy()
    clf.fit(X, y, sample_weight=final_w)
    return clf


class RidgeRegressionClassifier:
    """Ridge regression used as a binary classifier with MP as the low score."""

    def __init__(self, coef, intercept):
        self.coef_ = np.asarray(coef, dtype=np.float64).reshape(1, -1)
        self.intercept_ = np.asarray([intercept], dtype=np.float64)
        self.classes_ = np.array([0, 1], dtype=np.int32)

    def decision_function(self, X):
        return np.asarray(X) @ self.coef_.ravel() + self.intercept_[0]

    def predict(self, X):
        # y=0 is MP, y=1 is MR. The ridge target is -1 for MP and +1 for MR.
        return (self.decision_function(X) >= 0.0).astype(np.int32)

    def predict_proba(self, X):
        scores = np.clip(self.decision_function(X), -50.0, 50.0)
        p_mr = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - p_mr, p_mr])


def train_ridge_classifier(X, y, lambda_MP=1.0, alpha=1.0, sample_weight=None,
                           target_sum=None):
    """Train weighted ridge regression on targets MP=-1, MR=+1."""
    final_w = _class_ratio_sample_weights(y, lambda_MP, sample_weight,
                                          target_sum=target_sum)
    X = np.asarray(X, dtype=np.float64)
    target = np.where(y == 0, -1.0, 1.0).astype(np.float64)

    X_pad = np.column_stack([X, np.ones(len(X), dtype=np.float64)])
    sqrt_w = np.sqrt(final_w)[:, None]
    Xw = X_pad * sqrt_w
    yw = target[:, None] * sqrt_w

    gram = Xw.T @ Xw
    rhs = Xw.T @ yw
    reg = max(float(alpha), 0.0) * np.eye(X_pad.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    gram += reg

    try:
        sol = np.linalg.solve(gram, rhs).ravel()
    except np.linalg.LinAlgError:
        sol = (np.linalg.pinv(gram) @ rhs).ravel()

    return RidgeRegressionClassifier(sol[:-1], sol[-1])


class XGBoostBinaryClassifier:
    """Thin adapter exposing the classifier interface used by this script."""

    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([0, 1], dtype=np.int32)

    @property
    def feature_importances_(self):
        return getattr(self.model, "feature_importances_", None)

    def predict_proba(self, X):
        proba = np.asarray(self.model.predict_proba(X), dtype=np.float64)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        return proba

    def decision_function(self, X):
        p_mr = np.clip(self.predict_proba(X)[:, 1], 1e-12, 1.0 - 1e-12)
        return np.log(p_mr / (1.0 - p_mr))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)


def train_xgboost_classifier(X, y, lambda_MP=1.0, sample_weight=None, *,
                             n_estimators=400, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=1.0, gamma=0.0,
                             reg_lambda=1.0, tree_method="hist",
                             device="auto", n_jobs=-1, random_state=42,
                             target_sum=None):
    """Train an XGBoost boosted-tree classifier, following Yao et al.'s model family.

    The paper uses XGBoost multiclass classifiers over metallicity bins.  The
    active-learning loop here has only binary MP/MR labels, so this implements
    the same boosted decision-tree family as a binary classifier while preserving
    the existing class-ratio and reweighting semantics.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "--model xgboost requires the optional 'xgboost' package. "
            "Install it in this environment, e.g. `pip install xgboost`, "
            "then rerun with --model xgboost."
        ) from exc

    final_w = _class_ratio_sample_weights(y, lambda_MP, sample_weight,
                                          target_sum=target_sum)
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        min_child_weight=float(min_child_weight),
        gamma=float(gamma),
        reg_lambda=float(reg_lambda),
        tree_method=str(tree_method),
        n_jobs=int(n_jobs),
        random_state=int(random_state),
    )
    if device is not None and str(device).lower() != "auto":
        params["device"] = str(device)

    try:
        model = XGBClassifier(use_label_encoder=False, **params)
    except TypeError:
        model = XGBClassifier(**params)

    model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32),
              sample_weight=final_w)
    return XGBoostBinaryClassifier(model)


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
    print(f"[Query {m['n_queries']:4d}] Acc={m['accuracy']:.4f}  "
          f"Loss(test MP={m['loss_MP']:.4f} MR={m['loss_MR']:.4f} "
          f"avg={m['avg_test_loss']:.4f})  "
          f"labeled={m['n_labeled']} (MP={m['n_labeled_MP']}, MR={m['n_labeled_MR']})")


# ── Plotting ─────────────────────────────────────────────


def compute_pr_auc(clf, X_eval, y_eval):
    """Compute Precision-Recall AUC for the MP class."""
    from sklearn.metrics import precision_recall_curve, auc
    y_true_mp = (y_eval == 0).astype(int)
    y_scores = clf.predict_proba(X_eval)[:, 0]
    precision, recall, _ = precision_recall_curve(y_true_mp, y_scores)
    precision, recall = precision[:-1], recall[:-1]
    return auc(recall, precision)


def compute_average_precision(clf, X_eval, y_eval):
    """Compute sklearn average precision for the MP class."""
    from sklearn.metrics import average_precision_score
    y_true_mp = (y_eval == 0).astype(int)
    y_scores = clf.predict_proba(X_eval)[:, 0]
    return float(average_precision_score(y_true_mp, y_scores))


def _save_auc_trials_plot(auc_query_points, all_trial_aucs, out_dir, n_trials):
    """Plot PR-AUC across trials with confidence region (mean ± std)."""
    # Pad any short trial lists with NaN (e.g. if clf was None at a snapshot)
    max_len = max(len(t) for t in all_trial_aucs)
    padded = [t + [float('nan')] * (max_len - len(t)) for t in all_trial_aucs]
    aucs = np.array(padded, dtype=float)  # (n_trials, n_snapshots)
    mean_auc = np.nanmean(aucs, axis=0)
    std_auc  = np.nanstd(aucs, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(auc_query_points, mean_auc, 'o-', color='#4A90D9', lw=2, markersize=6,
            label='Mean PR-AUC')
    ax.fill_between(auc_query_points, mean_auc - std_auc, mean_auc + std_auc,
                    alpha=0.25, color='#4A90D9', label='±1 std')

    # Overlay individual trial lines faintly
    for t in range(n_trials):
        ax.plot(auc_query_points, aucs[t], '-', color='#999999', alpha=0.3, lw=0.8)

    ax.set_xlabel('Number of Queries', fontsize=12)
    ax.set_ylabel('PR-AUC (MP Class)', fontsize=12)
    ax.set_title(f'PR-AUC across {n_trials} Trials', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_file = os.path.join(out_dir, 'auc_trials.png')
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved AUC trials plot to {out_file}")


def _save_average_precision_trials_plot(query_points, all_trial_aps, out_dir, n_trials):
    """Plot average precision across trials with confidence region (mean ± std)."""
    max_len = max(len(t) for t in all_trial_aps)
    padded = [t + [float('nan')] * (max_len - len(t)) for t in all_trial_aps]
    aps = np.array(padded, dtype=float)
    mean_ap = np.nanmean(aps, axis=0)
    std_ap = np.nanstd(aps, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(query_points, mean_ap, 'o-', color='#5A9E7A', lw=2, markersize=6,
            label='Mean AP')
    ax.fill_between(query_points, mean_ap - std_ap, mean_ap + std_ap,
                    alpha=0.25, color='#5A9E7A', label='±1 std')

    for t in range(n_trials):
        ax.plot(query_points, aps[t], '-', color='#999999', alpha=0.3, lw=0.8)

    ax.set_xlabel('Number of Queries', fontsize=12)
    ax.set_ylabel('Average Precision (MP Class)', fontsize=12)
    ax.set_title(f'Average Precision across {n_trials} Trials', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_file = os.path.join(out_dir, 'average_precision_trials.png')
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved average-precision trials plot to {out_file}")


def _save_test_loss_trials_plot(eval_query_points, all_trial_test_losses, out_dir, n_trials):
    """Plot per-class test loss across trials with mean ± std (log-scale y-axis).

    Generates two plots: test_loss_MP_trials.png and test_loss_MR_trials.png.
    Each plot shows the average test loss evolution over training with variance bands.
    """
    max_len = max(len(t) for t in all_trial_test_losses)
    # Each element in all_trial_test_losses is a list of dicts
    # with keys "loss_MP", "loss_MR", "avg_test_loss"

    for class_key, class_label, color, filename in [
        ("loss_MP", "MP", "#E07070", "test_loss_MP_trials.png"),
        ("loss_MR", "MR", "#4A90D9", "test_loss_MR_trials.png"),
    ]:
        # Extract per-trial loss arrays
        trial_losses = []
        for trial_data in all_trial_test_losses:
            losses = [d[class_key] for d in trial_data]
            # Pad with NaN if needed
            losses += [float('nan')] * (max_len - len(losses))
            trial_losses.append(losses)

        arr = np.array(trial_losses, dtype=float)  # (n_trials, n_evals)
        mean_loss = np.nanmean(arr, axis=0)
        std_loss  = np.nanstd(arr, axis=0)

        # Trim eval_query_points to match
        qp = eval_query_points[:max_len]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(qp, mean_loss, 'o-', color=color, lw=2, markersize=5,
                label=f'Mean Test Loss ({class_label})')
        ax.fill_between(qp, mean_loss - std_loss, mean_loss + std_loss,
                        alpha=0.25, color=color, label='±1 std')

        # Overlay individual trial lines faintly
        for t in range(n_trials):
            ax.plot(qp, arr[t], '-', color='#999999', alpha=0.3, lw=0.8)

        ax.set_xlabel('Number of Queries', fontsize=12)
        ax.set_ylabel(f'Test Log-Loss ({class_label})', fontsize=12)
        ax.set_yscale('log')
        ax.set_title(f'{class_label} Test Loss across {n_trials} Trials',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_file = os.path.join(out_dir, filename)
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
        print(f"Saved {class_label} test loss trials plot to {out_file}")


def _compute_reweight_stats(sample_weight, y_labeled, n_queries):
    """Summarize concentration of reweighting weights before class-ratio scaling."""
    if sample_weight is None:
        return None

    w = np.asarray(sample_weight, dtype=np.float64).ravel()
    if len(w) == 0:
        return None

    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        return None

    p = w / total
    l2_sq = float(np.dot(p, p))
    l2_norm = float(np.sqrt(l2_sq))
    ess = float(1.0 / l2_sq) if l2_sq > 0.0 else float("inf")
    nonzero = p > 0

    def top_mass(k):
        k = min(int(k), len(p))
        if k <= 0:
            return 0.0
        if k == len(p):
            return 1.0
        return float(np.partition(p, -k)[-k:].sum())

    y_arr = np.asarray(y_labeled)
    mp_mask = y_arr == 0
    mr_mask = y_arr == 1

    return {
        "n_queries": int(n_queries),
        "n_labeled": int(len(w)),
        "weight_sum": total,
        "objective_l2_norm": l2_norm,
        "objective_l2_sq": l2_sq,
        "effective_sample_size": ess,
        "effective_sample_fraction": float(ess / len(w)) if len(w) else float("nan"),
        "max_mass": float(np.max(p)),
        "top10_mass": top_mass(10),
        "top100_mass": top_mass(100),
        "nonzero_count": int(np.count_nonzero(nonzero)),
        "nonzero_fraction": float(np.mean(nonzero)),
        "mp_mass": float(p[mp_mask].sum()) if len(mp_mask) == len(p) else float("nan"),
        "mr_mass": float(p[mr_mask].sum()) if len(mr_mask) == len(p) else float("nan"),
        "returned_weight_l2_norm": float(np.linalg.norm(w)),
        "returned_weight_l2_sq": float(np.dot(w, w)),
    }


def _save_weight_stats_trials_plot(all_trial_weight_stats, out_dir, n_trials):
    if not all_trial_weight_stats or not any(all_trial_weight_stats):
        return

    query_points = sorted({
        int(d["n_queries"])
        for trial_data in all_trial_weight_stats
        for d in trial_data
        if "n_queries" in d
    })
    if not query_points:
        return

    q_to_col = {q: i for i, q in enumerate(query_points)}
    metrics = [
        ("objective_l2_norm", "Objective Weight L2 Norm ||p||_2", "weight_l2_norm_trials.png"),
        ("objective_l2_sq", "Objective Weight L2 Squared ||p||_2^2", "weight_l2_sq_trials.png"),
        ("effective_sample_size", "Effective Sample Size 1 / ||p||_2^2", "weight_effective_sample_size_trials.png"),
    ]

    for metric, ylabel, filename in metrics:
        arr = np.full((len(all_trial_weight_stats), len(query_points)), np.nan, dtype=float)
        for t, trial_data in enumerate(all_trial_weight_stats):
            for d in trial_data:
                if metric in d and "n_queries" in d:
                    arr[t, q_to_col[int(d["n_queries"])]] = float(d[metric])

        if np.all(np.isnan(arr)):
            continue

        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(query_points, mean, "-o", color="#2563eb", lw=2.0,
                label=f"Mean over {n_trials} trials")
        ax.fill_between(query_points, mean - std, mean + std,
                        color="#2563eb", alpha=0.18, label="±1 std")
        for row in arr:
            ax.plot(query_points, row, "-", color="#93c5fd", alpha=0.20, lw=0.8)

        ax.set_xlabel("Number of Queried Points", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel + " vs. Query Count", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, ls="--")
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()
        out_file = os.path.join(out_dir, filename)
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
        print(f"Saved weight-stat trials plot to {out_file}")


def _save_mp_trials_plot(auc_query_points, all_trial_mp_counts, out_dir, n_trials):
    """Plot queried MP fraction across trials with confidence region (mean ± std)."""
    max_len = max(len(t) for t in all_trial_mp_counts)
    padded = [t + [float('nan')] * (max_len - len(t)) for t in all_trial_mp_counts]
    counts = np.array(padded, dtype=float)  # (n_trials, n_snapshots)

    # Convert cumulative MP counts to fractions: mp_count / total_queries
    queries_arr = np.array(auc_query_points[:max_len], dtype=float)
    fractions = counts / queries_arr[np.newaxis, :]

    mean_frac = np.nanmean(fractions, axis=0)
    std_frac  = np.nanstd(fractions, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(auc_query_points, mean_frac, 'o-', color='#E07070', lw=2, markersize=6,
            label='Mean MP Fraction')
    ax.fill_between(auc_query_points, mean_frac - std_frac, mean_frac + std_frac,
                    alpha=0.25, color='#E07070', label='±1 std')

    # Overlay individual trial lines faintly
    for t in range(n_trials):
        ax.plot(auc_query_points, fractions[t], '-', color='#999999', alpha=0.3, lw=0.8)

    ax.set_xlabel('Number of Queries', fontsize=12)
    ax.set_ylabel('MP Fraction in Queried Samples', fontsize=12)
    ax.set_title(f'Queried MP Fraction across {n_trials} Trials', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_file = os.path.join(out_dir, 'mp_fraction_trials.png')
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved MP fraction trials plot to {out_file}")


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


def save_final_model_summary(clf, full_data_file, out_dir):
    """Save linear weights or tree feature importances for the final classifier."""
    with h5py.File(full_data_file, "r") as f:
        cols = _feature_cols(list(f.keys()))

    out_file = os.path.join(out_dir, "final_weights.csv")
    if hasattr(clf, "coef_") and hasattr(clf, "intercept_"):
        w, b = clf.coef_.flatten(), clf.intercept_[0]
        with open(out_file, "w") as f:
            f.write("feature,weight\n" + f"BIAS,{b}\n")
            f.writelines(f"{name},{wv}\n" for name, wv in zip(cols, w))
        return

    importances = getattr(clf, "feature_importances_", None)
    if importances is not None:
        importances = np.asarray(importances, dtype=np.float64).ravel()
        with open(out_file, "w") as f:
            f.write("feature,importance\n")
            f.writelines(f"{name},{val}\n" for name, val in zip(cols, importances))
        return

    with open(out_file, "w") as f:
        f.write("feature,value\n")
        f.write("MODEL_HAS_NO_LINEAR_WEIGHTS_OR_FEATURE_IMPORTANCES,nan\n")


# ── Main Loop ────────────────────────────────────────────

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
    auc_query_points = [snap_interval * (i + 1) for i in range(args.n_snapshots)]
    auc_query_set = set(auc_query_points)

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
    else:
        if eval_source_id_mode:
            pool_mask = ~np.isin(sid_full, sid_warm)
        else:
            print("  Warning: source_id unavailable; using approximate dedup.")
            warm_set = {tuple(np.round(x, 6)) for x in X_warm}
            pool_mask = np.array([tuple(np.round(x, 6)) not in warm_set for x in X_full])

        X_pool, y_pool = X_full[pool_mask].copy(), y_full[pool_mask].copy()
        eval_n = min(args.eval_size, len(X_pool))
        eval_idx = eval_rng.choice(len(X_pool), eval_n, replace=False)
        X_eval, y_eval = X_pool[eval_idx], y_pool[eval_idx]

    # Free the full arrays (only pool & eval are needed hereafter)
    del X_full, y_full, sid_full, sid_warm

    for tag, n, mp in [("Warm-start", len(X_warm), (y_warm == 0).sum()),
                       ("Pool", len(X_pool), (y_pool == 0).sum()),
                       ("Eval set", eval_n, (y_eval == 0).sum())]:
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

    # 4. Pre-allocate labeled arrays (reused across trials)
    max_labeled = len(X_warm) + args.total_queries
    n_features = X_warm.shape[1]
    X_labeled = np.empty((max_labeled, n_features), dtype=np.float32)
    y_labeled = np.empty(max_labeled, dtype=np.int32)

    # ── Multi-trial loop ──
    all_trial_aucs = []          # legacy trapezoidal PR-AUC values
    all_trial_average_precisions = []  # sklearn average_precision_score values
    all_trial_mp_counts = []     # list of lists, cumulative MP counts at each snapshot
    all_trial_test_losses = []   # list of lists of dicts, test losses at each eval point
    all_trial_weight_stats = []  # list of lists of reweighting concentration stats
    first_trial_results = None

    for trial in range(args.n_trials):
        if args.n_trials > 1:
            print(f"\n{'=' * 60}")
            print(f"Trial {trial + 1} / {args.n_trials}  (seed={args.seed + trial})")
            print(f"{'=' * 60}")

        rng = np.random.RandomState(args.seed + trial)

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

        # Pre-compute fixed reweight-pool subsample for this trial (reused across snapshots
        # so the incremental top-K cache stays valid)
        if args.reweighting in ("soft", "voronoi_l2", "kl", "moment_l2") and args.reweight_pool_size and args.reweight_pool_size < len(X_pool):
            reweight_pool_idx = rng.choice(len(X_pool), args.reweight_pool_size, replace=False)
            X_reweight_pool = X_pool[reweight_pool_idx]
        else:
            X_reweight_pool = X_pool

        trial_aucs = []
        trial_average_precisions = []
        trial_mp_counts = []
        trial_test_losses = []   # test losses at each eval point for this trial
        trial_weight_stats = []
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
                print(f"  [Voronoi-Hard] Computing sample weights ({n_labeled} labeled vs {len(X_pool)} pool)...")
                sw, voronoi_state = compute_voronoi_weights(X_pool, Xl, voronoi_state)
            elif args.reweighting == "soft":
                reweight_t0 = time.perf_counter()
                reweight_label = "Voronoi-soft weights"
                print(f"  [Voronoi-Soft] Computing sample weights (τ={args.temperature}, "
                      f"{n_labeled} labeled vs {len(X_reweight_pool)} pool"
                      f"{f' (subsampled from {len(X_pool)})' if len(X_reweight_pool) < len(X_pool) else ''})...")
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
                      f"{n_labeled} labeled vs {len(X_reweight_pool)} pool"
                      f"{f' (subsampled from {len(X_pool)})' if len(X_reweight_pool) < len(X_pool) else ''})...")
                sw = compute_voronoi_l2_weights(X_reweight_pool, Xl, args.reweight_lambda,
                                                state=voronoi_l2_state,
                                                max_iter=l2_max_iter)
            elif args.reweighting == "kl":
                reweight_t0 = time.perf_counter()
                reweight_label = "KL weights"
                print(f"  [KL] Computing sample weights (λ={args.reweight_lambda}, "
                      f"{n_labeled} labeled vs {len(X_reweight_pool)} pool"
                      f"{f' (subsampled from {len(X_pool)})' if len(X_reweight_pool) < len(X_pool) else ''})...")
                sw = compute_kl_weights(X_reweight_pool, Xl, args.reweight_lambda,
                                        state=kl_state,
                                        max_iter=args.voronoi_l2_max_iter)
            elif args.reweighting == "moment_l2":
                reweight_t0 = time.perf_counter()
                reweight_label = "Moment-L2 weights"
                print(f"  [Moment-L2] Computing sample weights (λ={args.reweight_lambda}, "
                      f"{n_labeled} labeled vs {len(X_reweight_pool)} pool"
                      f"{f' (subsampled from {len(X_pool)})' if len(X_reweight_pool) < len(X_pool) else ''})...")
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
                yl, args.lambda_MP, sw, target_sum=target_weight_sum
            )
            weight_summary = _final_weight_summary(
                yl, final_train_w, target_weight_sum, args.lambda_MP
            )
            print(
                "  [TrainWeights] "
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
                                     target_sum=target_weight_sum)
            elif args.model == "ridge":
                clf = train_ridge_classifier(Xl, yl, args.lambda_MP,
                                             alpha=args.ridge_alpha,
                                             sample_weight=sw,
                                             target_sum=target_weight_sum)
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
        else:
            clf = None

        # 6. Active learning loop
        queried = 0

        while queried < args.total_queries and available.any():
            batch = min(args.eval_every, args.total_queries - queried, int(available.sum()))
            query_t0 = time.perf_counter()

            if args.strategy in ("wasserstein", "wasserstein_l2"):
                pool_idx = strategy_fn(X_pool, clf, batch, rng,
                                       X_labeled=X_labeled[:n_labeled],
                                       state=strategy_state,
                                       pool_size=args.wass_pool_size,
                                       plan_size=args.wass_plan_size,
                                       available_mask=available,
                                       reweight_lambda=args.reweight_lambda,
                                       temperature=args.eot_temperature)
            else:
                avail_idx = np.where(available)[0]
                sel = strategy_fn(X_pool[avail_idx], clf, batch, rng,
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
                n_queried_mp = int(np.sum(y_labeled[warm_n:n_labeled] == 0))
                trial_mp_counts.append(n_queried_mp)

        all_trial_aucs.append(trial_aucs)
        all_trial_average_precisions.append(trial_average_precisions)
        all_trial_mp_counts.append(trial_mp_counts)
        all_trial_test_losses.append(trial_test_losses)
        all_trial_weight_stats.append(trial_weight_stats)

        # 7. Save detailed outputs (first trial only)
        if trial == 0:
            first_trial_results = results
            t_trial = time.perf_counter() - t0
            print(f"\nTrial 1 runtime: {t_trial:.1f}s  (data loading: {t_load:.1f}s)")

            with open(os.path.join(args.out_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)

            # Linear models write coefficients; tree models write feature importances.
            save_final_model_summary(clf, args.full_data_file, args.out_dir)

            with open(os.path.join(args.out_dir, "params.json"), "w") as f:
                json.dump(vars(args), f, indent=2)



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

            # PR curve from snapshot classifiers
            pr_curves = [(f"{q} queries", c) for q, c in clf_snapshots]
            generate_pr_curve(pr_curves, X_eval, y_eval, args.out_dir)

    # 8. Summary & multi-trial AUC plot
    t_total = time.perf_counter() - t0
    print(f"\nTotal runtime ({args.n_trials} trial(s)): {t_total:.1f}s")

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

    print(f"\nAll outputs saved to {args.out_dir}/")
    return first_trial_results


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
    a("--total-queries",  type=int, default=3000, help="Total points to query.")
    a("--eval-every",     type=int, default=200,  help="Retrain & evaluate every k queries.")

    # Model
    a("--model", default="logistic", choices=["logistic", "ridge", "xgboost"],
      help="Final classifier: logistic regression, ridge-regression classifier, or XGBoost boosted trees.")
    a("--lambda-MP", type=float, default=1.0, help="Desired total-weight ratio MP/MR. Per-sample weights are auto-scaled so n_MP*w_MP / n_MR*w_MR = lambda_MP.")
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
    a("--voronoi-l2-max-iter", type=int, default=15,
       help="Maximum LBFGS iterations for voronoi_l2/kl reweighting.")
    a("--voronoi-l2-initial-max-iter", type=int, default=None,
       help="Maximum LBFGS iterations for the first voronoi_l2 reweighting solve in each trial. Defaults to --voronoi-l2-max-iter.")
    a("--temperature", type=float, default=1.0,
       help="Temperature τ for soft reweighting. τ→0 = hard, τ→∞ = uniform. Only used when --reweighting=soft.")
    a("--soft-topk", type=int, default=0,
       help="Top-K for soft reweighting. 0=auto (calibrate K per snapshot). Only used when --reweighting=soft.")
    a("--reweight-pool-size", dest="reweight_pool_size", type=int, default=None,
       help="Subsample pool to this size for soft/voronoi_l2/kl/moment_l2 reweighting. By default (None) uses the full pool. "
            "Setting e.g. 500000 computes weights on a 500k subsample instead of the full pool. "
            "Hard reweighting is unaffected.")
    a("--softmax-pool-size", dest="reweight_pool_size", type=int, default=None,
       help=argparse.SUPPRESS)

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
    a("--out-dir",         default=None, help="Output directory (default: al_{strategy}).")

    args = p.parse_args()
    if args.wass_plan_size is None:
        args.wass_plan_size = args.eval_every
    if args.wass_plan_size <= 0:
        p.error("--wass-plan-size must be positive.")
    if args.voronoi_l2_initial_max_iter is None:
        args.voronoi_l2_initial_max_iter = args.voronoi_l2_max_iter
    if args.voronoi_l2_initial_max_iter <= 0:
        p.error("--voronoi-l2-initial-max-iter must be positive.")
    if args.train_weight_sum <= 0:
        p.error("--train-weight-sum must be positive.")
    if args.out_dir is None:
        args.out_dir = f"al_{args.strategy}"
    run_active_learning(args)


if __name__ == "__main__":
    main()
