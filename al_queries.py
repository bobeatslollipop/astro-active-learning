"""Active-learning query strategies and geometric planning helpers."""

import time

import numpy as np
from scipy.spatial.distance import cdist

from al_data import _timing, mp_probability


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
    probs = mp_probability(clf, X_pool)
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
    probs = mp_probability(clf, X_pool)
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
