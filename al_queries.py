"""Active-learning query strategies and geometric planning helpers."""

import hashlib
import time

import numpy as np
from scipy.spatial.distance import cdist

from al_data import _timing, mp_probability


WASSERSTEIN_L2_QUERY_OBJECTIVE = "power_diagram_restricted_dual_drop"
WASSERSTEIN_L2_IMPLEMENTATION_VERSION = 3
WASSERSTEIN_L2_DUAL_UPDATE = "fixed_old_z_block_corrective_new_coordinates"
KMEDIANPP_QUERY_OBJECTIVE = "distance_proportional_sampling"
KMEDIANPP_IMPLEMENTATION_VERSION = 2
KMEDIANPP_CANDIDATE_POOL = "fresh_random_subpool_per_query_batch"


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
                         reweight_lambda=1.0, reweight_target=None,
                         voronoi_l2_state=None, coordinate_steps=32,
                         corrective_max_sweeps=128,
                         corrective_dual_relative_tol=1e-8,
                         corrective_z_relative_tol=1e-6,
                         corrective_patience=2, candidate_chunk_size=None,
                         **kw):
    """Power-cell greedy insertion with fixed old dual variables (v3).

    The preceding Voronoi-L2 reweighting solve supplies the old support dual
    ``z*`` and adjusted target costs ``b_t = min_i(d(t, s_i) + z_i*)``.  Each
    candidate optimises only its new scalar coordinate.  After selection, all
    coordinates introduced in the current query batch are jointly corrected
    by cyclic exact-coordinate updates while the old ``z*`` remains fixed.

    The score is a restricted-dual drop, not a certified primal-objective
    decrease and not the exact candidate-wise regularized-OT greedy objective.
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
    if reweight_target is None:
        raise ValueError(
            "wasserstein_l2 v3 requires the actual Voronoi-L2 reweight target."
        )
    if voronoi_l2_state is None:
        raise ValueError(
            "wasserstein_l2 v3 requires the preceding Voronoi-L2 optimizer state."
        )
    if coordinate_steps <= 0:
        raise ValueError("coordinate_steps must be positive")
    if corrective_max_sweeps <= 0:
        raise ValueError("corrective_max_sweeps must be positive")
    if corrective_dual_relative_tol < 0 or corrective_z_relative_tol < 0:
        raise ValueError("corrective tolerances must be non-negative")
    if corrective_patience <= 0:
        raise ValueError("corrective_patience must be positive")

    from al_reweighting import ensure_voronoi_l2_power_state

    power_state = ensure_voronoi_l2_power_state(
        reweight_target, X_labeled, lam, voronoi_l2_state
    )
    t_plan = time.perf_counter()
    planned, trace = _build_wasserstein_l2_plan(
        X_pool,
        n_pick,
        rng,
        pool_size,
        available_idx,
        reweight_target,
        power_state,
        lam,
        coordinate_steps=coordinate_steps,
        corrective_max_sweeps=corrective_max_sweeps,
        corrective_dual_relative_tol=corrective_dual_relative_tol,
        corrective_z_relative_tol=corrective_z_relative_tol,
        corrective_patience=corrective_patience,
        candidate_chunk_size=candidate_chunk_size,
    )
    selected = np.asarray(planned, dtype=np.intp)
    for step, pool_index in zip(trace["steps"], selected):
        step["pool_index"] = int(pool_index)
    source_trace = voronoi_l2_state.get("last_optimizer_trace", {})
    trace["source_reweight_solve"] = {
        key: source_trace.get(key)
        for key in (
            "trial", "seed", "n_queries", "solve", "stop_reason",
            "termination_class", "converged", "certified",
            "stable_not_certified", "accepted_updates_completed",
            "final_relative_primal_dual_gap", "final_grad_inf",
        )
        if key in source_trace
    }
    trace["elapsed_seconds"] = float(time.perf_counter() - t_plan)
    state.clear()
    state["last_plan_trace"] = trace
    _timing("Wasserstein-L2 v3 plan build", t_plan)
    return selected


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


def _build_wasserstein_l2_plan(
    X_pool,
    n_plan,
    rng,
    pool_size,
    available_idx,
    reweight_target,
    power_state,
    reweight_lambda,
    *,
    coordinate_steps=32,
    corrective_max_sweeps=128,
    corrective_dual_relative_tol=1e-8,
    corrective_z_relative_tol=1e-6,
    corrective_patience=2,
    candidate_chunk_size=None,
):
    """Build one v3 plan; no plan tail survives the next reweight solve."""
    effective_ps = min(pool_size, len(available_idx))
    if effective_ps <= 0:
        return np.empty(0, dtype=np.intp), {"steps": []}

    n_plan = min(n_plan, effective_ps)
    subpool_idx = rng.choice(available_idx, effective_ps, replace=False)
    candidates = X_pool[subpool_idx]
    target = np.asarray(reweight_target)
    base_cost = np.asarray(power_state["base_cost"], dtype=np.float32)

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    coupling_kwargs = {
        "coordinate_steps": coordinate_steps,
        "corrective_max_sweeps": corrective_max_sweeps,
        "corrective_dual_relative_tol": corrective_dual_relative_tol,
        "corrective_z_relative_tol": corrective_z_relative_tol,
        "corrective_patience": corrective_patience,
        "candidate_chunk_size": candidate_chunk_size,
    }
    if has_torch and torch.cuda.is_available():
        chosen, trace = _wasserstein_l2_coupling_torch(
            candidates, target, base_cost, n_plan, reweight_lambda,
            **coupling_kwargs,
        )
    else:
        chosen, trace = _wasserstein_l2_coupling_numpy(
            candidates, target, base_cost, n_plan, reweight_lambda,
            **coupling_kwargs,
        )

    trace.update({
        "schema_version": 1,
        "query_objective": WASSERSTEIN_L2_QUERY_OBJECTIVE,
        "query_implementation_version": WASSERSTEIN_L2_IMPLEMENTATION_VERSION,
        "query_dual_update": WASSERSTEIN_L2_DUAL_UPDATE,
        "target_source": "reweight_target",
        "target_rows": int(len(target)),
        "candidate_rows": int(len(candidates)),
        "candidate_subpool_sha256": hashlib.sha256(
            np.asarray(subpool_idx, dtype="<i8").tobytes()
        ).hexdigest(),
        "power_diagram_generation": int(power_state["generation"]),
    })
    return subpool_idx[np.asarray(chosen, dtype=np.intp)], trace


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


def _compact_voronoi_cells_numpy(raw_cell_ids):
    """Map occupied support IDs to a compact range and count their targets."""
    raw_cell_ids = np.asarray(raw_cell_ids, dtype=np.intp)
    compact_ids = np.full(len(raw_cell_ids), -1, dtype=np.intp)
    occupied = raw_cell_ids >= 0
    if not np.any(occupied):
        return compact_ids, np.empty(0, dtype=np.float64)
    _, inverse = np.unique(raw_cell_ids[occupied], return_inverse=True)
    compact_ids[occupied] = inverse
    counts = np.bincount(inverse).astype(np.float64, copy=False)
    return compact_ids, counts


def _wasserstein_l2_base_cells_numpy(T, X_labeled):
    """Return nearest-support distances and compact Voronoi cell state."""
    ps = len(T)
    base_min = np.full(ps, np.inf, dtype=np.float32)
    raw_cell_ids = np.full(ps, -1, dtype=np.intp)
    if X_labeled is not None and len(X_labeled) > 0:
        chunk_size = 5000
        for start in range(0, len(X_labeled), chunk_size):
            end = min(start + chunk_size, len(X_labeled))
            dists = cdist(T, X_labeled[start:end], metric="euclidean").astype(
                np.float32
            )
            local_argmin = dists.argmin(axis=1)
            local_min = dists[np.arange(ps), local_argmin]
            improved = local_min < base_min
            base_min[improved] = local_min[improved]
            raw_cell_ids[improved] = start + local_argmin[improved]
            del dists
    cell_ids, cell_counts = _compact_voronoi_cells_numpy(raw_cell_ids)
    return base_min, cell_ids, cell_counts


def _wasserstein_l2_full_penalties_numpy(
    base_min, intra_dists, cell_ids, cell_counts, row_chunk=None
):
    """Evaluate full updated Voronoi mass-squared penalties for all candidates.

    The returned values exclude lambda.  Counts are divided by ``len(T)`` so
    each result is ``m_u**2 + sum_i (w_i - c_i,u)**2`` in probability-mass
    units.  Candidate rows are chunked to avoid a global candidate-by-cell
    count matrix.
    """
    ps = len(base_min)
    if ps == 0:
        return np.empty(0, dtype=np.float64)
    cell_ids = np.asarray(cell_ids, dtype=np.intp)
    cell_counts = np.asarray(cell_counts, dtype=np.float64)
    n_cells = len(cell_counts)
    if len(cell_ids) != ps:
        raise ValueError("cell_ids must have one entry per target point")
    if np.any(cell_ids >= n_cells):
        raise ValueError("cell_ids contains an out-of-range compact cell ID")

    if row_chunk is None:
        target_bytes = 192 * 1024 * 1024
        bytes_per_row = max(ps * 10 + n_cells * 16, 1)
        row_chunk = max(1, min(ps, target_bytes // bytes_per_row))
    elif row_chunk <= 0:
        raise ValueError("row_chunk must be positive")

    valid_targets = cell_ids >= 0
    valid_cell_ids = cell_ids[valid_targets]
    penalties = np.empty(ps, dtype=np.float64)
    denom_sq = float(ps) ** 2

    for start in range(0, ps, row_chunk):
        end = min(start + row_chunk, ps)
        n_rows = end - start
        captured = intra_dists[start:end] < base_min[np.newaxis, :]
        captured_totals = captured.sum(axis=1, dtype=np.float64)

        if n_cells > 0:
            captured_valid = captured[:, valid_targets]
            flat_indices = (
                np.arange(n_rows, dtype=np.int64)[:, np.newaxis] * n_cells
                + valid_cell_ids[np.newaxis, :]
            ).reshape(-1)
            captured_by_cell = np.bincount(
                flat_indices,
                weights=captured_valid.reshape(-1),
                minlength=n_rows * n_cells,
            ).reshape(n_rows, n_cells)
            captured_by_cell *= -1.0
            captured_by_cell += cell_counts[np.newaxis, :]
            np.square(captured_by_cell, out=captured_by_cell)
            residual_sq = captured_by_cell.sum(axis=1)
        else:
            residual_sq = np.zeros(n_rows, dtype=np.float64)

        penalties[start:end] = (
            np.square(captured_totals) + residual_sq
        ) / denom_sq

    return penalties


def _update_voronoi_cells_numpy(cell_ids, cell_counts, changed):
    """Move changed target columns into one newly selected candidate cell."""
    changed = np.asarray(changed, dtype=np.intp)
    if len(changed) == 0:
        return cell_counts
    old_ids = cell_ids[changed]
    occupied_old = old_ids >= 0
    if np.any(occupied_old):
        removed = np.bincount(
            old_ids[occupied_old], minlength=len(cell_counts)
        ).astype(np.float64, copy=False)
        cell_counts = np.asarray(cell_counts, dtype=np.float64).copy()
        cell_counts -= removed
    else:
        cell_counts = np.asarray(cell_counts, dtype=np.float64).copy()
    new_cell_id = len(cell_counts)
    cell_ids[changed] = new_cell_id
    return np.concatenate([cell_counts, np.array([float(len(changed))])])


def _wasserstein_l2_initial_capture_counts_numpy(base_min, intra_dists):
    """Legacy-v1 total captured mass, retained for objective diagnostics."""
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


def _compact_voronoi_cells_torch(raw_cell_ids):
    """Torch equivalent of ``_compact_voronoi_cells_numpy``."""
    import torch
    compact_ids = torch.full_like(raw_cell_ids, -1)
    occupied = raw_cell_ids >= 0
    if not torch.any(occupied).item():
        return compact_ids, torch.empty(
            0, dtype=torch.float32, device=raw_cell_ids.device
        )
    _, inverse, counts = torch.unique(
        raw_cell_ids[occupied], sorted=True, return_inverse=True, return_counts=True
    )
    compact_ids[occupied] = inverse
    return compact_ids, counts.to(dtype=torch.float32)


def _wasserstein_l2_base_cells_torch(T_t, T_sq, X_labeled):
    """Return nearest-support distances and compact CUDA Voronoi cell state."""
    import torch
    device = T_t.device
    ps = len(T_t)
    base_min = torch.full((ps,), float("inf"), device=device)
    raw_cell_ids = torch.full((ps,), -1, dtype=torch.long, device=device)
    if X_labeled is not None and len(X_labeled) > 0:
        X_l = torch.as_tensor(
            X_labeled, dtype=torch.float32, device=device
        ).contiguous()
        X_l_sq = (X_l ** 2).sum(dim=1)
        chunk_size = 10000
        for start in range(0, len(X_l), chunk_size):
            end = min(start + chunk_size, len(X_l))
            dists = T_sq.unsqueeze(1) + X_l_sq[start:end].unsqueeze(0)
            dists.addmm_(T_t, X_l[start:end].T, beta=1.0, alpha=-2.0)
            dists.clamp_(min=0.0).sqrt_()
            local_min, local_argmin = dists.min(dim=1)
            improved = local_min < base_min
            base_min[improved] = local_min[improved]
            raw_cell_ids[improved] = start + local_argmin[improved]
            del dists
        del X_l, X_l_sq
        torch.cuda.empty_cache()
    cell_ids, cell_counts = _compact_voronoi_cells_torch(raw_cell_ids)
    return base_min, cell_ids, cell_counts


def _wasserstein_l2_full_penalties_torch(
    base_min, intra_dists, cell_ids, cell_counts, row_chunk=None
):
    """CUDA full updated Voronoi mass-squared penalties, without lambda."""
    import torch
    ps = len(base_min)
    if ps == 0:
        return torch.empty(0, dtype=intra_dists.dtype, device=intra_dists.device)
    n_cells = len(cell_counts)
    if len(cell_ids) != ps:
        raise ValueError("cell_ids must have one entry per target point")
    if n_cells > 0 and torch.any(cell_ids >= n_cells).item():
        raise ValueError("cell_ids contains an out-of-range compact cell ID")

    if row_chunk is None:
        try:
            free_bytes, _ = torch.cuda.mem_get_info(intra_dists.device)
            target_bytes = int(free_bytes * 0.12)
        except Exception:
            target_bytes = 256 * 1024 * 1024
        bytes_per_row = max(ps * 5 + n_cells * 8, 1)
        row_chunk = max(1, min(ps, target_bytes // bytes_per_row))
    elif row_chunk <= 0:
        raise ValueError("row_chunk must be positive")

    valid_targets = cell_ids >= 0
    valid_cell_ids = cell_ids[valid_targets]
    penalties = torch.empty(ps, dtype=intra_dists.dtype, device=intra_dists.device)
    denom_sq = float(ps) ** 2

    for start in range(0, ps, row_chunk):
        end = min(start + row_chunk, ps)
        n_rows = end - start
        captured = intra_dists[start:end] < base_min.unsqueeze(0)
        captured_totals = captured.sum(dim=1).to(intra_dists.dtype)

        if n_cells > 0:
            captured_by_cell = torch.zeros(
                (n_rows, n_cells),
                dtype=intra_dists.dtype,
                device=intra_dists.device,
            )
            captured_by_cell.scatter_add_(
                1,
                valid_cell_ids.unsqueeze(0).expand(n_rows, -1),
                captured[:, valid_targets].to(intra_dists.dtype),
            )
            captured_by_cell.neg_().add_(cell_counts.unsqueeze(0)).square_()
            residual_sq = captured_by_cell.sum(dim=1)
        else:
            residual_sq = torch.zeros(
                n_rows, dtype=intra_dists.dtype, device=intra_dists.device
            )

        penalties[start:end] = (
            torch.square(captured_totals) + residual_sq
        ) / denom_sq

    return penalties


def _update_voronoi_cells_torch(cell_ids, cell_counts, changed):
    """CUDA equivalent of ``_update_voronoi_cells_numpy``."""
    import torch
    if changed.numel() == 0:
        return cell_counts
    old_ids = cell_ids[changed]
    occupied_old = old_ids >= 0
    updated_counts = cell_counts.clone()
    if torch.any(occupied_old).item():
        removed = torch.bincount(
            old_ids[occupied_old], minlength=len(cell_counts)
        ).to(cell_counts.dtype)
        updated_counts -= removed
    new_cell_id = len(updated_counts)
    cell_ids[changed] = new_cell_id
    return torch.cat([
        updated_counts,
        torch.tensor(
            [float(changed.numel())],
            dtype=cell_counts.dtype,
            device=cell_counts.device,
        ),
    ])


def _wasserstein_l2_initial_capture_counts_torch(base_min, intra_dists):
    """Legacy-v1 total captured mass, retained for objective diagnostics."""
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


def _wasserstein_l2_v2_coupling_numpy(T, X_labeled, n_pick, reweight_lambda):
    """Historical CPU full-Voronoi-L2 greedy coupling (version 2)."""
    ps = len(T)
    lam = float(reweight_lambda)

    intra_dists = cdist(T, T, metric='euclidean').astype(np.float32)
    base_min, cell_ids, cell_counts = _wasserstein_l2_base_cells_numpy(
        T, X_labeled
    )

    print(f"  [CPU] Full-Voronoi-L2 coupling v2: pool_size={ps}, "
          f"lambda={lam:g}, selecting {n_pick}, "
          f"occupied_cells={len(cell_counts)}")

    wwds = _wasserstein_initial_wwds_numpy(base_min, intra_dists)

    chosen = []
    available = np.ones(ps, dtype=bool)

    for step in range(n_pick):
        score_t0 = time.perf_counter()
        penalties = _wasserstein_l2_full_penalties_numpy(
            base_min, intra_dists, cell_ids, cell_counts
        )
        scores = wwds.astype(np.float64, copy=False) + lam * penalties
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
            cell_counts = _update_voronoi_cells_numpy(
                cell_ids, cell_counts, changed
            )
            _wasserstein_delta_update_numpy(
                wwds, intra_dists, old_base, base_min, changed
            )

        print(
            f"    [CPU] Full-Voronoi-L2 step {step + 1}/{n_pick}: "
            f"best={best}, score={scores[best]:.7g}, "
            f"transport={float(wwds[best]):.7g}, "
            f"mass_penalty={penalties[best]:.7g}, "
            f"captured={len(changed)}, occupied_cells={len(cell_counts)}, "
            f"score_time={time.perf_counter() - score_t0:.3f}s"
        )

    return chosen


def _wasserstein_l2_v2_coupling_torch(T, X_labeled, n_pick, reweight_lambda):
    """Historical CUDA full-Voronoi-L2 greedy coupling (version 2)."""
    import torch
    device = torch.device('cuda')
    ps = len(T)
    lam = float(reweight_lambda)

    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    T_sq = (T_t ** 2).sum(dim=1)

    intra_dists = T_sq.unsqueeze(1) + T_sq.unsqueeze(0)
    intra_dists.addmm_(T_t, T_t.T, beta=1.0, alpha=-2.0)
    intra_dists.clamp_(min=0.0).sqrt_()

    base_min, cell_ids, cell_counts = _wasserstein_l2_base_cells_torch(
        T_t, T_sq, X_labeled
    )

    print(f"  [GPU] Full-Voronoi-L2 coupling v2: pool_size={ps}, "
          f"lambda={lam:g}, selecting {n_pick}, "
          f"occupied_cells={len(cell_counts)}")

    wwds = _wasserstein_initial_wwds_torch(base_min, intra_dists)

    chosen = []
    available = torch.ones(ps, dtype=torch.bool, device=device)

    for step in range(n_pick):
        score_t0 = time.perf_counter()
        penalties = _wasserstein_l2_full_penalties_torch(
            base_min, intra_dists, cell_ids, cell_counts
        )
        scores = wwds + lam * penalties
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
            cell_counts = _update_voronoi_cells_torch(
                cell_ids, cell_counts, changed
            )
            _wasserstein_delta_update_torch(
                wwds, intra_dists, old_base, base_min, changed
            )

        print(
            f"    [GPU] Full-Voronoi-L2 step {step + 1}/{n_pick}: "
            f"best={best}, score={scores[best].item():.7g}, "
            f"transport={wwds[best].item():.7g}, "
            f"mass_penalty={penalties[best].item():.7g}, "
            f"captured={changed.numel()}, occupied_cells={len(cell_counts)}, "
            f"score_time={time.perf_counter() - score_t0:.3f}s"
        )

    del T_t, intra_dists, wwds, penalties, cell_ids, cell_counts
    torch.cuda.empty_cache()
    return chosen


def _restricted_coordinate_numpy(residuals, reweight_lambda, steps=32):
    """Solve independent one-coordinate insertion problems by bisection.

    ``residuals[c, t]`` is ``b_t - d(t, u_c)``.  Strict ``residual > z``
    implements old-cell wins on adjusted-cost ties.
    """
    residuals = np.asarray(residuals)
    if residuals.ndim == 1:
        residuals = residuals[np.newaxis, :]
    if residuals.ndim != 2 or residuals.shape[1] == 0:
        raise ValueError("residuals must have shape (candidates, nonempty_target)")
    lam = float(reweight_lambda)
    if lam <= 0 or int(steps) <= 0:
        raise ValueError("reweight_lambda and steps must be positive")

    n_target = residuals.shape[1]
    lo = np.zeros(residuals.shape[0], dtype=np.float64)
    hi = np.minimum(
        2.0 * lam,
        np.maximum(residuals.max(axis=1).astype(np.float64), 0.0),
    )
    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        mass = np.count_nonzero(
            residuals > mid.astype(residuals.dtype)[:, np.newaxis], axis=1
        ).astype(np.float64) / float(n_target)
        derivative = mid / (2.0 * lam) - mass
        move_lo = derivative < 0.0
        lo[move_lo] = mid[move_lo]
        hi[~move_lo] = mid[~move_lo]

    def objective(z):
        hinge = np.maximum(
            residuals - z.astype(residuals.dtype)[:, np.newaxis], 0.0
        ).sum(axis=1, dtype=np.float64) / float(n_target)
        return np.square(z) / (4.0 * lam) + hinge

    objective_lo = objective(lo)
    objective_hi = objective(hi)
    use_hi = objective_hi < objective_lo
    z = np.where(use_hi, hi, lo)
    value = np.where(use_hi, objective_hi, objective_lo)
    return z, value


def _restricted_coordinate_torch(residuals, reweight_lambda, steps=32):
    """CUDA counterpart of :func:`_restricted_coordinate_numpy`."""
    import torch

    if residuals.ndim == 1:
        residuals = residuals.unsqueeze(0)
    if residuals.ndim != 2 or residuals.shape[1] == 0:
        raise ValueError("residuals must have shape (candidates, nonempty_target)")
    lam = float(reweight_lambda)
    if lam <= 0 or int(steps) <= 0:
        raise ValueError("reweight_lambda and steps must be positive")

    n_target = residuals.shape[1]
    work_dtype = residuals.dtype
    lo = torch.zeros(
        residuals.shape[0], dtype=work_dtype, device=residuals.device
    )
    hi = torch.minimum(
        torch.full_like(lo, 2.0 * lam),
        torch.clamp(residuals.max(dim=1).values, min=0.0),
    )
    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        mass = torch.sum(
            residuals > mid.unsqueeze(1), dim=1, dtype=torch.float64
        ) / float(n_target)
        derivative = mid.to(torch.float64) / (2.0 * lam) - mass
        move_lo = derivative < 0.0
        lo = torch.where(move_lo, mid, lo)
        hi = torch.where(move_lo, hi, mid)

    def objective(z):
        hinge = torch.clamp(
            residuals - z.unsqueeze(1), min=0.0
        ).sum(dim=1, dtype=torch.float64) / float(n_target)
        return z.to(torch.float64).square() / (4.0 * lam) + hinge

    objective_lo = objective(lo)
    objective_hi = objective(hi)
    use_hi = objective_hi < objective_lo
    z = torch.where(use_hi, hi, lo).to(torch.float64)
    value = torch.where(use_hi, objective_hi, objective_lo)
    return z, value


def _power_block_metrics_numpy(base_cost, selected_dists, z, reweight_lambda):
    """Return restricted block objective, adjusted base, and gradient norm."""
    base_cost = np.asarray(base_cost, dtype=np.float64)
    selected_dists = np.asarray(selected_dists, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    adjusted = selected_dists + z[:, np.newaxis]
    new_argmin = adjusted.argmin(axis=0)
    new_min = adjusted[new_argmin, np.arange(adjusted.shape[1])]
    captured = new_min < base_cost
    current_base = np.minimum(base_cost, new_min)
    objective = (
        np.dot(z, z) / (4.0 * float(reweight_lambda))
        + np.maximum(base_cost - new_min, 0.0).mean(dtype=np.float64)
    )
    counts = np.bincount(
        new_argmin[captured], minlength=len(z)
    ).astype(np.float64)
    gradient = z / (2.0 * float(reweight_lambda)) - counts / len(base_cost)
    return (
        float(objective),
        current_base.astype(np.float32, copy=False),
        float(np.max(np.abs(gradient), initial=0.0)),
    )


def _power_block_metrics_torch(base_cost, selected_dists, z, reweight_lambda):
    """CUDA restricted block objective, adjusted base, and gradient norm."""
    import torch

    base64 = base_cost.to(torch.float64)
    dists64 = selected_dists.to(torch.float64)
    z64 = z.to(torch.float64)
    adjusted = dists64 + z64.unsqueeze(1)
    new_min, new_argmin = adjusted.min(dim=0)
    captured = new_min < base64
    current_base = torch.minimum(base64, new_min)
    objective = (
        torch.dot(z64, z64) / (4.0 * float(reweight_lambda))
        + torch.clamp(base64 - new_min, min=0.0).mean()
    )
    counts = torch.zeros(len(z64), dtype=torch.float64, device=z64.device)
    if torch.any(captured):
        counts.scatter_add_(
            0,
            new_argmin[captured],
            torch.ones_like(new_argmin[captured], dtype=torch.float64),
        )
    gradient = z64 / (2.0 * float(reweight_lambda)) - counts / len(base64)
    return (
        float(objective.item()),
        current_base.to(torch.float32),
        float(torch.max(torch.abs(gradient)).item()),
    )


def _correct_power_block_numpy(
    base_cost,
    selected_dists,
    z_init,
    reweight_lambda,
    *,
    coordinate_steps=32,
    max_sweeps=128,
    dual_relative_tol=1e-8,
    z_relative_tol=1e-6,
    patience=2,
):
    """Cyclically correct only the query-batch coordinates on CPU."""
    base64 = np.asarray(base_cost, dtype=np.float64)
    dists64 = np.asarray(selected_dists, dtype=np.float64)
    z = np.asarray(z_init, dtype=np.float64).copy()
    if dists64.ndim != 2 or dists64.shape != (len(z), len(base64)):
        raise ValueError("selected_dists, z_init, and base_cost shapes do not align")

    objective, current_base, grad_inf = _power_block_metrics_numpy(
        base64, dists64, z, reweight_lambda
    )
    if not np.isfinite(objective):
        raise FloatingPointError("non-finite initial block objective")
    best_z = z.copy()
    best_objective = objective
    history = [float(objective)]
    stable_streak = 0
    last_relative_improvement = None
    last_z_relative_change = None
    stop_reason = "max_sweeps_not_converged"

    for sweep in range(1, int(max_sweeps) + 1):
        previous_z = z.copy()
        previous_objective = objective
        for coordinate in range(len(z)):
            adjusted = dists64 + z[:, np.newaxis]
            adjusted[coordinate] = np.inf
            baseline_without = np.minimum(base64, adjusted.min(axis=0))
            residual = baseline_without - dists64[coordinate]
            coordinate_z, _ = _restricted_coordinate_numpy(
                residual, reweight_lambda, steps=coordinate_steps
            )
            z[coordinate] = coordinate_z[0]

        objective, current_base, grad_inf = _power_block_metrics_numpy(
            base64, dists64, z, reweight_lambda
        )
        if not np.isfinite(objective) or not np.all(np.isfinite(z)):
            raise FloatingPointError("non-finite block-correction iterate")
        allowed_increase = 1e-9 * max(1.0, abs(previous_objective))
        if objective > previous_objective + allowed_increase:
            raise FloatingPointError(
                "block correction objective increased beyond numerical tolerance"
            )
        if objective > previous_objective:
            z = previous_z
            objective, current_base, grad_inf = _power_block_metrics_numpy(
                base64, dists64, z, reweight_lambda
            )
        if objective < best_objective:
            best_objective = objective
            best_z = z.copy()

        improvement = previous_objective - objective
        scale = max(abs(previous_objective), abs(objective), 1e-12)
        last_relative_improvement = max(improvement, 0.0) / scale
        z_scale = max(
            1.0,
            float(np.max(np.abs(previous_z), initial=0.0)),
            float(np.max(np.abs(z), initial=0.0)),
        )
        last_z_relative_change = float(
            np.max(np.abs(z - previous_z), initial=0.0) / z_scale
        )
        history.append(float(objective))
        stable = (
            improvement >= -allowed_increase
            and last_relative_improvement <= float(dual_relative_tol)
            and last_z_relative_change <= float(z_relative_tol)
        )
        stable_streak = stable_streak + 1 if stable else 0
        if stable_streak >= int(patience):
            stop_reason = "corrective_stability"
            break

    sweeps = len(history) - 1
    if stop_reason == "max_sweeps_not_converged":
        z = best_z
        objective, current_base, grad_inf = _power_block_metrics_numpy(
            base64, dists64, z, reweight_lambda
        )
    return z, current_base, {
        "sweeps": int(sweeps),
        "stop_reason": stop_reason,
        "converged": bool(stop_reason == "corrective_stability"),
        "block_restricted_dual_drop": float(objective),
        "relative_improvement": (
            None if last_relative_improvement is None
            else float(last_relative_improvement)
        ),
        "z_relative_change": (
            None if last_z_relative_change is None
            else float(last_z_relative_change)
        ),
        "grad_inf": float(grad_inf),
        "objective_history": history,
    }


def _correct_power_block_torch(
    base_cost,
    selected_dists,
    z_init,
    reweight_lambda,
    *,
    coordinate_steps=32,
    max_sweeps=128,
    dual_relative_tol=1e-8,
    z_relative_tol=1e-6,
    patience=2,
):
    """CUDA cyclic correction of query-batch coordinates."""
    import torch

    base64 = base_cost.to(torch.float64)
    dists64 = selected_dists.to(torch.float64)
    z = z_init.to(torch.float64).clone()
    if dists64.ndim != 2 or tuple(dists64.shape) != (len(z), len(base64)):
        raise ValueError("selected_dists, z_init, and base_cost shapes do not align")

    objective, current_base, grad_inf = _power_block_metrics_torch(
        base64, dists64, z, reweight_lambda
    )
    if not np.isfinite(objective):
        raise FloatingPointError("non-finite initial block objective")
    best_z = z.clone()
    best_objective = objective
    history = [float(objective)]
    stable_streak = 0
    last_relative_improvement = None
    last_z_relative_change = None
    stop_reason = "max_sweeps_not_converged"

    for sweep in range(1, int(max_sweeps) + 1):
        previous_z = z.clone()
        previous_objective = objective
        for coordinate in range(len(z)):
            adjusted = dists64 + z.unsqueeze(1)
            adjusted[coordinate] = float("inf")
            baseline_without = torch.minimum(base64, adjusted.min(dim=0).values)
            residual = baseline_without - dists64[coordinate]
            coordinate_z, _ = _restricted_coordinate_torch(
                residual, reweight_lambda, steps=coordinate_steps
            )
            z[coordinate] = coordinate_z[0]

        objective, current_base, grad_inf = _power_block_metrics_torch(
            base64, dists64, z, reweight_lambda
        )
        if not np.isfinite(objective) or not torch.all(torch.isfinite(z)).item():
            raise FloatingPointError("non-finite block-correction iterate")
        allowed_increase = 1e-9 * max(1.0, abs(previous_objective))
        if objective > previous_objective + allowed_increase:
            raise FloatingPointError(
                "block correction objective increased beyond numerical tolerance"
            )
        if objective > previous_objective:
            z = previous_z
            objective, current_base, grad_inf = _power_block_metrics_torch(
                base64, dists64, z, reweight_lambda
            )
        if objective < best_objective:
            best_objective = objective
            best_z = z.clone()

        improvement = previous_objective - objective
        scale = max(abs(previous_objective), abs(objective), 1e-12)
        last_relative_improvement = max(improvement, 0.0) / scale
        z_scale = max(
            1.0,
            float(torch.max(torch.abs(previous_z)).item()),
            float(torch.max(torch.abs(z)).item()),
        )
        last_z_relative_change = float(
            torch.max(torch.abs(z - previous_z)).item() / z_scale
        )
        history.append(float(objective))
        stable = (
            improvement >= -allowed_increase
            and last_relative_improvement <= float(dual_relative_tol)
            and last_z_relative_change <= float(z_relative_tol)
        )
        stable_streak = stable_streak + 1 if stable else 0
        if stable_streak >= int(patience):
            stop_reason = "corrective_stability"
            break

    sweeps = len(history) - 1
    if stop_reason == "max_sweeps_not_converged":
        z = best_z
        objective, current_base, grad_inf = _power_block_metrics_torch(
            base64, dists64, z, reweight_lambda
        )
    return z, current_base, {
        "sweeps": int(sweeps),
        "stop_reason": stop_reason,
        "converged": bool(stop_reason == "corrective_stability"),
        "block_restricted_dual_drop": float(objective),
        "relative_improvement": (
            None if last_relative_improvement is None
            else float(last_relative_improvement)
        ),
        "z_relative_change": (
            None if last_z_relative_change is None
            else float(last_z_relative_change)
        ),
        "grad_inf": float(grad_inf),
        "objective_history": history,
    }


def _wasserstein_l2_coupling_numpy(
    candidates,
    target,
    base_cost,
    n_pick,
    reweight_lambda,
    *,
    coordinate_steps=32,
    corrective_max_sweeps=128,
    corrective_dual_relative_tol=1e-8,
    corrective_z_relative_tol=1e-6,
    corrective_patience=2,
    candidate_chunk_size=None,
):
    """CPU power-cell restricted-dual greedy coupling (version 3)."""
    candidates = np.asarray(candidates, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    old_base = np.asarray(base_cost, dtype=np.float32)
    if candidates.ndim != 2 or target.ndim != 2:
        raise ValueError("candidates and target must be two-dimensional")
    if candidates.shape[1] != target.shape[1] or len(old_base) != len(target):
        raise ValueError("candidate, target, and power-base shapes do not align")
    if candidate_chunk_size is None:
        target_bytes = 256 * 1024 * 1024
        candidate_chunk_size = max(
            1,
            min(
                len(candidates),
                target_bytes // max(len(target) * 4 * 3, 1),
            ),
        )
    if candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")

    print(
        f"  [CPU] Power-cell Wasserstein-L2 v3: candidates={len(candidates)}, "
        f"target={len(target)}, lambda={float(reweight_lambda):g}, "
        f"selecting={n_pick}, chunk={candidate_chunk_size}"
    )
    chosen = []
    available = np.ones(len(candidates), dtype=bool)
    selected_dists = np.empty((0, len(target)), dtype=np.float32)
    z_block = np.empty(0, dtype=np.float64)
    current_base = old_base.copy()
    steps_trace = []

    for step_index in range(min(int(n_pick), len(candidates))):
        score_t0 = time.perf_counter()
        drops = np.full(len(candidates), -np.inf, dtype=np.float64)
        candidate_z = np.zeros(len(candidates), dtype=np.float64)
        for start in range(0, len(candidates), candidate_chunk_size):
            end = min(start + candidate_chunk_size, len(candidates))
            dists = cdist(
                candidates[start:end], target, metric="euclidean"
            ).astype(np.float32)
            residuals = current_base[np.newaxis, :] - dists
            del dists
            z_chunk, drop_chunk = _restricted_coordinate_numpy(
                residuals, reweight_lambda, steps=coordinate_steps
            )
            candidate_z[start:end] = z_chunk
            drops[start:end] = drop_chunk
        drops[~available] = -np.inf
        best = int(np.argmax(drops))
        if not np.isfinite(drops[best]):
            break
        score_seconds = time.perf_counter() - score_t0

        best_dist = cdist(
            candidates[best:best + 1], target, metric="euclidean"
        ).astype(np.float32)[0]
        chosen.append(best)
        available[best] = False
        selected_dists = np.vstack([selected_dists, best_dist])
        z_block = np.append(z_block, candidate_z[best])

        corrective_t0 = time.perf_counter()
        z_block, current_base, correction = _correct_power_block_numpy(
            old_base,
            selected_dists,
            z_block,
            reweight_lambda,
            coordinate_steps=coordinate_steps,
            max_sweeps=corrective_max_sweeps,
            dual_relative_tol=corrective_dual_relative_tol,
            z_relative_tol=corrective_z_relative_tol,
            patience=corrective_patience,
        )
        corrective_seconds = time.perf_counter() - corrective_t0
        step_trace = {
            "step": int(step_index + 1),
            "candidate_subpool_index": best,
            "insertion_z": float(candidate_z[best]),
            "selected_corrected_z": float(z_block[-1]),
            "corrected_block_z": [float(value) for value in z_block],
            "restricted_dual_drop": float(drops[best]),
            "block_restricted_dual_drop": correction[
                "block_restricted_dual_drop"
            ],
            "corrective_sweeps": correction["sweeps"],
            "corrective_stop_reason": correction["stop_reason"],
            "corrective_converged": correction["converged"],
            "corrective_relative_improvement": correction[
                "relative_improvement"
            ],
            "corrective_z_relative_change": correction["z_relative_change"],
            "corrective_grad_inf": correction["grad_inf"],
            "corrective_objective_history": correction["objective_history"],
            "score_seconds": float(score_seconds),
            "corrective_seconds": float(corrective_seconds),
        }
        steps_trace.append(step_trace)
        print(
            f"    [CPU] v3 step {step_index + 1}/{n_pick}: best={best}, "
            f"restricted_drop={drops[best]:.9e}, z={candidate_z[best]:.7g}, "
            f"block_drop={correction['block_restricted_dual_drop']:.9e}, "
            f"correction={correction['stop_reason']}/"
            f"{correction['sweeps']}, grad_inf={correction['grad_inf']:.3e}, "
            f"score_time={score_seconds:.3f}s, "
            f"correct_time={corrective_seconds:.3f}s"
        )

    return chosen, {
        "backend": "numpy_cpu",
        "coordinate_steps": int(coordinate_steps),
        "corrective_max_sweeps": int(corrective_max_sweeps),
        "corrective_dual_relative_tol": float(corrective_dual_relative_tol),
        "corrective_z_relative_tol": float(corrective_z_relative_tol),
        "corrective_patience": int(corrective_patience),
        "candidate_chunk_size": int(candidate_chunk_size),
        "steps": steps_trace,
    }


def _wasserstein_l2_coupling_torch(
    candidates,
    target,
    base_cost,
    n_pick,
    reweight_lambda,
    *,
    coordinate_steps=32,
    corrective_max_sweeps=128,
    corrective_dual_relative_tol=1e-8,
    corrective_z_relative_tol=1e-6,
    corrective_patience=2,
    candidate_chunk_size=None,
):
    """CUDA power-cell restricted-dual greedy coupling (version 3)."""
    import torch

    device = torch.device("cuda")
    candidates_t = torch.as_tensor(
        candidates, dtype=torch.float32, device=device
    ).contiguous()
    target_t = torch.as_tensor(
        target, dtype=torch.float32, device=device
    ).contiguous()
    old_base = torch.as_tensor(
        base_cost, dtype=torch.float32, device=device
    ).contiguous()
    if candidates_t.ndim != 2 or target_t.ndim != 2:
        raise ValueError("candidates and target must be two-dimensional")
    if candidates_t.shape[1] != target_t.shape[1] or len(old_base) != len(target_t):
        raise ValueError("candidate, target, and power-base shapes do not align")

    if candidate_chunk_size is None:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        budget = min(int(free_bytes * 0.20), 1536 * 1024 * 1024)
        candidate_chunk_size = max(
            1,
            min(
                len(candidates_t),
                budget // max(len(target_t) * 4 * 3, 1),
            ),
        )
    if candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")

    candidate_sq = torch.sum(candidates_t ** 2, dim=1)
    target_sq = torch.sum(target_t ** 2, dim=1)
    print(
        f"  [GPU] Power-cell Wasserstein-L2 v3: candidates={len(candidates_t)}, "
        f"target={len(target_t)}, lambda={float(reweight_lambda):g}, "
        f"selecting={n_pick}, chunk={candidate_chunk_size}"
    )
    torch.cuda.reset_peak_memory_stats(device)
    chosen = []
    available = np.ones(len(candidates_t), dtype=bool)
    selected_dists = torch.empty(
        (0, len(target_t)), dtype=torch.float32, device=device
    )
    z_block = torch.empty(0, dtype=torch.float64, device=device)
    current_base = old_base.clone()
    steps_trace = []

    for step_index in range(min(int(n_pick), len(candidates_t))):
        score_t0 = time.perf_counter()
        drops = np.full(len(candidates_t), -np.inf, dtype=np.float64)
        candidate_z = np.zeros(len(candidates_t), dtype=np.float64)
        for start in range(0, len(candidates_t), candidate_chunk_size):
            end = min(start + candidate_chunk_size, len(candidates_t))
            chunk = candidates_t[start:end]
            dists = candidate_sq[start:end].unsqueeze(1) + target_sq.unsqueeze(0)
            dists.addmm_(chunk, target_t.T, beta=1.0, alpha=-2.0)
            dists.clamp_(min=0.0).sqrt_()
            residuals = current_base.unsqueeze(0) - dists
            del dists
            z_chunk, drop_chunk = _restricted_coordinate_torch(
                residuals, reweight_lambda, steps=coordinate_steps
            )
            candidate_z[start:end] = z_chunk.cpu().numpy()
            drops[start:end] = drop_chunk.cpu().numpy()
            del residuals, z_chunk, drop_chunk
        drops[~available] = -np.inf
        best = int(np.argmax(drops))
        if not np.isfinite(drops[best]):
            break
        score_seconds = time.perf_counter() - score_t0

        best_dist = candidate_sq[best] + target_sq - 2.0 * torch.mv(
            target_t, candidates_t[best]
        )
        best_dist.clamp_(min=0.0).sqrt_()
        chosen.append(best)
        available[best] = False
        selected_dists = torch.cat([selected_dists, best_dist.unsqueeze(0)], dim=0)
        z_block = torch.cat([
            z_block,
            torch.as_tensor(
                [candidate_z[best]], dtype=torch.float64, device=device
            ),
        ])

        corrective_t0 = time.perf_counter()
        z_block, current_base, correction = _correct_power_block_torch(
            old_base,
            selected_dists,
            z_block,
            reweight_lambda,
            coordinate_steps=coordinate_steps,
            max_sweeps=corrective_max_sweeps,
            dual_relative_tol=corrective_dual_relative_tol,
            z_relative_tol=corrective_z_relative_tol,
            patience=corrective_patience,
        )
        corrective_seconds = time.perf_counter() - corrective_t0
        step_trace = {
            "step": int(step_index + 1),
            "candidate_subpool_index": best,
            "insertion_z": float(candidate_z[best]),
            "selected_corrected_z": float(z_block[-1].item()),
            "corrected_block_z": [
                float(value) for value in z_block.detach().cpu().tolist()
            ],
            "restricted_dual_drop": float(drops[best]),
            "block_restricted_dual_drop": correction[
                "block_restricted_dual_drop"
            ],
            "corrective_sweeps": correction["sweeps"],
            "corrective_stop_reason": correction["stop_reason"],
            "corrective_converged": correction["converged"],
            "corrective_relative_improvement": correction[
                "relative_improvement"
            ],
            "corrective_z_relative_change": correction["z_relative_change"],
            "corrective_grad_inf": correction["grad_inf"],
            "corrective_objective_history": correction["objective_history"],
            "score_seconds": float(score_seconds),
            "corrective_seconds": float(corrective_seconds),
        }
        steps_trace.append(step_trace)
        print(
            f"    [GPU] v3 step {step_index + 1}/{n_pick}: best={best}, "
            f"restricted_drop={drops[best]:.9e}, z={candidate_z[best]:.7g}, "
            f"block_drop={correction['block_restricted_dual_drop']:.9e}, "
            f"correction={correction['stop_reason']}/"
            f"{correction['sweeps']}, grad_inf={correction['grad_inf']:.3e}, "
            f"score_time={score_seconds:.3f}s, "
            f"correct_time={corrective_seconds:.3f}s"
        )

    peak_memory = int(torch.cuda.max_memory_allocated(device))
    del candidates_t, target_t, old_base, candidate_sq, target_sq
    del selected_dists, z_block, current_base
    torch.cuda.empty_cache()
    return chosen, {
        "backend": "torch_cuda",
        "coordinate_steps": int(coordinate_steps),
        "corrective_max_sweeps": int(corrective_max_sweeps),
        "corrective_dual_relative_tol": float(corrective_dual_relative_tol),
        "corrective_z_relative_tol": float(corrective_z_relative_tol),
        "corrective_patience": int(corrective_patience),
        "candidate_chunk_size": int(candidate_chunk_size),
        "peak_cuda_memory_bytes": peak_memory,
        "steps": steps_trace,
    }


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

def query_kmedianpp(X_pool, clf, n, rng, *, X_labeled=None, state=None,
                    pool_size=None, **kw):
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

    When ``pool_size`` is smaller than the available pool, each query batch
    draws a fresh candidate subpool without replacement. Distances are then
    initialized from the full labeled support, so no cache from a different
    candidate subpool is reused.
    """
    if state is None:
        state = {}

    n_pool = len(X_pool)
    n_pick = min(int(n), n_pool)
    if n_pick <= 0:
        return np.empty(0, dtype=np.intp)
    if pool_size is None:
        effective_ps = n_pool
    else:
        effective_ps = min(n_pool, max(n_pick, int(pool_size)))

    if effective_ps < n_pool:
        subpool_idx = rng.choice(n_pool, effective_ps, replace=False)
        X_candidates = X_pool[subpool_idx]
        candidate_state = {}
        state.clear()
        print(f"  [k-Median++] Candidate subpool: {effective_ps} / {n_pool}")
    else:
        subpool_idx = None
        X_candidates = X_pool
        candidate_state = state

    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch and torch.cuda.is_available():
        chosen = _query_kmedianpp_torch(
            X_candidates, n_pick, rng, X_labeled, candidate_state
        )
    else:
        chosen = _query_kmedianpp_numpy(
            X_candidates, n_pick, rng, X_labeled, candidate_state
        )
    if subpool_idx is None:
        return chosen
    return subpool_idx[np.asarray(chosen, dtype=np.intp)]


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
