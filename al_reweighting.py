"""Covariate-shift and support-weighting methods for active learning."""

import numpy as np
from scipy.spatial.distance import cdist


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
