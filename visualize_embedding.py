import h5py, os, re, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try: import cuml; HAS_CUDA = True
except ImportError: HAS_CUDA = False

def numerical_sort_key(s):
    """Sort strings with embedded numbers numerically."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def visualize_large_h5(
    file_path, 
    n_samples=20000, 
    method='umap',       # 'umap', 'tsne', or 'pca'
    random_state=42,
    feh_classify=False,  # If True: binary red/blue; If False: continuous heatmap
    feh_threshold=None,  # If None: random sampling; If value: balanced class sampling
    red_blue_ratio=2.0,  # Ratio of red to blue samples
    eval_weights=None,   # Path to linear classifier weights CSV
):
    """
    Sample a subset of data from a large normalized H5 file (columnar or 2D) and visualize.
    Also loads the 'feh' column for the sampled indices.
    - feh_classify=False  → continuous [Fe/H] colorbar heatmap (default)
    - feh_classify=True   → binary scatter: feh > feh_threshold = red, feh ≤ feh_threshold = blue
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    feh_values = None
    y_pred = None

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Inspecting file: {file_path}")
            
            # 1. Detect Data Structure
            keys = list(f.keys())
            bp_cols = sorted((k for k in keys if k.startswith('bp_')), key=numerical_sort_key)
            rp_cols = sorted((k for k in keys if k.startswith('rp_')), key=numerical_sort_key)
            feature_cols = bp_cols + rp_cols
            
            if feature_cols:
                print(f"Found {len(feature_cols)} feature columns (bp_*/rp_*).")
                total_rows = f[feature_cols[0]].shape[0]
                is_columnar = True
            else:
                dataset_name = f.visititems(lambda n, o: n if isinstance(o, h5py.Dataset) and len(o.shape) == 2 else None)
                
                if dataset_name:
                    dset = f[dataset_name]
                    total_rows = dset.shape[0]
                    is_columnar = False
                    print(f"Found 2D dataset: {dataset_name}")
                else:
                    print("Error: Could not find bp/rp columns AND could not find any 2D dataset.")
                    print("Available keys:", keys[:20], "...")
                    return

            print(f"Total rows: {total_rows}")

            # Check for feh column
            has_feh = 'feh' in keys
            if has_feh:
                print("Found 'feh' column.")
            else:
                print("Warning: 'feh' column not found. Skipping feh plot.")

            # 2. Sampling
            np.random.seed(random_state)

            if feh_threshold is not None and has_feh:
                # ---- Balanced class sampling ----
                # Read the full feh column first to determine class membership,
                # then draw k = min(N_blue, n_samples) samples from each class.
                print("feh_classify=True: loading full feh to build balanced sample...")
                full_feh = f['feh'][:].astype(np.float64)

                all_idx = np.arange(total_rows)
                blue_pool = all_idx[full_feh <  feh_threshold]   # feh <  threshold
                red_pool  = all_idx[full_feh >= feh_threshold]   # feh >= threshold
                N_blue = len(blue_pool)
                N_red  = len(red_pool)

                
                # Calculates max possible blue samples satisfying all constraints:
                # 1. k_blue <= N_blue
                # 2. k_blue <= n_samples
                # 3. k_blue * ratio <= N_red
                k_blue = min(N_blue, n_samples, int(N_red / red_blue_ratio))
                k_red = int(k_blue * red_blue_ratio)
                
                print(f"  Blue pool (feh < {feh_threshold}): {N_blue}  |  "
                      f"Red pool (feh >= {feh_threshold}): {N_red}")
                print(f"  Sampling {k_blue} Blue and {k_red} Red (Ratio: {red_blue_ratio})")

                blue_idx = np.random.choice(blue_pool, k_blue, replace=False)
                red_idx  = np.random.choice(red_pool,  k_red, replace=False)
                indices  = np.concatenate([blue_idx, red_idx])
                indices.sort()

                # Keep per-sample class labels aligned with indices order
                # (we'll rebuild from full_feh after sorting)
                feh_values = full_feh[indices]
                del full_feh

            else:
                # ---- Plain random sampling (Triggered if feh_threshold is None) ----
                if total_rows > n_samples:
                    print(f"No threshold provided. Sampling {n_samples} random indices...")
                    indices = np.random.choice(total_rows, n_samples, replace=False)
                    indices.sort()
                else:
                    indices = np.arange(total_rows)
                    print(f"Using all {total_rows} rows.")

            # 3. Load Feature Data
            print("Loading feature data (optimized)...")
            if is_columnar:
                data_list = []
                total_cols = len(feature_cols)

                for i, col_name in enumerate(feature_cols):
                    if i % 10 == 0:
                        print(f"Reading column {i}/{total_cols}: {col_name}...")

                    # OPTIMIZATION:
                    # H5py random access (fancy indexing) is very slow for large files.
                    # Reading the whole column then slicing is much faster.
                    data_list.append(f[col_name][:][indices])

                data_sampled = np.column_stack(data_list)
            else:
                data_sampled = f[dataset_name][:][indices]

            print(f"Data shape for visualization: {data_sampled.shape}")

            if np.isnan(data_sampled).any():
                print("Warning: NaNs found in feature data. Filling with 0.")
                data_sampled = np.nan_to_num(data_sampled)

            # 4. Load feh values (only needed if not already loaded above)
            if has_feh and feh_values is None:
                print("Loading feh values for sampled indices...")
                full_feh = f['feh'][:]
                feh_values = full_feh[indices].astype(np.float64)
                del full_feh

            if feh_values is not None:
                print(f"feh range: [{np.nanmin(feh_values):.3f}, {np.nanmax(feh_values):.3f}]")

            # 5. Evaluate Weights if provided
            if eval_weights is not None:
                import csv
                if not os.path.exists(eval_weights):
                    print(f"Error: Weights file '{eval_weights}' not found. Skipping evaluation.")
                else:
                    print(f"Evaluating linear classifier using '{eval_weights}'...")
                    with open(eval_weights, 'r') as cf:
                        rows = list(csv.DictReader(cf))
                    bias = float(next(r['weight'] for r in rows if r['feature'] == 'BIAS'))
                    wt_dict = {r['feature']: float(r['weight']) for r in rows if r['feature'] != 'BIAS'}
                    
                    logits = np.zeros(len(indices), dtype=np.float32) + bias
                    sum_sq = np.zeros(len(indices), dtype=np.float64)
                    
                    # Compute norm for BP/RP features used by the model
                    for i, col_name in enumerate(feature_cols):
                        if col_name in wt_dict:
                            sum_sq += data_sampled[:, i] ** 2
                    norms = np.sqrt(sum_sq) + 1e-8
                    
                    # Compute logits
                    for i, col_name in enumerate(feature_cols):
                        if col_name in wt_dict:
                            logits += (data_sampled[:, i] / norms) * wt_dict[col_name]
                            
                    # Handle other features like ebv
                    for feat, w in wt_dict.items():
                        if feat not in feature_cols:
                            if feat in f.keys():
                                col_data = np.nan_to_num(f[feat][:][indices])
                                logits += col_data * w
                            else:
                                print(f"Warning: feature '{feat}' needed by weights but not in H5.")
                                
                    y_pred = (logits > 0.0).astype(int)

    except Exception as e:
        print(f"Error processing H5 file: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Dimensionality Reduction
    print(f"Running {method.upper()}...")
    
    embedding = None
    
    if method == 'umap':
        if HAS_CUDA:
            print("Using cuML UMAP (GPU)...")
            embedding = cuml.UMAP(n_neighbors=15, n_components=2, random_state=random_state).fit_transform(data_sampled)
        else:
            try:
                import umap
                embedding = umap.UMAP(n_neighbors=15, n_components=2, random_state=random_state, n_jobs=-1).fit_transform(data_sampled)
            except ImportError:
                print("UMAP not installed. Please run `pip install umap-learn`.")
                return
    elif method == 'tsne':
        if HAS_CUDA:
            print("Using cuML t-SNE (GPU)...")
            embedding = cuml.TSNE(n_components=2, random_state=random_state, method='fft').fit_transform(data_sampled)
        else:
            from sklearn.manifold import TSNE
            embedding = TSNE(n_components=2, random_state=random_state, init='pca', learning_rate='auto').fit_transform(data_sampled)
    elif method == 'pca':
        if HAS_CUDA:
            print("Using cuML PCA (GPU)...")
            pca = cuml.PCA(n_components=2)
            embedding = pca.fit_transform(data_sampled)
            if hasattr(pca, 'explained_variance_ratio_'): print(f"Explained variance: {pca.explained_variance_ratio_}")
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embedding = pca.fit_transform(data_sampled)
            print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    else:
        print(f"Unknown method: {method}")
        return

    # 6. Plotting

    # Create output directory
    output_dir = 'data_visualization'
    os.makedirs(output_dir, exist_ok=True)

    # --- Plot B: feh-colored scatter ---
    if feh_values is not None:
        # Mask out NaN feh entries for coloring
        valid_mask = np.isfinite(feh_values)
        n_valid = valid_mask.sum()
        n_nan   = (~valid_mask).sum()
        if n_nan > 0:
            print(f"  {n_nan} samples have NaN feh — they will be shown in grey.")

        fig, ax = plt.subplots(figsize=(11, 9))

        # Plot NaN-feh points in grey first (background)
        if n_nan > 0:
            ax.scatter(
                embedding[~valid_mask, 0], embedding[~valid_mask, 1],
                s=1, alpha=0.3, c='lightgrey', label='feh = NaN'
            )

        # Decide final plotting mode: Binary mode requires both feh_classify=True AND a valid threshold
        use_binary_plot = feh_classify and (feh_threshold is not None)

        if use_binary_plot:
            # ---- Binary classification mode ----
            above = valid_mask & (feh_values >= feh_threshold)
            below = valid_mask & (feh_values <  feh_threshold)
            n_above = above.sum()
            n_below = below.sum()
            print(f"  feh >= {feh_threshold}: {n_above} (red),  "
                  f"feh < {feh_threshold}: {n_below} (blue)")

            # Plot metal-rich (red) on top so rare metal-poor stand out
            ax.scatter(
                embedding[below, 0], embedding[below, 1],
                s=2, alpha=0.5, c='royalblue',
                label=f'[Fe/H] < {feh_threshold}  (n={n_below})',
            )
            ax.scatter(
                embedding[above, 0], embedding[above, 1],
                s=2, alpha=0.5, c='crimson',
                label=f'[Fe/H] >= {feh_threshold}  (n={n_above})',
            )

            ax.legend(markerscale=4, fontsize=11, loc='best')
            ax.set_title(
                f"{method.upper()} projection of {len(indices)} samples\n"
                f"[Fe/H] classification  (threshold = {feh_threshold})",
                fontsize=13
            )

            feh_img = os.path.join(output_dir, f"{method}_feh_classify.png")

        else:
            # ---- Continuous heatmap mode ----
            vmin = np.nanpercentile(feh_values, 1)
            vmax = np.nanpercentile(feh_values, 99)
            sc = ax.scatter(
                embedding[valid_mask, 0], embedding[valid_mask, 1],
                s=1, alpha=0.6,
                c=feh_values[valid_mask],
                cmap='RdYlBu',   # blue = metal-poor, red = metal-rich
                vmin=vmin, vmax=vmax
            )

            cbar = fig.colorbar(sc, ax=ax, pad=0.01)
            cbar.set_label('[Fe/H]', fontsize=13)

            if n_nan > 0:
                ax.legend(markerscale=5, fontsize=10, loc='best')

            ax.set_title(
                f"{method.upper()} projection of {len(indices)} samples\n"
                f"colored by [Fe/H]  (valid: {n_valid}, NaN: {n_nan})",
                fontsize=13
            )

            feh_img = os.path.join(output_dir, f"{method}_feh_heatmap.png")

        # ---- Evaluation Overlay ----
        if eval_weights is not None and y_pred is not None:
            eval_threshold = feh_threshold if feh_threshold is not None else -2.0
            y_true = (feh_values >= eval_threshold).astype(int)
            
            # MP = 0, MR = 1
            mp_as_mr = valid_mask & (y_true == 0) & (y_pred == 1)
            mr_as_mp = valid_mask & (y_true == 1) & (y_pred == 0)
            
            # Helper to determine marker colors
            # If continuous: use the color from the heatmap for that feh value
            # If binary: use the standard crimson/royalblue
            if not use_binary_plot:
                # Need the same normalization as the main scatter
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.get_cmap('RdYlBu')
                colors_mp_as_mr = cmap(norm(feh_values[mp_as_mr]))
                colors_mr_as_mp = cmap(norm(feh_values[mr_as_mp]))
            else:
                colors_mp_as_mr = 'crimson'
                colors_mr_as_mp = 'royalblue'

            if mp_as_mr.sum() > 0:
                ax.scatter(
                    embedding[mp_as_mr, 0], embedding[mp_as_mr, 1],
                    marker='^', s=30, alpha=0.9, c=colors_mp_as_mr, edgecolors='black', linewidths=0.6,
                    label=f'Err: MP \u2192 MR (n={mp_as_mr.sum()})'
                )
            if mr_as_mp.sum() > 0:
                ax.scatter(
                    embedding[mr_as_mp, 0], embedding[mr_as_mp, 1],
                    marker='^', s=30, alpha=0.9, c=colors_mr_as_mp, edgecolors='black', linewidths=0.6,
                    label=f'Err: MR \u2192 MP (n={mr_as_mp.sum()})'
                )
                
            ax.legend(markerscale=1.5, fontsize=11, loc='best')
            feh_img = feh_img.replace('.png', '_eval.png')

        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)

        fig.savefig(feh_img, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved feh plot to {feh_img}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualize embedding from H5 file.")
    p.add_argument("--file", type=str, default='./bp_rp_lamost_normalized.h5', help="Path to input")
    p.add_argument("--n_samples", type=int, default=20000)
    p.add_argument("--method", type=str, default='umap', choices=['umap', 'tsne', 'pca'])
    p.add_argument("--threshold", type=float, default=None, help="Fe/H threshold for balanced sampling. If None, uses random sampling.")
    p.add_argument("--continuous", action="store_true", help="Use continuous heatmap for plotting")
    p.add_argument("--ratio", type=float, default=2.0, help="Red to Blue ratio for balanced sampling")
    p.add_argument("--eval_weights", nargs='?', const='linear_model_weights_ebv.csv', default=None, 
                   help="Path to weights CSV to evaluate and highlight classification errors (default: linear_model_weights_ebv.csv)")
    args = p.parse_args()
    
    TARGET_FILE = args.file
    
    if os.path.exists(TARGET_FILE):
        # feh_classify: True by default unless --continuous is passed (logic inverted for CLI convenience?)
        # Logic in existing code: feh_classify=True -> binary.
        # So if we want binary by default, we use feh_classify=not args.continuous
        
        visualize_large_h5(
            TARGET_FILE,
            n_samples=args.n_samples,
            method=args.method,
            feh_classify=not args.continuous,
            feh_threshold=args.threshold,
            red_blue_ratio=args.ratio,
            eval_weights=args.eval_weights,
        )
    else:
        print(f"File {TARGET_FILE} not found.")
