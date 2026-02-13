import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def numerical_sort_key(s):
    """Sort strings with embedded numbers numerically."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def visualize_large_h5(
    file_path, 
    n_samples=20000, 
    method='umap',  # 'umap', 'tsne', or 'pca'
    random_state=42
):
    """
    Sample a subset of data from a large H5 file (columnar or 2D) and visualize.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Inspecting file: {file_path}")
            
            # 1. Detect Data Structure
            keys = list(f.keys())
            bp_cols = [k for k in keys if k.startswith('bp_')]
            rp_cols = [k for k in keys if k.startswith('rp_')]
            
            # Sort columns numerically (e.g., bp_2 comes after bp_1, not after bp_10)
            bp_cols.sort(key=numerical_sort_key)
            rp_cols.sort(key=numerical_sort_key)
            
            feature_cols = bp_cols + rp_cols
            
            if feature_cols:
                print(f"Found {len(feature_cols)} feature columns (bp_*/rp_*).")
                # Assume all have same length, check first one
                total_rows = f[feature_cols[0]].shape[0]
                is_columnar = True
            else:
                # Fallback: look for a single 2D dataset
                dataset_name = None
                def find_first_2d(name, obj):
                    if isinstance(obj, h5py.Dataset) and len(obj.shape) == 2:
                        return name
                dataset_name = f.visititems(find_first_2d)
                
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

            # 2. Random Sampling
            if total_rows > n_samples:
                print(f"Sampling {n_samples} indices from {total_rows}...")
                np.random.seed(random_state)
                indices = np.random.choice(total_rows, n_samples, replace=False)
                indices.sort()
            else:
                indices = np.arange(total_rows)
                print(f"Using all {total_rows} rows.")

            # 3. Load Data
            print("Loading data (optimized)...")
            if is_columnar:
                # Load each column for the sampled indices
                data_list = []
                total_cols = len(feature_cols)
                
                for i, col_name in enumerate(feature_cols):
                    if i % 10 == 0:
                        print(f"Reading column {i}/{total_cols}: {col_name}...")
                        
                    # OPTIMIZATION: 
                    # H5py random access (fancy indexing) is very slow for large files because it does many seeks.
                    # Since each column (5M rows) is only ~40MB (float64) or ~20MB (float32), 
                    # it is MUCH faster to read the whole column sequentially into RAM, sample it, 
                    # and then discard the full column.
                    full_col = f[col_name][:]
                    col_data = full_col[indices]
                    data_list.append(col_data)
                    
                    # Help GC
                    del full_col
                
                # Stack to (n_samples, n_features)
                data_sampled = np.column_stack(data_list)
            else:
                # 2D dataset case (if it fits in memory, read all then sample)
                # If it's huge, this might still be slow, but usually 2D chunking is better handled.
                # For safety, let's try reading all if < 1GB.
                # 5M * 100 * 8 bytes = 4GB. Might be tight. 
                # But user case is likely columnar based on previous output.
                data_sampled = f[dataset_name][:][indices]

            print(f"Data shape for visualization: {data_sampled.shape}")
            
            # Simple check for NaNs/Infs
            if np.isnan(data_sampled).any():
                print("Warning: NaNs found in data. Filling with 0.")
                data_sampled = np.nan_to_num(data_sampled)

    except Exception as e:
        print(f"Error processing H5 file: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Dimensionality Reduction
    print(f"Running {method.upper()}...")
    
    embedding = None
    
    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_state, n_jobs=-1)
            embedding = reducer.fit_transform(data_sampled)
        except ImportError:
            print("UMAP not installed. Please run `pip install umap-learn`.")
            return
            
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=random_state, init='pca', learning_rate='auto')
        embedding = tsne.fit_transform(data_sampled)
        
    elif method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(data_sampled)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    else:
        print(f"Unknown method: {method}")
        return

    # 5. Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.5, c='blue')
    plt.title(f"{method.upper()} projection of {len(indices)} samples")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    output_img = f"{method}_projection.png"
    plt.savefig(output_img, dpi=300)
    print(f"Saved visualization to {output_img}")

if __name__ == "__main__":
    TARGET_FILE = './bp_rp_lamost.h5'
    
    if os.path.exists(TARGET_FILE):
        visualize_large_h5(TARGET_FILE, n_samples=20000, method='pca')
    else:
        print(f"File {TARGET_FILE} not found.")
