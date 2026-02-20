import h5py
import numpy as np
import argparse
import os
import re

def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def main():
    parser = argparse.ArgumentParser(description="Generate train and test datasets.")
    p_add = parser.add_argument
    p_add('--seed', type=int, default=42, help="Random seed for reproducibility.")
    p_add('--file-path', type=str, default='../bp_rp_lamost_normalized.h5', help="Path to the input H5 file.")
    p_add('--feh-threshold', type=float, default=-2.0, help="[Fe/H] threshold defining the boundary between MP and MR classes.")
    p_add('--train-frac', type=float, default=0.8, help="Fraction of metal-poor stars used for training (remaining used for test).")
    p_add('--mr-ratio', type=int, default=1, help="Ratio of Metal-Rich to Metal-Poor stars in the training/test sets.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    file_path = args.file_path
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print("Loading data...")
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        
        full_feh = f['feh'][:].astype(np.float64)
        valid_mask = np.isfinite(full_feh)
        
        valid_idx = np.where(valid_mask)[0]
        valid_feh = full_feh[valid_mask]
        
        thr = args.feh_threshold
        mp_idx = valid_idx[valid_feh < thr]
        mr_idx = valid_idx[valid_feh >= thr]

        np.random.shuffle(mp_idx)
        np.random.shuffle(mr_idx)

        N_mp = len(mp_idx)
        if N_mp == 0:
            print("No metal-poor data found!")
            return

        N_tr_mp = int(args.train_frac * N_mp)
        N_te_mp = N_mp - N_tr_mp

        ratio   = args.mr_ratio
        N_tr_mr = ratio * N_tr_mp
        N_te_mr = ratio * N_te_mp
        
        print(f"Total metal-poor data count: {N_mp}")
        print(f"Train MP (0): {N_tr_mp}, Train MR (1): {N_tr_mr}  [Total Train: {N_tr_mp + N_tr_mr}]")
        print(f"Test MP (0): {N_te_mp}, Test MR (1): {N_te_mr}   [Total Test:  {N_te_mp + N_te_mr}]")
        
        if len(mr_idx) < N_tr_mr + N_te_mr:
            print(f"Warning: Not enough metal-rich data! Have {len(mr_idx)}, need {N_tr_mr + N_te_mr}.")
            print("Adjusting test metal-rich limit to match available sizes.")
            N_te_mr = len(mr_idx) - N_tr_mr
            if N_te_mr < 0:
                print("Error: Too little MR data even for training.")
                return
            print(f"New Test MR (1): {N_te_mr}")

        tr_mp_idx = mp_idx[:N_tr_mp]
        te_mp_idx = mp_idx[N_tr_mp:]
        
        tr_mr_idx = mr_idx[:N_tr_mr]
        te_mr_idx = mr_idx[N_tr_mr:N_tr_mr+N_te_mr]
        
        train_idx = np.concatenate([tr_mp_idx, tr_mr_idx])
        test_idx = np.concatenate([te_mp_idx, te_mr_idx])
        
        train_shuffle = np.random.permutation(len(train_idx))
        train_idx = train_idx[train_shuffle]
        
        test_shuffle = np.random.permutation(len(test_idx))
        test_idx = test_idx[test_shuffle]

        print("Writing to random_train_set.h5 and random_test_set.h5...")
        with h5py.File("random_train_set.h5", "w") as f_train, h5py.File("random_test_set.h5", "w") as f_test:
            for col in keys:
                data = f[col][:]
                if isinstance(data, np.ndarray) and len(data.shape) > 0 and data.shape[0] == len(full_feh):
                    f_train.create_dataset(col, data=data[train_idx])
                    f_test.create_dataset(col, data=data[test_idx])

    print("Datasets saved successfully.")

if __name__ == "__main__":
    main()
