import h5py
import numpy as np
import os

def create_subsets(source_file, train_file, test_file, seed=42):
    np.random.seed(seed)
    
    print(f"Loading metadata from {source_file}...")
    with h5py.File(source_file, 'r') as f:
        teff = f['teff'][:]
        feh = f['feh'][:]
        keys = list(f.keys())
        
    print(f"Total entries: len={len(teff)}")
    
    # Valid entries
    valid_mask = np.isfinite(teff) & np.isfinite(feh)
    valid_indices = np.where(valid_mask)[0]
    
    # 1. Training Set: all Teff < 4500 MP stars, and 2x randomly selected MR stars (from Teff < 4500 pool)
    mp_mask = valid_mask & (feh < -2.0)
    mr_mask = valid_mask & (feh >= -2.0)
    
    low_temp_mask = teff < 4500
    high_temp_mask = teff >= 4500
    
    # Indices
    train_mp_indices = np.where(mp_mask & low_temp_mask)[0]
    cand_train_mr_indices = np.where(mr_mask & low_temp_mask)[0]
    
    num_train_mr = len(train_mp_indices) * 2
    if len(cand_train_mr_indices) < num_train_mr:
        print(f"Warning: Not enough MR stars with Teff < 4500. Needed {num_train_mr}, have {len(cand_train_mr_indices)}")
        train_mr_indices = cand_train_mr_indices.copy()
    else:
        train_mr_indices = np.random.choice(cand_train_mr_indices, size=num_train_mr, replace=False)
        
    train_indices = np.concatenate([train_mp_indices, train_mr_indices])
    np.random.shuffle(train_indices) # Shuffle for training
    
    # 2. Test Set: all OTHER MP stars (Teff >= 4500), and 2x randomly selected MR stars (not in training set)
    test_mp_indices = np.where(mp_mask & high_temp_mask)[0]
    
    all_mr_indices = np.where(mr_mask)[0]
    cand_test_mr_indices = np.setdiff1d(all_mr_indices, train_mr_indices)
    
    num_test_mr = len(test_mp_indices) * 2
    if len(cand_test_mr_indices) < num_test_mr:
        print(f"Warning: Not enough remaining MR stars. Needed {num_test_mr}, have {len(cand_test_mr_indices)}")
        test_mr_indices = cand_test_mr_indices.copy()
    else:
        test_mr_indices = np.random.choice(cand_test_mr_indices, size=num_test_mr, replace=False)
        
    test_indices = np.concatenate([test_mp_indices, test_mr_indices])
    np.random.shuffle(test_indices)
    
    print(f"Train Set - MP: {len(train_mp_indices)}, MR: {len(train_mr_indices)}, Total: {len(train_indices)}")
    print(f"Test Set  - MP: {len(test_mp_indices)}, MR: {len(test_mr_indices)}, Total: {len(test_indices)}")
    
    print(f"Writing data to {train_file} and {test_file}...")
    with h5py.File(source_file, 'r') as f_in, \
         h5py.File(train_file, 'w') as f_train, \
         h5py.File(test_file, 'w') as f_test:
             
        for key in keys:
            print(f"Processing column: {key}")
            data = f_in[key][:]
            
            # Save to train
            data_train = data[train_indices]
            f_train.create_dataset(key, data=data_train)
            
            # Save to test
            data_test = data[test_indices]
            f_test.create_dataset(key, data=data_test)

    print("Success!")

if __name__ == '__main__':
    source = '../bp_rp_lamost_normalized.h5'
    train_out = 'low_temp_training_set.h5'
    test_out = 'low_temp_test_set.h5'
    create_subsets(source, train_out, test_out, seed=42)
