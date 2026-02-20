import h5py
import numpy as np
import random

FILE_ORIGINAL = './bp_rp_lamost.h5'
FILE_NORMALIZED = './bp_rp_lamost_normalized.h5'

def check_h5_differences():
    with h5py.File(FILE_ORIGINAL, 'r') as f_orig, h5py.File(FILE_NORMALIZED, 'r') as f_norm:
        keys = list(f_orig.keys())
        total_rows = f_orig[keys[0]].shape[0]
        
        print("=== 1. Checking Data Types and Compression ===")
        # Check a few representative columns
        cols_to_check = ['bp_2', 'rp_2', 'feh', 'source_id']
        # Filter cols to ensure they exist
        cols_to_check = [c for c in cols_to_check if c in keys]
        
        for col in cols_to_check:
            dset_orig = f_orig[col]
            dset_norm = f_norm[col]
            print(f"Column: {col}")
            print(f"  Original   - dtype: {dset_orig.dtype}, compression: {dset_orig.compression}")
            print(f"  Normalized - dtype: {dset_norm.dtype}, compression: {dset_norm.compression}")
            print()

        print("=== 2. Random Row Content Verification ===")
        # Randomly select 3 rows
        num_samples = 3
        random_indices = random.sample(range(total_rows), num_samples)
        print(f"Randomly selected row indices: {random_indices}")
        
        for idx in random_indices:
            print(f"\n--- Row {idx} ---")
            
            # Check a non-normalized column (e.g., feh)
            if 'feh' in keys:
                val_orig_feh = f_orig['feh'][idx]
                val_norm_feh = f_norm['feh'][idx]
                print(f"feh (Should be identical):")
                print(f"  Orig: {val_orig_feh}")
                print(f"  Norm: {val_norm_feh}")
                if np.isnan(val_orig_feh) and np.isnan(val_norm_feh):
                    print("  Match: True (Both NaN)")
                else:
                    print(f"  Match: {val_orig_feh == val_norm_feh}")
            
            # Check bp_1/rp_1 in normalized file (should be close to 1.0)
            orig_bp1 = f_orig['bp_1'][idx]
            norm_bp1 = f_norm['bp_1'][idx]
            print(f"\nbp_1 (Orig: {orig_bp1}, Norm: {norm_bp1} - should be ~1.0 if orig != 0)")
            
            orig_rp1 = f_orig['rp_1'][idx]
            norm_rp1 = f_norm['rp_1'][idx]
            print(f"rp_1 (Orig: {orig_rp1}, Norm: {norm_rp1} - should be ~1.0 if orig != 0)")

            # Check normalization logic for another column, e.g., bp_2
            if 'bp_2' in keys:
                orig_bp2 = f_orig['bp_2'][idx]
                norm_bp2 = f_norm['bp_2'][idx]
                
                # Reconstruct expected normalized value
                # Note: orig_bp2 converted to float32 as done in the script, then divided by safe bp1
                safe_bp1 = np.nan if orig_bp1 == 0 else orig_bp1
                expected_norm = np.float32(orig_bp2) / np.float32(safe_bp1)
                
                print(f"\nbp_2 Check:")
                print(f"  Orig bp_2          : {orig_bp2}")
                print(f"  Norm bp_2          : {norm_bp2}")
                print(f"  Expected (Orig/bp1): {expected_norm}")
                
                if np.isnan(norm_bp2) and np.isnan(expected_norm):
                    print(f"  Match: True (Both NaN)")
                else:
                    # Use np.isclose because of floating point arithmetic differences
                    match = np.isclose(norm_bp2, expected_norm, rtol=1e-5, equal_nan=True)
                    print(f"  Match: {match}")

if __name__ == '__main__':
    check_h5_differences()
