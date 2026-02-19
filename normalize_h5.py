"""
normalize_h5.py
---------------
Normalize the spectral columns in bp_rp_lamost.h5:
  - Every bp_i  →  bp_i / bp_1   (per data point)
  - Every rp_i  →  rp_i / rp_1   (per data point)
  - All other columns (e.g. feh, source_id, …) are copied as-is.

Memory strategy: bp_1 and rp_1 (~40 MB each) are loaded once;
every other column is processed one at a time, so peak RAM is ~120 MB.
"""

import h5py
import numpy as np
import os
import time

INPUT_FILE  = './bp_rp_lamost.h5'
OUTPUT_FILE = './bp_rp_lamost_normalized.h5'

def normalize_h5(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: '{input_path}' not found.")
        return

    if os.path.exists(output_path):
        print(f"Output file '{output_path}' already exists. Remove it first.")
        return

    t0 = time.time()

    with h5py.File(input_path, 'r') as fin:
        keys = list(fin.keys())
        total_rows = fin[keys[0]].shape[0]
        print(f"Input : {input_path}")
        print(f"Rows  : {total_rows:,}")
        print(f"Cols  : {len(keys)}")

        # --- Load normalizers ---
        if 'bp_1' not in keys or 'rp_1' not in keys:
            print("Error: 'bp_1' or 'rp_1' not found in file.")
            return

        print("\nLoading bp_1 and rp_1 for normalization…")
        bp1 = fin['bp_1'][:].astype(np.float32)
        rp1 = fin['rp_1'][:].astype(np.float32)

        # Guard against zero-division (flag bad rows; set result to NaN)
        bp1_safe = np.where(bp1 == 0, np.nan, bp1)
        rp1_safe = np.where(rp1 == 0, np.nan, rp1)

        n_bp_zero = int(np.sum(bp1 == 0))
        n_rp_zero = int(np.sum(rp1 == 0))
        if n_bp_zero:
            print(f"  Warning: {n_bp_zero:,} rows have bp_1 == 0  → will become NaN")
        if n_rp_zero:
            print(f"  Warning: {n_rp_zero:,} rows have rp_1 == 0  → will become NaN")

        # Identify column groups
        bp_cols = sorted([k for k in keys if k.startswith('bp_')],
                         key=lambda s: int(s.split('_')[1]))
        rp_cols = sorted([k for k in keys if k.startswith('rp_')],
                         key=lambda s: int(s.split('_')[1]))
        other_cols = [k for k in keys if k not in bp_cols and k not in rp_cols]

        print(f"\nbp_* columns : {len(bp_cols)}")
        print(f"rp_* columns : {len(rp_cols)}")
        print(f"Other columns: {len(other_cols)}  {other_cols}")

        # --- Write output ---
        with h5py.File(output_path, 'w') as fout:

            def copy_col(name, data, dtype=np.float32):
                ds = fout.create_dataset(
                    name, shape=(total_rows,), dtype=dtype,
                    chunks=(min(total_rows, 65536),),
                    compression='gzip', compression_opts=4
                )
                ds[:] = data

            # 1. Normalize and write bp_* columns
            print("\n[1/3] Normalizing bp_* columns…")
            for i, col in enumerate(bp_cols):
                raw = fin[col][:].astype(np.float32)
                normalized = raw / bp1_safe
                copy_col(col, normalized)
                if (i + 1) % 20 == 0 or (i + 1) == len(bp_cols):
                    print(f"  {i+1}/{len(bp_cols)}  {col}  "
                          f"(elapsed {time.time()-t0:.0f}s)")

            # 2. Normalize and write rp_* columns
            print("\n[2/3] Normalizing rp_* columns…")
            for i, col in enumerate(rp_cols):
                raw = fin[col][:].astype(np.float32)
                normalized = raw / rp1_safe
                copy_col(col, normalized)
                if (i + 1) % 20 == 0 or (i + 1) == len(rp_cols):
                    print(f"  {i+1}/{len(rp_cols)}  {col}  "
                          f"(elapsed {time.time()-t0:.0f}s)")

            # 3. Copy other columns as-is (e.g. feh, source_id)
            print("\n[3/3] Copying other columns as-is…")
            for col in other_cols:
                src = fin[col]
                raw = src[:]
                ds = fout.create_dataset(
                    col, data=raw,
                    chunks=(min(total_rows, 65536),),
                    compression='gzip', compression_opts=4
                )
                # Preserve attributes if any
                for attr_key, attr_val in src.attrs.items():
                    ds.attrs[attr_key] = attr_val
                print(f"  Copied: {col}  shape={raw.shape}  dtype={raw.dtype}")

    elapsed = time.time() - t0
    out_size = os.path.getsize(output_path) / 1e9
    print(f"\nDone! Output: {output_path}")
    print(f"File size : {out_size:.2f} GB")
    print(f"Total time: {elapsed:.1f} s")


if __name__ == '__main__':
    normalize_h5(INPUT_FILE, OUTPUT_FILE)
