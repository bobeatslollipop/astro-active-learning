import h5py
import numpy as np
import os

def extract_low_teff(input_file, output_file, threshold=4500):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading data from {input_file}...")
    with h5py.File(input_file, 'r') as f_in:
        if 'teff' not in f_in.keys():
            print("Error: 'teff' key not found in the file.")
            return
            
        teff = f_in['teff'][:]
        valid_mask = np.isfinite(teff)
        low_teff_mask = teff < threshold
        
        final_mask = valid_mask & low_teff_mask
        
        indices = np.where(final_mask)[0]
        
        print(f"Total entries: {len(teff)}")
        print(f"Entries with valid Teff < {threshold}: {len(indices)}")
        
        print(f"Writing extracted data to {output_file}...")
        with h5py.File(output_file, 'w') as f_out:
            for key in f_in.keys():
                print(f"Saving column: {key}")
                data = f_in[key][:]
                extracted_data = data[indices]
                f_out.create_dataset(key, data=extracted_data)
                
    print(f"Extraction successful! Saved to {output_file}")

if __name__ == '__main__':
    # You can change these file names if needed
    input_filename = 'bp_rp_lamost_normalized.h5'
    output_filename = 'bp_rp_lamost_normalized_low_teff.h5'
    
    extract_low_teff(input_filename, output_filename)
