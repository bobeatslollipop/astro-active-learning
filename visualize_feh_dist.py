import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_feh_distribution(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            if 'feh' not in f.keys():
                print("Error: 'feh' column not found in the file.")
                return
            
            print("Loading 'feh' values...")
            feh_values = f['feh'][:]
            
            # Remove NaNs and infinities
            valid_feh = feh_values[np.isfinite(feh_values)]
            print(f"Total samples: {len(feh_values)}")
            print(f"Valid samples (non-NaN): {len(valid_feh)}")
            print(f"min [Fe/H]: {valid_feh.min():.3f}")
            print(f"max [Fe/H]: {valid_feh.max():.3f}")
            
            threshold = -2.0
            num_below = np.sum(valid_feh <= threshold)
            proportion = (num_below / len(valid_feh)) * 100
            print(f"Samples with [Fe/H] <= {threshold}: {num_below} ({proportion:.4f}%)")
            
            print("Plotting histogram...")
            plt.figure(figsize=(10, 6))
            plt.hist(valid_feh, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
            plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'[Fe/H] = {threshold}')
            plt.legend()
            plt.title('Distribution of [Fe/H]')
            plt.xlabel('[Fe/H]')
            plt.ylabel('Frequency')
            plt.yscale('log') # Optional log scale could be good, but let's stick to linear first, or just provide both/one
            plt.grid(axis='y', alpha=0.75)
            
            output_dir = 'data_visualization'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_file = os.path.join(output_dir, 'feh_distribution.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Histogram saved to {output_file}")
            
    except Exception as e:
        print(f"Error processing H5 file: {e}")

if __name__ == "__main__":
    TARGET_FILE = './bp_rp_lamost_normalized.h5'
    plot_feh_distribution(TARGET_FILE)
