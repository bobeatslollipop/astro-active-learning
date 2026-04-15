import h5py

def explore_hdf5(filepath):
    """Read and explore the contents of an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        print(f"File: {filepath}")
        print("=" * 50)
        
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📄 Dataset: {name}")
                print(f"{indent}   Shape: {obj.shape}, Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}📁 Group: {name}")
            # Print attributes
            for key, val in obj.attrs.items():
                print(f"{indent}   Attr [{key}]: {val}")
        
        # Print top-level attributes
        for key, val in f.attrs.items():
            print(f"Root Attr [{key}]: {val}")
        
        # Recursively visit all groups and datasets
        f.visititems(print_structure)
        
        print("\n" + "=" * 50)
        print("To access a specific dataset, use:")
        print('  data = f["your/dataset/path"][:]')

def read_dataset(filepath, dataset_path):
    """Read a specific dataset from the HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        data = f[dataset_path][:]
        return data

# --- Usage ---
filepath = "bp_rp_lamost.h5"  # Change this to your file path

# Explore structure
explore_hdf5(filepath)

# Example: read a specific dataset
# data = read_dataset(filepath, "/group/dataset_name")
# print(data)