import h5py
import os

def inspect_h5_file(filename):
    """
    Reads an H5 file and prints the shape (rows, columns) of each dataset found.
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    try:
        with h5py.File(filename, 'r') as f:
            print(f"Inspecting file: {filename}")
            print("=" * 40)

            def print_dataset_info(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}")
                    print(f"Total Shape: {obj.shape}")
                    
                    if len(obj.shape) >= 2:
                        rows = obj.shape[0]
                        cols = obj.shape[1]
                        print(f"Rows: {rows}")
                        print(f"Columns: {cols}")
                    elif len(obj.shape) == 1:
                        print(f"Rows: {obj.shape[0]} (1D array)")
                        print("Columns: 1")
                    
                    print(f"Data Type: {obj.dtype}")
                    print("-" * 40)

            f.visititems(print_dataset_info)

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    # The large H5 file identified in the directory
    target_file = 'bp_rp_lamost.h5' 
    
    # Check if h5py is installed
    try:
        import h5py
        inspect_h5_file(target_file)
    except ImportError:
        print("The 'h5py' library is required but not installed.")
        print("Please run: pip install h5py")
