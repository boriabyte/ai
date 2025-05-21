import numpy as np
import os

def init_npz_file(npz_file_path):
    # Check if the .npz file exists and is not empty
    if os.path.exists(npz_file_path):
        try:
            # Try loading the .npz file
            data = np.load(npz_file_path, allow_pickle=True)
            if len(data['X']) == 0:  # Check if no data is available
                print("Warning: No data found in the existing .npz file. Initializing new dataset.")
                return [], []
            else:
                return data['X'], data['y']
        except Exception as e:
            # In case of any error (e.g., file is corrupt or empty), initialize new dataset
            print(f"Error loading .npz file: {e}. Initializing new dataset.")
            return [], []
    else:
        # If the file doesn't exist, initialize new dataset
        print("No existing .npz file found. Initializing new dataset.")
        return [], []

def save_npz_dataset(X, y, filename):
    X_array = np.array(X, dtype=object)  # Ensure variable-length sequences are handled
    y_array = np.array(y, dtype=object)
    np.savez(filename, X=X_array, y=y_array)