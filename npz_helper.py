import numpy as np
import os

"""
Helper function for creating the storing .npz file for training.

Intialization needs to first take place in case the file is missing.

Saving the data after parsing it is done in save_npz_dataset.
"""

def init_npz_file(NPZ_FILE_PATH):
    if os.path.exists(NPZ_FILE_PATH):
        try:
            data = np.load(NPZ_FILE_PATH, allow_pickle=True)
            if len(data['X1']) == 0 or len(data['X2']) == 0:
                print("Warning: No data found in the existing .npz file. Initializing new dataset.")
                return [], [], []
            else:
                return data['X1'], data['X2'], data['y']
        except Exception as e:
            print(f"Error loading .npz file: {e}. Initializing new dataset.")
            return [], [], []
    else:
        print("No existing .npz file found. Initializing new dataset.")
        return [], [], []

def save_npz_dataset(X1, X2, y, FILENAME):
    X1_array = np.array(X1, dtype=object)
    X2_array = np.array(X2, dtype=object)
    y_array = np.array(y, dtype=object)
    np.savez(FILENAME, X1=X1_array, X2=X2_array, y=y_array)