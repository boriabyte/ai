import numpy as np
import os

def init_npz_file(npz_file_path):
    if os.path.exists(npz_file_path):
        try:
            data = np.load(npz_file_path, allow_pickle=True)
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

def save_npz_dataset(X1, X2, y, filename):
    X1_array = np.array(X1, dtype=object)
    X2_array = np.array(X2, dtype=object)
    y_array = np.array(y, dtype=object)
    np.savez(filename, X1=X1_array, X2=X2_array, y=y_array)