import numpy as np
from collections import defaultdict

# Load the .npz file with allow_pickle=True
npz_file_path = 'Sign language translator/gesture_data.npz'
data = np.load(npz_file_path, allow_pickle=True)

# Check the keys
print("Keys in the .npz file:", data.keys())

# Load X1, X2, and y arrays
X1_data = data['X1']
X2_data = data['X2']
y_data = data['y']

# Grouping the samples by their labels
label_samples = defaultdict(list)

for x1, x2, label in zip(X1_data, X2_data, y_data):
    label_samples[label].append((x1, x2))

# Open a file to log the output
log_file_path = 'samples_log.txt'

with open(log_file_path, 'w') as log_file:
    log_file.write("Logging samples by labels:\n\n")

    for label, samples in label_samples.items():
        log_file.write(f"Label {label}:\n")
        log_file.write(f"  Number of samples: {len(samples)}\n")

        for idx, (x1, x2) in enumerate(samples):
            x1_len = x1.shape[0] if hasattr(x1, 'shape') else 'Unknown'
            x2_len = x2.shape[0] if hasattr(x2, 'shape') else 'Unknown'

            log_file.write(f"  Sample {idx + 1}:\n")
            log_file.write(f"    -> X1 shape: {x1.shape}\n")
            log_file.write(f"    -> X2 shape: {x2.shape}\n")
            log_file.write(f"    -> X1 content:\n")
            for i, vec in enumerate(x1):
                log_file.write(f"      [{i}]: {vec}\n")
            log_file.write(f"    -> X2 content:\n")
            for i, vec in enumerate(x2):
                log_file.write(f"      [{i}]: {vec}\n")
        log_file.write("\n")

print(f"Output has been logged to {log_file_path}")
