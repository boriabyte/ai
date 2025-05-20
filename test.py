import numpy as np
from collections import defaultdict

# Load the .npz file with allow_pickle=True
npz_file_path = 'Sign language translator/gesture_data.npz'
data = np.load(npz_file_path, allow_pickle=True)

# Check the keys (it should include 'X' and 'y')
print("Keys in the .npz file:", data.keys())

# Load the X and y arrays
X_data = data['X']
y_data = data['y']

# Check the shapes of the arrays
print("Shape of X:", X_data.shape)
print("Shape of y:", y_data.shape)

# Grouping the samples by their labels
label_samples = defaultdict(list)

for features, label in zip(X_data, y_data):
    label_samples[label].append(features)

# Open a file to log the output
log_file_path = 'samples_log.txt'  # Change this to your desired file path

with open(log_file_path, 'w') as log_file:
    # Write the header
    log_file.write("Logging samples by labels:\n\n")
    
    # Now, write the samples grouped by their labels in the requested format
    for label, samples in label_samples.items():
        log_file.write(f"Label {label}:\n")
        log_file.write(f"  Number of samples: {len(samples)}\n")
        for idx, sample in enumerate(samples):
            num_feature_vectors = sample.shape[0] if hasattr(sample, 'shape') and len(sample.shape) > 0 else 'Unknown'
            log_file.write(f"  Sample {idx + 1}: {sample}\n")
            log_file.write(f"    -> Feature vectors: {num_feature_vectors}\n")
        log_file.write("\n")  # Add a newline after each label's samples

print(f"Output has been logged to {log_file_path}")
