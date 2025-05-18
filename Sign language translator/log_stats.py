import numpy as np
import re

# Path to the input log file
log_file_path = 'samples_log.txt'

# Read the log file
with open(log_file_path, 'r') as log_file:
    log_data = log_file.read()

# Extract the arrays from the log file using regex
# This assumes the arrays are written in the format similar to [array([...])]
array_pattern = r"array\(\[([^\]]+)\]\)"
arrays = re.findall(array_pattern, log_data)

# Convert each array string to a list of floats
arrays = [np.array(list(map(float, arr.split(','))), dtype=np.float32) for arr in arrays]

# Now we calculate the statistics for each array
statistics = []

for idx, arr in enumerate(arrays):
    mean_val = np.mean(arr)
    variance_val = np.var(arr)
    std_val = np.std(arr)
    statistics.append({
        'index': idx + 1,
        'mean': mean_val,
        'variance': variance_val,
        'std': std_val
    })

# Write the statistics back to the log file
with open(log_file_path, 'a') as log_file:
    log_file.write("\n\nStatistics of samples:\n")
    for stat in statistics:
        log_file.write(f"Sample {stat['index']}:\n")
        log_file.write(f"  Mean: {stat['mean']}\n")
        log_file.write(f"  Variance: {stat['variance']}\n")
        log_file.write(f"  Std Dev: {stat['std']}\n")
        log_file.write("\n")

print(f"Statistics have been appended to {log_file_path}")
