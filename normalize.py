import numpy as np

def z_score_norm(buffer, value, min_samples=10):
    """
    Z-score normalization is a data standardization technique that ensures standard deviation of 1 and mean of 0. 
    
    The formula [z = (current_value - mean_of_data) / standard deviation] ensures centering around 0, standardizing
    measurements in a common range and robustness to noise. It is especially useful for the present case due to
    its nature, in terms of dynamic adaptation for streaming data (seeing as SL translation needs to be done
    in real time). 
    
    Other normalization techniques, like min-max scaling, requires knowing the maximum value in advance, which is
    either impossible or extremely complex in online learning.
    """
    
    buffer.append(value)
    if len(buffer) < min_samples:
        return 0.0 # Not enough data, return 0 to avoid spikes
    
    mean = sum(buffer) / len(buffer)
    
    variance = sum((x - mean) ** 2 for x in buffer) / len(buffer)
    std = variance ** 0.5
    if std == 0:
        return 0.0
    
    return (value - mean) / std
