import numpy as np

def z_score_norm(values, new_value):
    if new_value is None:
        return 0  # Or another default value (could be the mean of the values if that's desired)
    
    # Remove None values from the list before calculating mean and std
    values = [val for val in values if val is not None]
    
    if not values:  # If the list is empty after filtering out None, return the new_value
        return new_value  # Or return 0, depending on the default behavior you want

    mean = np.mean(values)
    std = np.std(values)
    
    if std == 0:  # Prevent division by zero if the std is 0
        return 0  # Or return mean, depending on your approach

    return (new_value - mean) / std

def normalize_angle(angle, min_value, max_value):
    return 2 * (angle - min_value) / (max_value - min_value) - 1

def extract_feature_vector(
    handedness, orientation, opnss,
    yaw, pitch, roll,
    curvature, mean_wrist_angle,
    mean_spread_angle, variance_spread,
    max_spread_angle, compactness,
    acceleration, velocity,
    elapsed_time, features_to_normalize
):
    def update_and_norm(key, val):
        if val is not None:
            features_to_normalize[key].append(val)
            return z_score_norm(features_to_normalize[key], val)
        return 0.0

    return [
        update_and_norm('elapsed', elapsed_time),
        *handedness,
        *orientation,
        opnss,
        yaw, pitch, roll,
        update_and_norm('crv', curvature),
        update_and_norm('mwa', mean_wrist_angle),
        update_and_norm('msa', mean_spread_angle),
        update_and_norm('sv', variance_spread),
        update_and_norm('mxsa', max_spread_angle),
        update_and_norm('cmpct', compactness),
        update_and_norm('acc', acceleration),
        update_and_norm('vcty', velocity)
    ]