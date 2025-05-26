import numpy as np

def z_score_norm(buffer, value, min_samples=10):
    buffer.append(value)
    if len(buffer) < min_samples:
        return 0.0  # Not enough data, return 0 to avoid spikes
    mean = sum(buffer) / len(buffer)
    variance = sum((x - mean) ** 2 for x in buffer) / len(buffer)
    std = variance ** 0.5
    if std == 0:
        return 0.0
    return (value - mean) / std

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