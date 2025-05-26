import numpy as np
import os

def extract_label_from_filename(filename):
    base_filename = os.path.splitext(filename)[0]
    return base_filename[0].upper()

def round_features(features, decimals=4):
    if isinstance(features, list):
        rounded_features = []
        for feature in features:
            if isinstance(feature, (list, np.ndarray)):
                rounded_features.append(np.round(feature, decimals=decimals))
            else:
                rounded_features.append(round(feature, decimals))
        return rounded_features
    else:
        return np.round(features, decimals=decimals)