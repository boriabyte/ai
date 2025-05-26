import numpy as np
import time
from collections import deque
import mediapipe as mp
from hand_aux import *
from normalize import *
from npz_helper import *

# Globals
previous_hand_positions = {}
curvature_arr = []
prev_vector = None
prev_prev_vector = None
START_TIME = None

VELOCITY_THRESHOLD = 2.0
ACCELERATION_THRESHOLD = 2.0

features_to_normalize = {
    'elapsed': deque(maxlen=30), 'crv': deque(maxlen=30), 'mwa': deque(maxlen=30),
    'msa': deque(maxlen=30), 'sv': deque(maxlen=30), 'mxsa': deque(maxlen=30),
    'cmpct': deque(maxlen=30), 'acc': deque(maxlen=30), 'vcty': deque(maxlen=30),
    'fd0': deque(maxlen=30), 'fd1': deque(maxlen=30), 'fd2': deque(maxlen=30),
    'fd3': deque(maxlen=30), 'fd4': deque(maxlen=30), 'fd5': deque(maxlen=30),
    'fd6': deque(maxlen=30), 'fd7': deque(maxlen=30), 'fd8': deque(maxlen=30),
    'fd9': deque(maxlen=30), 'fab': deque(maxlen=30)
}
for i in range(5):
    features_to_normalize[f'curl_{i}'] = deque(maxlen=30)
for i in range(3):
    features_to_normalize[f'dx_{i}'] = deque(maxlen=30)
    features_to_normalize[f'dy_{i}'] = deque(maxlen=30)

features_to_normalize_delta1 = {}
features_to_normalize_delta2 = {}

def extract_features_for_inference(image, mp_hands, mp_drawing, h, w, hand_results, fps):
    global previous_hand_positions, prev_vector, prev_prev_vector, START_TIME

    frame_time = 1.0 / fps
    features = []
    logic_features = []

    if hand_results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            landmarks_3d = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            wrist = landmarks[0]

            fingers = {
                "Thumb": (landmarks[4], landmarks[2]),
                "Index": (landmarks[8], landmarks[5]),
                "Middle": (landmarks[12], landmarks[9]),
                "Ring": (landmarks[16], landmarks[13]),
                "Pinky": (landmarks[20], landmarks[17])
            }

            fingers_states = {
                "Thumb": (landmarks[2], landmarks[3], landmarks[4]),
                "Index": (landmarks[5], landmarks[6], landmarks[8]),
                "Middle": (landmarks[9], landmarks[10], landmarks[12]),
                "Ring": (landmarks[13], landmarks[14], landmarks[16]),
                "Pinky": (landmarks[17], landmarks[18], landmarks[20])
            }

            wrist_angles = []
            finger_curvatures = {}
            for fname, (tip, mcp) in fingers.items():
                previous_hand_positions[fname], curvature = angle_curv(
                    image, frame_time, mcp, wrist, tip, wrist_angles, fname, previous_hand_positions, finger_curvatures
                )
                curvature_arr.append(curvature)

            spread_angles = []
            for f1, f2 in [("Thumb", "Index"), ("Index", "Middle"), ("Middle", "Ring"), ("Ring", "Pinky")]:
                inter_finger_angles(image, fingers, f1, f2, wrist, spread_angles)

            mean_wrist_angle, mean_spread_angle, var_spread, max_spread_angle, compactness = aggregate_parameters(
                image, wrist_angles, spread_angles, 150, 20
            )

            velocity, acceleration = (None, None)
            if hand_idx in previous_hand_positions:
                velocity, acceleration = compute_velocity_acc(
                    image, frame_time, hand_idx, previous_hand_positions,
                    wrist, VELOCITY_THRESHOLD, ACCELERATION_THRESHOLD, 20, 150
                )
            previous_hand_positions[hand_idx] = wrist

            landmarks_copy = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            finger_distances = compute_hand_distances(landmarks_copy)
            fd_norms = [z_score_norm(features_to_normalize[f'fd{i}'], d) for i, d in enumerate(finger_distances)]
            har = calculate_hand_aspect_ratio(hand_landmarks)

            # Normalize main features
            norm_feats = [
                z_score_norm(features_to_normalize['crv'], curvature if curvature is not None else 0.0),
                z_score_norm(features_to_normalize['mwa'], mean_wrist_angle if mean_wrist_angle is not None else 0.0),
                z_score_norm(features_to_normalize['msa'], mean_spread_angle if mean_spread_angle is not None else 0.0),
                z_score_norm(features_to_normalize['sv'], var_spread if var_spread is not None else 0.0),
                z_score_norm(features_to_normalize['mxsa'], max_spread_angle if max_spread_angle is not None else 0.0),
                z_score_norm(features_to_normalize['cmpct'], compactness if compactness is not None else 0.0),
                z_score_norm(features_to_normalize['acc'], acceleration if acceleration is not None else 0.0),
                z_score_norm(features_to_normalize['vcty'], velocity if velocity is not None else 0.0)
            ]

            if START_TIME is None:
                START_TIME = time.time()
            elapsed = time.time() - START_TIME
            el_tm = z_score_norm(features_to_normalize['elapsed'], elapsed)

            feature_vector = [el_tm] + norm_feats + fd_norms + [har]

            # Logic-based features
            thumb_in = thumb_inside_fist(landmarks_copy, image)
            between = 0.0
            for (l, r), val in { (5, 9): 0.2, (9, 13): 0.4, (13, 17): 0.6 }.items():
                if thumb_between_fingers(landmarks_copy, l, r, image):
                    between = val
                    break

            fab = fingers_above_thumb(landmarks_copy, image)
            fab_norm = fab / 4.0
            curls = average_finger_curl(landmarks_copy, image)
            curl_norms = [z_score_norm(features_to_normalize[f'curl_{i}'], c) for i, c in enumerate(curls)]

            offsets = finger_xy_offsets(landmarks_copy, image)
            offset_norms = []
            for i in range(3):
                offset_norms.append(z_score_norm(features_to_normalize[f'dx_{i}'], offsets[2*i]))
                offset_norms.append(z_score_norm(features_to_normalize[f'dy_{i}'], offsets[2*i+1]))

            feature_vector += curl_norms + offset_norms

            # Delta normalization
            full_vector = feature_vector.copy()
            for i in range(len(full_vector)):
                features_to_normalize_delta1.setdefault(f'd1_{i}', deque(maxlen=30))
                features_to_normalize_delta2.setdefault(f'd2_{i}', deque(maxlen=30))

            if prev_vector is not None and prev_prev_vector is not None:
                delta_1 = [a - b for a, b in zip(feature_vector, prev_vector)]
                delta_2 = [a - b for a, b in zip(prev_vector, prev_prev_vector)]
                delta_1_z = [z_score_norm(features_to_normalize_delta1[f'd1_{i}'], d) for i, d in enumerate(delta_1)]
                delta_2_z = [z_score_norm(features_to_normalize_delta2[f'd2_{i}'], d) for i, d in enumerate(delta_2)]
                for i, d in enumerate(delta_1): features_to_normalize_delta1[f'd1_{i}'].append(d)
                for i, d in enumerate(delta_2): features_to_normalize_delta2[f'd2_{i}'].append(d)
                full_vector += delta_1_z + delta_2_z

            prev_prev_vector = prev_vector
            prev_vector = feature_vector

            logic_vec = extract_finger_state_features(fingers_states, wrist, image)
            logic_vec += [thumb_in, between, fab_norm]

            features.append(full_vector)
            logic_features.append(logic_vec)

    return features, logic_features
