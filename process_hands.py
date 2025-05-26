# === Updated: process_hands.py ===
import cv2
import numpy as np
from hand_aux import *  # Updated auxiliary functions
import time
from collections import deque
from normalize import *
from npz_helper import *
import os
import csv
import datetime
import pandas as pd
import keyboard

class FingerActivityTracker:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.history = {i: deque(maxlen=window_size) for i in range(5)}

    def update(self, current_states):
        for i in range(5):
            self.history[i].append(current_states[i])

    def get_activity_scores(self):
        return [sum(self.history[i]) / len(self.history[i]) for i in range(5)]

previous_hand_positions = {}
curvature_arr = []

VELOCITY_THRESHOLD = 2.0
ACCELERATION_THRESHOLD = 2.0
START_TIME = None

KEY_LABEL_MAP = {chr(k): chr(k).upper() for k in range(97, 123)}
KEY_LABEL_MAP.update({str(i): str(i) for i in range(10)})

sequence_buffer = []
current_label = None
was_recording = False
har_norms = []

prev_vector = None
prev_prev_vector = None

X_dataset = []
y_dataset = []

npz_file_path = 'Sign language translator/gesture_data.npz'

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

def process_frame_features(image, mp_hands, mp_drawing, h, w, hand_results, fps):
    global previous_hand_positions, START_TIME, current_label, sequence_buffer
    global was_recording, X_dataset, y_dataset, prev_vector, prev_prev_vector

    frame_time = 1.0 / fps
    features = []
    finger_logic_features = []

    if hand_results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(int(landmark.x * w), int(landmark.y * h)) for landmark in hand_landmarks.landmark]
            landmarks_3d = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
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

            text_x = 20
            wrist_angles = []
            finger_curvatures = {}

            for finger_name, (tip, mcp) in fingers.items():
                previous_hand_positions[finger_name], curvature = angle_curv(
                    image, frame_time, mcp, wrist, tip, wrist_angles, finger_name,
                    previous_hand_positions, finger_curvatures
                )
                curvature_arr.append(curvature)

            spread_angles = []
            for finger1, finger2 in [("Thumb", "Index"), ("Index", "Middle"), ("Middle", "Ring"), ("Ring", "Pinky")]:
                inter_finger_angles(image, fingers, finger1, finger2, wrist, spread_angles)

            mean_wrist_angle, mean_spread_angle, variance_spread, max_spread_angle, compactness = aggregate_parameters(
                image, wrist_angles, spread_angles, 150, text_x
            )

            velocity, acceleration = compute_velocity_acc(image, frame_time, hand_idx, previous_hand_positions, wrist,
                                                          VELOCITY_THRESHOLD, ACCELERATION_THRESHOLD, text_x, 150) if hand_idx in previous_hand_positions else (None, None)
            previous_hand_positions[hand_idx] = wrist

            landmarks_copy = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            finger_distances = compute_hand_distances(landmarks_copy)
            fd_norms = [z_score_norm(features_to_normalize[f'fd{i}'], d) for i, d in enumerate(finger_distances)]
            draw_inter_fing_distances(image, fd_norms, text_x)

            har = calculate_hand_aspect_ratio(hand_landmarks)
            print_har(image, har)

            norm_feats = [
                z_score_norm(features_to_normalize['crv'], curvature if curvature is not None else 0.0),
                z_score_norm(features_to_normalize['mwa'], mean_wrist_angle if mean_wrist_angle is not None else 0.0),
                z_score_norm(features_to_normalize['msa'], mean_spread_angle if mean_spread_angle is not None else 0.0),
                z_score_norm(features_to_normalize['sv'], variance_spread if variance_spread is not None else 0.0),
                z_score_norm(features_to_normalize['mxsa'], max_spread_angle if max_spread_angle is not None else 0.0),
                z_score_norm(features_to_normalize['cmpct'], compactness if compactness is not None else 0.0),
                z_score_norm(features_to_normalize['acc'], acceleration if acceleration is not None else 0.0),
                z_score_norm(features_to_normalize['vcty'], velocity if velocity is not None else 0.0)
            ]

            if START_TIME is None:
                START_TIME = time.time()
            elapsed_time = time.time() - START_TIME
            el_tm = z_score_norm(features_to_normalize['elapsed'], elapsed_time)

            feature_vector = [el_tm] + norm_feats + fd_norms + [har]

            # === New Features ===
            thumb_in = thumb_inside_fist(landmarks_copy, image)
            between = 0.0
            for (left_idx, right_idx), value in {(5, 9): 0.2, (9, 13): 0.4, (13, 17): 0.6}.items():
                if thumb_between_fingers(landmarks_copy, left_idx, right_idx, image):
                    between = value
                    break

            fab = fingers_above_thumb(landmarks_copy, image)
            curls = average_finger_curl(landmarks_copy, image)
            offsets = finger_xy_offsets(landmarks_copy, image)

            fab_norm = fab / 4.0
            curl_norms = [z_score_norm(features_to_normalize[f'curl_{i}'], val) for i, val in enumerate(curls)]
            offset_norms = []
            for i in range(3):
                dx = offsets[2 * i]
                dy = offsets[2 * i + 1]
                dx_norm = z_score_norm(features_to_normalize[f'dx_{i}'], dx)
                dy_norm = z_score_norm(features_to_normalize[f'dy_{i}'], dy)
                offset_norms.extend([dx_norm, dy_norm])

            feature_vector.extend(curl_norms)
            feature_vector.extend(offset_norms)

            # === Delta normalization with fallback ===
            num_features = len(feature_vector)
            for i in range(num_features):
                features_to_normalize_delta1.setdefault(f'd1_{i}', deque(maxlen=30))
                features_to_normalize_delta2.setdefault(f'd2_{i}', deque(maxlen=30))

            delta_1_normed = [0.0] * num_features
            delta_2_normed = [0.0] * num_features

            if prev_vector is not None and prev_prev_vector is not None:
                delta_1 = [a - b for a, b in zip(feature_vector, prev_vector)]
                delta_2 = [a - b for a, b in zip(prev_vector, prev_prev_vector)]

                for i in range(num_features):
                    features_to_normalize_delta1[f'd1_{i}'].append(delta_1[i])
                    features_to_normalize_delta2[f'd2_{i}'].append(delta_2[i])

                    delta_1_normed[i] = z_score_norm(features_to_normalize_delta1[f'd1_{i}'], delta_1[i])
                    delta_2_normed[i] = z_score_norm(features_to_normalize_delta2[f'd2_{i}'], delta_2[i])

            full_vector = feature_vector + delta_1_normed + delta_2_normed

            prev_prev_vector = prev_vector
            prev_vector = feature_vector

            features.append(full_vector)

            logic_vec = extract_finger_state_features(fingers_states, wrist, image)
            logic_vec.extend([thumb_in, between, fab_norm])
            finger_logic_features.append(logic_vec)

    return features, finger_logic_features
