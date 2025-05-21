import cv2
import numpy as np
from hand_aux import *  # Assuming your custom hand gesture calculation functions
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
        self.history = {i: deque(maxlen=window_size) for i in range(5)}  # T, I, M, R, P

    def update(self, current_states):
        for i in range(5):
            self.history[i].append(current_states[i])

    def get_activity_scores(self):
        return [sum(self.history[i]) / len(self.history[i]) for i in range(5)]
    
# Dictionary to store previous frame landmarks, velocities, accelerations, timestamps for each hand
previous_hand_positions = {}
curvature_arr = []

# Thresholds for noise filtering and gesture recognition
VELOCITY_THRESHOLD = 2.0
ACCELERATION_THRESHOLD = 2.0
START_TIME = None

KEY_LABEL_MAP = {
    'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G', 'h': 'H', 'i': 'I', 'j': 'J',
    'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'O', 'p': 'P', 'q': 'Q', 'r': 'R', 's': 'S', 't': 'T',
    'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X', 'y': 'Y', 'z': 'Z',
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'
}

finger_activity = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

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
    'elapsed': deque(maxlen=30),
    'crv': deque(maxlen=30),
    'mwa': deque(maxlen=30),
    'msa': deque(maxlen=30),
    'sv': deque(maxlen=30),
    'mxsa': deque(maxlen=30),
    'cmpct': deque(maxlen=30),
    'acc': deque(maxlen=30),
    'vcty': deque(maxlen=30),
    'fd0': deque(maxlen=30),
    'fd1': deque(maxlen=30),
    'fd2': deque(maxlen=30),
    'fd3': deque(maxlen=30),
    'fd4': deque(maxlen=30),
    'fd5': deque(maxlen=30),
    'fd6': deque(maxlen=30),
    'fd7': deque(maxlen=30),
    'fd8': deque(maxlen=30),
    'fd9': deque(maxlen=30)
}

features_to_normalize_delta1 = {}
features_to_normalize_delta2 = {}

def process_frame_features(image, mp_hands, mp_drawing, h, w, hand_results, fps):
    global previous_hand_positions
    global START_TIME
    global current_label
    global sequence_buffer
    global was_recording
    global X_dataset
    global y_dataset
    global prev_vector
    global prev_prev_vector

    # Store per-feature history for delta normalization
    if 'features_to_normalize_delta1' not in globals():
        globals()['features_to_normalize_delta1'] = {}
    if 'features_to_normalize_delta2' not in globals():
        globals()['features_to_normalize_delta2'] = {}

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
                "Thumb": (landmarks[2], landmarks[3], landmarks[4]),     # mcp, ip, tip
                "Index": (landmarks[5], landmarks[6], landmarks[8]),     # mcp, pip, tip
                "Middle": (landmarks[9], landmarks[10], landmarks[12]),
                "Ring": (landmarks[13], landmarks[14], landmarks[16]),
                "Pinky": (landmarks[17], landmarks[18], landmarks[20])
            }
            
            text_x = 20

            wrist_angles = []
            finger_curvatures = {}
            for finger_name, (tip, mcp) in fingers.items():
                previous_hand_positions[finger_name], curvature = angle_curv(
                    image, frame_time, mcp, wrist, tip, wrist_angles, finger_name, previous_hand_positions, finger_curvatures
                )
                curvature_arr.append(curvature)

            spread_angles = []
            pairs = [("Thumb", "Index"), ("Index", "Middle"), ("Middle", "Ring"), ("Ring", "Pinky")]
            for finger1, finger2 in pairs:
                inter_finger_angles(image, fingers, finger1, finger2, wrist, spread_angles)

            mean_wrist_angle, mean_spread_angle, variance_spread, max_spread_angle, compactness = aggregate_parameters(
                image, wrist_angles, spread_angles, 150, text_x
            )

            velocity, acceleration = None, None
            if hand_idx in previous_hand_positions:
                velocity, acceleration = compute_velocity_acc(
                    image, frame_time, hand_idx, previous_hand_positions,
                    wrist, VELOCITY_THRESHOLD, ACCELERATION_THRESHOLD,
                    text_x, 150
                )

            previous_hand_positions[hand_idx] = wrist

            yaw, pitch, roll = calculate_yaw_pitch_roll(landmarks_3d)
            orientation = get_hand_orientation(yaw, pitch, roll)
            opnss = print_handedness(hand_landmarks, image, text_x, 50, 50, fingers, wrist, orientation)

            landmarks_copy = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            finger_distances = compute_hand_distances(landmarks_copy)

            fd_norms = []
            for i, d in enumerate(finger_distances):
                key = f'fd{i}'
                features_to_normalize[key].append(d)
                fd_norms.append(z_score_norm(features_to_normalize[key], d))

            draw_inter_fing_distances(image, fd_norms, text_x)

            har = calculate_hand_aspect_ratio(hand_landmarks)
            print_har(image, har)

            new_crv = curvature
            new_mwa = mean_wrist_angle
            new_msa = mean_spread_angle
            new_sv = variance_spread
            new_mxsa = max_spread_angle
            new_cmpct = compactness
            new_acc = acceleration
            new_vcty = velocity

            if new_crv is not None: features_to_normalize['crv'].append(new_crv)
            if new_mwa is not None: features_to_normalize['mwa'].append(new_mwa)
            if new_msa is not None: features_to_normalize['msa'].append(new_msa)
            if new_sv is not None: features_to_normalize['sv'].append(new_sv)
            if new_mxsa is not None: features_to_normalize['mxsa'].append(new_mxsa)
            if new_cmpct is not None: features_to_normalize['cmpct'].append(new_cmpct)
            if new_acc is not None: features_to_normalize['acc'].append(new_acc)
            if new_vcty is not None: features_to_normalize['vcty'].append(new_vcty)

            crv_norm = z_score_norm(features_to_normalize['crv'], new_crv)
            mwa_norm = z_score_norm(features_to_normalize['mwa'], new_mwa)
            msa_norm = z_score_norm(features_to_normalize['msa'], new_msa)
            sv_norm = z_score_norm(features_to_normalize['sv'], new_sv)
            mxsa_norm = z_score_norm(features_to_normalize['mxsa'], new_mxsa)
            cmpct_norm = z_score_norm(features_to_normalize['cmpct'], new_cmpct)
            acc_norm = z_score_norm(features_to_normalize['acc'], new_acc)
            vcty_norm = z_score_norm(features_to_normalize['vcty'], new_vcty)

            if START_TIME is None:
                START_TIME = time.time()

            elapsed_time = time.time() - START_TIME
            features_to_normalize['elapsed'].append(elapsed_time)
            el_tm = z_score_norm(features_to_normalize['elapsed'], elapsed_time)

            feature_vector = [
                el_tm, orientation, opnss, yaw, pitch, roll,
                crv_norm, mwa_norm, msa_norm, sv_norm,
                mxsa_norm, cmpct_norm, acc_norm, vcty_norm
            ]
            feature_vector.extend(fd_norms)
            feature_vector.append(har)

            # Initialize delta normalization stores if needed
            num_features = len(feature_vector)
            for i in range(num_features):
                features_to_normalize_delta1.setdefault(f'd1_{i}', deque(maxlen=30))
                features_to_normalize_delta2.setdefault(f'd2_{i}', deque(maxlen=30))

            # Compute and normalize deltas
            full_vector = feature_vector.copy()
            if prev_vector is not None and prev_prev_vector is not None:
                delta_1 = [a - b for a, b in zip(feature_vector, prev_vector)]
                delta_2 = [a - b for a, b in zip(prev_vector, prev_prev_vector)]

                delta_1_normed = []
                delta_2_normed = []

                for i in range(num_features):
                    features_to_normalize_delta1[f'd1_{i}'].append(delta_1[i])
                    features_to_normalize_delta2[f'd2_{i}'].append(delta_2[i])

                    d1_z = z_score_norm(features_to_normalize_delta1[f'd1_{i}'], delta_1[i])
                    d2_z = z_score_norm(features_to_normalize_delta2[f'd2_{i}'], delta_2[i])

                    delta_1_normed.append(d1_z)
                    delta_2_normed.append(d2_z)

                full_vector.extend(delta_1_normed)
                full_vector.extend(delta_2_normed)

            prev_prev_vector = prev_vector
            prev_vector = feature_vector

            features.append(full_vector)
            finger_logic_features.append(extract_finger_state_features(fingers_states, wrist, image))

    return features, finger_logic_features

