from collections import deque
import numpy as np
import pandas as pd
import datetime
import keyboard
import time
import cv2
import os
import csv

from normalize import *
from npz_helper import *
from hand_aux import *

"""
process_hands.py gathers the data from the video feed and creates the feature vectors used for training

A multitude of computations and feature engineering calculations are done on the data gathered from the MediaPipe hand
landmarks - neighbor finger distances and angles, finger-wrist angles, finger-palm angles, hand aspect ratio,
distances between each and every finger, acceleration and velocity for each finger, curvature for each finger
as well as their aggregate values that are meant to describe the form of the hand in a more compact manner,
such as: mean spread variance, maximum spread variance & compactness; these will form the main feature vector

The second feature vector consists of logic states for each finger - extension state, activity state, number of active 
fingers and fingers above thumb (for particular letters in ASL, such as E)

Thresholds for filtering out noisy variations that can cause jitter are used, while multiple
dictionaries and arrays are utilized for storing previous hand positions for computing velocity and acceleration
A Sequence buffer stores the gathered data and adds it at the end of the video
"""

VELOCITY_THRESHOLD = 2.0
ACCELERATION_THRESHOLD = 2.0

START_TIME = None

previous_hand_positions = {}

curvature_arr = []
sequence_buffer = [] 
current_label = None
was_recording = False
har_norms = [] 

prev_vector = None
prev_prev_vector = None

X_dataset = []
y_dataset = []

class FingerActivityTracker:
    """
    Tracks finger activiy over a set portion of frames to determine activity level - useful for signs like B, I and Y
    
    Returns an activity score for each finger based on the number of frames in which the finger remains extended
    """
    
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.history = {i: deque(maxlen=window_size) for i in range(5)}

    def update(self, current_states):
        for i in range(5):
            self.history[i].append(current_states[i])

    def get_activity_scores(self):
        return [sum(self.history[i]) / len(self.history[i]) for i in range(5)]

# Dictionaries storing features meant for normalizing values around a common value to minimize standard deviation and variance
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
    """
    Bulk of data preprocessing logic
    
    Input parameters: image = current frame, mp_hands, mp_drawing are MediaPipe landmarks crucial for computations
                      h, w = height and weight of the captured video
                      hand_results = values returned after processing the frames in camera.py
                      fps = number of frames used for processing certain features like acceleration
    
    Intermediary pipeline: multiple functions that compute the necessary parameters for the final returned feature
                           vectors
    
    Output parameters: features, finger_logic_features = main feature vector and finger logic states respectively
    """
    global previous_hand_positions, START_TIME, current_label, sequence_buffer
    global was_recording, X_dataset, y_dataset, prev_vector, prev_prev_vector

    frame_time = 1.0 / fps
    features = []
    finger_logic_features = []

    if hand_results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(int(landmark.x * w), int(landmark.y * h)) for landmark in hand_landmarks.landmark]
           
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

            text_x = 20 # Coordinates used for printing of information during data gathering
            wrist_angles = []
            finger_curvatures = {}

            """
            Calculation of angles and curvatures for each finger using the current frame, its duration, 
            relevant hand landmarks points etc.
            """
            for finger_name, (tip, mcp) in fingers.items():
                previous_hand_positions[finger_name], curvature = angle_curv(
                    image, frame_time, mcp, wrist, tip, wrist_angles, finger_name,
                    previous_hand_positions, finger_curvatures
                )
                curvature_arr.append(curvature)

            # Computation of finger-pair angles 
            spread_angles = []
            for finger1, finger2 in [("Thumb", "Index"), ("Index", "Middle"), ("Middle", "Ring"), ("Ring", "Pinky")]:
                inter_finger_angles(image, fingers, finger1, finger2, wrist, spread_angles)

            # Computation of aggregate parameters
            mean_wrist_angle, mean_spread_angle, variance_spread, max_spread_angle, compactness = aggregate_parameters(
                image, wrist_angles, spread_angles, 150, text_x
            )

            # Computation of velocity and acceleration
            velocity, acceleration = compute_velocity_acc(image, frame_time, hand_idx, previous_hand_positions, wrist,
                                                          VELOCITY_THRESHOLD, ACCELERATION_THRESHOLD, text_x, 150) if hand_idx in previous_hand_positions else (None, None)
            previous_hand_positions[hand_idx] = wrist

            # Auxiliary landmarks copy for preprocessing 
            landmarks_copy = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            
            # Computation of distances between every finger
            finger_distances = compute_hand_distances(landmarks_copy)
            
            # Normalization of distances for standardization
            fd_norms = [z_score_norm(features_to_normalize[f'fd{i}'], d) for i, d in enumerate(finger_distances)]
            draw_inter_fing_distances(image, fd_norms, text_x) # Print on screen distances for each finger (e.g.: fd0, fd1, fd2)

            # har = hand aspect ratio
            har = calculate_hand_aspect_ratio(hand_landmarks)
            display_har(image, har) # Print hand aspect ratio on screen

            """
            Array used for normalized values using z-score normalization; the normalized value will be set as 0 if, at any given time,
            the instantenous value of a parameter is 0; this is caused by division by 0 done by the z-score normalization which may crash the program
            
            Replacement with 0 is acceptable due to the fact that values that are divided by ever-decreasing values will increase 
            unreasonably and evade the standardized intervals set by normalization
            """
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

            # Measuring elapsed time and normalizing it - useful for composite gestures
            if START_TIME is None:
                START_TIME = time.time()
            elapsed_time = time.time() - START_TIME
            el_tm = z_score_norm(features_to_normalize['elapsed'], elapsed_time)

            # Compose feature_vector
            feature_vector = [el_tm] + norm_feats + fd_norms + [har]

            """
            Logical states of fingers for more discriminative features regarding certain signs that look similar 
            such as M, N, T, O, S, A which are all very alike in terms of angles and curvatures
            """
            thumb_in = thumb_inside_fist(landmarks_copy, image)
            
            """
            thumb_between_fingers a value, either 0.0 or 1 for each state in which the thumb may be
            0.0 = thumb is not between any finger, 1 if it is between any two fingers
            
            If "between" is returned as 1, depending between which finger the thumb is positioned, "between"
            gets one of the values from the hashmap used in the for loop; either 0.2, 0.4 or 0.6
            """
            between = 0.0
            for (left_idx, right_idx), value in {(5, 9): 0.2, (9, 13): 0.4, (13, 17): 0.6}.items():
                if thumb_between_fingers(landmarks_copy, left_idx, right_idx, image):
                    between = value
                    break

            fab = fingers_above_thumb(landmarks_copy, image) # fab = fingers above thumb
            curls = average_finger_curl(landmarks_copy, image)
            offsets = finger_xy_offsets(landmarks_copy, image)
            
            # z-score norming would oversimplify fab; thus a simple sum divided by the number of possible fingers above thumb is done
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

            # Delta normalization with fallback
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

            # Create full main vector
            features.append(full_vector)

            # Ensure proper structure for logic features vector
            logic_vec = extract_finger_state_features(fingers_states, wrist, image)
            logic_vec.extend([thumb_in, between, fab_norm])
            finger_logic_features.append(logic_vec)

    return features, finger_logic_features
