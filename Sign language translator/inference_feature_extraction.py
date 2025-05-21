import time
import numpy as np
from collections import defaultdict
from hand_aux import (
    get_hand_orientation, det_handedness, print_handedness, angle_curv,
    inter_finger_angles, aggregate_parameters, compute_velocity_acc,
    calculate_yaw_pitch_roll
)
from normalize import z_score_norm

# Store previous hand data
previous_hand_positions = defaultdict(lambda: None)

# Running statistics for normalization
features_to_normalize = {
    'elapsed': [], 'crv': [], 'mwa': [], 'msa': [],
    'sv': [], 'mxsa': [], 'cmpct': [], 'acc': [], 'vcty': []
}

VELOCITY_THRESHOLD = 2.0
ACCELERATION_THRESHOLD = 2.0

def extract_features_for_inference(image, mp_hands, mp_drawing, h, w, hand_results, frame_id, fps):
    global previous_hand_positions, features_to_normalize

    frame_time = 1.0 / fps

    if not hand_results.multi_hand_landmarks:
        return None

    for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
        landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in hand_landmarks.landmark]
        landmarks_3d = [(pt.x, pt.y, pt.z) for pt in hand_landmarks.landmark]
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        orientation, _ = get_hand_orientation(wrist, middle_mcp)
        fingers = {
            "Thumb": (landmarks[4], landmarks[2]),
            "Index": (landmarks[8], landmarks[5]),
            "Middle": (landmarks[12], landmarks[9]),
            "Ring": (landmarks[16], landmarks[13]),
            "Pinky": (landmarks[20], landmarks[17])
        }

        handedness, _ = det_handedness(hand_results, hand_idx, w)
        _, opnss = print_handedness(handedness, None, 0, 0, 0, fingers, wrist, orientation, "", draw=False)

        wrist_angles = []
        finger_curvatures = {}
        for finger_name, (tip, mcp) in fingers.items():
            previous_hand_positions[finger_name], curvature = angle_curv(
                None, frame_time, mcp, wrist, tip, wrist_angles, finger_name,
                previous_hand_positions, finger_curvatures
            )

        spread_angles = []
        for finger1, finger2 in [("Thumb", "Index"), ("Index", "Middle"),
                                 ("Middle", "Ring"), ("Ring", "Pinky")]:
            inter_finger_angles(None, fingers, finger1, finger2, wrist, spread_angles)

        mean_wrist_angle, mean_spread_angle, variance_spread, max_spread_angle, compactness = aggregate_parameters(
            None, wrist_angles, spread_angles, 0, 0
        )

        velocity, acceleration = compute_velocity_acc(
            None, frame_time, hand_idx, previous_hand_positions, wrist,
            VELOCITY_THRESHOLD, ACCELERATION_THRESHOLD, 0, 0
        ) or (0.0, 0.0)

        previous_hand_positions[hand_idx] = wrist
        yaw, pitch, roll = calculate_yaw_pitch_roll(landmarks_3d)

        def update_and_normalize(key, val):
            if val is not None:
                features_to_normalize[key].append(val)
                return z_score_norm(features_to_normalize[key], val)
            return 0.0

        elapsed_time = time.time()
        features_to_normalize['elapsed'].append(elapsed_time)
        el_tm = z_score_norm(features_to_normalize['elapsed'], elapsed_time)

        feature_vector = [
            el_tm,
            *handedness,
            *orientation,
            opnss,
            yaw, pitch, roll,
            update_and_normalize('crv', curvature),
            update_and_normalize('mwa', mean_wrist_angle),
            update_and_normalize('msa', mean_spread_angle),
            update_and_normalize('sv', variance_spread),
            update_and_normalize('mxsa', max_spread_angle),
            update_and_normalize('cmpct', compactness),
            update_and_normalize('acc', acceleration),
            update_and_normalize('vcty', velocity)
        ]

        return np.array(feature_vector)

    return None
