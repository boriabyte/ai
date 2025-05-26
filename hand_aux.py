import numpy as np
import math
import cv2
from normalize import *

# Angle between fingertips and wrist, used mainly for HAND OPENNESS
# Joint b is the vertex

def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return float('nan')
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def get_palm_center(landmarks):
    indices = [0, 1, 5, 9, 13, 17]
    points = [landmarks[i] for i in indices]
    return np.mean(points, axis=0)

def thumb_inside_fist(landmarks, image=None, text_x=10, y=420):
    thumb_tip = np.array(landmarks[4])
    thumb_ip = np.array(landmarks[3])
    thumb_mcp = np.array(landmarks[2])

    index_tip = np.array(landmarks[8])
    middle_tip = np.array(landmarks[12])
    ring_tip = np.array(landmarks[16])

    # Criterion 1: Z-depth (thumb is farther away)
    min_finger_z = min(index_tip[2], middle_tip[2], ring_tip[2])
    z_margin = 0.01
    depth_ok = thumb_tip[2] > min_finger_z + z_margin

    # Criterion 2: Palm proximity
    palm_center = get_palm_center(landmarks)
    palm_dist = np.linalg.norm(thumb_tip[:2] - palm_center[:2])
    palm_ok = palm_dist < 0.09

    # Criterion 3: Thumb curl
    v1 = thumb_ip[:2] - thumb_mcp[:2]
    v2 = thumb_tip[:2] - thumb_ip[:2]
    curl_angle = angle_between(v1, v2)
    curl_ok = np.degrees(curl_angle) < 65

    # Criterion 4: Fingers visually above thumb
    thumb_y = thumb_tip[1]
    fingers_above = sum(1 for i in [8, 12, 16, 20] if landmarks[i][1] < thumb_y)
    above_ok = fingers_above > 0

    inside = 1.0 if depth_ok and palm_ok and curl_ok and not above_ok else 0.0

    if image is not None:
        debug = f"Z:{depth_ok} P:{palm_ok} C:{curl_ok} A:{above_ok}"
        cv2.putText(image, f"Thumb Inside: {inside} ({debug})", (text_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

    return inside


def angle_between(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return np.pi
    unit_v1 = v1 / norm_v1
    unit_v2 = v2 / norm_v2
    dot_product = np.dot(unit_v1, unit_v2)
    return np.arccos(np.clip(dot_product, -1.0, 1.0))

def thumb_between_fingers(landmarks, left_idx, right_idx, image=None, text_x=10, y=440):
    thumb_x = landmarks[4][0]
    thumb_y = landmarks[4][1]
    thumb_z = landmarks[4][2]

    left_x, left_y, left_z = landmarks[left_idx]
    right_x, right_y, right_z = landmarks[right_idx]

    # X-axis check
    x_between = min(left_x, right_x) < thumb_x < max(left_x, right_x)

    # Y-axis alignment
    y_aligned = min(left_y, right_y) - 0.05 < thumb_y < max(left_y, right_y) + 0.05

    # Z-depth: thumb must not be significantly deeper (i.e. behind)
    avg_finger_z = (left_z + right_z) / 2
    z_close_enough = thumb_z < avg_finger_z + 0.015  # Threshold prevents thumb from being *behind*

    between = 1.0 if x_between and y_aligned and z_close_enough else 0.0

    if image is not None and between == 1.0:
        pair_labels = {
            (5, 9): "Index-Middle",
            (9, 13): "Middle-Ring",
            (13, 17): "Ring-Pinky"
        }
        label = pair_labels.get((left_idx, right_idx), f"{left_idx}-{right_idx}")
        cv2.putText(image, f"Thumb Between {label}: {between}", (text_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

    return between


def fingers_above_thumb(landmarks, image=None, text_x=10, y=460):
    thumb_y = landmarks[4][1]
    count = sum(1 for i in [8, 12, 16, 20] if landmarks[i][1] < thumb_y)
    if image is not None:
        cv2.putText(image, f"Fingers Above Thumb: {count}", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
    return float(count)

def average_finger_curl(landmarks, image=None, text_x=10, y=480):
    finger_joints = {
        "Thumb": [1, 2, 4],
        "Index": [5, 6, 8],
        "Middle": [9, 10, 12],
        "Ring": [13, 14, 16],
        "Pinky": [17, 18, 20]
    }
    curls = []
    for i, (name, joints) in enumerate(finger_joints.items()):
        a, b, c = [landmarks[j] for j in joints]
        angle = calculate_angle(a, b, c)
        angle_val = angle if not np.isnan(angle) else 0.0
        curls.append(angle_val)
        if image is not None:
            cv2.putText(image, f"{name} Curl: {angle_val:.2f}", (text_x, y + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 100, 255), 1)
    return curls

def finger_xy_offsets(landmarks, image=None, text_x=10, y=600):
    pairs = [(8, 12), (12, 16), (16, 20)]
    offsets = []
    for i, (a, b) in enumerate(pairs):
        dx = landmarks[b][0] - landmarks[a][0]
        dy = landmarks[b][1] - landmarks[a][1]
        offsets.extend([dx, dy])
        if image is not None:
            cv2.putText(image, f"dx{i}: {dx:.2f}, dy{i}: {dy:.2f}", (text_x, y + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 100), 1)
    return offsets

# Detection of HAND OPENNESS 
# Determine if the hand is making a fist based on angles and fingertip distances
def calculate_openness(hand_landmarks):
    
    WRIST = 0
    FINGERTIPS = [8, 12, 16, 20] 

    wrist = np.array([hand_landmarks.landmark[WRIST].x, hand_landmarks.landmark[WRIST].y])

    distances = []
    for tip_idx in FINGERTIPS:
        fingertip = np.array([hand_landmarks.landmark[tip_idx].x, hand_landmarks.landmark[tip_idx].y])
        distance = np.linalg.norm(fingertip - wrist)
        distances.append(distance)

    # Return the average distance (larger = more open)
    if len(distances) == 0:
        return float('nan')  # Return NaN if no distances are available

    return np.mean(distances)

# Angle between two fingers - shown as green between digits
# Equivalent to distance between two vectors
def calculate_vector_angle(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if np.isnan(cos_angle):
        return float('nan')  # Return NaN if the cosine angle calculation results in NaN
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def det_handedness(hand_results, hand_idx, w):
    handedness = hand_results.multi_handedness[hand_idx].classification[0].label
    handedness = "Left" if handedness == "Left" else "Right"
    handedness = [1,0] if handedness == "Left" else [0,1]
    text_x = 20 if handedness == [1,0] else (w - 250)
    
    return handedness, text_x

def print_handedness(hand_landmarks, image, text_x, text_y_offset_left, text_y_offset_right, fingers, wrist, draw=True):
    fist_opennness = calculate_openness(hand_landmarks)

    # Check for NaN in openness degree
    if np.isnan(fist_opennness):
        fist_opennness = 0  # Default value if NaN is encountered

    cv2.putText(image, f"Openness degree: {fist_opennness}", (text_x, text_y_offset_right),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    text_y_offset_right += 30
    
    return fist_opennness
        
def angle_curv(image, frame_time, mcp, wrist, tip, wrist_angles, finger_name, previous_hand_positions, finger_curvatures):
    angle = calculate_angle(mcp, wrist, tip)
    curvature = 0

    # Only process/display if the angle is valid
    if np.isnan(angle):
        return float('nan'), curvature  # Return NaN if the angle is invalid

    wrist_angles.append(angle)
    cv2.putText(image, f"{finger_name}: {int(angle)}°", (tip[0], tip[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if finger_name in previous_hand_positions:
        prev_angle = previous_hand_positions[finger_name]
        curvature = abs(angle - prev_angle) / frame_time
        finger_curvatures[finger_name] = curvature
        cv2.putText(image, f"{finger_name} Curvature: {curvature:.2f}", (tip[0], tip[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    previous_hand_positions[finger_name] = angle

    return previous_hand_positions.get(finger_name, float('nan')), curvature
    
def inter_finger_angles(image, fingers, finger1, finger2, wrist, spread_angles):
    tip1, tip2 = fingers[finger1][0], fingers[finger2][0]
    wrist_pos = np.array(wrist)
    vec1 = np.array(tip1) - wrist_pos
    vec2 = np.array(tip2) - wrist_pos

    angle = calculate_vector_angle(vec1, vec2)
    if np.isnan(angle):
        return float('nan')  # Return NaN if the angle is invalid
    spread_angles.append(angle)

    distance = np.linalg.norm(np.array(tip1) - np.array(tip2))
    mid_x, mid_y = (tip1[0] + tip2[0]) // 2, (tip1[1] + tip2[1]) // 2
    cv2.putText(image, f"{int(distance)} px", (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(image, f"{int(angle)}°", (mid_x, mid_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return distance
    
def aggregate_parameters(image, wrist_angles, spread_angles, motion_y_offset, text_x):
    mean_wrist_angle = np.mean(wrist_angles)
    mean_spread_angle = np.mean(spread_angles)
    variance_spread = np.var(spread_angles)
    max_spread_angle = np.max(spread_angles)
    compactness = mean_wrist_angle / (mean_spread_angle + 1e-5)

    aggregate_text_y_offset = motion_y_offset + 50
    cv2.putText(image, f"Mean Wrist Angle: {mean_wrist_angle:.2f}°", (text_x, aggregate_text_y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(image, f"Mean Spread: {mean_spread_angle:.2f}°", (text_x, aggregate_text_y_offset + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(image, f"Spread Variance: {variance_spread:.2f}", (text_x, aggregate_text_y_offset + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(image, f"Max Spread Angle: {max_spread_angle:.2f}°", (text_x, aggregate_text_y_offset + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(image, f"Compactness: {compactness:.2f}", (text_x, aggregate_text_y_offset + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return mean_wrist_angle, mean_spread_angle, variance_spread, max_spread_angle, compactness
    
def compute_velocity_acc(image, frame_time, hand_idx, previous_hand_positions, wrist, VELOCITY_THRESHOLD, ACCELERATION_THRESHOLD, text_x, motion_y_offset):
    if hand_idx in previous_hand_positions:
        prev_wrist = previous_hand_positions[hand_idx]
        velocity = np.linalg.norm(np.array(wrist) - np.array(prev_wrist)) / frame_time
        acceleration = (velocity - previous_hand_positions.get(f"velocity_{hand_idx}", 0)) / frame_time

        if velocity < VELOCITY_THRESHOLD:
            velocity = 0
        if acceleration < ACCELERATION_THRESHOLD:
            acceleration = 0

        cv2.putText(image, f"V: {velocity:.2f}", (text_x, motion_y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(image, f"A: {acceleration:.2f}", (text_x, motion_y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        previous_hand_positions[f"velocity_{hand_idx}"] = velocity
        
        return velocity, acceleration

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def get_palm_center(landmarks):
    # Simple palm center proxy: average of wrist + base joints of fingers
    indices = [0, 1, 5, 9, 13, 17]
    points = [landmarks[i] for i in indices]
    return np.mean(points, axis=0)

def compute_hand_distances(landmarks):
    """
    Computes key hand distances from MediaPipe landmarks.

    Args:
        landmarks: list of 21 (x, y, z) tuples or np arrays

    Returns:
        list of 10 floats representing key distances
    """
    palm = get_palm_center(landmarks)

    dists = []

    # Fingertip pair distances (5)
    fingertip_pairs = [
        (4, 8),   # Thumb ↔ Index
        (8, 12),  # Index ↔ Middle
        (12, 16), # Middle ↔ Ring
        (16, 20), # Ring ↔ Pinky
        (8, 20),  # Index ↔ Pinky
    ]
    for i, j in fingertip_pairs:
        dists.append(euclidean_distance(landmarks[i], landmarks[j]))

    # Palm center to fingertip distances (5)
    fingertip_indices = [4, 8, 12, 16, 20]
    for idx in fingertip_indices:
        dists.append(euclidean_distance(palm, landmarks[idx]))

    return dists

def draw_inter_fing_distances(image, fd_norms, text_x, base_y_offset=350):
    color = (0, 255, 255)  # Yellowish cyan
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20

    for i, fd in enumerate(fd_norms):
        label = f"FD{i}: {fd:.3f}"
        cv2.putText(image, label, (text_x, base_y_offset + i * line_height), font, font_scale, color, thickness)
        
def calculate_hand_aspect_ratio(hand_landmarks):    
    # Calculate the bounding box for the hand
    min_x = min(hand_landmarks.landmark, key=lambda lm: lm.x).x  # Leftmost
    max_x = max(hand_landmarks.landmark, key=lambda lm: lm.x).x  # Rightmost
    min_y = min(hand_landmarks.landmark, key=lambda lm: lm.y).y  # Topmost
    max_y = max(hand_landmarks.landmark, key=lambda lm: lm.y).y  # Bottommost
    
    # Calculate the width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y
    
    # Return the aspect ratio
    if height == 0:  # Avoid division by zero
        return 0
    return min(width, height) / max(width, height)

def print_har(image, har_norm):
    text_x = 10  # Position where the text will appear
    cv2.putText(image, f"Hand Aspect Ratio: {har_norm:.2f}", (text_x, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def compute_feature_deltas(current_vector, prev_vector=None, prev_prev_vector=None):
    """
    Compute first (Δf) and second (Δ²f) order temporal deltas for a feature vector.

    Parameters:
        current_vector (list or np.ndarray): Current feature vector (1D array).
        prev_vector (list or np.ndarray): Previous feature vector (1D array), or None if not available.
        prev_prev_vector (list or np.ndarray): Feature vector two steps back, or None if not available.

    Returns:
        full_vector (np.ndarray): Concatenated vector [f, Δf, Δ²f] (shape = 3 * len(current_vector))
        delta_1 (np.ndarray): First-order delta (Δf)
        delta_2 (np.ndarray): Second-order delta (Δ²f)
    """
    current_vector = np.array(current_vector)

    if prev_vector is None:
        delta_1 = np.zeros_like(current_vector)
    else:
        prev_vector = np.array(prev_vector)
        delta_1 = current_vector - prev_vector

    if prev_prev_vector is None or prev_vector is None:
        delta_2 = np.zeros_like(current_vector)
    else:
        prev_prev_vector = np.array(prev_prev_vector)
        delta_2 = current_vector - 2 * prev_vector + prev_prev_vector

    full_vector = np.concatenate([current_vector, delta_1, delta_2])
    return full_vector, delta_1, delta_2

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
TIP_IDS = [4, 8, 12, 16, 20]
frames_up_counter = {name: 0 for name in FINGER_NAMES}
activity_score = {name: 0 for name in FINGER_NAMES}
THRESHOLD = 5  # frames needed to count as sustained activity

def reset_finger_activity_tracking():
    global frames_up_counter, activity_score
    frames_up_counter = {name: 0 for name in FINGER_NAMES}
    activity_score = {name: 0 for name in FINGER_NAMES}
    
def detect_active_fingers(fingers, wrist, image=None):
    active_fingers = []

    for name, tip_id in zip(FINGER_NAMES, TIP_IDS):
        tip = np.array(fingers[name][2])
        pip = np.array(fingers[name][1])

        if name == "Thumb":
            is_extended = tip[0] < pip[0]
        else:
            is_extended = tip[1] < pip[1]

        active_fingers.append(1 if is_extended else 0)

        # Activity logic
        if is_extended:
            frames_up_counter[name] += 1
            if frames_up_counter[name] >= THRESHOLD:
                activity_score[name] += 1
                frames_up_counter[name] = 0
        else:
            frames_up_counter[name] = 0

        if image is not None:
            color = (0, 255, 0) if is_extended else (0, 0, 255)
            cv2.circle(image, tuple(map(int, tip)), 6, color, -1)
            cv2.putText(image, name[0], tuple(map(int, tip)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return active_fingers


def extract_finger_state_features(fingers, wrist, image=None, text_x=10, finger_y_offset=120):
    states = detect_active_fingers(fingers, wrist, image)

    states_float = [float(s) for s in states]
    normalized_up = sum(states_float) / 5.0
    states_float.append(normalized_up)

    # Activity vector before normalization
    activity_vector = [float(activity_score[name]) for name in FINGER_NAMES]

    # Min-Max normalization of activity_vector
    min_score = min(activity_vector)
    max_score = max(activity_vector)
    
    # If the max and min are the same, avoid division by zero by setting all values to 0
    if min_score != max_score:
        normalized_activity_vector = [(score - min_score) / (max_score - min_score) for score in activity_vector]
    else:
        normalized_activity_vector = [0.0] * len(activity_vector)  # All values are equal, so set to 0

    # Extend the states_float with the normalized activity vector
    states_float.extend(normalized_activity_vector)

    if image is not None:
        num_up = int(round(normalized_up * 5))
        finger_labels = ["T", "I", "M", "R", "P"]
        finger_states_str = " ".join([f"{label}:{int(state)}" for label, state in zip(finger_labels, states)])
        activity_scores_str = " ".join([f"{label}:{score:.2f}" for label, score in zip(finger_labels, normalized_activity_vector)])

        cv2.putText(image, f"Fingers up: {num_up}/5", (text_x, finger_y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 100), 1)
        cv2.putText(image, finger_states_str, (text_x, finger_y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 100), 1)
        cv2.putText(image, f"Activity: {activity_scores_str}", (text_x, finger_y_offset + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 200, 80), 1)

    return states_float
