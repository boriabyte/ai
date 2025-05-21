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
        return float('nan')  # Invalid angle due to zero vector

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to prevent domain errors
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

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

# Function to determine hand orientation based on wrist and middle finger MCP
def get_hand_orientation(yaw, pitch, roll):
    # Convert angles to radians if they are in degrees
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    
    # Create a unit vector that represents the hand's orientation in 3D space
    x = np.cos(pitch) * np.cos(yaw)
    y = np.cos(pitch) * np.sin(yaw)
    z = np.sin(pitch)

    # Reference vector (e.g., pointing straight up in the z-direction)
    reference_vector = np.array([0, 0, 1])

    # Dot product to get the projection of the hand orientation onto the reference vector
    orientation = np.dot([x, y, z], reference_vector)

    return orientation  # This gives a scalar between -1 and 1

def det_handedness(hand_results, hand_idx, w):
    handedness = hand_results.multi_handedness[hand_idx].classification[0].label
    handedness = "Left" if handedness == "Left" else "Right"
    handedness = [1,0] if handedness == "Left" else [0,1]
    text_x = 20 if handedness == [1,0] else (w - 250)
    
    return handedness, text_x

def print_handedness(hand_landmarks, image, text_x, text_y_offset_left, text_y_offset_right, fingers, wrist, orientation, draw=True):
    fist_opennness = calculate_openness(hand_landmarks)

    # Check for NaN in openness degree
    if np.isnan(fist_opennness):
        fist_opennness = 0  # Default value if NaN is encountered

    cv2.putText(image, f"Openness degree: {fist_opennness}", (text_x, text_y_offset_right),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    text_y_offset_right += 30
    cv2.putText(image, f"Orientation agg. value: {orientation}", (text_x, text_y_offset_right),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
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

def calculate_yaw_pitch_roll(landmarks_3d):
    # Use wrist (0), index MCP (5), pinky MCP (17)
    wrist = np.array(landmarks_3d[0])
    index_mcp = np.array(landmarks_3d[5])
    pinky_mcp = np.array(landmarks_3d[17])

    # Vector from wrist to index and wrist to pinky
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist

    # Hand normal vector (perpendicular to the palm)
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)

    # Forward vector (palm facing direction)
    forward = (index_mcp + pinky_mcp)/2 - wrist
    forward /= np.linalg.norm(forward)

    # Yaw: rotation around Y-axis (based on X-Z plane)
    yaw = np.arctan2(forward[0], forward[2])

    # Pitch: rotation around X-axis (based on Y-Z plane)
    pitch = np.arcsin(forward[1])

    # Roll: rotation around Z-axis (based on X-Y plane)
    roll = np.arctan2(normal[1], normal[0])

    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

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
