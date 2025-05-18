import cv2
import os
import time
import numpy as np
from meshes import *
from hand_aux import *
from process_hands import process_frame_features
from npz_helper import *

npz_file_path = 'Sign language translator/gesture_data.npz'
video_folder = r'C:\\Users\\horia\\Pictures\\Camera Roll'
sequence_length = 20
stride = 5

def extract_label_from_filename(filename):
    # Remove the file extension
    base_filename = os.path.splitext(filename)[0]
    # Extract just the letter (first character) and make it uppercase
    return base_filename[0].upper()

def round_features(features, decimals=4):
    # Check if features are 2D and handle them accordingly
    if isinstance(features, list):
        # Iterate over each feature vector and round each value
        rounded_features = []
        for feature in features:
            if isinstance(feature, (list, np.ndarray)):
                # Round each element of the sublist/array
                rounded_features.append(np.round(feature, decimals=decimals))
            else:
                # Directly round scalar features
                rounded_features.append(round(feature, decimals))
        return rounded_features
    else:
        # If features is not a list, directly round the entire array
        return np.round(features, decimals=decimals)


def filter_empty_data(X, y):
    """Remove empty lists/arrays from X and y."""
    X_filtered = [x for x in X if x]  # Keep only non-empty elements in X
    y_filtered = [y[i] for i in range(len(y)) if X[i]]  # Keep corresponding labels for non-empty X elements
    return X_filtered, y_filtered

def video_feed():
    X_dataset, y_dataset = init_npz_file(npz_file_path)

    # Initialize X_dataset as a list
    if isinstance(X_dataset, np.ndarray):
        X_dataset = X_dataset.tolist()
    if isinstance(y_dataset, np.ndarray):
        y_dataset = y_dataset.tolist()

    for filename in os.listdir(video_folder):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
            continue
        
        label = extract_label_from_filename(filename)
        print(f"\nProcessing file: {filename} | Label: {label}")

        video_path = os.path.join(video_folder, filename)
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            hand_results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Process the frame and extract features
            features = process_frame_features(image, mp_hands, mp_drawing, h, w, hand_results, fps)
            if features is not None:
                # Round the features to 4 decimal places
                features = round_features(features, decimals=4)
                frames.append(features)

            # Show the frame with landmarks in a window named "Hand and Face Tracking"
            cv2.imshow('Hand and Face Tracking', image)

            # Wait for the user to press 'q' to stop processing early if desired
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

        # Slice frames into sequences
        for start in range(0, len(frames) - sequence_length + 1, stride):
            clip = frames[start:start + sequence_length]
            if len(clip) == sequence_length:
                X_dataset.append(clip)
                y_dataset.append(label)

        print(f"Added {len(frames) // stride} samples for label '{label}'")

        # Remove empty data (if any) before saving to the .npz file
        X_dataset, y_dataset = filter_empty_data(X_dataset, y_dataset)

        save_npz_dataset(X_dataset, y_dataset, npz_file_path)
            
    print("\n✅ All videos processed and saved.")

    # Close all OpenCV windows
    cv2.destroyAllWindows()
