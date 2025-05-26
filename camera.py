import cv2
import os
import time
import numpy as np
from meshes import *
from hand_aux import *
from process_hands import *
from npz_helper import *
from hand_aux import *

npz_file_path = 'Sign language translator/gesture_data.npz'
video_folder = r'C:\\Users\\horia\\Pictures\\Camera Roll'
sequence_length = 20
stride = 5

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

def video_feed():
    X1_dataset, X2_dataset, y_dataset = init_npz_file(npz_file_path)

    if isinstance(X1_dataset, np.ndarray):
        X1_dataset = X1_dataset.tolist()
    if isinstance(X2_dataset, np.ndarray):
        X2_dataset = X2_dataset.tolist()
    if isinstance(y_dataset, np.ndarray):
        y_dataset = y_dataset.tolist()

    for filename in os.listdir(video_folder):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        label = extract_label_from_filename(filename)
        print(f"\nProcessing file: {filename} | Label: {label}")

        reset_finger_activity_tracking()
        video_path = os.path.join(video_folder, filename)
        cap = cv2.VideoCapture(video_path)

        main_frames, logic_frames = [], []
        fps = cap.get(cv2.CAP_PROP_FPS)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            features, finger_logic_features = process_frame_features(image, mp_hands, mp_drawing, h, w, hand_results, fps)

            # Only append if features are present and valid
            if features and finger_logic_features:
                # Append only the first hand's features
                main_frames.append(round_features(features[0]))
                logic_frames.append(round_features(finger_logic_features[0]))

        cap.release()

        # Generate sliding window samples
        added = 0
        for start in range(0, len(main_frames) - sequence_length + 1, stride):
            main_clip = main_frames[start:start + sequence_length]
            logic_clip = logic_frames[start:start + sequence_length]

            if len(main_clip) == sequence_length and len(logic_clip) == sequence_length:
                X1_dataset.append(main_clip)
                X2_dataset.append(logic_clip)
                y_dataset.append(label)
                added += 1

        print(f"✅ Added {added} samples for label '{label}'")

        save_npz_dataset(X1_dataset, X2_dataset, y_dataset, npz_file_path)

    print("\n🎉 All videos processed and saved.")
    cv2.destroyAllWindows()
