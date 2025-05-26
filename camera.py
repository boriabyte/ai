import numpy as np
import time
import cv2
import os

from meshes import *
from hand_aux import *
from process_hands import *
from npz_helper import *
from hand_aux import *
from preprocess import *

"""
The main function that kicks off preprocessing; camera.py parses a given folder path and supplies
fetched data to process_frame_features() inside process_hands.py. The function returns two feature
vectors that are then saved for each frame of each video to the .npz file in NPZ_FILE_PATH.
Afterwards, the data will be processes for training.

A simple while loop is active while parsing the folder, in which the necessary and relevant functions
are called. Using the functions found in npz_helper.py, the obtained data is stored so the rest of the
pipeline can process it.

In order to better automate the whole gathering process, the corresponding labels to each of the letter
signed in the videos in the folder path are extracted from the name of the videos themselves (A.mp4, A2.mp4).
The label is then obtained by taking only the first letter of the filename and added to the y dataset vector
which is also stored in the .npz file, used as a target for training.
"""

NPZ_FILE_PATH = 'Sign language translator/gesture_data.npz'
VIDEO_FOLDER = 'C:/Users/horia/Pictures/Camera Roll'
SEQ_LENGTH = 20
STRIDE = 5

def video_feed():
    X1_dataset, X2_dataset, y_dataset = init_npz_file(NPZ_FILE_PATH)

    # Conversion to lists for better flexibility and processing power
    if isinstance(X1_dataset, np.ndarray):
        X1_dataset = X1_dataset.tolist()
    if isinstance(X2_dataset, np.ndarray):
        X2_dataset = X2_dataset.tolist()
    if isinstance(y_dataset, np.ndarray):
        y_dataset = y_dataset.tolist()

    # Parse video folder
    for FILENAME in os.listdir(VIDEO_FOLDER):
        if not FILENAME.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        # Extract label name
        label = extract_label_from_filename(FILENAME)
        print(f"\nProcessing file: {FILENAME} | Label: {label}")

        # For every video, finger activity must be reset after storing it for the previous one
        reset_finger_activity_tracking()
        video_path = os.path.join(VIDEO_FOLDER, FILENAME)
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

            # Obtain feature vectors
            features, finger_logic_features = process_frame_features(image, mp_hands, mp_drawing, h, w, hand_results, fps)

            # Only append if features are present and valid
            if features and finger_logic_features:
                main_frames.append(round_features(features[0]))
                logic_frames.append(round_features(finger_logic_features[0]))

        cap.release()

        # Generate sliding window samples
        added = 0
        for start in range(0, len(main_frames) - SEQ_LENGTH + 1, STRIDE):
            main_clip = main_frames[start:start + SEQ_LENGTH]
            logic_clip = logic_frames[start:start + SEQ_LENGTH]

            if len(main_clip) == SEQ_LENGTH and len(logic_clip) == SEQ_LENGTH:
                X1_dataset.append(main_clip)
                X2_dataset.append(logic_clip)
                y_dataset.append(label)
                added += 1

        print(f"Added {added} samples for label '{label}'")

        save_npz_dataset(X1_dataset, X2_dataset, y_dataset, NPZ_FILE_PATH)

    print("\nAll videos processed and saved.")
    cv2.destroyAllWindows()
