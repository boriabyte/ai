import time
import numpy as np
import cv2
import pickle
from collections import deque
from keras.models import load_model
import mediapipe as mp
from decoder import *  # Assuming decode_prediction is in decoder.py
from utils import *  # Assuming utilities are in utils.py
from hand_aux import *  # Assuming hand_aux contains necessary hand functions
from inference_feature_extraction import * 
from collections import defaultdict 
from model import AttentionLayer

# Store previous hand data
previous_hand_positions = defaultdict(lambda: None)

# Running statistics for normalization
features_to_normalize = {
    'elapsed': [], 'crv': [], 'mwa': [], 'msa': [], 'sv': [],
    'mxsa': [], 'cmpct': [], 'acc': [], 'vcty': []
}

VELOCITY_THRESHOLD = 2.0
ACCELERATION_THRESHOLD = 2.0

def load_label_encoder(path='saved_model/label_encoder.pkl'):
    with open(path, 'rb') as f:
        label_encoder = pickle.load(f)
    int_to_label = dict(enumerate(label_encoder.classes_))
    return int_to_label

def decode_prediction(prediction, int_to_label):
    predicted_class = np.argmax(prediction, axis=-1)
    return int_to_label.get(predicted_class[0], 'Unknown')


def run_live_inference():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Load the trained model and label encoder
    model = load_model('saved_model/best_model.keras', custom_objects={'AttentionLayer': AttentionLayer})
    int_to_label = load_label_encoder()

    feature_buffer = deque(maxlen=100)
    fps = 30  # You can adjust based on your system

    last_prediction = ""  # For displaying on screen

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        features = extract_features_for_inference(frame, mp_hands, mp_drawing, h, w, results, 0, fps)

        if features is not None and np.all(np.isfinite(features)) and np.any(features):
            feature_buffer.append(features)

        if len(feature_buffer) == 100:
            buffer_array = np.array(feature_buffer)
            velocities = buffer_array[:, 8]  # Assuming velocity is index 8

            if np.mean(velocities) > 0.5:
                input_data = np.expand_dims(buffer_array, axis=0)  # (1, 100, 19) for 19 features
                prediction = model.predict(input_data, verbose=0)
                predicted_label = decode_prediction(prediction, int_to_label)
                last_prediction = predicted_label  # Update what to show on screen

        # Draw the prediction text on screen
        cv2.putText(frame, f"Prediction: {last_prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
