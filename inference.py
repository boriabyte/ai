import cv2
import numpy as np
import pickle
from meshes import *
from collections import deque, defaultdict
from keras.models import load_model
import mediapipe as mp
from hand_aux import *  
from inference_feature_extraction import extract_features_for_inference
from model import AttentionLayer

# === Constants ===
VELOCITY_THRESHOLD = 0.5  # Lowered for better responsiveness
SEQ_LENGTH = 20
FPS = 30

# === Label decoder ===
def load_label_encoder(path='saved_model/label_encoder.pkl'):
    with open(path, 'rb') as f:
        label_encoder = pickle.load(f)
    return dict(enumerate(label_encoder.classes_))

def decode_prediction(prediction, int_to_label):
    predicted_class = np.argmax(prediction, axis=-1)
    return int_to_label.get(predicted_class[0], 'Unknown')

# === Real-time Inference Function ===
def run_live_inference():
    cap = cv2.VideoCapture(0)
    mp_hands_det = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    model = load_model('saved_model/refined_model.keras', custom_objects={'AttentionLayer': AttentionLayer})
    int_to_label = load_label_encoder()

    feature_buffer = deque(maxlen=SEQ_LENGTH)
    last_prediction = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands_det.process(frame_rgb)

        features = extract_features_for_inference(frame, mp_hands, mp_drawing, h, w, results, FPS)

        if features and isinstance(features, (list, np.ndarray)):
            vec = np.asarray(features[0], dtype=np.float32)
            if vec.shape == (75,) and np.all(np.isfinite(vec)):
                feature_buffer.append(vec)

        # Inference when buffer is full
        if len(feature_buffer) == SEQ_LENGTH:
            input_data = np.expand_dims(np.array(feature_buffer), axis=0)  # shape: (1, 20, 75)
            prediction = model.predict(input_data, verbose=0)
            predicted_label = decode_prediction(prediction, int_to_label)
            last_prediction = predicted_label

            # Optional: clear or partially slide the buffer
            feature_buffer.clear()  # or keep last few frames for smoother sliding

        # === Display prediction ===
        cv2.putText(frame, f"Prediction: {last_prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Live Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
