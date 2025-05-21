import cv2
import numpy as np
import pickle
from collections import deque
import mediapipe as mp
from keras.models import load_model
from hand_aux import *
from inference_feature_extraction import extract_features_for_inference
from model import AttentionLayer

# === Constants ===
SEQ_LENGTH = 20
MODEL_PATH = "saved_model/refined_model.keras"
LABEL_ENCODER_PATH = "saved_model/label_encoder.pkl"

# === Load model and label encoder ===
def load_label_encoder(path):
    with open(path, 'rb') as f:
        le = pickle.load(f)
    return dict(enumerate(le.classes_))

def decode_prediction(pred, label_map):
    return label_map.get(np.argmax(pred, axis=-1)[0], "Unknown")

# === Main Live Inference Function ===
def run_live_inference():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    model = load_model(MODEL_PATH, custom_objects={'AttentionLayer': AttentionLayer})
    label_map = load_label_encoder(LABEL_ENCODER_PATH)

    feature_buffer = deque(maxlen=SEQ_LENGTH)
    logic_buffer = deque(maxlen=SEQ_LENGTH)
    last_prediction = ""

    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image)

        main_feat, logic_feat = extract_features_for_inference(image, mp_hands, mp_draw, h, w, results, fps)

        if main_feat and logic_feat:
            vec_main = np.asarray(main_feat[0], dtype=np.float32)
            vec_logic = np.asarray(logic_feat[0], dtype=np.float32)

            if vec_main.shape[0] == 75 and vec_logic.shape[0] == 11:
                feature_buffer.append(vec_main)
                logic_buffer.append(vec_logic)

        if len(feature_buffer) == SEQ_LENGTH:
            input_main = np.expand_dims(np.array(feature_buffer), axis=0)  # (1, 20, 75)
            input_logic = np.expand_dims(np.array(logic_buffer), axis=0)  # (1, 20, 11)

            pred = model.predict([input_main, input_logic], verbose=0)
            last_prediction = decode_prediction(pred, label_map)

            # Optional: slide or clear
            feature_buffer.clear()
            logic_buffer.clear()

        cv2.putText(frame, f"Prediction: {last_prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Live Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_inference()
