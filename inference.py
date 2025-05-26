from collections import deque
import mediapipe as mp
import numpy as np
import pickle
import cv2

from keras.models import load_model                                                                                                                         # type: ignore

from inference_feature_extraction import extract_features_for_inference
from model import AttentionLayer
from hand_aux import *
from meshes import *

"""
inference.py is the final pipeline of the Sign Language Translator system.

Inference needs to actively predict new data, in real time, using unseen video feed.

The function is basically the same as the camera.py file, minus the parsing of the pre-recorded videos for gathering data.
While camera.py functions in headless mode for less overhead & less resource intensive processing, inference.py needs to
actively use the camera to test the compiled model.
"""

"""
SEQ_LENGTH is synonymous to the number of frames per sample, each label having associated 1400-1600 samples;
20 was a good value in order to have as many training samples as possible, considering the relatively limited dataset. 
Furthermore, each frame has two feature vectors.
"""

SEQ_LENGTH = 20
MODEL_PATH = "Sign language translator/saved_model/refined_model.keras"
LABEL_ENCODER_PATH = "Sign langauage translator/saved_model/label_encoder.pkl"

def load_label_encoder(path):
    """
    Label encoder loading, meant for loading an encoder - after which a decoder - whose purpose is to encode the label
    to parsable data. For example, A is encoded to an integer (for instance, 13, after shuffling).
    
    It will open the saved encoder, stored after training.
    """
    
    with open(path, 'rb') as f:
        le = pickle.load(f)
    return dict(enumerate(le.classes_))

def decode_prediction(pred, label_map):
    """
    The prediction needs to be decoded, just like it is encoded.
    """
    
    return label_map.get(np.argmax(pred, axis=-1)[0], "Unknown")

def run_live_inference():
    """
    Main function for inference; this focuses on running the associated webcam of the local station and using the
    compiled model to use for prediction on new, unseen data - effectively utilizing the application.
    """
    
    cap = cv2.VideoCapture(0)

    # Loading saved model; custom_objects tells the function that it contains attention heads
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
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Similar to the principle for data gathering in camera.py
        main_feat, logic_feat = extract_features_for_inference(image, mp_hands, mp_drawing, h, w, results, fps)

        if main_feat and logic_feat:
            vec_main = np.asarray(main_feat[0], dtype=np.float32)
            vec_logic = np.asarray(logic_feat[0], dtype=np.float32)

            if vec_main.shape[0] == 93 and vec_logic.shape[0] == 14:
                feature_buffer.append(vec_main)
                logic_buffer.append(vec_logic)

        if len(feature_buffer) == SEQ_LENGTH:
            input_main = np.expand_dims(np.array(feature_buffer), axis=0)
            input_logic = np.expand_dims(np.array(logic_buffer), axis=0)

            pred = model.predict([input_main, input_logic], verbose=0)
            last_prediction = decode_prediction(pred, label_map)
            
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
