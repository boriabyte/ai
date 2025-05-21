import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Drawing specifications
hand_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

# Init meshes for hands and face
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)