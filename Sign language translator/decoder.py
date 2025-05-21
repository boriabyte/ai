import numpy as np 

def decode_prediction(prediction):
    # Assuming your model output is a probability vector (e.g., softmax output)
    predicted_class = np.argmax(prediction, axis=-1)  # This gives the class index
    # Map the class index to the corresponding label (e.g., 'A', 'B', etc.)
    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'X', 23: 'Y', 24: 'Z'}
    return label_map.get(predicted_class[0], 'Unknown')  # Return 'Unknown' if the class is not found