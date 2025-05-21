import numpy as np 

def decode_prediction(prediction, int_to_label):
    import numpy as np
    predicted_class = np.argmax(prediction, axis=-1)
    return int_to_label.get(predicted_class[0], 'Unknown')