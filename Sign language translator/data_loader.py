import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dataset(npz_path, max_sequence_length=None, one_hot=False, test_size=0.2):
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    y = data['y']

    # Convert to list if needed
    if not isinstance(X, list):
        X = X.tolist()

    # Filter out sequences that have any empty frames or are empty
    clean_X, clean_y = [], []
    for seq, label in zip(X, y):
        if not seq:  # Empty sequence
            continue
        if any(len(frame) == 0 for frame in seq):  # Sequence with empty frame
            continue
        clean_X.append(seq)
        clean_y.append(label)

    if len(clean_X) == 0:
        raise ValueError("No valid sequences found in the dataset!")

    # Infer feature size (number of features per frame)
    num_features = len(clean_X[0][0])

    # Ensure all frames have consistent feature count (truncate or pad if needed)
    for i in range(len(clean_X)):
        clean_X[i] = [np.pad(frame, (0, num_features - len(frame))) if len(frame) < num_features else frame[:num_features]
                      for frame in clean_X[i]]

    # Pad sequences to max length
    if max_sequence_length is None:
        max_sequence_length = max(len(seq) for seq in clean_X)

    # Pad sequences to make sure they all have the same length
    X_padded = pad_sequences(clean_X, maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')

    # Reshape X_padded to remove the extra dimension (1)
    X_padded = np.squeeze(X_padded, axis=-2)  # Remove the second-to-last dimension (1)

    # Ensure the shape of X_padded is (samples, sequence_length, features)
    print(f"X_padded shape: {X_padded.shape}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(clean_y)

    if one_hot:
        y_encoded = to_categorical(y_encoded)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_padded, y_encoded, test_size=test_size, stratify=y_encoded)
    
    return X_train, X_val, y_train, y_val, label_encoder, max_sequence_length
