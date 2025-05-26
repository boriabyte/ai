def load_dataset(npz_path, max_sequence_length=None, one_hot=None, test_size=None, sequence_length=None, stride=None):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    if sequence_length is None or stride is None:
        raise ValueError("Both `sequence_length` and `stride` must be provided.")

    data = np.load(npz_path, allow_pickle=True)
    X1 = data['X1'].tolist()
    X2 = data['X2'].tolist()
    y = data['y']

    # Remove samples with invalid frames
    clean_X1, clean_X2, clean_y = [], [], []
    for x1_seq, x2_seq, label in zip(X1, X2, y):
        if not x1_seq or not x2_seq:
            continue
        if any(len(frame) == 0 for frame in x1_seq) or any(len(frame) == 0 for frame in x2_seq):
            continue
        clean_X1.append(x1_seq)
        clean_X2.append(x2_seq)
        clean_y.append(label)

    if len(clean_X1) == 0:
        raise ValueError("No valid sequences found in the dataset!")

    # Infer feature dimensions
    num_features1 = len(clean_X1[0][0])
    num_features2 = len(clean_X2[0][0])

    # Optional: padding individual frames if needed (not necessary in your case)
    for i in range(len(clean_X1)):
        clean_X1[i] = [np.pad(f, (0, num_features1 - len(f))) if len(f) < num_features1 else f[:num_features1] for f in clean_X1[i]]
        clean_X2[i] = [np.pad(f, (0, num_features2 - len(f))) if len(f) < num_features2 else f[:num_features2] for f in clean_X2[i]]

    # Sliding window over sequences
    seqs1, seqs2, labels = [], [], []
    for s1, s2, label in zip(clean_X1, clean_X2, clean_y):
        if len(s1) < sequence_length:
            continue  # skip short sequences
        for start in range(0, len(s1) - sequence_length + 1, stride):
            seqs1.append(s1[start:start + sequence_length])
            seqs2.append(s2[start:start + sequence_length])
            labels.append(label)

    if max_sequence_length is None:
        max_sequence_length = sequence_length

    # Pad sequences
    X1_padded = pad_sequences(seqs1, maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')
    X2_padded = pad_sequences(seqs2, maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')

    # No squeeze!
    X1_padded = np.array(X1_padded, dtype='float32')
    X2_padded = np.array(X2_padded, dtype='float32')

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    if one_hot:
        y_encoded = to_categorical(y_encoded)

    # Train/validation split
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        X1_padded, X2_padded, y_encoded, test_size=test_size, stratify=y_encoded
    )

    return (X1_train, X2_train), (X1_val, X2_val), y_train, y_val, label_encoder, max_sequence_length
