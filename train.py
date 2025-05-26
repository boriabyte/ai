import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pickle
import os

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau                                                                        # type: ignore
from tensorflow.keras.optimizers import Adam                                                                                                                    # type: ignore

from model import build_dual_input_model
from data_loader import load_dataset

"""
train.py will load the data inside of gesture_data.npz 
and run it through the usual training pipeline using
load_dataset() and train_model()

80/20 split on training and validation
"""

NPZ_PATH = 'Sign language translator/gesture_data.npz'
MODEL_SAVE_PATH = 'Sign language translator/saved_model/refined_model.keras'
LABEL_ENCODER_PATH = 'Sign language translator/saved_model/label_encoder.pkl'
TRAINING_PLOTS_PATH = 'Sign language translator/training_plots'

def train_model(train_data, val_data, y_train, y_val, input_shape_1, input_shape_2, num_classes,
                model_save_path=None,
                label_encoder_path=None,
                batch_size=None, epochs=None, lr=None):
    """
    Loading the saved model from path_to_model/name_of_model.keras,
    including output encoder for transforming the integer encoded
    output into the associated value

    Batch size for model training will be 64 for moderation
    50 epochs reasonable and yielded great results (see training_plots)

    Learning rate 1*10^(-3) is also reasonable and proven to yield good results.
    Change for fine-tuning.
    """
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)

    joblib.dump(label_encoder, label_encoder_path)

    # Log input info
    print("\nTraining data info:")
    print(f"X1 shape (full features): {train_data[0].shape}")                                  
    print(f"X2 shape (logic features): {train_data[1].shape}")  
    print(f"y_train shape: {y_train.shape}")
    print(f"Input shape 1 (for model): {input_shape_1}")
    print(f"Input shape 2 (for model): {input_shape_2}")
    print(f"Number of classes: {num_classes}\n")

    model = build_dual_input_model(input_shape_1, input_shape_2, num_classes)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    '''
    Will stop training after 5 epochs where val_loss didn't improve from previous epochs
    The best model will be the one with the best val_loss (monitor='val_loss')
    The learning rate will also be fine-tuned by monitoring val_loss, to a minimum of 1*10^(-6)
    '''
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
    lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        [train_data[0], train_data[1]], y_train,
        validation_data=([val_data[0], val_data[1]], y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stop, lr_plateau, checkpoint],
        verbose=1
    )

    return model, history

def plot_training_history(history, save_dir=None):
    """
    Plots training and validation accuracy and loss for performance monitoring and eventual fine-tuning
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_history.png')) # Saved to path defined in save_dir as training_history.png
    plt.show()

def run_training():
    (X1_train, X2_train), (X1_val, X2_val), y_train, y_val, label_encoder, max_seq_len = load_dataset(
        NPZ_PATH, max_sequence_length=None, one_hot=False, test_size=0.2, sequence_length=20, stride=5)

    input_shape_1 = (max_seq_len, X1_train.shape[2])  # shape of main features
    input_shape_2 = (max_seq_len, X2_train.shape[2])  # shape of logic features
    num_classes = len(label_encoder.classes_)

    print("X1_train shape:", X1_train.shape)
    print("X2_train shape:", X2_train.shape)

    model, history = train_model(
        (X1_train, X2_train), (X1_val, X2_val), y_train, y_val,
        input_shape_1=input_shape_1,
        input_shape_2=input_shape_2,
        num_classes=num_classes,
        model_save_path=MODEL_SAVE_PATH,
        label_encoder_path=LABEL_ENCODER_PATH, 
        batch_size=32,
        epochs=50,
        lr=1e-4
    )

    plot_training_history(history, save_dir=TRAINING_PLOTS_PATH)

    os.makedirs('saved_model', exist_ok=True)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)