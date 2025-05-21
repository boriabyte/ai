import argparse
from camera import video_feed
from data_loader import load_dataset
from train import *
from inference import *
import pickle
import os

def run_training():
    npz_path = 'Sign language translator/gesture_data.npz'
    (X1_train, X2_train), (X1_val, X2_val), y_train, y_val, label_encoder, max_seq_len = load_dataset(
        npz_path, max_sequence_length=None, one_hot=False, test_size=0.2, sequence_length=20, stride=5)

    input_shape_1 = (max_seq_len, X1_train.shape[2])  # shape of main features (75)
    input_shape_2 = (max_seq_len, X2_train.shape[2])  # shape of logic features (11)
    num_classes = len(label_encoder.classes_)

    print("X1_train shape:", X1_train.shape)
    print("X2_train shape:", X2_train.shape)

    model, history = train_model(
        (X1_train, X2_train), (X1_val, X2_val), y_train, y_val,
        input_shape_1=input_shape_1,
        input_shape_2=input_shape_2,
        num_classes=num_classes,
        model_save_path='saved_model/refined_model.keras',
        label_encoder_path='saved_model/label_encoder.pkl', 
        batch_size=32,
        epochs=50,
        lr=1e-4
    )

    plot_training_history(history, save_dir='training_plots')

    os.makedirs('saved_model', exist_ok=True)
    with open('saved_model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)


def main():
    parser = argparse.ArgumentParser(description='Sign Language System')
    parser.add_argument('--mode', choices=['record', 'train', 'run'], required=True,
                        help='Select mode: record from camera or train model')

    args = parser.parse_args()

    if args.mode == 'record':
        video_feed()
    elif args.mode == 'train':
        run_training()
    elif args.mode == 'run':
        run_live_inference()

if __name__ == '__main__':
    main()