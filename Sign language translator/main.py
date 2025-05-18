import argparse
from camera import video_feed
from data_loader import load_dataset
from train import *
from inference import *
import pickle
import os

def run_training():
    npz_path = 'Sign language translator/gesture_data.npz'
    X_train, X_val, y_train, y_val, label_encoder, max_seq_len = load_dataset(
        npz_path, max_sequence_length=500, one_hot=False)

    input_shape = (max_seq_len, X_train.shape[2])
    num_classes = len(label_encoder.classes_)  # corrected

    model, history = train_model(
        X_train, X_val, y_train, y_val,
        input_shape=input_shape,
        num_classes=num_classes,
        model_save_path='saved_model/best_model.keras',
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