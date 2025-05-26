import argparse
from train import *
from inference import *
import pickle
import os

from camera import video_feed

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