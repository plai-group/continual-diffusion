import argparse
import cv2
import json
from numpy import *
import os
import random


def video_to_npy(video_path, save_dir, resolution=(64, 64), chunk_size=10000):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Failed to open the video file.")

    n_frames = 0
    chunk_idx = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to desired dimensions
        frame = cv2.resize(frame, resolution)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255
        frames.append(frame)
        n_frames += 1

        if len(frames) == chunk_size:
            with open(f"{save_dir}/{chunk_idx}.npy", "wb") as f:
                save(f, stack(frames))
            frames = []
            chunk_idx += 1
    cap.release()
    return n_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_video_path", type=str, default="datasets/continual_minecraft/MC16min.mp4",
                        help="video stream to train on.")
    parser.add_argument("--test_video_path", type=str, default="datasets/continual_minecraft/MC16min.mp4",
                        help="video stream to test on.")
    parser.add_argument("--save_dir", type=str, default="datasets/continual_minecraft")
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    train_path, test_path = f"{args.save_dir}/train", f"{args.save_dir}/test"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    n_frames_train = video_to_npy(video_path=args.train_video_path, resolution=(args.resolution, args.resolution),
                                  chunk_size=args.chunk_size, save_dir=train_path)
    n_frames_test = video_to_npy(video_path=args.test_video_path, resolution=(args.resolution, args.resolution),
                                 chunk_size=args.chunk_size, save_dir=test_path)
    args.T_total = n_frames_train

    with open(f"{args.save_dir}/config.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)
