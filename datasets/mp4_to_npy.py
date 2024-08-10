import argparse
import cv2
import json
from numpy import *
import os
import random


"""
Maze data preprocessing

== Video 1 ==
yt-dlp "https://www.youtube.com/watch?v=MHGnSqr9kK8&ab_channel=Dprotp" -S res:256
ffmpeg -i '10 Hours of Windows 3D Maze [MHGnSqr9kK8].webm' -filter:v "fps=30,crop=240:240:30:0,scale=64:64" windows_maze_10h_r64.mp4
ffmpeg -to 35973.73 -i windows_maze_10h_r64_v1.mp4 windows_maze_10h_r64_v1_trimmed.mp4

== Video 2 ==
yt-dlp "https://www.youtube.com/watch?v=Hs5pyyPTzDE&ab_channel=TheBestClassic%26RetroScreensavers" -S res:720
ffmpeg -i '10 Hours of Windows 3D Maze [MHGnSqr9kK8].webm' -filter:v fps=30 windows_maze_10h_r64_v2.mp4
ffmpeg -i windows_maze_10h_r64_v2.mp4 -aspect 300:240 windows_maze_10h_r64_v2_edit.mp4
ffmpeg -i windows_maze_10h_r64_v2_edit.mp4 -filter:v "fps=30,crop=240:240:30:0,scale=64:64" windows_maze_10h_r64_v2_final.mp4

ffmpeg -ss 0.02 -to 71970.02 -i windows_maze_20h_r64.mp4 windows_maze_20h_r64_final1.mp4

ffmpeg -i windows_maze_20h_r64_fps30.mp4 -filter:v "fps=20" windows_maze_20h_r64_fps20.mp4
ffmpeg -ss 00:00:00 -to 18:00:00 -i windows_maze_20h_r64_fps20.mp4 -c copy windows_maze_20h_r64_fps20_train.mp4
ffmpeg -ss 18:00:00 -to 19:59:30 -i windows_maze_20h_r64_fps20.mp4 -c copy windows_maze_20h_r64_fps20_test.mp4
"""

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
