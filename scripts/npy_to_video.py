import argparse
import numpy as np
import sys
import torch as th
import uuid

sys.path.append('..')
from improved_diffusion.test_util import tensor2mp4

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", type=str, default='datasets/ball/train/0.npy')
parser.add_argument("--save_path", type=str, default='output.mp4')
args = parser.parse_args()

# Load the .npy file
video_frames = np.load(args.in_path)

random_str = uuid.uuid4()
video_frames = (video_frames * 255).astype(np.uint8)
tensor2mp4(th.tensor(video_frames).permute(0, 3, 1, 2), args.save_path, drange=[0, 255], random_str=random_str)
# tensor2mp4(th.tensor(video_frames), args.save_path, drange=[0, 255], random_str=random_str)
