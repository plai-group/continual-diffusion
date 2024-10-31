import argparse
import os
import pandas as pd
import shutil
import imageio
import numpy as np
from PIL import Image, ImageDraw

from improved_diffusion.script_util import str2bool


"""
This script collects metrics and optionally videos of multiple runs trained on the same datastream.

Assumes samples are in results/<WANDB_ID>/<EVAL_DIR_SUFFIX>/samples folder.
Output is written to summarized/<OUTPUT_DIR>.

Example Command

python scripts/collect_results.py \
--wandb_ids dhwmkuiq 1fyc5svh 9ewarb2a 79vf3yni piexv54k \
--nicknames auto joint flex50 flex50attentive flex20 \
--eval_dir_suffix=ema_0.9999_500000/hierarchy-3_10_5_50_10 \
--metric_prefixes fvd-500 \
--output_dir=ball \
--video_name=4_1.gif
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_ids', type=str, nargs='+', required=True)
    parser.add_argument('--nicknames', type=str, nargs='+', default=None)
    parser.add_argument('--eval_dir_suffix', type=str, required=True,
                        help="Path to the results directory of an evaluation, excluding 'results/<WANDB ID>'.")
    parser.add_argument('--metric_prefixes', type=str, nargs='+', required=True,
                        help="Multiple prefixes that various metric files may start with.")
    parser.add_argument('--output_dir', type=str, default='ball')
    parser.add_argument('--video_name', type=str, default=None,
                        help="Optional video filename to be collected.")
    return parser.parse_args()


def stack_gifs_with_labels(input_paths, nicknames, output_path):
    # Open the first GIF to get dimensions
    gif_reader = imageio.get_reader(input_paths[0])
    frame_height, frame_width = gif_reader.get_data(0).shape[:2]
    num_frames = len(gif_reader)
    gif_reader.close()

    # Calculate the total height of the stacked frames
    boundary_width = 2
    boundary_color = (255, 255, 255)
    total_width = frame_width + 50
    total_height = frame_height * len(input_paths) + (len(input_paths) - 1) * boundary_width

    # Initialize an array to hold the stacked frames
    stacked_frames = np.zeros((num_frames, total_height, total_width, 3), dtype=np.uint8)
    stacked_frames[:, :, frame_width:, :] = boundary_color

    # Loop through each GIF and stack its frames
    current_height = 0
    for path, nickname in zip(input_paths, nicknames):
        gif_reader = imageio.get_reader(path)
        current_frame_height = current_height+frame_height
        for frame_idx, frame in enumerate(gif_reader):
            stacked_frames[frame_idx, current_height:current_frame_height, :frame_width, :] = frame
            text_frame = stacked_frames[frame_idx, current_height:current_frame_height, frame_width:, :]
            text_frame = add_text_to_image(Image.fromarray(text_frame), nickname, position=(5, 5))
            stacked_frames[frame_idx, current_height:current_frame_height, frame_width:, :] = np.array(text_frame)
            if current_height + frame_height < total_height:
                stacked_frames[frame_idx, current_frame_height:current_frame_height+boundary_width, :, :] = boundary_color
        current_height += frame_height + boundary_width
        gif_reader.close()

    # Write the stacked frames to output GIF
    fps = gif_reader.get_meta_data().get('fps', 5)
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in stacked_frames:
        writer.append_data(frame)
    writer.close()


def add_text_to_image(image, text, font_size=20, position=(10, 10), font_path=None):
    """
    Adds text to an image.

    Args:
        image (PIL.Image.Image): Input image.
        text (str): Text to be added to the image.
        font_size (int): Font size of the text.
        position (tuple): Position (x, y) where the text will be added.
        font_path (str): Path to the font file.

    Returns:
        PIL.Image.Image: Image with added text.
    """
    draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, fill='black')
    return image


def main(args):
    output_dir = os.path.join("summarized", args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    nicknames = args.nicknames if args.nicknames else [f"config_{i}" for i in range(len(args.wandb_ids))]

    result = {'nickname': [], 'wandb': []}

    for id, nickname in zip(args.wandb_ids, nicknames):
        input_dir = os.path.join("results", id, args.eval_dir_suffix)
        result['nickname'].append(nickname)
        result['wandb'].append(id)

        metric_paths = []
        for dirpath, _, filenames in os.walk(input_dir):
            for filename in filenames:
                rel_path = os.path.join(dirpath, filename)
                if not os.path.isfile(rel_path):
                    continue
                for metric_prefix in args.metric_prefixes:
                    if filename.startswith(metric_prefix):
                        metric_paths.append(rel_path)
                        break

        for path in metric_paths:
            metric = os.path.basename(path).split('.')[0]
            try:
                print(f"Reading {path} ({nickname})")
                with open(path, 'r') as f:
                    content = f.read()
            except FileNotFoundError:
                print(f"WARNING - File for {nickname} not found: {path}")
                result[metric].append(float('nan'))
                continue
            if metric in result:
                result[metric].append(float(content))
            else:
                result[metric] = [float(content)]


    print(result)
    df = pd.DataFrame(result)
    out_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(out_path, index=False)
    print(f"saved results to {out_path}.")

    if args.video_name:
        os.makedirs(f"{output_dir}/gifs", exist_ok=True)
        gif_paths, gif_nicknames = [], []
        for nickname, id in zip(nicknames, args.wandb_ids):
            try:
                video_path = os.path.join("results", id, args.eval_dir_suffix, "videos", args.video_name)
                extension = args.video_name.split('.')[-1]
                video_out_path = os.path.join(output_dir, "gifs", f"{nickname}.{extension}")
                shutil.copy(video_path, video_out_path)
            except Exception as e:
                print(f"WARNING - Video copy failed for {nickname} with error: {e}")
                continue
            gif_nicknames.append(nickname)
            gif_paths.append(video_path)
        out_path = f"{output_dir}/summary.gif"
        stack_gifs_with_labels(gif_paths, gif_nicknames, f"{output_dir}/summary.gif")
        print(f"saved gifs to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
