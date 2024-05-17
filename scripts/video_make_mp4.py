import torch as th
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import uuid
import argparse
import json

from improved_diffusion.video_datasets import get_test_dataset
from improved_diffusion.test_util import mark_as_observed, tensor2gif, tensor2mp4
from improved_diffusion.script_util import str2bool


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--add_gt", type=str2bool, default=True)
    parser.add_argument("--do_n", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--obs_length", type=int, default=0,
                        help="Number of observed images. If positive, marks the first obs_length frames in output gifs by a red border.")
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4", "avi"])
    parser.add_argument("--eval_on_train", type=str2bool, default=False)
    args = parser.parse_args()

    samples_prefix = "samples_train" if args.eval_on_train else "samples"
    videos_prefix = "videos_train" if args.eval_on_train else "videos"
    if args.add_gt:
        try:  # Infer T from sampled video
            T = len(np.load(Path(args.eval_dir) / samples_prefix / f"sample_{args.start_idx:04d}-0.npy"))
        except PermissionError:
            T = None
        model_args_path = Path(args.eval_dir) / "model_config.json"
        with open(model_args_path, "r") as f:
            model_args = argparse.Namespace(**json.load(f))
        dataset = get_test_dataset(model_args.dataset, T=T)
        if args.eval_on_train:
            dataset.is_test = False
    out_dir = (Path(args.out_dir) if args.out_dir is not None else Path(args.eval_dir)) / videos_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.do_n}_{args.n_seeds}.{args.format}"

    videos = []
    for data_idx in range(args.start_idx, args.start_idx+args.do_n):
        if args.add_gt:

            gt_drange = [-1, 1]
            gt_video, _ = dataset[data_idx]
            gt_video = (gt_video.numpy() - gt_drange[0]) / (gt_drange[1] - gt_drange[0])  * 255
            gt_video = gt_video.astype(np.uint8)
            mark_as_observed(gt_video)
            videos.append([gt_video])
        else:
            videos.append([])
        seed = 0
        done = 0
        while done < args.n_seeds:
            filename = Path(args.eval_dir) / samples_prefix / f"sample_{data_idx:04d}-{seed}.npy"
            print(filename)
            try:
                video = np.load(filename)
                mark_as_observed(video[:args.obs_length])
                videos[-1].append(video)
                done += 1
            except PermissionError:
                pass
            seed += 1
            assert seed < 100, f'Not enough seeds for idx {data_idx} (found {done} after trying {seed} seeds)'
        videos[-1] = np.concatenate(videos[-1], axis=-2)
    video = np.concatenate(videos, axis=-1)

    random_str = uuid.uuid4()
    if args.format == "gif":
        tensor2gif(th.tensor(video), out_path, drange=[0, 255], random_str=random_str)
    elif args.format == "mp4":
        print(th.tensor(video).shape, th.tensor(video).dtype)
        tensor2mp4(th.tensor(video), out_path, drange=[0, 255], random_str=random_str)
    else:
        raise ValueError(f"Unknown format {args.format}")
    print(f"Saved to {out_path}")