"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.

Sample Command
python scripts/video_sample.py checkpoints/p9lrebju/ema_0.9999_050000.pt --T=50 --stop_index=3 --max_frames=10 --n_obs=5 --sampling_scheme=autoreg --batch_size=1 --eval_on_train=True --sampler=heun-80-inf-0-1-1000-0.002-7-100
"""

import argparse
from operator import is_
import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch as th

from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    str2bool,
)
from improved_diffusion.test_util import get_model_results_path, get_eval_run_identifier, Protect
from improved_diffusion.sampling_schemes import sampling_schemes
from improved_diffusion.video_datasets import get_eval_dataset, eval_dataset_configs
from improved_diffusion.resample import create_named_schedule_sampler


def compute_loss(batch, args, model, diffusion, schedule_sampler, trials=10):
    B, T, *_ = batch.shape
    batch = batch.to(dist_util.dev())

    def get_sub_batch_kwargs_iterator(batch):
        frame_indices_iterator = iter(sampling_schemes['autoreg'](
            video_length=T, num_obs=args.n_obs,
            max_frames=args.max_frames, step_size=args.max_latent_frames,
        ))
        frame_indices_iterator.set_videos(batch)  # ignored for non-adaptive sampling schemes
        for obs_frame_indices, latent_frame_indices in frame_indices_iterator:
            frame_indices = th.cat([th.tensor(obs_frame_indices), th.tensor(latent_frame_indices)], dim=1).long()
            x0 = th.stack([batch[i, fi] for i, fi in enumerate(frame_indices)], dim=0).clone()
            obs_mask = th.cat([
                    th.ones_like(th.tensor(obs_frame_indices)),
                    th.zeros_like(th.tensor(latent_frame_indices))],
                dim=1).view(B, -1, 1, 1, 1).float()
            latent_mask = 1 - obs_mask
            obs_mask, latent_mask, frame_indices = map(lambda t: t.to(x0.device), [obs_mask, latent_mask, frame_indices])
            yield dict(
                frame_indices=frame_indices, obs_mask=obs_mask,
                latent_mask=latent_mask, x0=x0
            )
    loss_parts = []
    n_latent_parts = []
    for model_kwargs in get_sub_batch_kwargs_iterator(batch):
        x0 = model_kwargs['x0']
        latent_mask = model_kwargs['latent_mask']
        loss = []
        for _ in range(trials):
            t, weights = schedule_sampler.sample(x0.shape[0], dist_util.dev())
            loss_trial = diffusion.training_losses(
                model, x0, t, model_kwargs=model_kwargs,
                latent_mask=latent_mask, eval_mask=latent_mask)['loss']
            loss_trial *= weights
            loss.append(loss_trial)
        loss = th.stack(loss).mean(dim=0)
        n_latents = latent_mask.view(latent_mask.shape[:2]).sum(dim=-1)
        loss_parts.append((loss * n_latents).cpu())
        n_latent_parts.append(n_latents.cpu())
    loss_sum = th.stack(loss_parts, dim=0).sum(dim=0)
    cnt = th.stack(n_latent_parts, dim=0).sum(dim=0)
    return loss_sum / cnt


@th.no_grad()
def main(args):
    loss_save_path = Path(args.eval_dir) / f"loss-{args.trials}.txt"
    if loss_save_path.exists():
        loss = np.loadtxt(loss_save_path).squeeze()
        print(f"Losses are already computed: {loss}")
        exit()
    loss_save_path.parent.mkdir(parents=True, exist_ok=True)
    args.indices = list(range(args.start_index, args.stop_index))
    if args.num_sampled_videos is None:
        args.num_sampled_videos = len(args.indices)
    print(f"Sampling for indices {args.start_index} to {args.stop_index}.")

    # Load the checkpoint (state dictionary and config)
    data = dist_util.load_state_dict(args.checkpoint_path, map_location="cpu")
    state_dict = data["state_dict"]
    model_args = data["config"]
    model_args.update({"use_ddim": args.sampler == "ddim",
                       "timestep_respacing": args.timestep_respacing})
    model_args["diffusion_space_kwargs"]["enable_decoding"] = True
    model_args = argparse.Namespace(**model_args)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(model_args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    schedule_sampler = create_named_schedule_sampler(model_args.schedule_sampler, diffusion)
    args.image_size = model_args.image_size
    if args.max_frames is None:
        args.max_frames = model_args.max_frames
    if args.max_latent_frames is None:
        args.max_latent_frames = args.max_frames // 2
    if model_args.diffusion_space == "latent":
        args.clip_denoised = False

    # Load the dataset (to get observations from)
    eval_dataset_args = dict(dataset_name=model_args.dataset, T=args.T, train=args.eval_on_train,
                             eval_dataset_config=args.eval_dataset_config)
    # if args.eval_dataset_config == eval_dataset_configs["default"]:
    if args.eval_dataset_config != eval_dataset_configs["continuous"]:
        spacing_kwargs = dict(n_data=args.num_sampled_videos,
                              frame_range=(args.lower_frame_range, args.upper_frame_range))
        eval_dataset_args["spacing_kwargs"] = spacing_kwargs
    dataset = get_eval_dataset(**eval_dataset_args)
    dataset = th.utils.data.Subset(dataset=dataset, indices=args.indices)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    avg_loss = []
    for batch, _ in tqdm(dataloader):
        loss = compute_loss(
            batch=batch,
            args=args,
            model=model,
            diffusion=diffusion,
            schedule_sampler=schedule_sampler,
            trials=args.trials,
        )
        avg_loss.append(loss)
    avg_loss = th.cat(avg_loss, dim=0).mean().item()
    print(avg_loss)
    np.savetxt(loss_save_path, np.array([avg_loss]))


def create_sampling_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--stop_index", type=int, required=True)
    parser.add_argument("--num_sampled_videos", type=int, default=None,
                        help="Total number of samples (default: args.stop_index-args.start_index)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--n_obs", type=int, default=36, help="Number of observed frames at the beginning of the video. The rest are sampled.")
    parser.add_argument("--T", type=int, default=None, help="Length of the videos. If not specified, it will be inferred from the dataset.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Denoted K in the paper. Maximum number of (observed or latent) frames input to the model at once. Defaults to what the model was trained with.")
    parser.add_argument("--max_latent_frames", type=int, default=None, help="Number of frames to sample in each stage. Defaults to max_frames/2.")
    parser.add_argument("--sampler", type=str, default="heun-80-inf-0-1-1000-0.002-7-50")
    parser.add_argument("--use_ddim", type=str2bool, default=False)
    parser.add_argument("--eval_on_train", type=str2bool, default=False)
    parser.add_argument("--timestep_respacing", type=str, default="")
    parser.add_argument("--clip_denoised", type=str2bool, default=True, help="If true, diffusion model generates data between [-1,1].")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sampled images will have this specific index. Used for sampling multiple videos with the same observations.")
    parser.add_argument("--optimality", type=str, default=None,
                        choices=["linspace-t", "random-t", "linspace-t-force-nearby", "random-t-force-nearby"],
                        help="Type of optimised sampling scheme to use for choosing observed frames. By default uses non-optimized sampling scheme. The optimal indices should be computed before use via video_optimal_schedule.py.")
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")

    parser.add_argument("--eval_dataset_config", type=str, default=eval_dataset_configs["default"], choices=list(eval_dataset_configs.keys()))
    parser.add_argument("--lower_frame_range", type=int, default=0, help="Lower bound of frame index used for SpacedDatasets.")
    parser.add_argument("--upper_frame_range", type=int, default=None, help="Upper bound of frame index used for SpacedDatasets.")
    parser.add_argument("--decode_chunk_size", type=int, default=10)
    parser.add_argument("--trials", type=int, default=10)
    return parser


if __name__ == "__main__":
    parser = create_sampling_parser()
    args = parser.parse_args()
    main(args)
