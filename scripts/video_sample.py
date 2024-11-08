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


@th.no_grad()
def sample_video(args, model, diffusion, batch, just_get_indices=False):
    """
    batch has a shape of BxTxCxHxW where
    B: batch size
    T: video length
    CxWxH: image size
    """
    B, T, C, H, W = batch.shape
    samples = th.zeros_like(batch)
    samples[:, :args.n_obs] = batch[:, :args.n_obs]

    # Observation-level samples
    visualized_samples = None

    # Intilise sampling scheme
    optimal_schedule_path = None if args.optimality is None else args.eval_dir / "optimal_schedule.pt"
    frame_indices_iterator = iter(sampling_schemes[args.sampling_scheme](
        video_length=T, num_obs=args.n_obs,
        max_frames=args.max_frames, step_size=args.max_latent_frames,
        optimal_schedule_path=optimal_schedule_path,
    ))

    indices_used = []
    while True:
        frame_indices_iterator.set_videos(samples.to(args.device))  # ignored for non-adaptive sampling schemes
        try:
            obs_frame_indices, latent_frame_indices = next(frame_indices_iterator)
        except StopIteration:
            break
        print(f"Conditioning on {sorted(obs_frame_indices)} frames, predicting {sorted(latent_frame_indices)}.")
        # Prepare network's input
        frame_indices = th.cat([th.tensor(obs_frame_indices), th.tensor(latent_frame_indices)], dim=1).long()
        x0 = th.stack([samples[i, fi] for i, fi in enumerate(frame_indices)], dim=0).clone()
        obs_mask = th.cat([th.ones_like(th.tensor(obs_frame_indices)),
                              th.zeros_like(th.tensor(latent_frame_indices))], dim=1).view(B, -1, 1, 1, 1).float()
        latent_mask = 1 - obs_mask
        if just_get_indices:
            local_samples = th.stack([batch[i, ind] for i, ind in enumerate(frame_indices)])
        else:
            # Prepare masks
            print(f"{'Frame indices':20}: {frame_indices[0].cpu().numpy()}.")
            print(f"{'Observation mask':20}: {obs_mask[0].cpu().int().numpy().squeeze()}")
            print(f"{'Latent mask':20}: {latent_mask[0].cpu().int().numpy().squeeze()}")
            print("-" * 40)
            # Move tensors to the correct device
            x0, obs_mask, latent_mask, frame_indices = (t.to(args.device) for t in [x0, obs_mask, latent_mask, frame_indices])
            # Run the network
            sampler, *sampler_args = args.sampler.split('-')
            if sampler == "ddpm":
                sample_func = diffusion.p_sample_loop
                sampler_kwargs = {}
            elif sampler == "ddim":
                sample_func = diffusion.ddim_sample_loop
                sampler_kwargs = {}
            elif sampler == "heun":
                sample_func = diffusion.heun_sample
                (S_churn, S_max, S_min, S_noise,
                 sigma_max, sigma_min, rho, num_steps) = sampler_args
                sampler_kwargs = dict(
                    S_churn=float(S_churn), S_max=float(S_max),
                    S_min=float(S_min), S_noise=float(S_noise),
                    sigma_max=float(sigma_max), sigma_min=float(sigma_min),
                    rho=int(rho), num_steps=int(num_steps)
                )
            print('sample_func', sample_func)
            local_samples, _ = sample_func(
                model, x0.shape, clip_denoised=args.clip_denoised,
                model_kwargs=dict(frame_indices=frame_indices,
                                  x0=x0,
                                  obs_mask=obs_mask,
                                  latent_mask=latent_mask),
                latent_mask=latent_mask,
                return_attn_weights=False,
                decode_chunk_size=args.decode_chunk_size,
                **sampler_kwargs,
            )

            if isinstance(local_samples, tuple):
                # Edge case: Encoded sample
                visualized_local_samples = local_samples[1]
                local_samples = local_samples[0].to(x0.dtype)
            else:
                # No encoded samples
                visualized_local_samples = local_samples.to(x0.dtype)

            if visualized_samples is None:
                if local_samples.shape == visualized_local_samples.shape:
                    decoded_obs_batch = batch[:, :args.n_obs].to(batch.device)
                else:
                    decoded_obs_batch = diffusion.decode(batch[:, :args.n_obs].to(th.float16),
                                                         chunk_size=args.decode_chunk_size).to(batch.device)
                C_d, H_d, W_d = decoded_obs_batch.shape[2:]
                visualized_samples = th.zeros(B, T, *decoded_obs_batch.shape[2:]).to(batch.device).to(batch.dtype)
                visualized_samples[:, :args.n_obs] = decoded_obs_batch

            print('local samples', local_samples.min(), local_samples.max())

        # Fill in the generated frames
        for i, li in enumerate(latent_frame_indices):
            samples[i, li] = local_samples[i, -len(li):].cpu().to(samples.dtype)
            visualized_samples[i, li] = visualized_local_samples[i, -len(li):].cpu().to(visualized_samples.dtype)
        indices_used.append((obs_frame_indices, latent_frame_indices))
    return visualized_samples, indices_used


def main(args, model, diffusion, dataset, samples_prefix):
    not_done = list(args.indices)
    while len(not_done) > 0:
        batch_indices = not_done[:args.batch_size]
        not_done = not_done[args.batch_size:]
        output_filenames = [args.eval_dir / samples_prefix / f"sample_{i:04d}-{args.sample_idx}.npy" for i in batch_indices]
        todo = [not p.exists() for p in output_filenames]
        if not any(todo):
            print(f"Nothing to do for the batches {min(batch_indices)} - {max(batch_indices)}, sample #{args.sample_idx}.")
            continue
        batch = th.stack([dataset[i][0] for i in batch_indices])
        samples, _ = sample_video(args, model, diffusion, batch)
        drange = [-1, 1]
        samples = (samples.clamp(*drange).numpy() - drange[0]) / (drange[1] - drange[0]) * 255
        samples = samples.astype(np.uint8)
        for i in range(len(batch_indices)):
            if todo[i]:
                np.save(output_filenames[i], samples[i])
                print(f"*** Saved {output_filenames[i]} ***")


def main_outer(args):
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
    args.image_size = model_args.image_size
    if args.max_frames is None:
        args.max_frames = model_args.max_frames
    if args.max_latent_frames is None:
        args.max_latent_frames = args.max_frames // 2 
    if model_args.diffusion_space == "latent":
        args.clip_denoised = False

    # Prepare samples directory
    args.eval_dir = get_model_results_path(args) / get_eval_run_identifier(args)
    samples_prefix = "samples"

    # Load the dataset (to get observations from)
    eval_dataset_args = dict(dataset_name=model_args.dataset, T=args.T, train=args.eval_on_train,
                             eval_dataset_config=args.eval_dataset_config,
                             spacing_kwargs=dict(n_data=args.num_sampled_videos),
                             frame_range=(args.lower_frame_range, args.upper_frame_range))
    dataset = get_eval_dataset(**eval_dataset_args)

    (args.eval_dir / samples_prefix).mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to {args.eval_dir / samples_prefix}")

    # Store model configs in a JSON file
    json_path = args.eval_dir / "model_config.json"
    if not json_path.exists():
        with Protect(json_path): # avoids race conditions
            to_save = vars(model_args)
            with open(json_path, "w") as f:
                json.dump(to_save, f, indent=4)
        print(f"Saved model config at {json_path}")

    main(args, model, diffusion, dataset, samples_prefix)


def create_sampling_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--stop_index", type=int, required=True)
    parser.add_argument("--num_sampled_videos", type=int, default=None,
                        help="Total number of samples (default: args.stop_index-args.start_index)")
    parser.add_argument("--sampling_scheme", required=True, choices=sampling_schemes.keys())
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
    return parser


if __name__ == "__main__":
    parser = create_sampling_parser()
    args = parser.parse_args()
    main_outer(args)
