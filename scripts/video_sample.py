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
from improved_diffusion.video_datasets import get_test_dataset, get_vis_dataset


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
        x0 = diffusion.encode(x0)
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
                **sampler_kwargs,
            )

            if isinstance(local_samples, tuple):
                # Edge case: Encoded sample
                visualized_local_samples = local_samples[1]
                local_samples = local_samples[0]
            else:
                # No encoded samples
                visualized_local_samples = local_samples

            if visualized_samples is None:
                if local_samples.shape == visualized_local_samples.shape:
                    decoded_obs_batch = batch[:, args.n_obs].to(batch.device)
                else:
                    decoded_obs_batch = diffusion.decode(batch[:, :args.n_obs].to(th.float16)).to(batch.device)
                C_d, H_d, W_d = decoded_obs_batch.shape[2:]
                visualized_samples = th.zeros(B, T, *decoded_obs_batch.shape[2:]).to(batch.device).to(batch.dtype)
                visualized_samples[:, :args.n_obs] = decoded_obs_batch

            print('local samples', local_samples.min(), local_samples.max())

        # Fill in the generated frames
        for i, li in enumerate(latent_frame_indices):
            samples[i, li] = local_samples[i, -len(li):].cpu().to(samples.dtype)
            visualized_samples[i, li] = visualized_local_samples[i, -len(li):].cpu().to(visualized_local_samples.dtype)
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
        samples = (samples.numpy() - drange[0]) / (drange[1] - drange[0]) * 255
        samples = samples.astype(np.uint8)
        for i in range(len(batch_indices)):
            if todo[i]:
                np.save(output_filenames[i], samples[i])
                print(f"*** Saved {output_filenames[i]} ***")


# def visualise(args, model, diffusion, dataset):
#     """
#     batch has a shape of BxTxCxHxW where
#     B: batch size
#     T: video length
#     CxWxH: image size
#     """
#     is_adaptive = "adaptive" in args.sampling_scheme
#     bs = args.batch_size if is_adaptive else 1
#     batch = next(iter(th.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False)))[0]
#     _, indices = sample_video(args, model, diffusion, batch, just_get_indices=True)

#     def visualise_obs_lat_sequence(sequence, index):
#         """ if index is None, expects sequence to be a list of tuples of form (list, list)
#             if index is given, expects sequence to be a list of tuples of form (list of lists (from which index i is taken), list of lists (from which index i is taken))
#         """
#         vis = []
#         exist_indices = list(range(args.n_obs))
#         for obs_frame_indices, latent_frame_indices in sequence:
#             obs_frame_indices, latent_frame_indices = obs_frame_indices[index], latent_frame_indices[index]
#             exist_indices.extend(latent_frame_indices)
#             new_layer = th.zeros((args.T, 3)).int()
#             border_colour = th.tensor([0, 0, 0]).int()
#             not_sampled_colour = th.tensor([255, 255, 255]).int()
#             exist_colour = th.tensor([50, 50, 50]).int()
#             obs_colour = th.tensor([50, 50, 255]).int()
#             latent_colour = th.tensor([255, 69, 0]).int()
#             new_layer = th.zeros((args.T, 3)).int()
#             new_layer[:, :] = not_sampled_colour
#             new_layer[exist_indices, :] = exist_colour
#             new_layer[obs_frame_indices, :] = obs_colour
#             new_layer[latent_frame_indices, :] = latent_colour
#             scale = 4
#             new_layer = new_layer.repeat_interleave(scale+1, dim=0)
#             new_layer[::(scale+1)] = border_colour
#             new_layer = th.cat([new_layer, new_layer[:1]], dim=0)
#             vis.extend([new_layer.clone() for _ in range(scale+1)])
#             vis[-1][:] = border_colour
#         vis = th.stack([vis[-1], *vis])
#         if not is_adaptive:
#             assert index == 0
#         fname = f"vis_{args.sampling_scheme}_sampling-{args.T}-given-{args.n_obs}_{args.max_latent_frames}-{args.max_frames}-chunks"
#         if args.optimality is not None:
#             fname += f"_optimal-{args.optimality}"
#         if is_adaptive:
#             fname += f"_index-{index}"
#         else:
#             assert index == 0
#         fname += '.png'
#         dir = Path("visualisations")
#         dir.mkdir(parents=True, exist_ok=True)
#         Image.fromarray(vis.numpy().astype(np.uint8)).save(dir / fname)
#         print(f"Saved to {str(dir / fname)}")

#     for i in range(len(batch)):
#         visualise_obs_lat_sequence(indices, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--sampling_scheme", required=True, choices=sampling_schemes.keys())
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--n_obs", type=int, default=36, help="Number of observed frames at the beginning of the video. The rest are sampled.")
    parser.add_argument("--T", type=int, default=None, help="Length of the videos. If not specified, it will be inferred from the dataset.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Denoted K in the paper. Maximum number of (observed or latent) frames input to the model at once. Defaults to what the model was trained with.")
    parser.add_argument("--max_latent_frames", type=int, default=None, help="Number of frames to sample in each stage. Defaults to max_frames/2.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--stop_index", type=int, default=None)
    parser.add_argument("--sampler", type=str, default="heun-80-inf-0-1-1000-0.002-7-50")
    parser.add_argument("--use_ddim", type=str2bool, default=False)
    parser.add_argument("--eval_on_train", type=str2bool, default=False)
    parser.add_argument("--timestep_respacing", type=str, default="")
    parser.add_argument("--clip_denoised", type=str2bool, default=True)
    parser.add_argument("--sample_idx", type=int, default=0, help="Sampled images will have this specific index. Used for sampling multiple videos with the same observations.")
    parser.add_argument("--optimality", type=str, default=None,
                        choices=["linspace-t", "random-t", "linspace-t-force-nearby", "random-t-force-nearby"],
                        help="Type of optimised sampling scheme to use for choosing observed frames. By default uses non-optimized sampling scheme. The optimal indices should be computed before use via video_optimal_schedule.py.")
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("--visualize_mode", type=str2bool, default=False, help="Used to produce videos of specific video subsequences.")
    args = parser.parse_args()

    # HACK: Do this for now
    # if not args.visualize_mode:
    #     assert args.start_index == 0, "Start index must be 0 due to the test set potentially being evenly spaced out from the entire test video."

    # Prepare which indices to sample (for unconditional generation index does nothing except change file name)
    if args.stop_index is None:
        # assume we're in a slurm batch job, set start and stop_index accordingly
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        else:
            print("Warning: assuming we're not in a slurm batch job, only sampling first batch.")
            task_id = 0
        args.start_index = task_id * args.batch_size
        args.stop_index = (task_id + 1) * args.batch_size
    args.indices = list(range(args.start_index, args.stop_index))
    print(f"Sampling for indices {args.start_index} to {args.stop_index}.")

    # Load the checkpoint (state dictionary and config)
    data = dist_util.load_state_dict(args.checkpoint_path, map_location="cpu")
    state_dict = data["state_dict"]
    model_args = data["config"]
    model_args.update({"use_ddim": args.sampler == "ddim",
                       "timestep_respacing": args.timestep_respacing})
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

    # Prepare samples directory
    args.eval_dir = get_model_results_path(args) / get_eval_run_identifier(args)
    samples_prefix = "samples_train" if args.eval_on_train else "samples"

    # Load the dataset (to get observations from)
    if args.visualize_mode:
        dataset = get_vis_dataset(dataset_name=model_args.dataset, T=args.T)
        samples_prefix += "_vis"
    else:
        dataset = get_test_dataset(dataset_name=model_args.dataset, T=args.T, n_data=len(args.indices))

    if args.eval_on_train:
        dataset.is_test = False


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
