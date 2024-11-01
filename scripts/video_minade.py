import argparse
import json
import numpy as np
import os
from pathlib import Path
import torch
from typing import List

from improved_diffusion.video_datasets import get_eval_dataset, eval_dataset_configs
from improved_diffusion.script_util import str2bool
from improved_diffusion.test_util import parse_eval_run_identifier, Protect
from video_fvd import SampleDataset


# from torchvision.utils import save_image
# save_image((frame+1)/2, 'true.png')

def ar(x, y, z):
    return z/2+np.arange(x, y, z, dtype='float')


def kl_div(ps, qs):
    results = []
    for p1, p2 in zip(ps, qs):
        result = 0.
        for b1, b2 in zip(p1, p2):
            if b1 > 0:
                result += b1 * (b1 / b2).log() if b2>0 else float('inf')
        results.append(result)
    return sum(results)/len(results)


def compute_color_transition_stats(sample_colors, n_obs):
    """
    Compute how many times the ball transitioned from one color to another for each color pairs.
    - sample_colors shape: <n_seeds x n_timesteps, x n_balls>
    - output shape: <n_seed>
    """
    n_seeds, n_timesteps, n_balls = sample_colors.shape
    transition_stats = torch.zeros(n_seeds, 3, 3)
    # transition_accs = torch.zeros(n_seeds)
    # gt_dict = {(0): (1,2), (1): (0), (2): (0), (0,1): (0), (0,2): (0), (1,0): (2), (2,0): (1),
    #            (0,1,0): (2), (1,0,2): (0), (0,2,0): (1), (2,0,1): (0)}

    for s in range(n_seeds):
        for b in range(n_balls):
            prev_color = sample_colors[s, 0, b]
            # color_window = (prev_color)
            for t in range(1, n_timesteps):
                curr_color = sample_colors[s, t, b]
                if prev_color != curr_color:
                    # color change detected during model sampling
                    if t >= n_obs:
                        # reached model generated part of the video
                        # true_color = gt_dict[color_window] if color_window in gt_dict else -1
                        # transition_accs[s] += 1 if curr_color == true_color else 0
                        transition_stats[s, prev_color, curr_color] += 1
                    # color_window = (*color_window[-2:], curr_color)
                prev_color = curr_color

    # transition_accs = transition_accs/(n_timesteps-n_obs)
    transition_stats = torch.nan_to_num(transition_stats/transition_stats.sum(dim=-1, keepdim=True))
    return transition_stats


def get_kernel_maxacts(frame, kernels, n_max=1):
    kernel_size = kernels.size(-1)
    _, C, width, height = frame.shape
    frame_unfolded = frame.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1).permute(0, 2, 3, 1, 4, 5)
    frame_unfolded = frame_unfolded.reshape(-1, C, kernel_size, kernel_size)

    best_stats = [(None, float('inf'), None)] * n_max  # location, distance, kernel index
    for k, kernel in enumerate(kernels):
        distances = (kernel - frame_unfolded).norm(2, dim=(1, 2, 3))
        topk_result = torch.topk(distances, k=n_max * 4, largest=False)
        best_match_indices, min_dists = topk_result.indices, topk_result.values
        min_locs = torch.unravel_index(best_match_indices, (width - kernel_size + 1, height - kernel_size + 1))
        min_locs = torch.stack(min_locs[::-1], dim=0).transpose(0, 1)
        curr_stats = [(loc.tolist(), dist.item(), k) for loc, dist in zip(min_locs, min_dists)]

        # HACK: Ensure the centers of balls are at least 4 pixels away from the best match location for this ball.
        curr_stats = [curr_stats[0]] + [stat for stat in curr_stats if (torch.tensor(stat[0])-min_locs[0]).float().norm()>4]

        best_stats = sorted(best_stats+curr_stats, key=lambda e: e[1])[:n_max]
    locs, kernel_idxs = [e[0] for e in best_stats], [e[2] for e in best_stats]
    return locs, kernel_idxs


def align_balls(traj, color):
    first_frame_info = sorted(zip(traj[0], color[0]), key=lambda e: e[0])
    result = [([e[0] for e in first_frame_info], [e[1] for e in first_frame_info])]

    for t, (locs_next, c_next) in enumerate(zip(traj[1:], color[1:])):
        n_balls = len(locs_next)
        min_locs = torch.zeros(n_balls).to(torch.int)
        available_new_ball_indices = set(range(n_balls))
        distances = torch.cdist(torch.FloatTensor(result[t][0]), torch.FloatTensor(locs_next))
        for old_ball_index in range(n_balls):
            # For each current ball position, select the next timestep ball position that is closest and not taken
            best_match, ordered_matches = None, torch.topk(distances[old_ball_index], k=n_balls, largest=False).indices
            for match in ordered_matches.tolist():
                if match in available_new_ball_indices:
                    best_match = match
                    break
            min_locs[old_ball_index] = best_match
            available_new_ball_indices.remove(best_match)

        assert len(min_locs) == len(min_locs.unique()), "Each location must uniquely be assigned to a ball."
        to_add_traj, to_add_color = [], []
        for min_loc in min_locs:
            to_add_traj.append(locs_next[min_loc])
            to_add_color.append(c_next[min_loc])
        result.append((to_add_traj, to_add_color))
    return [e[0] for e in result], [e[1] for e in result]


def compute_minADE_exemplar(test_frames: torch.Tensor, all_sample_frames: List[torch.Tensor],
                            kernels: torch.Tensor, T: int, n_balls: int, n_obs: int):
    num_samples = len(all_sample_frames)
    test_traj = []
    test_color = []
    sample_trajs = [[] for _ in range(num_samples)]
    sample_colors = [[] for _ in range(num_samples)]
    for t in range(T):
        test_frame = test_frames[0, t]
        test_loc, test_kernel = get_kernel_maxacts(test_frame.unsqueeze(0), kernels, n_balls)
        test_traj.append(test_loc)
        test_color.append(test_kernel)

        for sample_idx in range(num_samples):
            sample_frame = all_sample_frames[sample_idx][0, t]
            sample_loc, sample_kernel = get_kernel_maxacts(sample_frame.unsqueeze(0), kernels, n_balls)
            sample_trajs[sample_idx].append(sample_loc)
            sample_colors[sample_idx].append(sample_kernel)

    test_traj, test_color = align_balls(test_traj, test_color)
    sample_results = [align_balls(traj, color) for traj, color in zip(sample_trajs, sample_colors)]
    sample_trajs, sample_colors = [e[0] for e in sample_results], [e[1] for e in sample_results]

    test_traj = torch.tensor(test_traj).unsqueeze(0).float()
    sample_trajs = torch.tensor(sample_trajs).float()
    ADEs = (test_traj[:, n_obs:] - sample_trajs[:, n_obs:]).norm(2, dim=-1).mean(dim=(-1,-2))
    minADE = torch.min(ADEs).item()

    test_color = torch.tensor(test_color).unsqueeze(0)
    sample_colors = torch.tensor(sample_colors)
    color_accs = (test_color[:, n_obs:] == sample_colors[:, n_obs:]).float().mean(dim=(-1,-2))
    max_color_acc = torch.max(color_accs).item()
    transition_stats = compute_color_transition_stats(sample_colors, n_obs)

    return minADE, max_color_acc, transition_stats


def compute_minADE(test_dataset, sample_datasets, n_obs, T, n_balls):
    minADE = 0.
    test_loader = iter(torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                   shuffle=False, drop_last=False))
    sample_loaders = [iter(torch.utils.data.DataLoader(sample_dataset, batch_size=1,
                                                       shuffle=False, drop_last=False))
                      for sample_dataset in sample_datasets]

    kernel_width = 8
    [I, J] = np.meshgrid(ar(-1.25, 1.25, 2.5/kernel_width), ar(-1.25, 1.25, 2.5/kernel_width))  # Ball radius
    # Make a 3x8x8 kernel based on the bouncing ball dataset's ball shape.
    gaussian_bump = np.exp(-((I**2+J**2)/(1.2**2))**4)
    gaussian_bump = torch.tensor(gaussian_bump).to(torch.float32)
    kernels = (
        torch.stack([1. * gaussian_bump, 0. * gaussian_bump, 0. * gaussian_bump], dim=0),  # red ball
        torch.stack([1. * gaussian_bump, 1. * gaussian_bump, 0. * gaussian_bump], dim=0),  # yellow ball
        torch.stack([0. * gaussian_bump, 1. * gaussian_bump, 0. * gaussian_bump], dim=0),  # green ball
    )
    kernels = torch.stack(kernels, dim=0) * 2 - 1

    # TODO: Have 3 kernels for red, white, yellow

    all_ADEs, all_accs, all_transitions = [], [], []
    dataset_obj = test_dataset.dataset
    for i in range(len(test_dataset)):
        test_frames = next(test_loader)[0]
        sample_frames = [next(sample_loader)[0] for sample_loader in sample_loaders]
        exemplar_kernel = kernels.clone()
        if "ball_nstn" in str(dataset_obj.path):
            #  HACK: For adjusting kernel blue channels to account for nonstationarity
            exemplar_kernel[:,-1] += i/dataset_obj.n_data
        ADE, acc, transitions = compute_minADE_exemplar(test_frames, sample_frames, exemplar_kernel, T, n_balls, n_obs)
        all_ADEs.append(ADE)
        all_accs.append(acc)
        all_transitions.append(transitions)

    minADE = sum(all_ADEs)/len(all_ADEs)
    max_color_acc = sum(all_accs)/len(all_accs)

    total_transition_stats = sum(all_transitions).mean(dim=0)
    total_transition_stats = torch.nan_to_num(total_transition_stats/total_transition_stats.sum(dim=-1, keepdim=True))
    true_transition_stats = torch.tensor([[0.,0.5,0.5], [1.,0.,0.], [1.,0.,0.]])
    transition_kl = kl_div(true_transition_stats, total_transition_stats)
    return minADE, max_color_acc, transition_kl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--num_videos", type=int, default=None, help="Number of test video to evaluate on.")
    parser.add_argument("--n_balls", type=int, default=2, help="Number of bouncing balls in a video frame.")
    parser.add_argument("--sample_indices", type=int, nargs='+', default=[0])
    args = parser.parse_args()

    parsed = parse_eval_run_identifier(os.path.basename(args.eval_dir))
    T, obs_length, eval_on_train, eval_dataset_config = parsed["T"], parsed["n_obs"], parsed["eval_on_train"], parsed["eval_dataset_config"]

    save_path = Path(args.eval_dir) / f"minADE-{args.num_videos}.txt"
    save_path_color = Path(args.eval_dir) / f"color-acc-{args.num_videos}.txt"
    save_path_transition = Path(args.eval_dir) / f"trans-kl-{args.num_videos}.txt"

    # if save_path.exists():
    #     content = np.loadtxt(save_path).squeeze()
    #     print(f"minADE/max_color_acc already computed: {content}")
    #     print(f"Results at\n{save_path}\n{save_path_color}")
    #     exit(0)

    # Load model args
    model_args_path = Path(args.eval_dir) / "model_config.json"
    with open(model_args_path, "r") as f:
        model_args = argparse.Namespace(**json.load(f))

    args.dataset = model_args.dataset

    # Load the dataset (to get observations from)
    eval_dataset_args = dict(dataset_name=model_args.dataset, T=T, train=eval_on_train,
                             eval_dataset_config=eval_dataset_config)
    if eval_dataset_config != eval_dataset_configs["continuous"]:
        spacing_kwargs = dict(frame_range=(parsed["lower_frame_range"], parsed["upper_frame_range"]), n_data=args.num_videos)
        eval_dataset_args["spacing_kwargs"] = spacing_kwargs
    test_dataset_full = get_eval_dataset(**eval_dataset_args)
    test_dataset = torch.utils.data.Subset(dataset=test_dataset_full, indices=list(range(args.num_videos)))
    sample_datasets = [SampleDataset(samples_path=(Path(args.eval_dir) / "samples"),
                                     sample_idx=i, length=args.num_videos) for i in args.sample_indices]

    minADE, max_color_acc, transition_kl = compute_minADE(test_dataset, sample_datasets, n_obs=obs_length, T=T, n_balls=args.n_balls)
    np.savetxt(save_path, np.array([minADE]))
    np.savetxt(save_path_color, np.array([max_color_acc]))
    np.savetxt(save_path_transition, np.array([transition_kl]))
    print(f"minADE: {minADE:.5f}, color-acc: {max_color_acc:.5f}, trans-kl: {transition_kl:.5f}")
    print(f"Results saved to\n{save_path}\n{save_path_color}\n{save_path_transition}")
