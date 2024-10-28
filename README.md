# Forked from [official repository](https://github.com/plai-group/flexible-video-diffusion-modeling) for [Flexible Diffusion Modeling of Long Videos](https://arxiv.org/abs/2205.11495)


# Full overview

# Installation

Tested with Python 3.10 in a conda environment. We require the Python packages `mpi4py torch torchvision wandb blobfile tqdm moviepy imageio diffusers opencv-python ffmpeg lpips tensorflow==2.15 tensorflow_hub==0.15.0 diffusers transformers pandas`.
This repository itself should also be installed by running
```
pip install -e .
```

This repo logs to wandb, using the wandb entity/username and project name set by:
```
export WANDB_ENTITY=<...>
export WANDB_PROJECT=<...>
```

And add a directory for checkpoints to be saved in:
```
mkdir checkpoints
```

## Data Preparation

### Running Bouncing Ball datasets

#### Stationary

1. Go to the project directory.
2. Make the dataset: `python datasets/ball.py --save_path=datasets/ball_stn`.

#### Non-Stationary

1. Go to the project directory.
2. Make the dataset: `python datasets/ball.py --save_path=datasets/ball_stn --color_shift`.

### Running Windows Maze Screensaver 95

1. Get the zipped dataset file from Jason and unzip it at `datasets` folder.

### Running PLAICraft on UBC-ML or PLAI clusters

1. Go to the project directory.
2. Make a symlink in `datasets` foler that points to VQVAE-encoded video data: `ln -s /ubc/cs/research/plai-scratch/plaicraft/data/processed datasets/plaicraft`.


# Sample Commands

Sample SLURM scripts for training and evaluation are at `sample_slurm_scripts`.

TLDR
- `scripts/video_train.py` is the model training script.
- `scripts/video_sample.py` is the sampling script.
- `scripts/video_make_mp4.py` is the script that makes MP4 videos from model samples.
- `scripts/video_fvd.py` calculates FVD and saves it the the `results/...` folder.
- `scripts/collect_results.py` can group metrics from multiple runs.


# Dataset Objects Explained

Datasets used to construct continual learning data streams are at `improved_diffusion/video_datasets.py` and `improved_diffusion/plaicraft_dataset.py`. There are three types of datasets: ContinuousDataset, ChunkedDataset, and SpacedDataset. ContinuousDataset returns a sliding window of size `T` indexed by the first frame index (ex. [0,1,2,3,4], [1,2,3,4,5], ... when `T=5`). ChunkedDataset return a window of size `T` that are mutually exclusive (ex. [0,1,2,3,4], [5,6,7,8,9], ... when `T=5`). SpacedDataset returns a window of size `T` that are evenly spaced out across the data stream (ex. [0,1,2,3,4], [10,11,12,13,14], ..., [90,91,92,93,94] when we want 10 videos with length `T=5` from a dataset with 100 total frames).


# Artifact-Related Directory Structure

Model and optimizer checkpoints locations: `checkpoints/<WANDB RUN ID>`

Model sample and results locations: `results/<WANDB RUN ID>/<MODEL NAME>_<SAMPLER CONFIGS>/<DATA STREAM CONFIG>`

Run summary (that can group multiple runs) locations: `summarized/<SUMMARY NAME>`
