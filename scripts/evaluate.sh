#!/bin/bash

if [[ ! -v WANDB_ID ]]; then
    echo "WANDB_ID is not set."
    exit 1
fi
if [[ ! -v MODEL_NAME ]]; then
    echo "MODEL_NAME is not set."
    exit 1
fi

NUM_GENERATIONS=500
MAX_FRAMES=10
MAX_LATENT_FRAMES=5
VID_LENGTH=50
N_START_FRAMES=10
NUM_FVD_VIDEOS=100

CHECKPOINT_PATH=checkpoints/${WANDB_ID}/${MODEL_NAME}.pt
RESULT_PATH=results/${WANDB_ID}/${MODEL_NAME}/autoreg_${MAX_FRAMES}_${MAX_LATENT_FRAMES}_${VID_LENGTH}_${N_START_FRAMES}

conda activate fdm
python scripts/video_sample.py $CHECKPOINT_PATH --T=$VID_LENGTH --stop_index=$NUM_GENERATIONS --max_frames=$MAX_FRAMES --n_obs=$N_START_FRAMES --sampling_scheme=autoreg --batch_size=50
python scripts/video_make_mp4.py --eval_dir=$RESULT_PATH --do_n=8 --add_gt=False --obs_length=$N_START_FRAMES
python scripts/video_fvd.py --eval_dir=$RESULT_PATH --T=$VID_LENGTH --num_videos=$NUM_GENERATIONS --batch_size=25
