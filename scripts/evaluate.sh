#!/bin/bash

if [[ ! -v WANDB_ID ]]; then
    echo "WANDB_ID is not set."
    exit 1
fi
if [[ ! -v MODEL_NAME ]]; then
    echo "MODEL_NAME is not set."
    exit 1
fi

NUM_GENERATIONS=100
MAX_FRAMES=10
MAX_LATENT_FRAMES=5
VID_LENGTH=100
N_START_FRAMES=36
NUM_FVD_VIDEOS=100

CHECKPOINT_PATH=checkpoints/${WANDB_ID}/${MODEL_NAME}.pt
RESULT_PATH=results/${WANDB_ID}/${MODEL_NAME}/autoreg_${MAX_FRAMES}_${MAX_LATENT_FRAMES}_${VID_LENGTH}_${N_START_FRAMES}

conda activate fdm
#python scripts/video_sample.py $CHECKPOINT_PATH --sampling_scheme=autoreg --T=$VID_LENGTH --stop_index=$NUM_GENERATIONS --max_latent_frames=$MAX_LATENT_FRAMES
#python scripts/video_make_mp4.py --eval_dir=$RESULT_PATH --do_n=$NUM_GENERATIONS --add_gt=False --obs_length=$N_START_FRAMES
python scripts/video_fvd.py --eval_dir=$RESULT_PATH --num_videos=$NUM_FVD_VIDEOS --batch_size=20
