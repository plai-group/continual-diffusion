"""
Train a diffusion model on images.
"""

import os
import sys
import argparse
import wandb
import torch
import torch.distributed as dist

from improved_diffusion import dist_util
from improved_diffusion.video_datasets import load_data, default_T_dict, default_image_size_dict
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.logger import logger

os.environ["MY_WANDB_DIR"] = "none"
if "--unobserve" in sys.argv:
    sys.argv.remove("--unobserve")
    os.environ["WANDB_MODE"] = "dryrun"
    if "WANDB_DIR_DRYRUN" in os.environ:
        os.environ["MY_WANDB_DIR"] = os.environ["WANDB_DIR_DRYRUN"]


def init_wandb(config, id):
    if dist.get_rank() != 0:
        return
    wandb_dir = os.environ.get("MY_WANDB_DIR", "none")
    if wandb_dir == "none":
        wandb_dir = None
    wandb.init(entity=os.environ['WANDB_ENTITY'],
               project=os.environ['WANDB_PROJECT'],
               config=config, dir=wandb_dir, id=id)
    print(f"Wandb run id: {wandb.run.id}")
    num_nodes = 1
    if "SLURM_JOB_NODELIST" in os.environ:
        assert "SLURM_JOB_NUM_NODES" in os.environ
        num_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        print(f"Node list: {os.environ['SLURM_JOB_NODELIST']}")
    logger.logkv("num_nodes", num_nodes)
    print(f"Number of nodes: {num_nodes}") 


def num_available_cores():
    # Copied from pytorch source code https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    max_num_worker_suggest = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if max_num_worker_suggest is None:
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            max_num_worker_suggest = cpu_count
    return max_num_worker_suggest or 1


def main():
    args = create_argparser().parse_args()
    if args.num_workers == -1:
        # Set the number of workers automatically.
        args.num_workers = max(num_available_cores() - 1, 1)
        print(f"num_workers is not specified. It is automatically set to \"number of cores - 1\" = {args.num_workers}")

    # Set T and image size
    default_image_size = default_image_size_dict[args.dataset]
    args.T = default_T_dict[args.dataset] if args.T == -1 else args.T
    args.image_size = {
        "pixel": default_image_size,
        "latent": default_image_size,
    }[args.diffusion_space]
    args.in_channels = {
        "pixel": 3,
        "latent": 4,
    }[args.diffusion_space]

    args.diffusion_space_kwargs = {
        "diffusion_space": args.diffusion_space,
        "pre_encoded": args.diffusion_space == "latent",
    }

    dist_util.setup_dist()
    resume = bool(args.resume_id)
    init_wandb(config=args, id=args.resume_id if resume else None)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    print("creating data loader...")
    data = load_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        T=args.T,
        num_workers=args.num_workers,
        resume_id=args.resume_id,
        seed=args.data_seed,
        buffer_size=args.ltm_size,
        n_sequential=args.n_sample_stm,
        save_every=args.save_interval,
    )

    print("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        diffusion_space_kwargs=args.diffusion_space_kwargs,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        sample_interval=args.sample_interval,
        pad_with_random_frames=args.pad_with_random_frames,
        max_frames=args.max_frames,
        enc_dec_chunk_size=args.enc_dec_chunk_size,
        steps_per_experience=args.steps_per_experience,
        masking_mode=args.masking_mode,
        args=args,
    ).run_loop()


def create_argparser():
    defaults = dict(
        dataset="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=100000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        diffusion_space="pixel",
        resume_id='',  # set this to a previous run's wandb id to resume training
        num_workers=-1,
        pad_with_random_frames=True,
        max_frames=20,
        enc_dec_chunk_size=20,
        T=-1,
        sample_interval=50000,
        stm_size=-1,  # Only used for flexible and streaming sampling (masking_mode == "flexible")
        ltm_size=-1,
        n_sample_stm=-1,
        n_sample_ltm=1,
        steps_per_experience=1,
        masking_mode="autoregressive",
        save_replay_mem=False,
        attentive_er=False,  # If true, the model attends to replay frames
        data_seed=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
