import copy
import functools
import os
import wandb

import blobfile as bf
import glob
from pathlib import Path
from time import time
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.optim import AdamW

from . import dist_util
from .logger import logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .rng_util import rng_decorator, RNG

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16,
        diffusion_space_kwargs,
        fp16_scale_growth,
        schedule_sampler,
        weight_decay,
        lr_anneal_steps,
        sample_interval,
        pad_with_random_frames,
        max_frames,
        enc_dec_chunk_size,
        steps_per_experience,
        masking_mode,
        args,
    ):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.diffusion_space_kwargs = diffusion_space_kwargs
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.sample_interval = sample_interval
        self.pad_with_random_frames = pad_with_random_frames
        self.enc_dec_chunk_size = enc_dec_chunk_size
        self.vis_batch = None
        self.max_frames = max_frames

        self.step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()
        self.original_dtype = None

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

        if dist.get_rank() == 0:
            Path(get_blob_logdir(self.args.resume_id)).mkdir(parents=True, exist_ok=True)

        if self.args.resume_id != '':
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            # self.ddp_model = DP(self.model)
        else:
            if dist.get_world_size() > 1:
                print(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        if dist.get_rank() == 0:
            logger.logkv("num_parameters", sum(p.numel() for p in model.parameters()), distributed=False)

        self.steps_per_experience = steps_per_experience
        self.masking_mode = masking_mode

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint(self.args) or self.resume_checkpoint

        if resume_checkpoint:
            self.step = parse_resume_step_from_filename(resume_checkpoint)
            print(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )['state_dict']
            )

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint(self.args) or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.step, rate)
        if ema_checkpoint:
            print(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )['state_dict']
            ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint(self.args) or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            print(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    
    def sample_some_indices(self, max_indices, T):
        s = th.randint(low=1, high=max_indices+1, size=())
        max_scale = T / (s-0.999)
        scale = np.exp(np.random.rand() * np.log(max_scale))
        pos = th.rand(()) * (T - scale*(s-1))
        indices = [int(pos+i*scale) for i in range(s)]
        # do some recursion if we have somehow failed to satisfy the consrtaints
        if all(i<T and i>=0 for i in indices):
            return indices
        else:
            print('warning: sampled invalid indices', [int(pos+i*scale) for i in range(s)], 'trying again')
            return self.sample_some_indices(max_indices, T)

    def sample_all_masks(self, batch1, batch2=None, gather=True, set_masks={'obs': (), 'latent': ()}):
        B, T, *_ = batch1.shape
        N = min(T, self.max_frames)

        masks = {k: th.zeros_like(batch1[:, :, :1, :1, :1]) for k in ['obs', 'latent']}
        for obs_row, latent_row in zip(*[masks[k] for k in ['obs', 'latent']]):
            latent_row[self.sample_some_indices(max_indices=N, T=T)] = 1.
            n_repeats = 0
            while True:
                mask = obs_row if th.rand(()) < 0.5 else latent_row
                indices = th.tensor(self.sample_some_indices(max_indices=N, T=T))
                taken = (obs_row[indices] + latent_row[indices]).view(-1)
                indices = indices[taken == 0]  # remove indices that are already used in a mask
                if len(indices) > N - sum(obs_row) - sum(latent_row)\
                    or len(indices) == N - sum(obs_row) - sum(latent_row) == 0:
                    break
                mask[indices] = 1.

        if len(set_masks['obs']) > 0:  # set_masks allow us to choose informative masks for logging
            for k in masks:
                set_values = set_masks[k]
                n_set = min(len(set_values), len(masks[k]))
                masks[k][:n_set] = set_values[:n_set]
        any_mask = (masks['obs'] + masks['latent']).to(th.float32).clip(max=1).to(masks['obs'].dtype)
        if not gather:
            return batch1, masks['obs'], masks['latent']
        batch, (obs_mask, latent_mask), frame_indices =\
            self.prepare_training_batch(any_mask, batch1, batch2, (masks['obs'], masks['latent']))

        return batch, frame_indices, obs_mask, latent_mask

    def get_autoregressive_masks(self, batch1, gather=True, set_masks={'obs': (), 'latent': ()}):
        # Same as above, but always sets the last element as latent and all others as observed
        masks = {'latent': th.zeros_like(batch1[:, :, :1, :1, :1]), 'obs': th.ones_like(batch1[:, :, :1, :1, :1])}
        masks['latent'][:, masks['latent'].size(1)//2:] = 1.
        masks['obs'][:, masks['obs'].size(1)//2:] = 0.
        if len(set_masks['obs']) > 0:  # set_masks allow us to choose informative masks for logging
            for k in masks:
                set_values = set_masks[k]
                n_set = min(len(set_values), len(masks[k]))
                masks[k][:n_set] = set_values[:n_set]
        any_mask = (masks['obs'] + masks['latent']).to(th.float32).clip(max=1).to(masks['obs'].dtype)
        if not gather:
            return batch1, masks['obs'], masks['latent']
        batch, (obs_mask, latent_mask), frame_indices =\
            self.prepare_training_batch(any_mask, batch1, None, (masks['obs'], masks['latent']))
        return batch, frame_indices, obs_mask, latent_mask

    def get_joint_masks(self, batch1, gather=True, set_masks={'obs': (), 'latent': ()}):
        # Same as above, but always sets the last element as latent and all others as observed
        masks = {'latent': th.ones_like(batch1[:, :, :1, :1, :1]), 'obs': th.zeros_like(batch1[:, :, :1, :1, :1])}
        if len(set_masks['obs']) > 0:  # set_masks allow us to choose informative masks for logging
            for k in masks:
                set_values = set_masks[k]
                n_set = min(len(set_values), len(masks[k]))
                masks[k][:n_set] = set_values[:n_set]
        any_mask = (masks['obs'] + masks['latent']).to(th.float32).clip(max=1).to(masks['obs'].dtype)
        if not gather:
            return batch1, masks['obs'], masks['latent']
        batch, (obs_mask, latent_mask), frame_indices =\
            self.prepare_training_batch(any_mask, batch1, None, (masks['obs'], masks['latent']))
        return batch, frame_indices, obs_mask, latent_mask

    def get_autoregressive_flexible_masks(self, batch1, batch2=None, gather=True, set_masks={'obs': (), 'latent': ()}):
        # Same as self.sample_all_masks but sets first half indices to observed and the latter half to latent
        B, T, *_ = batch1.shape
        N = min(T, self.max_frames)
        masks = {k: th.zeros_like(batch1[:, :, :1, :1, :1]) for k in ['obs', 'latent']}
        for obs_row, latent_row in zip(*[masks[k] for k in ['obs', 'latent']]):
            selected_indices = set()
            while True:
                indices = th.tensor(self.sample_some_indices(max_indices=N, T=T))
                new_indices = [i.item() for i in indices if i not in selected_indices]
                if len(new_indices) > N - len(selected_indices)\
                    or len(new_indices) == N - len(selected_indices) == 0:
                    break
                selected_indices.update(new_indices)
            selected_indices = sorted(list(selected_indices))
            obs_row[selected_indices[:N//2]] = 1.
            latent_row[selected_indices[N//2:]] = 1.

        if len(set_masks['obs']) > 0:  # set_masks allow us to choose informative masks for logging
            for k in masks:
                set_values = set_masks[k]
                n_set = min(len(set_values), len(masks[k]))
                masks[k][:n_set] = set_values[:n_set]
        any_mask = (masks['obs'] + masks['latent']).to(th.float32).clip(max=1).to(masks['obs'].dtype)
        if not gather:
            return batch1, masks['obs'], masks['latent']
        batch, (obs_mask, latent_mask), frame_indices =\
            self.prepare_training_batch(any_mask, batch1, batch2, (masks['obs'], masks['latent']))

        return batch, frame_indices, obs_mask, latent_mask

    def prepare_training_batch(self, mask, batch1, batch2, tensors):
        """
        Prepare training batch by selecting frames from batch1 according to mask, appending uniformly sampled frames
        from batch2, and selecting the corresponding elements from tensors (usually obs_mask and latent_mask).
        """
        B, T, *_ = mask.shape
        mask = mask.view(B, T)  # remove unit C, H, W dims
        effective_T = self.max_frames if self.pad_with_random_frames else mask.sum(dim=1).max().int()
        indices = th.zeros_like(mask[:, :effective_T], dtype=th.int64)
        new_batch = th.zeros_like(batch1[:, :effective_T])
        new_tensors = [th.zeros_like(t[:, :effective_T]) for t in tensors]
        for b in range(B):
            instance_T = mask[b].sum().int()
            indices[b, :instance_T] = mask[b].nonzero().flatten()
            indices[b, instance_T:] = th.randint_like(indices[b, instance_T:], high=T) if self.pad_with_random_frames else 0
            new_batch[b, :instance_T] = batch1[b][mask[b]==1]
            new_batch[b, instance_T:] = (batch1 if batch2 is None else batch2)[b][indices[b, instance_T:]]
            for new_t, t in zip(new_tensors, tensors):
                new_t[b, :instance_T] = t[b][mask[b]==1]
                new_t[b, instance_T:] = t[b][indices[b, instance_T:]]
        return new_batch, new_tensors, indices

    def get_next_batch(self):
        frames, absolute_index_map = next(self.data)
        if self.vis_batch is None or self.vis_batch.size(1) < self.max_frames:
            with RNG(0):  # Initialize datapoint to log here in case data is deterministic
                self.vis_batch = frames
        return frames, absolute_index_map

    def run_loop(self):
        last_sample_time = None
        while not self.lr_anneal_steps or self.step < self.lr_anneal_steps:
            try:
                frames, absolute_index_map = self.get_next_batch()
                # print(f"rank: {dist.get_rank()}, indices: {absolute_index_map[:,0].tolist()}, device: {dist_util.dev()}")
            except RuntimeError as e:
                print(e)
                # self.step += 1
                break

            for _ in range(self.steps_per_experience):
                self.run_step(frames, None, absolute_index_map)
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
                if self.sample_interval is not None and self.step != 0 and (self.step % self.sample_interval == 0 or self.step == self.max_frames):
                    if last_sample_time is not None:
                        logger.logkv('timing/time_between_samples', time()-last_sample_time)
                    self.log_samples()
                    last_sample_time = time()
                self.step += 1
        self.save()

    def run_step(self, batch1, batch2, absolute_index_map=None):
        t0 = time()
        self.forward_backward(batch1, batch2, absolute_index_map)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()
        logger.logkv("timing/step_time", time() - t0)
    
    def forward_backward(self, batch1, batch2, absolute_index_map=None):
        zero_grad(self.model_params)

        batch_size = batch1.shape[0]
        self.microbatch = batch_size  # HACK: Disable microbatches
        for i in range(0, batch_size, self.microbatch):
            micro1 = batch1[i : i + self.microbatch]
            micro2 = batch2[i : i + self.microbatch] if batch2 is not None else None
            if self.masking_mode == "autoregressive":
                micro, frame_indices, obs_mask, latent_mask = self.get_autoregressive_masks(micro1)
            elif self.masking_mode == "none":
                micro, frame_indices, obs_mask, latent_mask = self.get_joint_masks(micro1)
            elif self.masking_mode == "flexible":
                micro, frame_indices, obs_mask, latent_mask = self.sample_all_masks(micro1, micro2)
            elif self.masking_mode == "autoflex":
                micro, frame_indices, obs_mask, latent_mask = self.get_autoregressive_flexible_masks(micro1)
            else:
                raise ValueError(f"Unknown masking mode: {self.masking_mode}")

            if absolute_index_map is not None:
                # Convert frame indices to indices in one long video frame for RPE attention to work with.
                frame_indices = th.stack([idx_map[idx] for idx_map, idx in zip(absolute_index_map, frame_indices)], dim=0)

            micro = micro.to(dist_util.dev())
            frame_indices = frame_indices.to(dist_util.dev())
            obs_mask = obs_mask.to(dist_util.dev())
            latent_mask = latent_mask.to(dist_util.dev())

            last_batch = (i + self.microbatch) >= batch1.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs={'frame_indices': frame_indices, 'obs_mask': obs_mask,
                              'latent_mask': latent_mask, 'x0': micro},
                latent_mask=(1-obs_mask) if self.pad_with_random_frames else latent_mask,
                eval_mask=latent_mask,
            )
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if losses['loss'].isnan().sum() > 0:
                raise Exception("Loss is nan.")

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            print(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.step / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        if dist.get_rank() == 0:
            Path(get_blob_logdir(self.args.resume_id)).mkdir(parents=True, exist_ok=True)
        def save_checkpoint(rate, params):
            if dist.get_rank() == 0:
                print(f"saving model {rate}...")
                if not rate:
                    filename = f"model{self.step:06d}.pt"
                else:
                    filename = f"ema_{rate}_{self.step:06d}.pt"
                to_save = {
                    "state_dict": self._master_params_to_state_dict(params),
                    "config": self.args.__dict__,
                    "step": self.step
                }
                with bf.BlobFile(bf.join(get_blob_logdir(self.args.resume_id), filename), "wb") as f:
                    th.save(to_save, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(self.args.resume_id), f"opt{self.step:06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params

    def encode(self, video):
        return self.diffusion.encode(video, chunk_size=self.enc_dec_chunk_size)

    def decode(self, video):
        return self.diffusion.decode(video, chunk_size=self.enc_dec_chunk_size)

    @rng_decorator(seed=0)
    def log_samples(self):
        if dist.get_rank() == 0:
            sample_start = time()
            self.model.eval()
            orig_state_dict = copy.deepcopy(self.model.state_dict())
            self.model.load_state_dict(copy.deepcopy(self._master_params_to_state_dict(self.ema_params[0])))

            print("sampling...")
            # construct simple masks for our vis batch
            obs_mask = th.zeros_like(self.vis_batch[:, :, :1, :1, :1])
            latent_mask = obs_mask.clone()
            n_obs = self.max_frames // 3
            obs_mask[0, :n_obs] = 1.
            latent_mask[0, n_obs:self.max_frames] = 1.
            # if self.batch_size > 1:
            #     spacing = len(self.vis_batch[0]) // self.max_frames
            #     obs_mask[1, :n_obs*spacing:spacing] = 1.
            #     latent_mask[1, n_obs*spacing:self.max_frames*spacing:spacing] = 1.
            batch, frame_indices, obs_mask, latent_mask = self.sample_all_masks(
                self.vis_batch, None, gather=True,
                 set_masks={'obs': obs_mask, 'latent': latent_mask}
            )
            samples, _ = self.diffusion.heun_sample(
                self.model,
                batch.shape,
                clip_denoised=True,
                model_kwargs={
                    'frame_indices': frame_indices.to(dist_util.dev()),
                    'x0': batch.to(dist_util.dev()),
                    'obs_mask': obs_mask.to(dist_util.dev()),
                    'latent_mask': latent_mask.to(dist_util.dev())},
                latent_mask=latent_mask,
                return_attn_weights=True,
                return_decoded=False,
            )
            # NOTE: Don't decode latent samples
            samples = (samples.cpu() * latent_mask + batch * obs_mask).float()
            _mark_as_observed(samples[:, :n_obs])
            samples = ((samples + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()
            for i, video in enumerate(samples):
                logger.logkv(f'video-{i}', wandb.Video(video), distributed=False)
            logger.logkv("timing/sampling_time", time() - sample_start, distributed=False)

            # restore model to original state
            self.model.train()
            self.model.load_state_dict(orig_state_dict)
            print("finished sampling")
        dist.barrier()


def _mark_as_observed(images, color=[1., -1., -1.]):
    for i, c in enumerate(color):
        images[..., i, :, 1:2] = c
        images[..., i, 1:2, :] = c
        images[..., i, :, -2:-1] = c
        images[..., i, -2:-1, :] = c


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir(resume_id=''):
    root_dir = "checkpoints"
    assert os.path.exists(root_dir), "Must create directory 'checkpoints'"
    wandb_id = resume_id if len(resume_id) > 0 else wandb.run.id
    return os.path.join(root_dir, wandb_id)


def find_resume_checkpoint(args):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    if not args.resume_id:
        return
    ckpts = glob.glob(os.path.join(get_blob_logdir(args.resume_id), "model*.pt"))
    if len(ckpts) == 0:
        return None
    iters_fnames = {int(Path(fname).stem.replace('model', '')): fname for fname in ckpts}
    return iters_fnames[max(iters_fnames.keys())]


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
