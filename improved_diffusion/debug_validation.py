"""
Issue #58 validation for the plaicraft-debug toy world.

Each row of the validation DB names a moment (``R_start``) in a held-out
recording.  We hand VDT the 10 video frames immediately before that moment and
ask it to generate the next 10, at exactly the observed/generated split the
model was trained on -- so nothing here is off-distribution.

The model is *not* action-conditioned.  The action-driven part of each task
(the jump, the strafe) is information VDT was never shown, so per-task scores
measure world reconstruction, not action following, and are not comparable to
action-conditioned baselines.  Frame 10 -- the first generated frame, only
100ms past the last observed one -- is the cleanest read, because 100ms of
unknown action moves the camera very little.
"""

import os
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import torch as th
import torch.nn as nn

from . import debug_actions
from .logger import logger
from .rng_util import RNG
from .decode_debug import (
    render_overlay,
    get_frame_actions,
    _to_uint8_frame,
    _overlay_frame,
    DECODE_VIDEO_FPS,
)

VIDEO_FPS = 10
MS_PER_FRAME = 1000 // VIDEO_FPS

# LPIPS runs a VGG backbone with five 2x downsamples; a 24x40 frame collapses to
# nothing by the last stage. Upsample (nearest, to avoid inventing detail that
# is not in the source pixels) before scoring.
LPIPS_UPSAMPLE = 4


class DebugValidationSet:
    """The validation rows plus the frames each one needs."""

    def __init__(self, db_path, data_root, T=20, n_observed=10):
        self.db_path = Path(db_path)
        self.data_root = Path(data_root)
        self.T = T
        self.n_observed = n_observed
        self.n_generated = T - n_observed
        self.rows = self._load_rows()

    def _load_rows(self):
        if not self.db_path.exists():
            raise FileNotFoundError(f"validation DB not found: {self.db_path}")
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        raw = conn.execute(
            'SELECT Num, Prompt, "Test Type", Session_ID, player_email, '
            '"R_start (ms)", "R_duration (ms)" FROM validation_rows ORDER BY Num'
        ).fetchall()
        conn.close()

        rows = []
        for r in raw:
            session_id = r["Session_ID"]
            session_dir = self.data_root / r["player_email"] / session_id
            r_start_ms = int(r["R_start (ms)"])
            start_frame = r_start_ms // MS_PER_FRAME  # first generated frame

            ctx_start = start_frame - self.n_observed
            if ctx_start < 0:
                raise ValueError(
                    f"row {r['Num']} ({r['Prompt']}): R_start={r_start_ms}ms leaves "
                    f"only {start_frame} frames of context, need {self.n_observed}"
                )
            rows.append(
                dict(
                    num=int(r["Num"]),
                    prompt=r["Prompt"],
                    test_type=r["Test Type"],
                    session_dir=session_dir,
                    session_db=session_dir / f"{session_id}.db",
                    hdf5=session_dir / "encoded_video_hdf5" / f"{session_id}_encoded_video.hdf5",
                    r_start_ms=r_start_ms,
                    r_duration_ms=int(r["R_duration (ms)"]),
                    window_start=ctx_start,
                )
            )
        return rows

    def slug(self, row):
        s = "".join(c if c.isalnum() else "-" for c in row["prompt"].lower())
        return f"{row['num']:02d}_{'-'.join(filter(None, s.split('-')))}"

    def load_window(self, row):
        """(T, 3, H, W) float32 in [-1,1]: n_observed context + n_generated GT."""
        import h5py

        with h5py.File(row["hdf5"], "r") as f:
            frames = f["frames"][row["window_start"] : row["window_start"] + self.T]
        if frames.shape[0] != self.T:
            raise ValueError(
                f"row {row['num']}: got {frames.shape[0]} frames, need {self.T} "
                f"(window {row['window_start']}..{row['window_start'] + self.T})"
            )
        return th.from_numpy(np.asarray(frames, dtype=np.float32))

    def load_all(self):
        return th.stack([self.load_window(r) for r in self.rows])

    def load_action_window(self, row):
        """(T, 10) float32: same window as load_window, causally-shifted actions."""
        arr = debug_actions.load_or_build(row["session_dir"])
        ws, we = row["window_start"], row["window_start"] + self.T
        if we > arr.shape[0]:
            raise ValueError(
                f"row {row['num']}: action array has {arr.shape[0]} frames, "
                f"need window {ws}..{we}"
            )
        return th.from_numpy(np.asarray(arr[ws:we], dtype=np.float32))

    def load_all_actions(self):
        return th.stack([self.load_action_window(r) for r in self.rows])


def _to01(x):
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


class _Metrics:
    """PSNR / SSIM / LPIPS on [-1,1] video tensors."""

    def __init__(self, device):
        from torchmetrics.image import (
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
        )
        import lpips as lpips_lib

        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        # Pin the feature extractor to eval and freeze it; a train-mode backbone
        # would drift with the model it is meant to score.
        self.lpips = lpips_lib.LPIPS(net="vgg").to(device).eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    @th.no_grad()
    def __call__(self, pred, gt):
        """pred, gt: (N, 3, H, W) in [-1,1]. Returns dict of floats."""
        pred, gt = pred.to(self.device).float(), gt.to(self.device).float()
        p01, g01 = _to01(pred), _to01(gt)
        # L2 in [0,1] pixel space so it is comparable across runs and to
        # model-pi0's reporting; the training `mse` key is diffusion-space.
        out = {
            "psnr": float(self.psnr(p01, g01)),
            "ssim": float(self.ssim(p01, g01)),
            "l2": float(th.mean((p01 - g01) ** 2)),
            "rmse": float(th.sqrt(th.mean((p01 - g01) ** 2))),
            "l1": float(th.mean(th.abs(p01 - g01))),
        }
        if LPIPS_UPSAMPLE > 1:
            up = lambda t: th.nn.functional.interpolate(t, scale_factor=LPIPS_UPSAMPLE, mode="nearest")
            pred, gt = up(pred), up(gt)
        out["lpips"] = float(self.lpips(pred, gt).mean())
        return out


_METRICS = None


def _get_metrics(device):
    global _METRICS
    if _METRICS is None:
        _METRICS = _Metrics(device)
    return _METRICS


class _CFGWrapper(nn.Module):
    """Wraps an action-conditioned VDT model to sample with classifier-free
    guidance: runs the model twice (conditional + null-action unconditional
    via ``force_action_drop``) and extrapolates. NOT vdt.py:forward_with_cfg
    (dead DiT-era code, incompatible signature) -- guidance is applied here,
    at the sampling site.
    """

    def __init__(self, model, w):
        super().__init__()
        self.model = model
        self.w = w

    def forward(self, x, timesteps, **kwargs):
        kwargs_cond = dict(kwargs)
        kwargs_cond["force_action_drop"] = False
        kwargs_uncond = dict(kwargs)
        kwargs_uncond["force_action_drop"] = True
        eps_cond, _ = self.model(x, timesteps=timesteps, **kwargs_cond)
        eps_uncond, _ = self.model(x, timesteps=timesteps, **kwargs_uncond)
        return eps_uncond + self.w * (eps_cond - eps_uncond), None


def _swap_actions(actions, n_obs):
    """Counterfactual actions on the generated half: the OPPOSITE action on
    every axis. Observed half untouched, since those frames are given.

        w <-> s          (dims 0, 2)  forward / back
        a <-> d          (dims 1, 3)  strafe left / right
        space <-> shift  (dims 4, 5)  up / down
        left <-> right   (dims 6, 7)  mouse buttons
        dx, dy negated   (dims 8, 9)  look direction

    A full inversion rather than a partial one: if the model is listening, every
    axis of the counterfactual should push the frame the other way, which makes
    the true-vs-swap divergence as large as this world allows.
    """
    out = actions.clone()
    gen = actions[:, n_obs:, :]
    swapped = gen.clone()
    for i, j in ((0, 2), (1, 3), (4, 5), (6, 7)):
        swapped[..., i] = gen[..., j]
        swapped[..., j] = gen[..., i]
    swapped[..., 8] = -gen[..., 8]
    swapped[..., 9] = -gen[..., 9]
    out[:, n_obs:, :] = swapped
    return out


def _zero_actions(actions, n_obs):
    """All-zero actions on the generated half. Observed half untouched."""
    out = actions.clone()
    out[:, n_obs:, :] = 0.0
    return out


#: order of the 6 key dims in the 10-d action vector; names must match the
#: labels decode_debug draws, i.e. the values of its KEY_ID_TO_NAME.
_ACTION_KEY_NAMES = ["w", "a", "s", "d", "space", "Shift_L"]
_ACTION_CLICK_NAMES = ["left", "right"]


def _action_vec_to_bar(vec):
    """One 10-d action vector -> the dict decode_debug._overlay_frame draws.

    Lets each row of the swap overlay show the actions it was ACTUALLY generated
    with. Reading the bar from the session DB instead would paint the true
    actions onto the swap and zero rows too, which hides the very thing the
    overlay exists to show.
    """
    vec = np.asarray(vec, dtype=np.float32)
    unsymlog = lambda v: float(np.sign(v) * np.expm1(abs(v)))
    return {
        "keys": [n for i, n in enumerate(_ACTION_KEY_NAMES) if vec[i] > 0.5],
        "clicks": [n for i, n in enumerate(_ACTION_CLICK_NAMES) if vec[6 + i] > 0.5],
        "mouseDX": unsymlog(vec[8]),
        "mouseDY": unsymlog(vec[9]),
    }


def _label_panel(panel, text):
    """Draw a panel name in the top bar's empty middle band."""
    cv2.putText(panel, text, (panel.shape[1] // 2 - 60, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3, cv2.LINE_AA)
    return panel


def _render_triple_overlay(frames_gt, frames_true, frames_swap, frames_zero,
                           actions_true, actions_swap, actions_zero,
                           n_observed, out_path, title=None):
    """2x2 grid mp4:  GT | TRUE  over  SWAP | ZERO.

    GT is included so the generated half can be judged against reality, not only
    against the other two continuations -- past frame n_observed the true/swap/
    zero rows are all model output and share no ground truth.

    Laid out as a grid rather than 4 stacked rows because stacking gives a
    1280x3472 video, which wandb renders unusably small.

    Each panel's action bar comes from that panel's own action tensor, so the
    swap panel visibly shows the swapped keys / reversed mouse it was given.
    GT carries the true actions, being the real recording.

    Same imageio/libx264 settings as decode_debug.render_overlay -- cv2's mp4v
    encodes fine and then will not play in wandb.
    """
    frames_gt = np.asarray(frames_gt)
    frames_true = np.asarray(frames_true)
    frames_swap = np.asarray(frames_swap)
    frames_zero = np.asarray(frames_zero)
    T = frames_true.shape[0]

    def _to_display(a):
        """Shift the causal action cache back to the DISPLAY convention.

        The cache is causal: row i is the action from window [i-1, i), i.e. the
        one that CAUSED frame i. That is right for conditioning the model, but
        drawing it as-is paints an action onto the very frame it produced -- a
        click appears simultaneously with its own effect, which reads as though
        causality were violated.

        decode_debug.get_frame_actions (used by the 2-row val/overlay) instead
        returns the RAW action for window [i, i+1), so the input renders one
        frame BEFORE its consequence. Match that here, or the two overlays
        disagree by one frame. cache[i+1] == raw[i], and the final frame has no
        successor so its bar is blank.
        """
        a = np.asarray(a)
        out = np.zeros_like(a)
        out[:-1] = a[1:]
        return out

    acts = [_to_display(a) for a in
            (actions_true, actions_true, actions_swap, actions_zero)]
    labels = ["GT", "TRUE", "SWAP", "ZERO"]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import imageio

    writer = imageio.get_writer(
        str(out_path), fps=DECODE_VIDEO_FPS, codec="libx264",
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    for t in range(T):
        border = t < n_observed
        panels = [
            _label_panel(
                _overlay_frame(_to_uint8_frame(frames[t]),
                               _action_vec_to_bar(act[t]), border=border),
                lab)
            for frames, act, lab in zip(
                (frames_gt, frames_true, frames_swap, frames_zero), acts, labels)
        ]
        writer.append_data(cv2.vconcat([cv2.hconcat(panels[:2]),
                                        cv2.hconcat(panels[2:])]))
    writer.close()
    return out_path


@th.no_grad()
def run_debug_validation(model, diffusion, valset, device, out_dir,
                         step=0, chunk_size=3, log_videos=True,
                         per_task_scalars=False, actions=True,
                         swap_test=True, cfg_scale=1.0):
    """Sample every validation row, render overlays, log metrics to wandb.

    Returns the aggregate metric dict (also logged via ``logger.logkv``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _get_metrics(device)

    action_conditioned = getattr(model, "action_embedder", None) is not None
    sampling_model = _CFGWrapper(model, cfg_scale) if cfg_scale != 1.0 else model

    T, n_obs = valset.T, valset.n_observed
    x0_all = valset.load_all()  # (N, T, 3, H, W)
    n_rows = x0_all.shape[0]
    actions_all = valset.load_all_actions() if (actions and action_conditioned) else None

    per_row, agg, swap_rows = [], {}, []
    for lo in range(0, n_rows, chunk_size):
        hi = min(lo + chunk_size, n_rows)
        x0 = x0_all[lo:hi].to(device)
        b = x0.shape[0]
        act_chunk = actions_all[lo:hi].to(device) if actions_all is not None else None

        obs_mask = th.zeros(b, T, 1, 1, 1, device=device)
        obs_mask[:, :n_obs] = 1.0
        latent_mask = th.zeros_like(obs_mask)
        latent_mask[:, n_obs:] = 1.0

        model_kwargs = {
            "frame_indices": None,
            "x0": x0,
            "obs_mask": obs_mask,
            "latent_mask": latent_mask,
        }
        if act_chunk is not None:
            model_kwargs["actions"] = act_chunk

        # Match the schedule's true max sigma; see the note in train_util.log_samples.
        sched_sigma_max = float(diffusion.timestep2sigma(diffusion.num_timesteps - 1))
        samples, _ = diffusion.heun_sample(
            sampling_model,
            x0.shape,
            sigma_max=sched_sigma_max,
            clip_denoised=True,
            model_kwargs=model_kwargs,
            latent_mask=latent_mask.cpu(),
            return_decoded=False,
        )
        samples = samples.to(device)
        # Keep the observed half exactly as given; only the generated half is model output.
        samples = samples * latent_mask + x0 * obs_mask

        for j in range(b):
            row = valset.rows[lo + j]
            pred, gt = samples[j], x0[j]

            # frame 10 == first generated frame: the cleanest world-reconstruction read
            m_next = metrics(pred[n_obs:n_obs + 1], gt[n_obs:n_obs + 1])
            # the full 1s continuation: drift trend, VDT-vs-VDT only
            m_roll = metrics(pred[n_obs:], gt[n_obs:])

            rec = {"row": row["num"], "prompt": row["prompt"], "type": row["test_type"]}
            rec.update({f"next/{k}": v for k, v in m_next.items()})
            rec.update({f"roll/{k}": v for k, v in m_roll.items()})
            per_row.append(rec)

            slug = valset.slug(row)
            # Off by default: 9 tasks x 6 metrics is 54 channels of mostly noise.
            # The aggregates say whether it is learning; the overlays say how it
            # fails. Per-task scalars are for drilling into one bad task.
            if per_task_scalars:
                for k, v in m_next.items():
                    logger.logkv(f"val/per_task/{slug}/{k}", v, distributed=False)

            if log_videos:
                mp4 = out_dir / f"step{step}_{slug}.mp4"
                try:
                    render_overlay(
                        gt_frames=gt.cpu().numpy(),
                        pred_frames=pred.cpu().numpy(),
                        session_db_path=str(row["session_db"]),
                        start_frame_idx=row["window_start"],
                        out_path=str(mp4),
                        n_observed=n_obs,
                        title=row["prompt"],
                    )
                    import wandb
                    logger.logkv(f"val/overlay/{slug}", wandb.Video(str(mp4)), distributed=False)
                except Exception as e:
                    # An overlay failure must never take down a training run.
                    print(f"[debug_validation] overlay failed for row {row['num']}: {e!r}")

            # THE SWAP TEST (acceptance criterion): same model, same context,
            # three action tensors differing only on the generated half. If
            # conditioning works these L2s are non-zero; if the model ignores
            # actions they collapse toward 0. Never let a failure here kill a
            # training run.
            if swap_test and action_conditioned and act_chunk is not None:
                try:
                    act_true_j = act_chunk[j:j + 1]
                    act_swap_j = _swap_actions(act_true_j, n_obs)
                    act_zero_j = _zero_actions(act_true_j, n_obs)
                    x0_j = x0[j:j + 1]
                    obs_mask_j = obs_mask[j:j + 1]
                    latent_mask_j = latent_mask[j:j + 1]
                    # Same starting noise for all three passes -- only the actions
                    # tensor should differ between them, or the comparison is
                    # dominated by heun_sample's noise rather than by whether the
                    # model listens to actions.
                    shared_noise = th.randn(*x0_j.shape, device=device)
                    # Pinning `noise=` is NOT sufficient. heun_sample is an EDM
                    # sampler with stochastic churn: with S_churn=80 over 50 steps
                    # gamma is min(80/50, sqrt(2)-1) = 0.414, and it draws a fresh
                    # th.randn_like from the GLOBAL rng at every step
                    # (gaussian_diffusion.py:829). Without a fixed seed around each
                    # pass those independent draws swamp the action effect and
                    # val/swap/* would measure the sampler, not the conditioning --
                    # it would look non-zero even for a model that ignores actions.
                    swap_seed = 20250813 + int(row["num"])

                    def _sample_with_actions(act):
                        with RNG(swap_seed):
                            s, _ = diffusion.heun_sample(
                                sampling_model, x0_j.shape, noise=shared_noise,
                                sigma_max=sched_sigma_max,
                                clip_denoised=True,
                                model_kwargs={
                                    "frame_indices": None, "x0": x0_j,
                                    "obs_mask": obs_mask_j, "latent_mask": latent_mask_j,
                                    "actions": act,
                                },
                                latent_mask=latent_mask_j.cpu(), return_decoded=False,
                            )
                        s = s.to(device)
                        return (s * latent_mask_j + x0_j * obs_mask_j)[0]

                    true_full = _sample_with_actions(act_true_j)
                    swap_full = _sample_with_actions(act_swap_j)
                    zero_full = _sample_with_actions(act_zero_j)

                    true01 = _to01(true_full[n_obs:])
                    swap01 = _to01(swap_full[n_obs:])
                    zero01 = _to01(zero_full[n_obs:])
                    gt01 = _to01(gt[n_obs:])

                    swap_rows.append(dict(
                        l2_true_swap=float(th.mean((true01 - swap01) ** 2)),
                        l2_true_zero=float(th.mean((true01 - zero01) ** 2)),
                        l2_swap_zero=float(th.mean((swap01 - zero01) ** 2)),
                        psnr_true=float(metrics.psnr(true01, gt01)),
                    ))

                    if log_videos:
                        mp4_swap = out_dir / f"step{step}_{slug}_swap.mp4"
                        _render_triple_overlay(
                            frames_gt=gt.cpu().numpy(),
                            frames_true=true_full.cpu().numpy(),
                            frames_swap=swap_full.cpu().numpy(),
                            frames_zero=zero_full.cpu().numpy(),
                            actions_true=act_true_j[0].cpu().numpy(),
                            actions_swap=act_swap_j[0].cpu().numpy(),
                            actions_zero=act_zero_j[0].cpu().numpy(),
                            n_observed=n_obs, out_path=str(mp4_swap),
                            title=row["prompt"],
                        )
                        import wandb
                        logger.logkv(f"val/swap_overlay/{slug}", wandb.Video(str(mp4_swap)), distributed=False)
                except Exception as e:
                    print(f"[debug_validation] swap test failed for row {row['num']}: {e!r}")

    # Key naming mirrors plaicraft-model-pi0's `val/video/*` so the same wandb
    # panels line up against the DiT runs.
    #   val/video/*      -> frame 10 only: next-frame reconstruction (the headline)
    #   val/video_roll/* -> frames 10-19: 1s drift, VDT-vs-VDT comparisons only
    METRIC_KEYS = ("psnr", "ssim", "lpips", "l2", "rmse", "l1")
    for scope, prefix in (("next", "val/video"), ("roll", "val/video_roll")):
        for k in METRIC_KEYS:
            vals = [r[f"{scope}/{k}"] for r in per_row]
            agg[f"{prefix}/{k}"] = float(np.mean(vals))

    if swap_rows:
        agg["val/swap/l2_true_vs_swap"] = float(np.mean([r["l2_true_swap"] for r in swap_rows]))
        agg["val/swap/l2_true_vs_zero"] = float(np.mean([r["l2_true_zero"] for r in swap_rows]))
        agg["val/swap/l2_swap_vs_zero"] = float(np.mean([r["l2_swap_zero"] for r in swap_rows]))
        agg["val/swap/psnr_true"] = float(np.mean([r["psnr_true"] for r in swap_rows]))

    for k, v in agg.items():
        logger.logkv(k, v, distributed=False)

    return {"aggregate": agg, "per_row": per_row}
