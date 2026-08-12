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

import numpy as np
import torch as th

from .logger import logger
from .decode_debug import render_overlay

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


@th.no_grad()
def run_debug_validation(model, diffusion, valset, device, out_dir,
                         step=0, chunk_size=3, log_videos=True,
                         per_task_scalars=False):
    """Sample every validation row, render overlays, log metrics to wandb.

    Returns the aggregate metric dict (also logged via ``logger.logkv``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _get_metrics(device)

    T, n_obs = valset.T, valset.n_observed
    x0_all = valset.load_all()  # (N, T, 3, H, W)
    n_rows = x0_all.shape[0]

    per_row, agg = [], {}
    for lo in range(0, n_rows, chunk_size):
        hi = min(lo + chunk_size, n_rows)
        x0 = x0_all[lo:hi].to(device)
        b = x0.shape[0]

        obs_mask = th.zeros(b, T, 1, 1, 1, device=device)
        obs_mask[:, :n_obs] = 1.0
        latent_mask = th.zeros_like(obs_mask)
        latent_mask[:, n_obs:] = 1.0

        # Match the schedule's true max sigma; see the note in train_util.log_samples.
        sched_sigma_max = float(diffusion.timestep2sigma(diffusion.num_timesteps - 1))
        samples, _ = diffusion.heun_sample(
            model,
            x0.shape,
            sigma_max=sched_sigma_max,
            clip_denoised=True,
            model_kwargs={
                "frame_indices": None,
                "x0": x0,
                "obs_mask": obs_mask,
                "latent_mask": latent_mask,
            },
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

    # Key naming mirrors plaicraft-model-pi0's `val/video/*` so the same wandb
    # panels line up against the DiT runs.
    #   val/video/*      -> frame 10 only: next-frame reconstruction (the headline)
    #   val/video_roll/* -> frames 10-19: 1s drift, VDT-vs-VDT comparisons only
    METRIC_KEYS = ("psnr", "ssim", "lpips", "l2", "rmse", "l1")
    for scope, prefix in (("next", "val/video"), ("roll", "val/video_roll")):
        for k in METRIC_KEYS:
            vals = [r[f"{scope}/{k}"] for r in per_row]
            agg[f"{prefix}/{k}"] = float(np.mean(vals))
    for k, v in agg.items():
        logger.logkv(k, v, distributed=False)

    return {"aggregate": agg, "per_row": per_row}
