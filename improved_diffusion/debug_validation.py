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
80ms past the last observed one -- is the cleanest read, because 80ms of
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
from .action_masks import frame_mask_to_action_mask
from .gaussian_diffusion import _unpack_action_mouse_out
from .logger import logger
from .rng_util import RNG
from .decode_debug import (
    render_overlay,
    get_frame_actions,
    _to_uint8_frame,
    _overlay_frame,
    DECODE_VIDEO_FPS,
)

VIDEO_FPS = 12.5  # plaicraft-debug#80: the corpus's session.fps
MS_PER_FRAME = 1000.0 / VIDEO_FPS

# 12.5Hz action grid (plaicraft-debug#80): a tick's raw action is broadcast across
# its 8 sub-bins when encoding an intervention live through the km tokenizer.
SUBBINS_PER_TICK = debug_actions.SUBBINS_PER_TICK

# LPIPS runs a VGG backbone with five 2x downsamples; a 24x40 frame collapses to
# nothing by the last stage. Upsample (nearest, to avoid inventing detail that
# is not in the source pixels) before scoring.
LPIPS_UPSAMPLE = 4


class DebugValidationSet:
    """The validation rows plus the frames each one needs."""

    def __init__(self, db_path, data_root, T=20, n_observed=10, action_encoding="raw", tokenizer_checkpoint=None):
        self.db_path = Path(db_path)
        self.data_root = Path(data_root)
        self.T = T
        self.n_observed = n_observed
        self.n_generated = T - n_observed
        self.action_encoding = action_encoding
        self.tokenizer_checkpoint = tokenizer_checkpoint
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
            start_frame = int(r_start_ms // MS_PER_FRAME)  # first generated frame

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
        """((T, 8), (T, 2)) float32: same window as load_window, causally-shifted keypress/mouse."""
        keypress, mouse = debug_actions.load_or_build(
            row["session_dir"], action_encoding=self.action_encoding, tokenizer_checkpoint=self.tokenizer_checkpoint,
        )
        ws, we = row["window_start"], row["window_start"] + self.T
        if we > keypress.shape[0]:
            raise ValueError(
                f"row {row['num']}: action array has {keypress.shape[0]} frames, "
                f"need window {ws}..{we}"
            )
        return (th.from_numpy(np.asarray(keypress[ws:we], dtype=np.float32)),
                th.from_numpy(np.asarray(mouse[ws:we], dtype=np.float32)))

    def load_all_actions(self):
        windows = [self.load_action_window(r) for r in self.rows]
        keypress = th.stack([k for k, _ in windows])
        mouse = th.stack([m for _, m in windows])
        return keypress, mouse

    def load_action_window_raw(self, row):
        """((T, 8), (T, 2)) float32, ALWAYS the tick-resolution raw keypress/mouse --
        independent of action_encoding. Overlays and interventions read this, never the
        model's native conditioning tensor, so a km_fsq run's swap/zero test still
        operates on real keys and pixels (plaicraft-debug#80)."""
        keypress, mouse = debug_actions.load_or_build_raw(row["session_dir"])
        ws, we = row["window_start"], row["window_start"] + self.T
        return (th.from_numpy(np.asarray(keypress[ws:we], dtype=np.float32)),
                th.from_numpy(np.asarray(mouse[ws:we], dtype=np.float32)))

    def load_all_actions_raw(self):
        windows = [self.load_action_window_raw(r) for r in self.rows]
        keypress = th.stack([k for k, _ in windows])
        mouse = th.stack([m for _, m in windows])
        return keypress, mouse


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


_KM_TOKENIZER = None


def _get_km_tokenizer(device, checkpoint=None):
    """Lazily load the frozen km tokenizer once, cached like _get_metrics -- validation is
    the only place this module needs it, for decoding a km_fsq run's generated codes and
    for encoding a swap/zero intervention live (plaicraft-debug#80)."""
    global _KM_TOKENIZER
    if _KM_TOKENIZER is None:
        from .km_tokenizer.model import DEFAULT_CHECKPOINT, load_tokenizer

        _KM_TOKENIZER = load_tokenizer(checkpoint_path=checkpoint or DEFAULT_CHECKPOINT, device=device)
    return _KM_TOKENIZER


def _symlog(v):
    """Metric-time-only compression so mouse_l1/mouse_mse stay on their historical scale
    even though the underlying targets are raw pixels for both raw and km_fsq (#80's B2)."""
    return th.sign(v) * th.log1p(th.abs(v))


def _decode_km_actions(tokenizer, codes):
    """(B, T, 36) quantized km codes -> ((B, T, 8) keys bool, (B, T, 2) mouse raw pixels).
    keys: sigmoid(key_logits) at _RAW_POSITIONS, mean over the tick's 8 sub-bins, thresholded.
    mouse: mouse_pred summed over the tick's 8 sub-bins."""
    from .km_tokenizer.keypress_scatter import _RAW_POSITIONS

    B, T, _ = codes.shape
    codes = codes.to(dtype=tokenizer.frame_pos.dtype)  # heun_sample's noise math can drift to float64
    key_logits, _mouse_logits, mouse_pred = tokenizer.decode_codes(
        codes.reshape(B, T, tokenizer.config.num_tokens, tokenizer.config.fsq_dim)
    )
    keys = (th.sigmoid(key_logits[..., _RAW_POSITIONS]).mean(dim=2) > 0.5).float()  # (B, T, 8)
    mouse = mouse_pred.sum(dim=2)  # (B, T, 2), raw pixels
    return keys, mouse


def _encode_km_actions(tokenizer, keys_raw, mouse_raw):
    """(B, T, 8) keys + (B, T, 2) mouse, tick-resolution raw -> (B, T, 36) km codes.

    Broadcasts each tick's single action across its 8 sub-bins: keys repeat (matching how
    debug's own tick-aligned intervals already look after 10ms binning, see debug_actions),
    mouse splits evenly so the sub-bins sum back to the tick's pixel total. This is the ONLY
    place an intervention touches the tokenizer -- it runs on raw arrays, never on a decoded
    round trip (see the module-level note on why #74 decode-then-re-encode was a bug)."""
    from .km_tokenizer.keypress_scatter import scatter_keypress

    B, T, _ = keys_raw.shape
    n = SUBBINS_PER_TICK
    dtype = tokenizer.frame_pos.dtype
    keys_raw, mouse_raw = keys_raw.to(dtype=dtype), mouse_raw.to(dtype=dtype)
    keys_sub = keys_raw.unsqueeze(2).expand(B, T, n, 8).reshape(B, T * n, 8)
    mouse_sub = (mouse_raw / n).unsqueeze(2).expand(B, T, n, 2).reshape(B, T * n, 2)
    key_press = scatter_keypress(keys_sub)
    with th.no_grad():
        prequantized, _frame_mask, _block_mask = tokenizer._encode_prequantized(key_press, mouse_sub)
        _token_ids, codes = tokenizer._quantize(prequantized)
    return codes.reshape(B, T, debug_actions.KM_CODE_DIM)


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


# _invert_actions/_swap_actions/_zero_actions always operate on the RAW 8+2 tick-resolution
# arrays (DebugValidationSet.load_all_actions_raw), never on a km_fsq run's native 36-dim
# codes and never on a decode_codes() round trip. Issue-74's keypress autoencoder decoded
# latents, intervened, then re-encoded -- feeding the encoder out-of-distribution logits,
# which made shift/space swaps silently vanish (fixed in 6a38b88). Keeping the raw arrays
# around and only encoding LIVE, after intervening (see _encode_km_actions), avoids that
# failure mode structurally rather than by discipline.
def _invert_actions(keypress, mouse):
    """The OPPOSITE action on every axis.

        w <-> s          (dims 0, 2)  forward / back
        a <-> d          (dims 1, 3)  strafe left / right
        space <-> shift  (dims 4, 5)  up / down
        left <-> right   (dims 6, 7)  mouse buttons
        dx, dy negated   (mouse dims 0, 1)  look direction

    A full inversion rather than a partial one: if the model is listening,
    every axis pushes the frame the other way, which makes the true-vs-swap
    divergence as large as this world allows.
    """
    swapped_k = keypress.clone()
    for i, j in ((0, 2), (1, 3), (4, 5), (6, 7)):
        swapped_k[..., i] = keypress[..., j]
        swapped_k[..., j] = keypress[..., i]
    return swapped_k, -mouse


def _swap_actions(keypress, mouse, where):
    """Invert the actions on the rows `where` selects; leave the rest alone.

    `where` is a (B, T, 1) 0/1 mask selecting rows n_obs..T-1: the actions that
    drive the generated frames. The history stays true, so every panel's action
    bar agrees with the ground-truth context frames underneath it.

    For an action-generating model row n_obs is pinned (it is `observed` in the
    action mask) and rows n_obs+1.. stay latent, so the swap lands on the
    boundary action and the model then generates its own future actions and
    frames under it. For a conditioned-only model there is nothing to generate
    and the whole swapped future is imposed directly.
    """
    inv_k, inv_m = _invert_actions(keypress, mouse)
    return th.where(where.bool(), inv_k, keypress), th.where(where.bool(), inv_m, mouse)


def _zero_actions(keypress, mouse, where):
    """Zero the actions on the rows `where` selects; leave the rest alone."""
    return (th.where(where.bool(), th.zeros_like(keypress), keypress),
            th.where(where.bool(), th.zeros_like(mouse), mouse))


#: order of the 6 key dims in the 8-d keypress vector; names must match the
#: labels decode_debug draws, i.e. the values of its KEY_ID_TO_NAME.
_ACTION_KEY_NAMES = ["w", "a", "s", "d", "space", "Shift_L"]
_ACTION_CLICK_NAMES = ["left", "right"]


def _action_vec_to_bar(key_vec, mouse_vec):
    """One 8-d keypress vector + one 2-d mouse vector (raw pixels, plaicraft-debug#80 --
    NOT symlog) -> the dict decode_debug._overlay_frame draws.

    Lets each row of the swap overlay show the actions it was ACTUALLY generated
    with. Reading the bar from the session DB instead would paint the true
    actions onto the swap and zero rows too, which hides the very thing the
    overlay exists to show.
    """
    key_vec = np.asarray(key_vec, dtype=np.float32)
    mouse_vec = np.asarray(mouse_vec, dtype=np.float32)
    return {
        "keys": [n for i, n in enumerate(_ACTION_KEY_NAMES) if key_vec[i] > 0.5],
        "clicks": [n for i, n in enumerate(_ACTION_CLICK_NAMES) if key_vec[6 + i] > 0.5],
        "mouseDX": float(mouse_vec[0]),
        "mouseDY": float(mouse_vec[1]),
    }


def _to_display_actions(a):
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
    if hasattr(a, 'detach'):
        a = a.detach().float().cpu().numpy()
    a = np.asarray(a)
    out = np.zeros_like(a)
    out[:-1] = a[1:]
    return out


def _action_bars(keypress, mouse):
    """(T, 8) + (T, 2) causal action arrays -> the T bar dicts render_overlay draws."""
    dk, dm = _to_display_actions(keypress), _to_display_actions(mouse)
    return [_action_vec_to_bar(dk[t], dm[t]) for t in range(dk.shape[0])]


_QUANTIZE_FNS = {
    "none": lambda x: (x > 0.5).float(),
    # Module-attribute lookups, not bound references, so tests can monkeypatch either function.
    "codebook": lambda x: debug_actions.quantize_keypress(x),
    "fsq": lambda x: debug_actions.quantize_km_fsq(x),
}


def _action_metrics(p_key, g_key, p_mouse, g_mouse, sl, quantize="none"):
    """Action metrics over one frame window, plus the all-zeros baseline.

    Keypress and mouse are computed independently -- either side may be None
    (that modality wasn't generated) and simply contributes no keys.

    Keys are pressed 2-33% of the time in this corpus, so the do-nothing
    predictor already scores ~0.93 key_acc and its mouse_mse is ~3.4. Without
    the *_trivial series a dead action head and a learning one look alike on
    the dashboard. They are measured from the GT rows in this batch rather
    than hard-coded, so they track whatever data the run actually saw.
    """
    out = {}
    if p_key is not None and g_key is not None:
        if quantize is True:
            quantize = "codebook"
        elif quantize is False:
            quantize = "none"
        quantize_fn = _QUANTIZE_FNS[quantize]
        p_k, g_k = quantize_fn(p_key[sl]), quantize_fn(g_key[sl])
        out["key_acc"] = float((p_k == g_k).float().mean().item())
        out["key_acc_trivial"] = float((g_k == 0).float().mean().item())
    if p_mouse is not None and g_mouse is not None:
        # symlog at metric time: p_mouse/g_mouse are raw pixels (#80's B2), but the wandb
        # metric stays on its historical, dynamic-range-compressed scale.
        p_m, g_m = _symlog(p_mouse[sl]), _symlog(g_mouse[sl])
        out["mouse_l1"] = float((p_m - g_m).abs().mean().item())
        out["mouse_mse"] = float(((p_m - g_m) ** 2).mean().item())
        out["mouse_l1_trivial"] = float(g_m.abs().mean().item())
        out["mouse_mse_trivial"] = float((g_m ** 2).mean().item())
    return out


def _label_panel(panel, text):
    """Draw a panel name in the top bar's empty middle band."""
    cv2.putText(panel, text, (panel.shape[1] // 2 - 60, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3, cv2.LINE_AA)
    return panel


def _render_triple_overlay(frames_gt, frames_true, frames_swap, frames_zero,
                           actions_gt, actions_true, actions_swap, actions_zero,
                           n_observed, out_path, title=None, true_label="TRUE"):
    """2x2 grid mp4:  GT | TRUE  over  SWAP | ZERO.

    GT is included so the generated half can be judged against reality, not only
    against the other two continuations -- past frame n_observed the true/swap/
    zero rows are all model output and share no ground truth.

    Laid out as a grid rather than 4 stacked rows because stacking gives a
    1280x3472 video, which wandb renders unusably small.

    Each panel's action bar comes from that panel's own action tensor, so the
    swap panel visibly shows the swapped keys / reversed mouse it was given.
    GT carries the recorded actions. When the model generates actions, the
    second panel carries the ones it GENERATED (label GEN), so GT vs GEN is a
    direct read of action-prediction quality; otherwise it carries the true
    actions it was conditioned on (label TRUE).

    Same imageio/libx264 settings as decode_debug.render_overlay -- cv2's mp4v
    encodes fine and then will not play in wandb.
    """
    frames_gt = np.asarray(frames_gt)
    frames_true = np.asarray(frames_true)
    frames_swap = np.asarray(frames_swap)
    frames_zero = np.asarray(frames_zero)
    T = frames_true.shape[0]

    # actions_* are each a (keypress, mouse) pair.
    acts = [(_to_display_actions(k), _to_display_actions(m)) for k, m in
            (actions_gt, actions_true, actions_swap, actions_zero)]
    labels = ["GT", true_label, "SWAP", "ZERO"]

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
                               _action_vec_to_bar(dk[t], dm[t]), border=border),
                lab)
            for frames, (dk, dm), lab in zip(
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
                         swap_test=True, cfg_scale=1.0,
                         teacher_force_actions=True):
    """Sample every validation row, render overlays, log metrics to wandb.

    Returns the aggregate metric dict (also logged via ``logger.logkv``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _get_metrics(device)

    _m = getattr(model, "module", model)
    # Generation mode has no action_embedder (the action is a denoised token), so gating on it hid the action path.
    generates_actions = (bool(getattr(_m, "generate_actions", False))
                         and getattr(_m, "action_dim", 0) > 0)
    generates_mouse = (bool(getattr(_m, "generate_mouse", False))
                       and getattr(_m, "mouse_dim", 0) > 0)
    # Token-cond mode has no action_embedder and no generation head, but actions are still its conditioning signal.
    action_conditioned = (getattr(_m, "action_embedder", None) is not None
                          or getattr(_m, "action_x_embedder", None) is not None
                          or getattr(_m, "mouse_x_embedder", None) is not None
                          or generates_actions or generates_mouse)
    action_quantization = getattr(diffusion, "action_quantization", "none")
    action_encoding = getattr(diffusion, "action_encoding", "raw")
    is_km_fsq = action_encoding == "km_fsq"
    km_tokenizer = _get_km_tokenizer(device, getattr(valset, "tokenizer_checkpoint", None)) if is_km_fsq else None
    sampling_model = _CFGWrapper(model, cfg_scale) if cfg_scale != 1.0 else model

    T, n_obs = valset.T, valset.n_observed
    x0_all = valset.load_all()  # (N, T, 3, H, W)
    n_rows = x0_all.shape[0]
    if actions and action_conditioned:
        keypress_all, mouse_all = valset.load_all_actions()  # native encoding: model conditioning
        keypress_raw_all, mouse_raw_all = valset.load_all_actions_raw()  # always 8+2: overlays/interventions
    else:
        keypress_all, mouse_all = None, None
        keypress_raw_all, mouse_raw_all = None, None

    per_row, agg, swap_rows = [], {}, []
    for lo in range(0, n_rows, chunk_size):
        hi = min(lo + chunk_size, n_rows)
        x0 = x0_all[lo:hi].to(device)
        b = x0.shape[0]
        keypress_chunk = keypress_all[lo:hi].to(device) if keypress_all is not None else None
        mouse_chunk = mouse_all[lo:hi].to(device) if mouse_all is not None else None
        keypress_raw_chunk = keypress_raw_all[lo:hi].to(device) if keypress_raw_all is not None else None
        mouse_raw_chunk = mouse_raw_all[lo:hi].to(device) if mouse_raw_all is not None else None

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
        # Teacher forcing pins the observed prefix's actions to GT (mask lags the frame mask by one row, see action_masks); False pins nothing, i.e. free rollout.
        if keypress_chunk is not None:
            model_kwargs["actions"] = keypress_chunk
            if generates_actions:
                obs_act_mask = frame_mask_to_action_mask(obs_mask)
                if not teacher_force_actions:
                    obs_act_mask = th.zeros_like(obs_act_mask)
                model_kwargs["actions0"] = keypress_chunk
                model_kwargs["obs_action_mask"] = obs_act_mask
        if mouse_chunk is not None:
            model_kwargs["mouse"] = mouse_chunk
            if generates_mouse:
                obs_mouse_mask = frame_mask_to_action_mask(obs_mask)
                if not teacher_force_actions:
                    obs_mouse_mask = th.zeros_like(obs_mouse_mask)
                model_kwargs["mouse0"] = mouse_chunk
                model_kwargs["obs_mouse_mask"] = obs_mouse_mask

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
        samples_act, samples_mouse = None, None
        if isinstance(samples, tuple):
            samples_video, second = samples
            samples_act, samples_mouse = _unpack_action_mouse_out(second, generates_actions, generates_mouse)
            samples = samples_video
        samples = samples.to(device)
        # Keep the observed half exactly as given; only the generated half is model output.
        samples = samples * latent_mask + x0 * obs_mask

        if is_km_fsq:
            # The "Key design decision": VDT regresses continuous codes; snap to the nearest
            # FSQ lattice point here, at inference, before decoding -- never during training.
            act_for_decode = (debug_actions.quantize_km_fsq(samples_act)
                              if action_quantization == "fsq" and samples_act is not None else samples_act)
            p_key_chunk, p_mouse_chunk = (_decode_km_actions(km_tokenizer, act_for_decode)
                                          if act_for_decode is not None else (None, None))
            g_key_chunk, g_mouse_chunk = (_decode_km_actions(km_tokenizer, keypress_chunk)
                                          if keypress_chunk is not None else (None, None))
            metrics_quantize = "none"  # already hard-thresholded booleans; nothing left to snap
        else:
            p_key_chunk, g_key_chunk = samples_act, keypress_chunk
            p_mouse_chunk, g_mouse_chunk = samples_mouse, mouse_chunk
            metrics_quantize = action_quantization

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
            p_key = p_key_chunk[j].to(device) if p_key_chunk is not None and g_key_chunk is not None else None
            g_key = g_key_chunk[j].to(device) if p_key is not None else None
            p_mouse = p_mouse_chunk[j].to(device) if p_mouse_chunk is not None and g_mouse_chunk is not None else None
            g_mouse = g_mouse_chunk[j].to(device) if p_mouse is not None else None
            if p_key is not None or p_mouse is not None:
                # Row n_obs is pinned GT (the action mask lags by one row), so the first genuinely generated action is row n_obs + 1.
                first_gen = n_obs + 1
                for scope, sl in (("next", slice(first_gen, first_gen + 1)),
                                  ("roll", slice(first_gen, None))):
                    rec.update({f"{scope}/{k}": v
                                for k, v in _action_metrics(p_key, g_key, p_mouse, g_mouse, sl, metrics_quantize).items()})
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
                    # Top row keeps the recorded actions; this row shows what the model produced (always
                    # 8+2/raw-pixels here, decoded already for km_fsq), falling back to the raw ground
                    # truth for any un-generated modality -- never to the native 36-dim km codes, which
                    # _action_bars cannot draw.
                    pred_bar_key = p_key_chunk[j] if p_key_chunk is not None else keypress_raw_chunk[j] if keypress_raw_chunk is not None else None
                    pred_bar_mouse = p_mouse_chunk[j] if p_mouse_chunk is not None else mouse_raw_chunk[j] if mouse_raw_chunk is not None else None
                    render_overlay(
                        gt_frames=gt.cpu().numpy(),
                        pred_frames=pred.cpu().numpy(),
                        pred_actions=(_action_bars(pred_bar_key, pred_bar_mouse)
                                      if pred_bar_key is not None and pred_bar_mouse is not None else None),
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

            # The swap test (acceptance criterion): true/swapped/zero actions on the same context.
            # Always intervenes on the RAW 8+2 tick-resolution arrays -- see the note above
            # _invert_actions -- and, for km_fsq, encodes the (possibly-intervened) result live
            # through the tokenizer only when building each pass's model_kwargs.
            if swap_test and action_conditioned and keypress_raw_chunk is not None and mouse_raw_chunk is not None:
                try:
                    key_true_j = keypress_raw_chunk[j:j + 1]
                    mouse_true_j = mouse_raw_chunk[j:j + 1]
                    x0_j = x0[j:j + 1]
                    obs_mask_j = obs_mask[j:j + 1]
                    latent_mask_j = latent_mask[j:j + 1]
                    obs_act_mask_j = frame_mask_to_action_mask(obs_mask_j)
                    # Rewrite only the actions driving the generated frames: rows n_obs..T-1, the frame-mask complement, not the action-mask complement (which lags one row) -- leave the history true.
                    intervene_on = 1.0 - obs_mask_j.reshape(
                        obs_mask_j.shape[0], obs_mask_j.shape[1], 1)
                    key_swap_j, mouse_swap_j = _swap_actions(key_true_j, mouse_true_j, intervene_on)
                    key_zero_j, mouse_zero_j = _zero_actions(key_true_j, mouse_true_j, intervene_on)
                    # Same starting noise for all three passes, so only the actions tensor differs.
                    shared_noise = th.randn(*x0_j.shape, device=device)
                    # heun_sample's churn draws from the global RNG each step; reseed per pass too.
                    swap_seed = 20250813 + int(row["num"])

                    def _sample_with_actions(key_raw, mouse_raw):
                        if is_km_fsq:
                            actions_in = _encode_km_actions(km_tokenizer, key_raw, mouse_raw)
                            mouse_in = key_raw.new_zeros(key_raw.shape[0], key_raw.shape[1], 0)
                        else:
                            actions_in, mouse_in = key_raw, mouse_raw
                        with RNG(swap_seed):
                            s, _ = diffusion.heun_sample(
                                sampling_model, x0_j.shape, noise=shared_noise,
                                sigma_max=sched_sigma_max,
                                clip_denoised=True,
                                model_kwargs={
                                    "frame_indices": None, "x0": x0_j,
                                    "obs_mask": obs_mask_j, "latent_mask": latent_mask_j,
                                    "actions": actions_in, "mouse": mouse_in,
                                    # Pin the history only; future action tokens are denoised jointly with the video.
                                    **({"actions0": actions_in, "obs_action_mask": obs_act_mask_j}
                                       if generates_actions else {}),
                                    **({"mouse0": mouse_in, "obs_mouse_mask": obs_act_mask_j}
                                       if generates_mouse else {}),
                                },
                                latent_mask=latent_mask_j.cpu(), return_decoded=False,
                            )
                        key_out, mouse_out = None, None
                        if isinstance(s, tuple):
                            s, second = s
                            key_out, mouse_out = _unpack_action_mouse_out(second, generates_actions, generates_mouse)
                        s = s.to(device)
                        video = (s * latent_mask_j + x0_j * obs_mask_j)[0]
                        if is_km_fsq and key_out is not None:
                            # Same snap-then-decode as the main pass, still batch=1 here.
                            act_out = debug_actions.quantize_km_fsq(key_out) if action_quantization == "fsq" else key_out
                            key_out, mouse_out = _decode_km_actions(km_tokenizer, act_out)
                        return (video, key_out[0] if key_out is not None else None,
                                mouse_out[0] if mouse_out is not None else None)

                    true_full, true_key, true_mouse = _sample_with_actions(key_true_j, mouse_true_j)
                    swap_full, swap_key, swap_mouse = _sample_with_actions(key_swap_j, mouse_swap_j)
                    zero_full, zero_key, zero_mouse = _sample_with_actions(key_zero_j, mouse_zero_j)

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
                            # All three passes share the GT frame context and true action history, differing only from row n_obs on.
                            frames_true=true_full.cpu().numpy(),
                            frames_swap=swap_full.cpu().numpy(),
                            frames_zero=zero_full.cpu().numpy(),
                            actions_gt=(key_true_j[0].cpu().numpy(), mouse_true_j[0].cpu().numpy()),
                            actions_true=((true_key if true_key is not None else key_true_j[0]).cpu().numpy(),
                                          (true_mouse if true_mouse is not None else mouse_true_j[0]).cpu().numpy()),
                            actions_swap=((swap_key if swap_key is not None else key_swap_j[0]).cpu().numpy(),
                                          (swap_mouse if swap_mouse is not None else mouse_swap_j[0]).cpu().numpy()),
                            actions_zero=((zero_key if zero_key is not None else key_zero_j[0]).cpu().numpy(),
                                          (zero_mouse if zero_mouse is not None else mouse_zero_j[0]).cpu().numpy()),
                            true_label=("GEN" if (true_key is not None or true_mouse is not None) else "TRUE"),
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
            vals = [r[f"{scope}/{k}"] for r in per_row if f"{scope}/{k}" in r]
            if vals:
                agg[f"{prefix}/{k}"] = float(np.mean(vals))
    ACT_METRIC_KEYS = ("key_acc", "mouse_l1", "mouse_mse",
                       "key_acc_trivial", "mouse_l1_trivial", "mouse_mse_trivial")
    for scope, prefix in (("next", "val/action"), ("roll", "val/action_roll")):
        for k in ACT_METRIC_KEYS:
            vals = [r[f"{scope}/{k}"] for r in per_row if f"{scope}/{k}" in r]
            if vals:
                agg[f"{prefix}/{k}"] = float(np.mean(vals))

    if swap_rows:
        agg["val/swap/l2_true_vs_swap"] = float(np.mean([r["l2_true_swap"] for r in swap_rows]))
        agg["val/swap/l2_true_vs_zero"] = float(np.mean([r["l2_true_zero"] for r in swap_rows]))
        agg["val/swap/l2_swap_vs_zero"] = float(np.mean([r["l2_swap_zero"] for r in swap_rows]))
        agg["val/swap/psnr_true"] = float(np.mean([r["psnr_true"] for r in swap_rows]))

    for k, v in agg.items():
        logger.logkv(k, v, distributed=False)

    return {"aggregate": agg, "per_row": per_row}
