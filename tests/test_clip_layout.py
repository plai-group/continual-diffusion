"""layout=clips corpora (independent fresh-encoded Wan clips): plaicraft-debug#99."""

import h5py
import numpy as np
import pytest
import torch

from improved_diffusion import debug_actions
from improved_diffusion.debug_dataset import ChunkedDebugDataset, ContinuousDebugDataset
from test_player_dataset import _session

N_TICKS, N_CLIPS, D = 100, 4, 2  # clip c = ticks [4+20c, 25+20c), so 4 clips fit in 100 ticks


def _clip_corpus(tmp_path, sids=("s00",), T=6):
    for sid in sids:
        d = _session(tmp_path, sid, n_ticks=N_TICKS, fps=10.0)
        path = d / "encoded_video_hdf5" / f"{sid}_encoded_video.hdf5"
        frames = np.arange(N_CLIPS * 3 * T * 2 * 2, dtype=np.float16).reshape(N_CLIPS, 3, T, 2, 2)
        with h5py.File(path, "w") as f:
            ds = f.create_dataset("frames", data=frames)
            ds.attrs.update(layout="clips", clip_ticks=21, clip_stride=20, clip_start=4, n_ticks=N_TICKS, fps=10, vae="wan22", ticks_per_latent=4)
    return tmp_path, frames


def _ds(root, **kw):
    ds = ContinuousDebugDataset(root, window_length=6, action_encoding="km_ctx14", **kw)
    kp = np.arange(N_TICKS * D, dtype=np.float32).reshape(N_TICKS, D)
    ds._get_action_arrays = lambda _d: (kp, -kp[:, :1])
    return ds, kp


def test_one_window_per_clip_with_clip_shapes(tmp_path):
    root, frames = _clip_corpus(tmp_path)
    ds, _ = _ds(root)
    assert len(ds) == N_CLIPS and ds.clips == (4, 20, 21)
    x, idx, keypress, mouse, _p = ds[2]
    assert x.shape == (6, 3, 2, 2) and x.dtype == torch.float32 and keypress.shape == (6, 4 * D) and mouse.shape == (6, 4)
    assert torch.equal(x, torch.from_numpy(frames[2]).float().transpose(0, 1))  # (C, T, h, w) -> (T, C, h, w)
    assert idx.shape == (6,)


def test_actions_cover_the_clip_plus_three_history_ticks(tmp_path):
    root, _ = _clip_corpus(tmp_path)
    ds, kp = _ds(root)
    for c in (0, 3):
        keypress = ds[c][2]
        lo = 4 + 20 * c - 3  # 3 history ticks, then the 21 clip ticks
        assert torch.equal(keypress.reshape(24, D), torch.from_numpy(kp[lo:lo + 24]))
    # latent L's last action tick is the tick of its frame, 4 + 20c + 4L
    assert torch.equal(ds[1][2][3].reshape(4, D)[-1], torch.from_numpy(kp[4 + 20 + 12]))


def test_chunked_dataset_keeps_one_window_per_clip_and_rejects_wrong_T(tmp_path):
    root, _ = _clip_corpus(tmp_path)
    assert len(ChunkedDebugDataset(root, window_length=6, action_encoding="km_ctx14")) == N_CLIPS
    with pytest.raises(AssertionError, match="window_length"):
        ContinuousDebugDataset(root, window_length=5, action_encoding="km_ctx14")


def test_windows_never_cross_sessions_and_players_align(tmp_path):
    root, _ = _clip_corpus(tmp_path, sids=("s00", "s01"))
    ds, _ = _ds(root)
    assert len(ds) == 2 * N_CLIPS and len(ds.window_players()) == 2 * N_CLIPS


def test_n_ticks_comes_from_the_attr(tmp_path):
    root, _ = _clip_corpus(tmp_path)
    assert debug_actions._n_ticks_from_hdf5(root / "s00") == N_TICKS
