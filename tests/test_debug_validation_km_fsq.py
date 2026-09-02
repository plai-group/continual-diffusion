"""plaicraft-debug#80: km_fsq wiring in debug_validation.py.

Overlays/metrics decode through the real tokenizer; interventions stay on raw
8+2 arrays and only ever encode LIVE (never decode-then-re-encode, see the
module-level note above _invert_actions in debug_validation.py).
"""
import sqlite3

import h5py
import numpy as np
import torch

from improved_diffusion import debug_actions as da
from improved_diffusion import debug_validation as dv
from improved_diffusion.km_tokenizer.keypress_scatter import scatter_keypress
from improved_diffusion.km_tokenizer.model import load_tokenizer


def test_frame_timing_is_80ms_12_5fps():
    assert dv.VIDEO_FPS == 12.5
    assert dv.MS_PER_FRAME == 80.0
    from improved_diffusion.decode_debug import DECODE_VIDEO_FPS, FRAME_DURATION_MS
    assert DECODE_VIDEO_FPS == 12.5
    assert FRAME_DURATION_MS == 80.0


def test_action_vec_to_bar_passes_pixels_through_unchanged():
    # plaicraft-debug#80: mouse is raw pixels now (B2), not symlog -- no inversion needed.
    bar = dv._action_vec_to_bar(np.zeros(8), np.array([12.0, -7.0]))
    assert bar["mouseDX"] == 12.0
    assert bar["mouseDY"] == -7.0


def test_get_km_tokenizer_is_cached_singleton():
    dv._KM_TOKENIZER = None
    t1 = dv._get_km_tokenizer("cpu")
    t2 = dv._get_km_tokenizer("cpu")
    assert t1 is t2


def test_action_metrics_symlogs_mouse_for_comparable_scale():
    p_mouse = torch.tensor([[100.0, -100.0]])
    g_mouse = torch.tensor([[0.0, 0.0]])
    out = dv._action_metrics(None, None, p_mouse, g_mouse, slice(None))
    expected_l1 = float(torch.log1p(torch.tensor(100.0)))
    assert abs(out["mouse_l1"] - expected_l1) < 1e-4


def test_decode_then_encode_km_actions_round_trip():
    tokenizer = load_tokenizer()
    torch.manual_seed(0)
    keys_raw = (torch.rand(1, 3, 8) > 0.7).float()
    mouse_raw = torch.randint(-20, 21, (1, 3, 2)).float()

    codes = dv._encode_km_actions(tokenizer, keys_raw, mouse_raw)
    assert codes.shape == (1, 3, da.KM_CODE_DIM)

    keys_hat, mouse_hat = dv._decode_km_actions(tokenizer, codes)
    assert keys_hat.shape == (1, 3, 8)
    assert mouse_hat.shape == (1, 3, 2)
    # A trained tokenizer should recover simple, sparse actions near-exactly.
    assert (mouse_hat - mouse_raw).abs().mean().item() < 2.0


def _make_session(tmp_path, n_ticks=3):
    session_dir = tmp_path / "sess"
    (session_dir / "encoded_video_hdf5").mkdir(parents=True)
    with h5py.File(session_dir / "encoded_video_hdf5" / "sess_encoded_video.hdf5", "w") as f:
        f.create_dataset("frames", data=np.zeros((n_ticks, 3, 2, 2), dtype=np.float32))
    con = sqlite3.connect(str(session_dir / "sess.db"))
    con.execute("CREATE TABLE session (fps REAL)")
    con.execute("INSERT INTO session VALUES (12.5)")
    con.execute("CREATE TABLE keyboard (key_id TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_click (mouse_key_type TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_movement (timestamp INTEGER, mouseDX INTEGER, mouseDY INTEGER)")
    con.execute("INSERT INTO keyboard VALUES ('87', 0, 80)")
    for t in range(n_ticks):
        for b in range(da.SUBBINS_PER_TICK):
            con.execute("INSERT INTO mouse_movement VALUES (?, ?, ?)", (t * 80 + b * 10, 1, -1))
    con.commit()
    con.close()
    return session_dir


def test_load_action_window_raw_ignores_action_encoding(tmp_path):
    session_dir = _make_session(tmp_path, n_ticks=3)
    row = dict(num=1, session_dir=session_dir, window_start=0)

    valset = dv.DebugValidationSet.__new__(dv.DebugValidationSet)
    valset.T = 3
    valset.action_encoding = "km_fsq"
    valset.tokenizer_checkpoint = None

    raw_keypress, raw_mouse = valset.load_action_window_raw(row)
    assert raw_keypress.shape == (3, da.KEYPRESS_DIM)
    assert raw_mouse.shape == (3, da.MOUSE_DIM)
    # Same as the always-raw accessor, independent of the encoding on the object.
    expected_keypress, expected_mouse = da.load_or_build_raw(session_dir)
    assert torch.equal(raw_keypress, torch.from_numpy(np.asarray(expected_keypress, dtype=np.float32)))
    assert torch.equal(raw_mouse, torch.from_numpy(np.asarray(expected_mouse, dtype=np.float32)))
