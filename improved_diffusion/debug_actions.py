# Per-frame action arrays for plaicraft-debug sessions, cached to <session_dir>/actions_{keypress,mouse}.npy.
import os
import sqlite3
from pathlib import Path

import h5py
import numpy as np
import torch as th

from improved_diffusion.decode_debug import FRAME_DURATION_MS

KEYPRESS_DIM = 8
MOUSE_DIM = 2

# Fixed key order for dims 0-5: [w, a, s, d, space, shift]
_KEY_IDS = ["87", "65", "83", "68", "32", "340"]


def _symlog(v):
    return np.sign(v) * np.log1p(np.abs(v))


def quantize_keypress(x):
    """Snap a continuous (..., 8) keypress prediction to the nearest of the 256 valid
    multi-hot vectors. Every codebook entry is a corner of the unit hypercube, so
    nearest-neighbour in L2 reduces to independent per-dim rounding (plaicraft-debug#77)."""
    return (x > 0.5).float()


def build_action_array(session_db_path, n_frames):
    """
    Returns (keypress, mouse): (n_frames, 8) and (n_frames, 2) float32.
      keypress 0-5: held keys [w,a,s,d,space,shift] during the frame's window
      keypress 6-7: held mouse clicks [left, right]
      mouse 0-1: symlog(sum mouseDX), symlog(sum mouseDY) over the window

    CAUSAL SHIFT: row i holds the action from window [i-1, i) -- the action
    that CAUSED frame i. Row 0 is all zeros.
    """
    con = sqlite3.connect(str(session_db_path))
    cur = con.cursor()
    cur.execute("SELECT key_id, start_timestamp, end_timestamp FROM keyboard")
    key_rows = cur.fetchall()
    cur.execute("SELECT mouse_key_type, start_timestamp, end_timestamp FROM mouse_click")
    click_rows = cur.fetchall()
    cur.execute("SELECT timestamp, mouseDX, mouseDY FROM mouse_movement")
    mouse_rows = cur.fetchall()
    con.close()

    # Raw per-window arrays: K[k]/M[k] is the action during window [k, k+1).
    K = np.zeros((n_frames, KEYPRESS_DIM), dtype=np.float32)
    M = np.zeros((n_frames, MOUSE_DIM), dtype=np.float32)
    for k in range(n_frames):
        win_start = k * FRAME_DURATION_MS
        win_end = win_start + FRAME_DURATION_MS

        for j, key_id in enumerate(_KEY_IDS):
            held = any(
                str(kid) == key_id and s < win_end and e > win_start
                for kid, s, e in key_rows
            )
            K[k, j] = 1.0 if held else 0.0

        for j, btn in enumerate(("left", "right")):
            held = any(
                b == btn and s < win_end and e > win_start
                for b, s, e in click_rows
            )
            K[k, 6 + j] = 1.0 if held else 0.0

        dx_sum = 0.0
        dy_sum = 0.0
        for ts, dx, dy in mouse_rows:
            if win_start <= ts < win_end:
                dx_sum += dx
                dy_sum += dy
        M[k, 0] = _symlog(dx_sum)
        M[k, 1] = _symlog(dy_sum)

    out_k = np.zeros_like(K)
    out_k[1:] = K[:-1]
    out_m = np.zeros_like(M)
    out_m[1:] = M[:-1]
    return out_k, out_m


def _n_frames_from_hdf5(session_dir):
    sid = Path(session_dir).name
    hdf5_path = Path(session_dir) / "encoded_video_hdf5" / f"{sid}_encoded_video.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        return f["frames"].shape[0]


def _load_cached(cache_path, n_frames, expected_dim):
    if cache_path.exists():
        arr = np.load(cache_path, mmap_mode="r")
        if arr.shape[0] == n_frames and arr.shape[1] == expected_dim:
            return arr
    return None  # missing, or stale (frame count or dim mismatch)


def load_or_build(session_dir):
    """Cache build_action_array's output to <session_dir>/actions_{keypress,mouse}.npy."""
    session_dir = Path(session_dir)
    sid = session_dir.name
    n_frames = _n_frames_from_hdf5(session_dir)
    keypress_path = session_dir / "actions_keypress.npy"
    mouse_path = session_dir / "actions_mouse.npy"

    keypress = _load_cached(keypress_path, n_frames, KEYPRESS_DIM)
    mouse = _load_cached(mouse_path, n_frames, MOUSE_DIM)
    if keypress is not None and mouse is not None:
        return keypress, mouse

    db_path = session_dir / f"{sid}.db"
    keypress, mouse = build_action_array(db_path, n_frames)
    for path, arr in ((keypress_path, keypress), (mouse_path, mouse)):
        tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp_path, arr)
        os.replace(tmp_path, path)  # atomic within same directory

    return np.load(keypress_path, mmap_mode="r"), np.load(mouse_path, mmap_mode="r")
