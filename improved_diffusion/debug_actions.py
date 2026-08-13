# improved_diffusion/debug_actions.py
#
# Per-frame action vectors for the plaicraft-debug sqlite session DBs, cached
# to disk as <session_dir>/actions_10d.npy. Reuses the same window-timing and
# table-schema logic as decode_debug.get_frame_actions (100 ms/frame; keyboard
# / mouse_click / mouse_movement tables) rather than re-deriving it.
import os
import sqlite3
from pathlib import Path

import h5py
import numpy as np

from improved_diffusion.decode_debug import FRAME_DURATION_MS

ACTION_DIM = 10

# Fixed key order for dims 0-5: [w, a, s, d, space, shift]
_KEY_IDS = ["87", "65", "83", "68", "32", "340"]


def _symlog(v):
    return np.sign(v) * np.log1p(np.abs(v))


def build_action_array(session_db_path, n_frames):
    """
    Returns (n_frames, 10) float32:
      0-5: held keys [w,a,s,d,space,shift] during the frame's window
      6-7: held mouse clicks [left, right]
      8-9: symlog(sum mouseDX), symlog(sum mouseDY) over the window

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

    # Raw per-window array A: A[k] is the action during window [k, k+1).
    A = np.zeros((n_frames, ACTION_DIM), dtype=np.float32)
    for k in range(n_frames):
        win_start = k * FRAME_DURATION_MS
        win_end = win_start + FRAME_DURATION_MS

        for j, key_id in enumerate(_KEY_IDS):
            held = any(
                str(kid) == key_id and s < win_end and e > win_start
                for kid, s, e in key_rows
            )
            A[k, j] = 1.0 if held else 0.0

        for j, btn in enumerate(("left", "right")):
            held = any(
                b == btn and s < win_end and e > win_start
                for b, s, e in click_rows
            )
            A[k, 6 + j] = 1.0 if held else 0.0

        dx_sum = 0.0
        dy_sum = 0.0
        for ts, dx, dy in mouse_rows:
            if win_start <= ts < win_end:
                dx_sum += dx
                dy_sum += dy
        A[k, 8] = _symlog(dx_sum)
        A[k, 9] = _symlog(dy_sum)

    out = np.zeros_like(A)
    out[1:] = A[:-1]
    return out


def _n_frames_from_hdf5(session_dir):
    sid = Path(session_dir).name
    hdf5_path = Path(session_dir) / "encoded_video_hdf5" / f"{sid}_encoded_video.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        return f["frames"].shape[0]


def load_or_build(session_dir):
    """Cache build_action_array's output to <session_dir>/actions_10d.npy."""
    session_dir = Path(session_dir)
    sid = session_dir.name
    cache_path = session_dir / "actions_10d.npy"
    n_frames = _n_frames_from_hdf5(session_dir)

    if cache_path.exists():
        arr = np.load(cache_path, mmap_mode="r")
        if arr.shape[0] == n_frames:
            return arr
        # stale cache (frame count mismatch) -- rebuild below

    db_path = session_dir / f"{sid}.db"
    arr = build_action_array(db_path, n_frames)

    tmp_path = session_dir / f".actions_10d.{os.getpid()}.tmp.npy"
    np.save(tmp_path, arr)
    os.replace(tmp_path, cache_path)  # atomic within same directory

    return np.load(cache_path, mmap_mode="r")
