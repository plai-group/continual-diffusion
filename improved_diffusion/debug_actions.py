# Per-tick action arrays for plaicraft-debug sessions, cached to <session_dir>/actions_{keypress,mouse}.npy.
import os
import sqlite3
from pathlib import Path

import h5py
import numpy as np

KEYPRESS_DIM = 8
MOUSE_DIM = 2

# The 12.5 Hz action grid (plaicraft-debug#80): one tick per video frame, one
# tokenizer control-frame every SUBBIN_MS ms within a tick.
TICK_MS = 80
SUBBIN_MS = 10
SUBBINS_PER_TICK = TICK_MS // SUBBIN_MS  # 8, matches the km tokenizer's block_size

# Fixed key order for dims 0-5: [w, a, s, d, space, shift]
_KEY_IDS = ["87", "65", "83", "68", "32", "340"]


def _symlog(v):
    """Compress unbounded pixel sums to ~+-5 so the model's mouse conditioning input keeps
    the DreamerV3 scale. Applied in load_or_build only, never in build_action_array."""
    return np.sign(v) * np.log1p(np.abs(v))


def quantize_keypress(x):
    """Snap a continuous (..., 8) keypress prediction to the nearest of the 256 valid
    multi-hot vectors. Every codebook entry is a corner of the unit hypercube, so
    nearest-neighbour in L2 reduces to independent per-dim rounding (plaicraft-debug#77)."""
    return (x > 0.5).float()


def build_action_array(session_db_path, n_ticks):
    """
    10 ms sub-bins over the whole window -> (key_press, mouse): (n_ticks*8, 8) and
    (n_ticks*8, 2) float32, one row per 10 ms sub-bin.
      keypress 0-5: held keys [w,a,s,d,space,shift] during that sub-bin
      keypress 6-7: held mouse clicks [left, right]
      mouse 0-1: raw pixel mouseDX, mouseDY summed at that sub-bin -- never symlogged
        here, the km tokenizer's own feature stem normalizes raw pixels itself, and
        load_or_build_raw needs this un-symlogged for ground truth. load_or_build applies
        symlog on top of this, but only for its own raw-mode model-conditioning output.

    This is the SAME code regardless of who wrote the DB: debug's key/click
    intervals happen to be tick-aligned, so all 8 sub-bins of a tick end up
    holding the same key state -- that "broadcast" emerges from the generic
    10 ms binning below, not from any special-cased repeat.

    CAUSAL SHIFT, applied per TICK (not per sub-bin): tick i's whole 8-row block
    holds the action from tick [i-1, i) -- the action that CAUSED frame i. The
    first tick's block is all zeros.
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

    n_sub = n_ticks * SUBBINS_PER_TICK
    sub_starts = np.arange(n_sub, dtype=np.float64) * SUBBIN_MS
    sub_ends = sub_starts + SUBBIN_MS

    K = np.zeros((n_sub, KEYPRESS_DIM), dtype=np.float32)
    M = np.zeros((n_sub, MOUSE_DIM), dtype=np.float32)

    def _fill(rows, id_list, out_col_offset):
        by_id = {}
        for ident, s, e in rows:
            by_id.setdefault(str(ident), []).append((float(s), float(e)))
        for j, ident in enumerate(id_list):
            intervals = by_id.get(ident)
            if not intervals:
                continue
            starts = np.array([s for s, _ in intervals])
            ends = np.array([e for _, e in intervals])
            overlap = (starts[None, :] < sub_ends[:, None]) & (ends[None, :] > sub_starts[:, None])
            K[:, out_col_offset + j] = overlap.any(axis=1)

    _fill(key_rows, _KEY_IDS, 0)
    _fill(click_rows, ("left", "right"), 6)

    # Containment, not exact-timestamp equality (debug data happens to land exactly on the
    # 10ms grid, but real PLAICraft timestamps are continuous) -- accumulate each row into
    # whichever sub-bin's [start, end) contains it. Vectorised via searchsorted/add.at rather
    # than a python loop over n_sub * n_rows, since the corpus is large.
    if mouse_rows:
        ts = np.array([r[0] for r in mouse_rows], dtype=np.float64)
        dx = np.array([r[1] for r in mouse_rows], dtype=np.float64)
        dy = np.array([r[2] for r in mouse_rows], dtype=np.float64)
        bin_idx = np.searchsorted(sub_starts, ts, side="right") - 1
        clamped = np.clip(bin_idx, 0, n_sub - 1)
        in_range = (bin_idx >= 0) & (bin_idx < n_sub) & (ts < sub_ends[clamped])
        bin_idx, dx, dy = bin_idx[in_range], dx[in_range], dy[in_range]
        np.add.at(M[:, 0], bin_idx, dx)
        np.add.at(M[:, 1], bin_idx, dy)

    K = K.reshape(n_ticks, SUBBINS_PER_TICK, KEYPRESS_DIM)
    M = M.reshape(n_ticks, SUBBINS_PER_TICK, MOUSE_DIM)
    out_k = np.zeros_like(K)
    out_k[1:] = K[:-1]
    out_m = np.zeros_like(M)
    out_m[1:] = M[:-1]
    return out_k.reshape(n_sub, KEYPRESS_DIM), out_m.reshape(n_sub, MOUSE_DIM)


def _n_ticks_from_hdf5(session_dir):
    sid = Path(session_dir).name
    hdf5_path = Path(session_dir) / "encoded_video_hdf5" / f"{sid}_encoded_video.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        return f["frames"].shape[0]


def _load_cached(cache_path, n_ticks, expected_dim):
    if cache_path.exists():
        arr = np.load(cache_path, mmap_mode="r")
        if arr.shape[0] == n_ticks and arr.shape[1] == expected_dim:
            return arr
    return None  # missing, or stale (tick count or dim mismatch)


def load_or_build_raw(session_dir):
    """Tick-resolution ground truth, for overlays and interventions: (n_ticks, 8) keys
    (OR-reduced over the tick's 8 sub-bins) + (n_ticks, 2) raw-pixel mouse (summed over
    the tick's 8 sub-bins). Already causally shifted -- see build_action_array."""
    session_dir = Path(session_dir)
    sid = session_dir.name
    n_ticks = _n_ticks_from_hdf5(session_dir)
    keypress_path = session_dir / "actions_keypress.npy"
    mouse_path = session_dir / "actions_mouse.npy"

    keypress = _load_cached(keypress_path, n_ticks, KEYPRESS_DIM)
    mouse = _load_cached(mouse_path, n_ticks, MOUSE_DIM)
    if keypress is not None and mouse is not None:
        return keypress, mouse

    db_path = session_dir / f"{sid}.db"
    key_sub, mouse_sub = build_action_array(db_path, n_ticks)
    keypress = key_sub.reshape(n_ticks, SUBBINS_PER_TICK, KEYPRESS_DIM).any(axis=1).astype(np.float32)
    mouse = mouse_sub.reshape(n_ticks, SUBBINS_PER_TICK, MOUSE_DIM).sum(axis=1).astype(np.float32)
    for path, arr in ((keypress_path, keypress), (mouse_path, mouse)):
        tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp_path, arr)
        os.replace(tmp_path, path)  # atomic within same directory

    return np.load(keypress_path, mmap_mode="r"), np.load(mouse_path, mmap_mode="r")


def load_or_build(session_dir):
    """(n_ticks, 8) keypress + (n_ticks, 2) symlog-compressed mouse -- the MODEL'S NATIVE
    CONDITIONING encoding, not ground truth (load_or_build_raw is that)."""
    keypress, mouse = load_or_build_raw(session_dir)
    return keypress, _symlog(np.asarray(mouse))
