# Per-tick action arrays for plaicraft-debug sessions, cached to <session_dir>/actions_{keypress,mouse}.npy.
import json
import os
import sqlite3
from pathlib import Path

import h5py
import numpy as np
import torch

KEYPRESS_DIM = 8
MOUSE_DIM = 2
KM_CODE_DIM = 36  # 12 FSQ groups x 3 dims (plaicraft-debug#80)
# Bump when build_action_array's binning rule changes, to force raw/km cache rebuilds.
SUBBIN_RULE_VERSION = "2"  # v2: symlog mouse in load_or_build's raw branch + containment binning

# 12.5 Hz action grid (plaicraft-debug#80): one tick per video frame, one tokenizer control-frame per SUBBIN_MS.
TICK_MS = 80
SUBBIN_MS = 10
SUBBINS_PER_TICK = TICK_MS // SUBBIN_MS  # 8, matches the km tokenizer's block_size

# Fixed key order for dims 0-5: [w, a, s, d, space, shift]
_KEY_IDS = ["87", "65", "83", "68", "32", "340"]


def _symlog(v):
    """Compress unbounded pixel sums to ~+-5 so the model's mouse conditioning input
    matches the 2c02s2pu-era scale (plaicraft-debug#80's B2 regression)."""
    return np.sign(v) * np.log1p(np.abs(v))


def quantize_keypress(x):
    """Snap a continuous (..., 8) keypress prediction to the nearest of the 256 valid
    multi-hot vectors. Every codebook entry is a corner of the unit hypercube, so
    nearest-neighbour in L2 reduces to independent per-dim rounding (plaicraft-debug#77)."""
    return (x > 0.5).float()


_FSQ_LEVELS = (8, 6, 5)


def quantize_km_fsq(x):
    """Closed-form nearest-lattice snap for a continuous (..., 36) km_fsq action prediction:
    12 groups of 3 dims, each snapped independently onto the 8x6x5 FSQ grid (plaicraft-
    debug#80). A direct scale+round+clamp -- exact, idempotent on lattice points, and cheap
    (no search over the 240 per-group codes). Mirrors km_tokenizer.model.FSQ's own
    codes_to_indices()/indices_to_codes() math exactly (skipping the tanh soft-bound FSQ.forward
    uses for TRAINING-time gradient flow, which is not idempotent right at the lattice edges --
    a snap has no gradient to protect, so the direct scale+round+clamp is the right one here)."""
    if x.shape[-1] != KM_CODE_DIM:
        raise ValueError(f"Expected last dim {KM_CODE_DIM}, got {x.shape[-1]}.")
    orig_shape = x.shape
    x = x.reshape(*orig_shape[:-1], KM_CODE_DIM // 3, 3)
    levels = torch.tensor(_FSQ_LEVELS, dtype=x.dtype, device=x.device)
    half_width = levels // 2
    level_idx = (x * half_width + half_width).round()
    level_idx = torch.minimum(level_idx.clamp(min=0), levels - 1)
    codes = (level_idx - half_width) / half_width
    return codes.reshape(orig_shape)


def read_session_fps(session_dir):
    """The session table's fps column -- the DB's own record of its tick rate."""
    session_dir = Path(session_dir)
    db_path = session_dir / f"{session_dir.name}.db"
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute("SELECT fps FROM session").fetchone()
    finally:
        con.close()
    return float(row[0]) if row is not None else None


def validate_action_encoding(action_encoding, fps=None, action_dim=None, mouse_dim=None):
    """km_fsq needs fps==12.5, action_dim==36, mouse_dim==0; raw needs action_dim==8,
    mouse_dim==2 -- except raw with both dims 0, which means no action conditioning
    at all (the video-only default) and skips the dim check entirely."""
    if action_encoding not in ("raw", "km_fsq"):
        raise ValueError(f"unknown action_encoding {action_encoding!r}, expected 'raw' or 'km_fsq'")
    if action_encoding == "raw" and action_dim == 0 and mouse_dim == 0:
        return
    expected_dim, expected_mouse = (KM_CODE_DIM, 0) if action_encoding == "km_fsq" else (KEYPRESS_DIM, MOUSE_DIM)
    if action_encoding == "km_fsq" and fps is not None and abs(fps - 12.5) > 1e-6:
        raise ValueError(f"action_encoding='km_fsq' requires session fps==12.5, got {fps}")
    if action_dim is not None and action_dim != expected_dim:
        raise ValueError(f"action_encoding={action_encoding!r} expects action_dim={expected_dim}, got {action_dim}")
    if mouse_dim is not None and mouse_dim != expected_mouse:
        raise ValueError(f"action_encoding={action_encoding!r} expects mouse_dim={expected_mouse}, got {mouse_dim}")


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


def _raw_cache_paths(session_dir):
    return (session_dir / "actions_keypress.npy", session_dir / "actions_mouse.npy",
            session_dir / "actions_raw.json")


def load_or_build_raw(session_dir):
    """Tick-resolution ground truth, for overlays and interventions: (n_ticks, 8) keys
    (OR-reduced over the tick's 8 sub-bins) + (n_ticks, 2) raw-pixel mouse (summed over
    the tick's 8 sub-bins). Already causally shifted -- see build_action_array.

    Cached to a sidecar recording the binning rule, same pattern as the km codes cache --
    a pre-#80 cache has the identical (n_ticks, dim) shape (100ms bins, exact-timestamp
    mouse lookup) so the shape check alone can't tell it apart; a missing or mismatched
    sidecar is just a stale cache, not corrupt, so it rebuilds rather than fails."""
    session_dir = Path(session_dir)
    sid = session_dir.name
    n_ticks = _n_ticks_from_hdf5(session_dir)
    keypress_path, mouse_path, sidecar_path = _raw_cache_paths(session_dir)
    expected_sidecar = {"subbin_rule_version": SUBBIN_RULE_VERSION}

    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text())
        except (json.JSONDecodeError, OSError):
            sidecar = None
        if sidecar == expected_sidecar:
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
    tmp_sidecar = sidecar_path.with_name(f".{sidecar_path.stem}.{os.getpid()}.tmp.json")
    tmp_sidecar.write_text(json.dumps(expected_sidecar))
    os.replace(tmp_sidecar, sidecar_path)

    return np.load(keypress_path, mmap_mode="r"), np.load(mouse_path, mmap_mode="r")


def _km_cache_paths(session_dir):
    return session_dir / "actions_km_codes.npy", session_dir / "actions_km_codes.json"


def _load_or_build_km_codes(session_dir, tokenizer_checkpoint, device):
    """Scatter the 8 compact dims to 79-wide at _RAW_POSITIONS, run the km tokenizer once,
    cache the (n_ticks, 36) post-FSQ quantized codes plus a sidecar recording which
    checkpoint and which binning rule produced them. A sidecar mismatch rebuilds, it
    never fails -- the cache is just stale, not corrupt."""
    from .km_tokenizer.keypress_scatter import scatter_keypress
    from .km_tokenizer.model import _sha256, load_tokenizer

    session_dir = Path(session_dir)
    sid = session_dir.name
    n_ticks = _n_ticks_from_hdf5(session_dir)
    codes_path, sidecar_path = _km_cache_paths(session_dir)

    checkpoint_path = Path(tokenizer_checkpoint)
    expected_sidecar = {
        "tokenizer_sha256": _sha256(checkpoint_path),
        "subbin_rule_version": SUBBIN_RULE_VERSION,
    }

    if codes_path.exists() and sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text())
        except (json.JSONDecodeError, OSError):
            sidecar = None
        if sidecar == expected_sidecar:
            codes = _load_cached(codes_path, n_ticks, KM_CODE_DIM)
            if codes is not None:
                return codes

    db_path = session_dir / f"{sid}.db"
    key_sub, mouse_sub = build_action_array(db_path, n_ticks)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(checkpoint_path=checkpoint_path, device=device)
    key_press = scatter_keypress(torch.from_numpy(key_sub).float()).unsqueeze(0).to(device)
    mouse_movement = torch.from_numpy(mouse_sub).float().unsqueeze(0).to(device)
    with torch.no_grad():
        prequantized, _frame_mask, _block_mask = tokenizer._encode_prequantized(key_press, mouse_movement)
        _token_ids, quantized_codes = tokenizer._quantize(prequantized)
    codes = quantized_codes[0].reshape(n_ticks, KM_CODE_DIM).cpu().numpy().astype(np.float32)

    tmp_codes = codes_path.with_name(f".{codes_path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp_codes, codes)
    os.replace(tmp_codes, codes_path)
    tmp_sidecar = sidecar_path.with_name(f".{sidecar_path.stem}.{os.getpid()}.tmp.json")
    tmp_sidecar.write_text(json.dumps(expected_sidecar))
    os.replace(tmp_sidecar, sidecar_path)

    return np.load(codes_path, mmap_mode="r")


def load_or_build(session_dir, action_encoding="raw", tokenizer_checkpoint=None, device=None):
    """raw -> (n_ticks, 8) keypress + (n_ticks, 2) symlog-compressed mouse -- the MODEL'S
    NATIVE CONDITIONING encoding, not ground truth (see load_or_build_raw for that).
    km_fsq -> (n_ticks, 36) quantized codes + (n_ticks, 0) empty mouse (folded into the codes)."""
    if action_encoding == "raw":
        keypress, mouse = load_or_build_raw(session_dir)
        return keypress, _symlog(np.asarray(mouse))
    if action_encoding == "km_fsq":
        from .km_tokenizer.model import DEFAULT_CHECKPOINT

        checkpoint = tokenizer_checkpoint or DEFAULT_CHECKPOINT
        codes = _load_or_build_km_codes(session_dir, checkpoint, device)
        n_ticks = codes.shape[0]
        return codes, np.zeros((n_ticks, 0), dtype=np.float32)
    raise ValueError(f"unknown action_encoding {action_encoding!r}, expected 'raw' or 'km_fsq'")
