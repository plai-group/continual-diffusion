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
# Bump when build_action_array's binning rule changes, to force km cache rebuilds.
SUBBIN_RULE_VERSION = "1"

# The 12.5 Hz action grid (plaicraft-debug#80): one tick per video frame, one
# tokenizer control-frame every SUBBIN_MS ms within a tick.
TICK_MS = 80
SUBBIN_MS = 10
SUBBINS_PER_TICK = TICK_MS // SUBBIN_MS  # 8, matches the km tokenizer's block_size

# Fixed key order for dims 0-5: [w, a, s, d, space, shift]
_KEY_IDS = ["87", "65", "83", "68", "32", "340"]


def quantize_keypress(x):
    """Snap a continuous (..., 8) keypress prediction to the nearest of the 256 valid
    multi-hot vectors. Every codebook entry is a corner of the unit hypercube, so
    nearest-neighbour in L2 reduces to independent per-dim rounding (plaicraft-debug#77)."""
    return (x > 0.5).float()


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
    mouse_dim==2. Every argument but action_encoding is optional, so this doubles as a
    dataset-construction guard (fps only) and a CLI guard (dims only)."""
    if action_encoding not in ("raw", "km_fsq"):
        raise ValueError(f"unknown action_encoding {action_encoding!r}, expected 'raw' or 'km_fsq'")
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
      mouse 0-1: raw pixel mouseDX, mouseDY summed at that sub-bin (no symlog --
        the km tokenizer's own feature stem does its own normalization, and raw
        mode wants pixels directly)

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

    # mouse_movement carries exactly one row per sub-bin, at tick*80 + b*10 ms.
    mouse_by_ts = {int(ts): (dx, dy) for ts, dx, dy in mouse_rows}
    for s in range(n_sub):
        dx, dy = mouse_by_ts.get(int(sub_starts[s]), (0.0, 0.0))
        M[s, 0], M[s, 1] = dx, dy

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
    """raw -> (n_ticks, 8) keypress + (n_ticks, 2) mouse, both tick-resolution ground truth.
    km_fsq -> (n_ticks, 36) quantized codes + (n_ticks, 0) empty mouse (folded into the codes)."""
    if action_encoding == "raw":
        return load_or_build_raw(session_dir)
    if action_encoding == "km_fsq":
        from .km_tokenizer.model import DEFAULT_CHECKPOINT

        checkpoint = tokenizer_checkpoint or DEFAULT_CHECKPOINT
        codes = _load_or_build_km_codes(session_dir, checkpoint, device)
        n_ticks = codes.shape[0]
        return codes, np.zeros((n_ticks, 0), dtype=np.float32)
    raise ValueError(f"unknown action_encoding {action_encoding!r}, expected 'raw' or 'km_fsq'")
