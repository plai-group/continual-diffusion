# Per-frame action arrays for plaicraft-debug sessions, cached to <session_dir>/{actions_keypress,encoded_keypress,actions_mouse}.npy.
import os
import pickle
import sqlite3
from pathlib import Path

import h5py
import numpy as np
import torch

from improved_diffusion.decode_debug import FRAME_DURATION_MS
from improved_diffusion.keypress_autoencoder.constants import id_to_index
from improved_diffusion.keypress_autoencoder.model import KeyPressAutoencoder

KEYPRESS_DIM = 8
MOUSE_DIM = 2
SUBFRAME_HZ = 10  # PLAICraft encoder's native 100ms-window / 10ms-bin resolution
RAW_INPUT_DIM = 79  # encoder's trained input width; our 8 keys occupy fixed positions within this
ENCODED_KEYPRESS_DIM = 80  # flattened 16 x 5 encoder output

_CHECKPOINT_PATH = Path(__file__).parent / "keypress_autoencoder" / "keyencoder_16_5_best_checkpoint.pt"

# Fixed key order for dims 0-5: [w, a, s, d, space, shift]
_KEY_IDS = ["87", "65", "83", "68", "32", "340"]

# Real encoder-input channel for each of the 8 compact dims [w,a,s,d,space,shift,left,right].
_RAW_POSITIONS = [id_to_index[k] for k in _KEY_IDS] + [id_to_index["left"], id_to_index["right"]]

_keypress_ae = None


def _get_keypress_autoencoder(device):
    """Frozen, eval-mode PLAICraft keypress autoencoder, cached across calls."""
    global _keypress_ae
    if _keypress_ae is None:
        ae = KeyPressAutoencoder(input_dim=RAW_INPUT_DIM)
        ae.load_state_dict(torch.load(_CHECKPOINT_PATH, map_location="cpu", weights_only=True))
        ae.eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        _keypress_ae = ae
    return _keypress_ae.to(device)


def _encode_windows(windows, device):
    """(N, 10, 8) raw keypress windows -> (N, 80) encoded latents, via the frozen encoder.

    Scatters the compact 8 dims into their real trained channel positions (_RAW_POSITIONS)
    within the encoder's 79-wide input; all other channels stay zero (never pressed here).
    """
    ae = _get_keypress_autoencoder(device)
    x = torch.zeros(*windows.shape[:-1], RAW_INPUT_DIM, dtype=windows.dtype, device=windows.device)
    x[..., _RAW_POSITIONS] = windows
    with torch.no_grad():
        z = ae.encoder(x.transpose(1, 2))  # (N, 16, 5)
    return z.reshape(z.shape[0], ENCODED_KEYPRESS_DIM)


def encode_keypress_live(keypress):
    """Encode an arbitrary (possibly intervened) raw (..., 8) keypress tensor into (..., 80) latents.

    Used only where the offline cache doesn't apply -- debug_validation's swap/invert/zero
    interventions, which mutate the raw tensor live and then need it re-encoded before sampling.
    """
    orig_shape = keypress.shape[:-1]
    flat = keypress.reshape(-1, KEYPRESS_DIM).float()
    tiled = flat.unsqueeze(1).expand(-1, SUBFRAME_HZ, -1).contiguous()
    z = _encode_windows(tiled, keypress.device)
    return z.reshape(*orig_shape, ENCODED_KEYPRESS_DIM)


def decode_keypress_latent(latent):
    """Decode an (..., 80) encoded latent back to raw (..., 8) keypress via the frozen decoder.

    Output is the decoder's raw (unbounded, logit-scale) reconstruction, not clipped to [0,1] --
    matching plaicraft-model-pi0's own decode.py, which thresholds this same output at 0.5;
    real pressed/not-pressed reconstructions separate cleanly (e.g. ~+10 vs ~-15), so 0.5 still works.
    Downsamples the decoder's 10 reconstructed sub-frame bins back to one row per frame by
    averaging them (the encoder input was a constant tile, so the bins should agree).
    Gathers the 8 compact dims back from their real trained channel positions (_RAW_POSITIONS).
    """
    orig_shape = latent.shape[:-1]
    ae = _get_keypress_autoencoder(latent.device)
    z = latent.reshape(-1, 16, 5).float()
    with torch.no_grad():
        x_recon = ae.decoder(z)  # (N, 79, 10)
    raw = x_recon[:, _RAW_POSITIONS, :].mean(dim=2)
    return raw.reshape(*orig_shape, KEYPRESS_DIM)


def _encode_offline(raw_keypress_100hz, chunk=4096):
    """(n_frames*10, 8) cached raw array -> (n_frames, 80) encoded array, batched to bound memory."""
    n_frames = raw_keypress_100hz.shape[0] // SUBFRAME_HZ
    windows = torch.from_numpy(np.array(raw_keypress_100hz, dtype=np.float32))
    windows = windows.reshape(n_frames, SUBFRAME_HZ, KEYPRESS_DIM)
    out = np.empty((n_frames, ENCODED_KEYPRESS_DIM), dtype=np.float32)
    for start in range(0, n_frames, chunk):
        end = min(start + chunk, n_frames)
        out[start:end] = _encode_windows(windows[start:end], "cpu").numpy()
    return out


def _symlog(v):
    return np.sign(v) * np.log1p(np.abs(v))


def quantize_keypress(x):
    """Snap a continuous (..., 8) keypress prediction to the nearest of the 256 valid
    multi-hot vectors. Every codebook entry is a corner of the unit hypercube, so
    nearest-neighbour in L2 reduces to independent per-dim rounding (plaicraft-debug#77)."""
    return (x > 0.5).float()


def build_action_array(session_db_path, n_frames):
    """
    Returns (keypress, mouse): (n_frames*10, 8) and (n_frames, 2) float32.
      keypress 0-5: held keys [w,a,s,d,space,shift] during the frame's window
      keypress 6-7: held mouse clicks [left, right]
      mouse 0-1: symlog(sum mouseDX), symlog(sum mouseDY) over the window

    CAUSAL SHIFT: row i (of the underlying per-frame array) holds the action
    from window [i-1, i) -- the action that CAUSED frame i. Row 0 is all zeros.

    Keypress is then tiled 10x per frame (100Hz) to match the PLAICraft encoder's
    native sub-frame resolution -- each of the 10 rows repeats the frame's single
    collapsed 100ms bit; there is no real sub-frame timing in this data to recover.
    Mouse stays at 10Hz (one row per frame): a continuous window-sum has no
    sub-frame timing to gain from tiling.
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
    out_k_100hz = np.repeat(out_k, SUBFRAME_HZ, axis=0)
    return out_k_100hz, out_m


def _n_frames_from_hdf5(session_dir):
    sid = Path(session_dir).name
    hdf5_path = Path(session_dir) / "encoded_video_hdf5" / f"{sid}_encoded_video.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        return f["frames"].shape[0]


def _load_cached(cache_path, expected_rows, expected_dim):
    """"rows" not "frames": keypress caches are n_frames*10 rows (100Hz) but mouse/encoded
    caches are n_frames rows -- this helper checks both against whatever count applies."""
    if cache_path.exists():
        arr = np.load(cache_path, mmap_mode="r")
        if arr.shape[0] == expected_rows and arr.shape[1] == expected_dim:
            return arr
    return None  # missing, or stale (row count or dim mismatch -- e.g. pre-#74 10Hz format)


def _atomic_save(path, arr):
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp_path, arr)
    os.replace(tmp_path, path)  # atomic within same directory


def _ensure_raw_cache(session_dir, n_frames):
    """Cache build_action_array's output to <session_dir>/actions_{keypress,mouse}.npy.

    keypress is the 100Hz raw array (n_frames*10, 8); mouse stays (n_frames, 2).
    """
    session_dir = Path(session_dir)
    keypress_path = session_dir / "actions_keypress.npy"
    mouse_path = session_dir / "actions_mouse.npy"

    keypress = _load_cached(keypress_path, n_frames * SUBFRAME_HZ, KEYPRESS_DIM)
    mouse = _load_cached(mouse_path, n_frames, MOUSE_DIM)
    if keypress is not None and mouse is not None:
        return keypress, mouse

    db_path = session_dir / f"{session_dir.name}.db"
    keypress, mouse = build_action_array(db_path, n_frames)
    _atomic_save(keypress_path, keypress)
    _atomic_save(mouse_path, mouse)
    return np.load(keypress_path, mmap_mode="r"), np.load(mouse_path, mmap_mode="r")


def load_or_build_raw(session_dir):
    """Raw per-frame (n_frames, 8) keypress + (n_frames, 2) mouse, for display/interventions.

    Downsamples the cached 100Hz keypress array back to one row per frame (any of the
    10 tiled sub-bins reconstructs it exactly, since they were tiled from the same bit).
    """
    session_dir = Path(session_dir)
    n_frames = _n_frames_from_hdf5(session_dir)
    keypress_100hz, mouse = _ensure_raw_cache(session_dir, n_frames)
    keypress = np.asarray(keypress_100hz).reshape(n_frames, SUBFRAME_HZ, KEYPRESS_DIM)[:, 0, :]
    return keypress, mouse


def _read_encoded_from_db(session_dir, n_frames):
    """Read the (n_frames, 80) encoded keypress array plaicraft-debug's generate_sessions.py
    already wrote into the session's key_press_encodings table, instead of re-deriving it.

    Table row i is the encoding of window [i, i+1) (same pre-shift convention as
    build_action_array's internal K); apply the same causal shift here so row i of the
    returned array holds the action from window [i-1, i) -- the action that CAUSED frame i.
    """
    session_dir = Path(session_dir)
    db_path = session_dir / f"{session_dir.name}.db"
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='key_press_encodings'")
    if cur.fetchone() is None:
        con.close()
        raise RuntimeError(f"{db_path} has no key_press_encodings table -- session predates plaicraft-debug#74's encoder pass")
    cur.execute("SELECT encoding FROM key_press_encodings ORDER BY start_timestamp")
    rows = cur.fetchall()
    con.close()
    if len(rows) != n_frames:
        raise RuntimeError(f"key_press_encodings has {len(rows)} rows, expected {n_frames} frames, in {db_path}")

    table = np.stack([pickle.loads(blob).reshape(ENCODED_KEYPRESS_DIM) for (blob,) in rows]).astype(np.float32)
    encoded = np.zeros_like(table)
    encoded[1:] = table[:-1]
    return encoded


def load_or_build(session_dir):
    """The encoded (n_frames, 80) keypress + (n_frames, 2) mouse arrays training reads directly.

    Cached to <session_dir>/encoded_keypress.npy, sourced from the session's own
    key_press_encodings table (already run through the frozen encoder at corpus-generation
    time) -- no live encoder call is needed on this path.
    """
    session_dir = Path(session_dir)
    n_frames = _n_frames_from_hdf5(session_dir)
    _, mouse = _ensure_raw_cache(session_dir, n_frames)

    encoded_path = session_dir / "encoded_keypress.npy"
    encoded = _load_cached(encoded_path, n_frames, ENCODED_KEYPRESS_DIM)
    if encoded is None:
        encoded = _read_encoded_from_db(session_dir, n_frames)
        _atomic_save(encoded_path, encoded)
        encoded = np.load(encoded_path, mmap_mode="r")

    return encoded, mouse
