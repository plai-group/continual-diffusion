"""issue #74: 100Hz keypress tiling in build_action_array, and the frozen
PLAICraft encoder/decoder round trip (encode_keypress_live / decode_keypress_latent)."""

import pickle
import sqlite3
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from improved_diffusion import debug_actions as da
from improved_diffusion.keypress_autoencoder.constants import id_to_index


def _make_session_db(path, key_events=(), n_frames=5):
    """Minimal keyboard/mouse_click/mouse_movement schema build_action_array reads."""
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE keyboard (key_id TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_click (mouse_key_type TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_movement (timestamp INTEGER, mouseDX REAL, mouseDY REAL)")
    con.executemany("INSERT INTO keyboard VALUES (?, ?, ?)", key_events)
    con.commit()
    con.close()


def test_build_action_array_tiles_100hz():
    n_frames = 5
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "session.db"
        # "w" (key_id 87) held during frame 2's window [200, 300).
        _make_session_db(db_path, key_events=[("87", 200, 300)])
        keypress, mouse = da.build_action_array(db_path, n_frames)

    assert keypress.shape == (n_frames * da.SUBFRAME_HZ, da.KEYPRESS_DIM)
    assert mouse.shape == (n_frames, da.MOUSE_DIM)

    # Every block of 10 consecutive sub-bins must be identical (tiled, not resampled).
    blocks = keypress.reshape(n_frames, da.SUBFRAME_HZ, da.KEYPRESS_DIM)
    for f in range(n_frames):
        for row in range(1, da.SUBFRAME_HZ):
            assert (blocks[f, row] == blocks[f, 0]).all()

    # Causal shift: row 0 is all zero; the "w" held in frame 2's window shows up shifted to frame 3.
    assert (blocks[0, 0] == 0).all()
    assert blocks[3, 0, 0] == 1.0
    assert blocks[2, 0, 0] == 0.0


def _rand_keypress(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(*shape, generator=g) > 0.5).float()


def test_encode_keypress_live_shape():
    keypress = _rand_keypress(2, 6, da.KEYPRESS_DIM)
    latent = da.encode_keypress_live(keypress)
    assert latent.shape == (2, 6, da.ENCODED_KEYPRESS_DIM)


def test_decode_keypress_latent_shape():
    latent = torch.randn(2, 6, da.ENCODED_KEYPRESS_DIM)
    raw = da.decode_keypress_latent(latent)
    assert raw.shape == (2, 6, da.KEYPRESS_DIM)


def test_encode_decode_round_trip_shape_and_determinism():
    keypress = _rand_keypress(3, 4, da.KEYPRESS_DIM)
    latent = da.encode_keypress_live(keypress)
    recon = da.decode_keypress_latent(latent)
    assert recon.shape == keypress.shape

    # The frozen model must be deterministic (eval mode, dropout off) so the offline
    # cache and any live re-encode of the same input always agree.
    latent2 = da.encode_keypress_live(keypress)
    assert torch.equal(latent, latent2)
    recon2 = da.decode_keypress_latent(latent)
    assert torch.equal(recon, recon2)


def test_all_zero_keypress_decodes_below_threshold():
    # Sanity check only (off-manifold behavior is deferred, see issue #74 plan).
    # The decoder's reconstruction is unbounded/logit-scale, not clipped to [0,1] --
    # matching plaicraft-model-pi0's own decode.py, which thresholds this same raw
    # decoder output at 0.5 (real "pressed"/"not pressed" logits separate cleanly,
    # e.g. ~+10..+14 vs ~-10..-27) -- so 0.5 is still the right decision boundary here.
    keypress = torch.zeros(1, 1, da.KEYPRESS_DIM)
    latent = da.encode_keypress_live(keypress)
    recon = da.decode_keypress_latent(latent)
    assert (recon < 0.5).all()


def test_held_key_decodes_above_threshold_others_below():
    keypress = torch.zeros(1, 1, da.KEYPRESS_DIM)
    keypress[..., 0] = 1.0  # "w" held
    latent = da.encode_keypress_live(keypress)
    recon = da.decode_keypress_latent(latent)
    assert recon[..., 0].item() > 0.5
    assert (recon[..., 1:] < 0.5).all()


def test_encode_keypress_live_matches_offline_encoding_path():
    # encode_keypress_live's live pad+tile+encode must exactly match the offline
    # cache-building path (_encode_offline) for the same underlying 100Hz window.
    keypress = _rand_keypress(1, 3, da.KEYPRESS_DIM, seed=1)[0]  # (3, 8)
    live = da.encode_keypress_live(keypress)  # (3, 80)

    tiled_100hz = keypress.repeat_interleave(da.SUBFRAME_HZ, dim=0).numpy()  # (30, 8)
    offline = da._encode_offline(tiled_100hz)  # (3, 80)

    assert torch.allclose(live, torch.from_numpy(offline), atol=1e-5)


def test_raw_positions_match_vendored_constants():
    # Pins the compact [w,a,s,d,space,shift,left,right] order to the autoencoder's real
    # trained channel positions -- an accidental edit to either side must break this.
    expected = [
        id_to_index["87"], id_to_index["65"], id_to_index["83"], id_to_index["68"],
        id_to_index["32"], id_to_index["340"], id_to_index["left"], id_to_index["right"],
    ]
    assert expected == [23, 1, 19, 4, 0, 51, 75, 76]
    assert da._RAW_POSITIONS == expected


def _make_full_session(session_dir, n_frames, encodings, key_events=()):
    """Session dir with a minimal hdf5 (for n_frames) and db (keyboard/mouse tables +
    optionally key_press_encodings, one pickled (16,5) array per frame)."""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    hdf5_dir = session_dir / "encoded_video_hdf5"
    hdf5_dir.mkdir()
    with h5py.File(hdf5_dir / f"{session_dir.name}_encoded_video.hdf5", "w") as f:
        f.create_dataset("frames", data=np.zeros((n_frames, 1), dtype=np.float32))

    db_path = session_dir / f"{session_dir.name}.db"
    _make_session_db(db_path, key_events=key_events, n_frames=n_frames)
    if encodings is not None:
        con = sqlite3.connect(str(db_path))
        con.execute(
            "CREATE TABLE key_press_encodings "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, "
            "start_timestamp INTEGER, end_timestamp INTEGER, encoding BLOB)"
        )
        for k, enc in enumerate(encodings):
            start = k * 100
            con.execute(
                "INSERT INTO key_press_encodings (start_timestamp, end_timestamp, encoding) VALUES (?, ?, ?)",
                (start, start + 100, pickle.dumps(enc)),
            )
        con.commit()
        con.close()
    return db_path


def test_load_or_build_applies_causal_shift_to_db_encodings():
    n_frames = 3
    encodings = [np.full((16, 5), float(k), dtype=np.float32) for k in range(n_frames)]
    with tempfile.TemporaryDirectory() as d:
        session_dir = Path(d) / "sess"
        _make_full_session(session_dir, n_frames, encodings)
        encoded, mouse = da.load_or_build(session_dir)

    assert encoded.shape == (n_frames, da.ENCODED_KEYPRESS_DIM)
    assert mouse.shape == (n_frames, da.MOUSE_DIM)
    # Row 0 is zeros (no preceding window); row i>0 is table row i-1, flattened.
    assert (np.asarray(encoded[0]) == 0).all()
    assert np.allclose(encoded[1], encodings[0].reshape(-1))
    assert np.allclose(encoded[2], encodings[1].reshape(-1))


def test_load_or_build_raises_when_table_missing():
    n_frames = 3
    with tempfile.TemporaryDirectory() as d:
        session_dir = Path(d) / "sess"
        _make_full_session(session_dir, n_frames, encodings=None)
        try:
            da.load_or_build(session_dir)
            assert False, "expected RuntimeError for missing key_press_encodings table"
        except RuntimeError:
            pass
