"""issue #74: 100Hz keypress tiling in build_action_array, and the frozen
PLAICraft encoder/decoder round trip (encode_keypress_live / decode_keypress_latent)."""

import sqlite3
import tempfile
from pathlib import Path

import torch

from improved_diffusion import debug_actions as da


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
