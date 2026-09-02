"""plaicraft-debug#80: 10 ms sub-bin action binning + the causal tick shift.

Highest-risk slice: a sub-bin/scatter alignment bug still yields plausible-
looking codes and training proceeds normally -- it just learns a scrambled
action space (see plaicraft-debug#74's cb507a9 for exactly that failure, a
release earlier, on the old keypress autoencoder). So this file pairs the
pure-array tests with an end-to-end check through the REAL trained tokenizer:
encode a synthetic session, decode it, and confirm keys recover exactly and
mouse recovers to within a couple of pixels.
"""
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from improved_diffusion import debug_actions as da
from improved_diffusion.km_tokenizer.keypress_scatter import scatter_keypress, _RAW_POSITIONS
from improved_diffusion.km_tokenizer.model import load_tokenizer


def _make_db(path, n_ticks, key_events=(), click_events=(), mouse_fn=None):
    """key_events/click_events: list of (id, start_ms, end_ms). mouse_fn(tick, subbin) -> (dx, dy)."""
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE keyboard (key_id TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_click (mouse_key_type TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_movement (timestamp INTEGER, mouseDX INTEGER, mouseDY INTEGER)")
    for key_id, s, e in key_events:
        con.execute("INSERT INTO keyboard VALUES (?, ?, ?)", (key_id, s, e))
    for btn, s, e in click_events:
        con.execute("INSERT INTO mouse_click VALUES (?, ?, ?)", (btn, s, e))
    mouse_fn = mouse_fn or (lambda t, b: (0, 0))
    for t in range(n_ticks):
        for b in range(da.SUBBINS_PER_TICK):
            ts = t * da.TICK_MS + b * da.SUBBIN_MS
            dx, dy = mouse_fn(t, b)
            con.execute("INSERT INTO mouse_movement VALUES (?, ?, ?)", (ts, dx, dy))
    con.commit()
    con.close()


def test_build_action_array_shape():
    n_ticks = 3
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "s.db"
        _make_db(db_path, n_ticks)
        key_sub, mouse_sub = da.build_action_array(db_path, n_ticks)
    assert key_sub.shape == (n_ticks * da.SUBBINS_PER_TICK, da.KEYPRESS_DIM)
    assert mouse_sub.shape == (n_ticks * da.SUBBINS_PER_TICK, da.MOUSE_DIM)


def test_causal_shift_first_tick_is_zero_and_ticks_are_repeated_blocks():
    n_ticks = 3
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "s.db"
        # 'w' held for the whole of raw tick 0.
        _make_db(db_path, n_ticks, key_events=[("87", 0, da.TICK_MS)])
        key_sub, _ = da.build_action_array(db_path, n_ticks)
    blocks = key_sub.reshape(n_ticks, da.SUBBINS_PER_TICK, da.KEYPRESS_DIM)
    assert blocks[0].sum() == 0  # row 0 (no prior tick) is all zeros
    assert np.all(blocks[1][:, 0] == 1.0)  # shifted tick 1 == raw tick 0 == w held
    assert blocks[2].sum() == 0


def test_key_and_click_intervals_land_in_the_right_compact_dims():
    n_ticks = 2
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "s.db"
        _make_db(
            db_path, n_ticks,
            key_events=[("32", 0, da.TICK_MS)],       # space, dim 4
            click_events=[("left", 0, da.TICK_MS)],   # dim 6
        )
        key_sub, _ = da.build_action_array(db_path, n_ticks)
    block1 = key_sub.reshape(n_ticks, da.SUBBINS_PER_TICK, da.KEYPRESS_DIM)[1]
    assert np.all(block1[:, 4] == 1.0)
    assert np.all(block1[:, 6] == 1.0)
    assert block1[:, [0, 1, 2, 3, 5, 7]].sum() == 0


def test_mouse_subbins_are_not_broadcast_and_shift_correctly():
    n_ticks = 2
    mouse_fn = lambda t, b: (b - 3, 2 * b)
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "s.db"
        _make_db(db_path, n_ticks, mouse_fn=mouse_fn)
        _, mouse_sub = da.build_action_array(db_path, n_ticks)
    block1 = mouse_sub.reshape(n_ticks, da.SUBBINS_PER_TICK, da.MOUSE_DIM)[1]  # == raw tick 0
    expected = np.array([[b - 3, 2 * b] for b in range(da.SUBBINS_PER_TICK)], dtype=np.float32)
    assert np.array_equal(block1, expected)


def test_load_or_build_raw_aggregates_or_and_sum_per_tick(tmp_path):
    session_dir = tmp_path / "sess"
    (session_dir / "encoded_video_hdf5").mkdir(parents=True)
    import h5py
    n_ticks = 3
    with h5py.File(session_dir / "encoded_video_hdf5" / "sess_encoded_video.hdf5", "w") as f:
        f.create_dataset("frames", data=np.zeros((n_ticks, 3, 2, 2), dtype=np.float32))
    _make_db(
        session_dir / "sess.db", n_ticks,
        key_events=[("87", 0, da.SUBBIN_MS)],  # 'w' held for ONE sub-bin only, of raw tick 0
        mouse_fn=lambda t, b: (1, -1),          # 1px/-1px every sub-bin -> 8/-8 per tick
    )
    keypress, mouse = da.load_or_build_raw(session_dir)
    assert keypress.shape == (n_ticks, da.KEYPRESS_DIM)
    assert mouse.shape == (n_ticks, da.MOUSE_DIM)
    assert keypress[0].sum() == 0  # no prior tick
    assert keypress[1, 0] == 1.0   # OR-reduced: w held during SOME sub-bin of raw tick 0
    assert np.allclose(mouse[1], [8.0, -8.0])  # summed over raw tick 0's 8 sub-bins

    # cached: a second call must return identical arrays without rebuilding differently.
    keypress2, mouse2 = da.load_or_build_raw(session_dir)
    assert np.array_equal(keypress, keypress2)
    assert np.array_equal(mouse, mouse2)


def test_load_or_build_dispatches_to_raw(tmp_path):
    session_dir = tmp_path / "sess"
    (session_dir / "encoded_video_hdf5").mkdir(parents=True)
    import h5py
    with h5py.File(session_dir / "encoded_video_hdf5" / "sess_encoded_video.hdf5", "w") as f:
        f.create_dataset("frames", data=np.zeros((2, 3, 2, 2), dtype=np.float32))
    _make_db(session_dir / "sess.db", 2)
    a = da.load_or_build(session_dir)
    b = da.load_or_build_raw(session_dir)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_km_tokenizer_round_trip_recovers_keys_exactly_and_mouse_near_zero():
    """The critical B2 verification: encode a synthetic session through the real
    trained tokenizer and decode it back."""
    n_ticks = 4
    key_events = [("87", 0, da.TICK_MS)]            # raw tick 0: w held
    click_events = [("left", da.TICK_MS, 2 * da.TICK_MS)]  # raw tick 1: left click held
    mouse_fn = lambda t, b: ((b - 3) + t, -(b) + 2 * t)
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "s.db"
        _make_db(db_path, n_ticks, key_events=key_events, click_events=click_events, mouse_fn=mouse_fn)
        key_sub, mouse_sub = da.build_action_array(db_path, n_ticks)

    scattered = scatter_keypress(torch.from_numpy(key_sub).float()).unsqueeze(0)
    mouse_t = torch.from_numpy(mouse_sub).float().unsqueeze(0)

    tokenizer = load_tokenizer()
    with torch.no_grad():
        out = tokenizer(scattered, mouse_t)

    key_hat = (torch.sigmoid(out.key_logits[0, :, :, _RAW_POSITIONS]) > 0.5).float()
    key_true = torch.from_numpy(key_sub).float().reshape(n_ticks, da.SUBBINS_PER_TICK, da.KEYPRESS_DIM)
    assert torch.equal(key_hat, key_true)

    mouse_pred = out.mouse_pred[0].reshape(-1, 2)
    mouse_true = torch.from_numpy(mouse_sub).float()
    assert (mouse_pred - mouse_true).abs().mean().item() < 1.0
