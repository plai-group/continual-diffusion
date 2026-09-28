"""plaicraft-debug#93 B1: build_action_array_79, the PLAIOmni-semantics 79-key builder for
real PLAICraft DBs (NULL end_timestamp, full key vocabulary, floor/ceil sub-bin marking)."""
import json
import sqlite3
from pathlib import Path

import h5py
import numpy as np

from improved_diffusion import debug_actions as da
from improved_diffusion.km_tokenizer.keypress_lookup import KEYPRESS_LOOKUP


def _make_db_79(path, key_events=(), click_events=(), mouse_rows=()):
    """key_events/click_events: list of (id, start_ms, end_ms_or_None). mouse_rows: list of
    (timestamp, dx, dy)."""
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE keyboard (key_id TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_click (mouse_key_type TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_movement (timestamp INTEGER, mouseDX INTEGER, mouseDY INTEGER)")
    for key_id, s, e in key_events:
        con.execute("INSERT INTO keyboard VALUES (?, ?, ?)", (key_id, s, e))
    for btn, s, e in click_events:
        con.execute("INSERT INTO mouse_click VALUES (?, ?, ?)", (btn, s, e))
    for ts, dx, dy in mouse_rows:
        con.execute("INSERT INTO mouse_movement VALUES (?, ?, ?)", (ts, dx, dy))
    con.commit()
    con.close()


def test_shape_matches_8key_builder(tmp_path):
    n_ticks = 3
    db_path = tmp_path / "s.db"
    _make_db_79(db_path)
    key_sub, mouse_sub = da.build_action_array_79(db_path, n_ticks)
    assert key_sub.shape == (n_ticks * da.SUBBINS_PER_TICK, da.KEYPRESS_DIM_79)
    assert mouse_sub.shape == (n_ticks * da.SUBBINS_PER_TICK, da.MOUSE_DIM)


def test_unknown_key_is_skipped(tmp_path):
    n_ticks = 2
    db_path = tmp_path / "s.db"
    # 'middle' mouse button is not in PLAIOmni's FIXED_MOUSE_BUTTONS -- must be silently dropped.
    _make_db_79(db_path, click_events=[("middle", 0, da.TICK_MS)])
    key_sub, _ = da.build_action_array_79(db_path, n_ticks)
    assert key_sub.sum() == 0


def test_zero_length_click_marks_a_single_bin(tmp_path):
    n_ticks = 2
    db_path = tmp_path / "s.db"
    # end == start ("zero-length"): PLAIOmni marks exactly bin floor(start).
    _make_db_79(db_path, click_events=[("left", 15, 15)])
    key_sub, _ = da.build_action_array_79(db_path, n_ticks)
    left_idx = KEYPRESS_LOOKUP["left"]
    hit_bins = np.nonzero(key_sub[:, left_idx])[0]
    # floor(15/10) == 1 (raw sub-bin 1, tick 0); causal shift moves tick 0's block to tick 1's
    # slot, so it lands at flat sub-bin 1*SUBBINS_PER_TICK + 1 == 9.
    assert list(hit_bins) == [1 * da.SUBBINS_PER_TICK + 1]


def test_null_end_extends_keyboard_to_session_end(tmp_path):
    n_ticks = 2
    db_path = tmp_path / "s.db"
    _make_db_79(db_path, key_events=[("87", 0, None)])
    key_sub, _ = da.build_action_array_79(db_path, n_ticks)
    w_idx = KEYPRESS_LOOKUP["87"]
    # NULL end -> session_duration_ms == n_ticks*TICK_MS: every raw sub-bin held, so after the
    # causal shift every tick from 1 on is held.
    blocks = key_sub.reshape(n_ticks, da.SUBBINS_PER_TICK, da.KEYPRESS_DIM_79)
    assert blocks[0].sum() == 0
    assert np.all(blocks[1][:, w_idx] == 1.0)


def test_null_end_click_is_zero_length_not_session_end(tmp_path):
    n_ticks = 2
    db_path = tmp_path / "s.db"
    _make_db_79(db_path, click_events=[("right", 5, None)])
    key_sub, _ = da.build_action_array_79(db_path, n_ticks)
    right_idx = KEYPRESS_LOOKUP["right"]
    hit_bins = np.nonzero(key_sub[:, right_idx])[0]
    # floor(5/10) == 0 (raw sub-bin 0, tick 0), a single bin -- not extended to session end.
    # Causal shift moves tick 0's block to tick 1's slot: flat sub-bin 1*SUBBINS_PER_TICK + 0 == 8.
    assert list(hit_bins) == [1 * da.SUBBINS_PER_TICK]


def test_mouse_clamp_overflow(tmp_path):
    n_ticks = 3  # raw tick 1 is not the last tick, so its shifted block survives into the output
    db_path = tmp_path / "s.db"
    _make_db_79(db_path, mouse_rows=[(da.TICK_MS, 5000.0, -5000.0)])  # raw tick 1, sub-bin 0
    _, mouse_sub = da.build_action_array_79(db_path, n_ticks)
    blocks = mouse_sub.reshape(n_ticks, da.SUBBINS_PER_TICK, da.MOUSE_DIM)
    assert np.allclose(blocks[2][0], [da.MOUSE_ABS_CLAMP_X, -da.MOUSE_ABS_CLAMP_Y])


def test_causal_shift_first_tick_zero(tmp_path):
    n_ticks = 3
    db_path = tmp_path / "s.db"
    _make_db_79(db_path, key_events=[("87", 0, da.TICK_MS)])
    key_sub, _ = da.build_action_array_79(db_path, n_ticks)
    blocks = key_sub.reshape(n_ticks, da.SUBBINS_PER_TICK, da.KEYPRESS_DIM_79)
    w_idx = KEYPRESS_LOOKUP["87"]
    assert blocks[0].sum() == 0
    assert np.all(blocks[1][:, w_idx] == 1.0)
    assert blocks[2].sum() == 0


def _make_session_dir_79(tmp_path, n_ticks=3, **kwargs):
    session_dir = tmp_path / "sess"
    (session_dir / "encoded_video_hdf5").mkdir(parents=True)
    with h5py.File(session_dir / "encoded_video_hdf5" / "sess_encoded_video.hdf5", "w") as f:
        f.create_dataset("frames", data=np.zeros((n_ticks, 3, 2, 2), dtype=np.float32))
    _make_db_79(session_dir / "sess.db", **kwargs)
    return session_dir


def test_load_or_build_raw_79_derives_8dim_at_raw_positions(tmp_path):
    session_dir = _make_session_dir_79(
        tmp_path, n_ticks=2,
        key_events=[("87", 0, da.TICK_MS)],  # dim 0 of the compact 8
        click_events=[("left", 0, da.TICK_MS)],  # dim 6 of the compact 8
    )
    keypress, mouse = da.load_or_build_raw(session_dir, action_builder="raw79")
    assert keypress.shape == (2, da.KEYPRESS_DIM)
    assert mouse.shape == (2, da.MOUSE_DIM)
    assert keypress[1, 0] == 1.0
    assert keypress[1, 6] == 1.0
    assert keypress[1, [1, 2, 3, 4, 5, 7]].sum() == 0


def test_builder_recorded_in_sidecar_and_default_is_legacy_shape(tmp_path):
    session_dir = _make_session_dir_79(tmp_path, n_ticks=2)
    da.load_or_build_raw(session_dir, action_builder="raw8")
    _, _, sidecar_path = da._raw_cache_paths(session_dir)
    # Default builder's sidecar is byte-identical to pre-#93: no "action_builder" key at all.
    assert json.loads(sidecar_path.read_text()) == {"subbin_rule_version": da.SUBBIN_RULE_VERSION}


def test_switching_builder_invalidates_the_cache(tmp_path):
    session_dir = _make_session_dir_79(
        tmp_path, n_ticks=2, key_events=[("87", 0, da.TICK_MS)],
    )
    keypress_8, _ = da.load_or_build_raw(session_dir, action_builder="raw8")
    keypress_79, _ = da.load_or_build_raw(session_dir, action_builder="raw79")  # must rebuild
    _, _, sidecar_path = da._raw_cache_paths(session_dir)
    assert json.loads(sidecar_path.read_text()) == {
        "subbin_rule_version": da.SUBBIN_RULE_VERSION, "action_builder": "raw79",
    }
    assert np.array_equal(np.asarray(keypress_8), np.asarray(keypress_79))  # same result, both paths


def test_n_ticks_from_hdf5_accepts_non_debug_filename(tmp_path):
    """plaicraft_morgan's hdf5 is named <sid>.hdf5, not debug world's <sid>_encoded_video.hdf5."""
    session_dir = tmp_path / "sess"
    (session_dir / "encoded_video_hdf5").mkdir(parents=True)
    with h5py.File(session_dir / "encoded_video_hdf5" / "sess.hdf5", "w") as f:
        f.create_dataset("frames", data=np.zeros((4, 4, 24, 40), dtype=np.float16))
    assert da._n_ticks_from_hdf5(session_dir) == 4


def test_km_codes_skip_scatter_for_raw79(tmp_path, monkeypatch):
    """raw79's key array is already 79-wide -- the km-codes path must feed it to the tokenizer
    directly, not through scatter_keypress (which would raise on a non-8-wide input)."""
    from improved_diffusion.km_tokenizer.model import DEFAULT_CHECKPOINT

    session_dir = _make_session_dir_79(
        tmp_path, n_ticks=2, key_events=[("87", 0, da.TICK_MS)],
    )
    codes, mouse = da.load_or_build(
        session_dir, action_encoding="km_fsq", action_builder="raw79",
    )
    assert codes.shape == (2, da.KM_CODE_DIM)
    assert mouse.shape == (2, 0)


def test_km_codes_are_identical_whether_or_not_the_session_is_chunked(tmp_path, monkeypatch):
    """Chunking the tokenizer along ticks (to fit hours-long sessions on the GPU) must not change a code."""
    events = [("87", 0, 5 * da.TICK_MS), ("32", 3 * da.TICK_MS, 9 * da.TICK_MS)]
    whole = np.asarray(da.load_or_build(_make_session_dir_79(tmp_path / "a", n_ticks=10, key_events=events),
                                        action_encoding="km_fsq", action_builder="raw79")[0])
    monkeypatch.setattr(da, "KM_ENCODE_CHUNK_TICKS", 3)
    chunked = np.asarray(da.load_or_build(_make_session_dir_79(tmp_path / "b", n_ticks=10, key_events=events),
                                          action_encoding="km_fsq", action_builder="raw79")[0])
    assert np.array_equal(whole, chunked)


def test_load_ticks_79_keeps_every_key_the_8key_slice_drops(tmp_path):
    from improved_diffusion.km_tokenizer.keypress_scatter import _RAW_POSITIONS
    session_dir = _make_session_dir_79(
        tmp_path, n_ticks=2, key_events=[("87", 0, da.TICK_MS), ("69", 0, da.TICK_MS)])  # w, and e (not a debug key)
    ticks = da.load_ticks_79(session_dir)
    keypress, _ = da.load_or_build_raw(session_dir, action_builder="raw79")
    assert ticks.shape == (2, da.KEYPRESS_DIM_79)
    np.testing.assert_array_equal(ticks[:, _RAW_POSITIONS], keypress)
    assert ticks[1, KEYPRESS_LOOKUP["e"]] == 1.0
