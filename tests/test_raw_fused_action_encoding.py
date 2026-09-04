"""plaicraft-debug#81: the raw_fused action_encoding (10-dim single token, mouse_dim=0).

Restores the pre-#16 layout best-run-69 used: [8 keypress, symlog(dx), symlog(dy)].
Must reuse load_or_build_raw's cache -- a raw and a raw_fused run share one warmed corpus.
"""
import sqlite3

import h5py
import numpy as np
import pytest
import torch

from improved_diffusion import debug_actions as da
from improved_diffusion.debug_actions import validate_action_encoding


def test_symlog_round_trip_numpy():
    v = np.array([-150.0, -1.0, 0.0, 1.0, 150.0], dtype=np.float32)
    assert np.allclose(da._inv_symlog(da._symlog(v)), v, atol=1e-3)


def test_symlog_round_trip_torch():
    v = torch.tensor([-150.0, -1.0, 0.0, 1.0, 150.0])
    assert torch.allclose(da._inv_symlog(da._symlog(v)), v, atol=1e-3)


def test_symlog_compresses_large_mouse_values():
    # +/-150px -> roughly +/-5.0, per the design note.
    assert abs(float(da._symlog(np.array([150.0]))[0]) - 5.0) < 0.05


def test_validate_action_encoding_raw_fused_dims_ok():
    validate_action_encoding("raw_fused", action_dim=10, mouse_dim=0)


def test_validate_action_encoding_raw_fused_rejects_two_headed_dims():
    with pytest.raises(ValueError, match="mouse_dim"):
        validate_action_encoding("raw_fused", action_dim=10, mouse_dim=2)


def test_validate_action_encoding_raw_fused_rejects_raw_dims():
    with pytest.raises(ValueError, match="action_dim"):
        validate_action_encoding("raw_fused", action_dim=8, mouse_dim=0)


def test_validate_action_encoding_raw_fused_has_no_fps_gate():
    validate_action_encoding("raw_fused", fps=10.0, action_dim=10, mouse_dim=0)


def test_validate_action_encoding_unknown_mode_still_raises():
    with pytest.raises(ValueError, match="unknown action_encoding"):
        validate_action_encoding("bogus")


def _make_session(tmp_path, n_ticks=3):
    session_dir = tmp_path / "sess"
    (session_dir / "encoded_video_hdf5").mkdir(parents=True)
    with h5py.File(session_dir / "encoded_video_hdf5" / "sess_encoded_video.hdf5", "w") as f:
        f.create_dataset("frames", data=np.zeros((n_ticks, 3, 2, 2), dtype=np.float32))
    con = sqlite3.connect(str(session_dir / "sess.db"))
    con.execute("CREATE TABLE keyboard (key_id TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_click (mouse_key_type TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_movement (timestamp INTEGER, mouseDX INTEGER, mouseDY INTEGER)")
    con.execute("INSERT INTO keyboard VALUES ('87', 0, 80)")  # w held during raw tick 0
    for t in range(n_ticks):
        for b in range(da.SUBBINS_PER_TICK):
            con.execute("INSERT INTO mouse_movement VALUES (?, ?, ?)", (t * 80 + b * 10, 20, -20))
    con.commit()
    con.close()
    return session_dir


def test_load_or_build_raw_fused_shape_and_keypress_columns_match_raw(tmp_path):
    session_dir = _make_session(tmp_path, n_ticks=3)
    keypress, mouse = da.load_or_build_raw(session_dir)
    fused, fused_mouse = da.load_or_build(session_dir, action_encoding="raw_fused")
    assert fused.shape == (3, da.RAW_FUSED_DIM)
    assert fused_mouse.shape == (3, 0)
    assert np.array_equal(np.asarray(fused[:, :8]), np.asarray(keypress))
    assert np.allclose(np.asarray(fused[:, 8:]), da._symlog(np.asarray(mouse)))


def test_load_or_build_raw_fused_does_not_create_a_third_cache_file(tmp_path):
    session_dir = _make_session(tmp_path, n_ticks=2)
    da.load_or_build(session_dir, action_encoding="raw_fused")
    files = sorted(p.name for p in session_dir.glob("actions_*"))
    assert files == ["actions_keypress.npy", "actions_mouse.npy"]


def test_raw_and_raw_fused_share_the_same_warmed_cache(tmp_path):
    session_dir = _make_session(tmp_path, n_ticks=2)
    da.load_or_build(session_dir, action_encoding="raw")  # warms the cache
    keypress_path = session_dir / "actions_keypress.npy"
    mtime_before = keypress_path.stat().st_mtime_ns
    da.load_or_build(session_dir, action_encoding="raw_fused")
    assert keypress_path.stat().st_mtime_ns == mtime_before  # not rebuilt
