"""plaicraft-debug#80: the raw/km_fsq action_encoding guard.

km_fsq needs a 12.5Hz corpus (action_dim=36, mouse_dim=0); raw needs the
pre-#80 8+2 split. Both directions must fail loudly rather than silently
training on a mismatched action space.
"""
import sqlite3
import warnings

import h5py
import numpy as np
import pytest

from improved_diffusion.debug_actions import validate_action_encoding
from improved_diffusion.debug_dataset import ContinuousDebugDataset


def test_validate_action_encoding_km_fsq_dims_ok():
    validate_action_encoding("km_fsq", fps=12.5, action_dim=36, mouse_dim=0)


def test_validate_action_encoding_km_fsq_wrong_fps_raises():
    with pytest.raises(ValueError, match="fps"):
        validate_action_encoding("km_fsq", fps=10.0)


def test_validate_action_encoding_km_fsq_wrong_action_dim_raises():
    with pytest.raises(ValueError, match="action_dim"):
        validate_action_encoding("km_fsq", action_dim=8)


def test_validate_action_encoding_km_fsq_wrong_mouse_dim_raises():
    with pytest.raises(ValueError, match="mouse_dim"):
        validate_action_encoding("km_fsq", mouse_dim=2)


def test_validate_action_encoding_raw_dims_ok():
    validate_action_encoding("raw", action_dim=8, mouse_dim=2)


def test_validate_action_encoding_raw_wrong_action_dim_raises():
    with pytest.raises(ValueError, match="action_dim"):
        validate_action_encoding("raw", action_dim=36)


def test_validate_action_encoding_raw_wrong_mouse_dim_raises():
    with pytest.raises(ValueError, match="mouse_dim"):
        validate_action_encoding("raw", mouse_dim=0)


def test_validate_action_encoding_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown action_encoding"):
        validate_action_encoding("bogus")


def _make_corpus(tmp_path, fps, n_sessions=2, n_ticks=5):
    for i in range(n_sessions):
        session_dir = tmp_path / f"session_{i:03d}"
        (session_dir / "encoded_video_hdf5").mkdir(parents=True)
        with h5py.File(session_dir / "encoded_video_hdf5" / f"session_{i:03d}_encoded_video.hdf5", "w") as f:
            f.create_dataset("frames", data=np.zeros((n_ticks, 3, 2, 2), dtype=np.float32))
        con = sqlite3.connect(str(session_dir / f"session_{i:03d}.db"))
        con.execute("CREATE TABLE session (fps REAL)")
        con.execute("INSERT INTO session VALUES (?)", (fps,))
        con.execute("CREATE TABLE keyboard (key_id TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
        con.execute("CREATE TABLE mouse_click (mouse_key_type TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
        con.execute("CREATE TABLE mouse_movement (timestamp INTEGER, mouseDX INTEGER, mouseDY INTEGER)")
        con.commit()
        con.close()
    return tmp_path


def test_dataset_construction_rejects_km_fsq_on_a_10fps_corpus(tmp_path):
    root = _make_corpus(tmp_path, fps=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="fps"):
            ContinuousDebugDataset(root, window_length=2, action_encoding="km_fsq")


def test_dataset_construction_accepts_km_fsq_on_a_12_5fps_corpus(tmp_path):
    root = _make_corpus(tmp_path, fps=12.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = ContinuousDebugDataset(root, window_length=2, action_encoding="km_fsq")
    assert ds.action_encoding == "km_fsq"


def test_dataset_construction_accepts_raw_regardless_of_fps(tmp_path):
    root = _make_corpus(tmp_path, fps=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = ContinuousDebugDataset(root, window_length=2, action_encoding="raw")
    assert ds.action_encoding == "raw"
