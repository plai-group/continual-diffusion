"""issue #76 slice 1: DebugCorpusWindowSet's session holdout, window sampling, seed."""

import pickle
import sqlite3
import tempfile
from pathlib import Path

import h5py
import numpy as np

from improved_diffusion.debug_dataset import ContinuousDebugDataset
from improved_diffusion.debug_validation import DebugCorpusWindowSet


def _make_session(session_dir, n_frames):
    """Minimal hdf5 + db (keyboard/mouse tables + key_press_encodings) for one session."""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    hdf5_dir = session_dir / "encoded_video_hdf5"
    hdf5_dir.mkdir()
    with h5py.File(hdf5_dir / f"{session_dir.name}_encoded_video.hdf5", "w") as f:
        f.create_dataset("frames", data=np.zeros((n_frames, 1), dtype=np.float32))

    db_path = session_dir / f"{session_dir.name}.db"
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE keyboard (key_id TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_click (mouse_key_type TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_movement (timestamp INTEGER, mouseDX REAL, mouseDY REAL)")
    con.execute(
        "CREATE TABLE key_press_encodings "
        "(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, "
        "start_timestamp INTEGER, end_timestamp INTEGER, encoding BLOB)"
    )
    for k in range(n_frames):
        start = k * 100
        con.execute(
            "INSERT INTO key_press_encodings (start_timestamp, end_timestamp, encoding) VALUES (?, ?, ?)",
            (start, start + 100, pickle.dumps(np.zeros((16, 5), dtype=np.float32))),
        )
    con.commit()
    con.close()


def _make_corpus(root, n_sessions, n_frames):
    ids = [f"s{k:03d}" for k in range(n_sessions)]
    for sid in ids:
        _make_session(root / sid, n_frames)
    return ids


def test_holdout_disjoint_from_train():
    n_sessions, n_frames, T = 25, 4, 2
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ids = _make_corpus(root, n_sessions, n_frames)
        ws = DebugCorpusWindowSet(root, T=T, n_observed=1, n_windows=1000, seed=0)

    held_out = set(ids[-ContinuousDebugDataset.N_TEST_SESSIONS:])
    train = set(ids[:-ContinuousDebugDataset.N_TEST_SESSIONS])
    seen = {r["session_id"] for r in ws.rows}
    assert seen <= held_out
    assert not (seen & train)


def test_windows_non_overlapping():
    T = 3
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_corpus(root, n_sessions=2, n_frames=9)  # 3 non-overlapping windows/session
        ws = DebugCorpusWindowSet(root, T=T, n_observed=1, n_windows=1000, seed=0)

    by_session = {}
    for r in ws.rows:
        by_session.setdefault(r["session_id"], []).append(r["window_start"])
    for starts in by_session.values():
        starts = sorted(starts)
        for a, b in zip(starts, starts[1:]):
            assert b - a >= T, "windows must not overlap"


def test_seed_reproducible():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_corpus(root, n_sessions=3, n_frames=10)
        ws_a = DebugCorpusWindowSet(root, T=2, n_observed=1, n_windows=5, seed=42)
        ws_b = DebugCorpusWindowSet(root, T=2, n_observed=1, n_windows=5, seed=42)
        ws_c = DebugCorpusWindowSet(root, T=2, n_observed=1, n_windows=5, seed=7)

    assert ws_a.rows == ws_b.rows
    assert ws_a.rows != ws_c.rows


def test_load_all_and_actions_shapes():
    T = 4
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_corpus(root, n_sessions=1, n_frames=8)
        ws = DebugCorpusWindowSet(root, T=T, n_observed=2, n_windows=10, seed=0)

        x0 = ws.load_all()
        keypress, mouse = ws.load_all_actions()
        raw = ws.load_all_keypress_raw()

    n = len(ws.rows)
    assert x0.shape == (n, T, 1)
    assert keypress.shape == (n, T, 80)
    assert mouse.shape == (n, T, 2)
    assert raw.shape == (n, T, 8)
    assert ws.slug(ws.rows[0])
