"""plaicraft-debug#93 B2: PlaicraftDataset's four overrides on a tmp corpus fixture."""
import json
import sqlite3

import h5py
import numpy as np
import pytest

from improved_diffusion.plaicraft_morgan_dataset import PlaicraftDataset, _hash_to_bucket


def _make_session(corpus_dir, sid, n_ticks=4, fps=12.5, session_fps=30.0, other_metadata=None):
    session_dir = corpus_dir / sid
    (session_dir / "encoded_video_hdf5").mkdir(parents=True)
    with h5py.File(session_dir / "encoded_video_hdf5" / f"{sid}.hdf5", "w") as f:
        f.create_dataset("frames", data=np.zeros((n_ticks, 4, 24, 40), dtype=np.float16))
        f.attrs["fps"] = fps
    con = sqlite3.connect(str(session_dir / f"{sid}.db"))
    con.execute("CREATE TABLE session (fps REAL, other_metadata TEXT)")
    con.execute(
        "INSERT INTO session VALUES (?, ?)",
        (session_fps, json.dumps(other_metadata) if other_metadata else None),
    )
    con.execute("CREATE TABLE keyboard (key_id TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_click (mouse_key_type TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_movement (timestamp INTEGER, mouseDX INTEGER, mouseDY INTEGER)")
    con.commit()
    con.close()
    return session_dir


def _make_global_db(path, rows):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE session_metadata (session_id TEXT, player_name TEXT)")
    con.executemany("INSERT INTO session_metadata VALUES (?, ?)", rows)
    con.commit()
    con.close()


def _make_validation_db(path, session_ids):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE validation_rows (Session_ID TEXT)")
    con.executemany("INSERT INTO validation_rows VALUES (?)", [(s,) for s in session_ids])
    con.commit()
    con.close()


@pytest.fixture
def corpus(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _make_session(corpus_dir, "trainA")
    _make_session(corpus_dir, "trainB")
    _make_session(corpus_dir, "valSess")
    global_db = tmp_path / "global.db"
    _make_global_db(global_db, [("trainA", "Morgan"), ("trainB", "Morgan")])
    # valSess predates the global DB (like d4e57...) -- name only in other_metadata.
    val_dir = corpus_dir / "valSess"
    con = sqlite3.connect(str(val_dir / "valSess.db"))
    con.execute("UPDATE session SET other_metadata = ?", (json.dumps({"name": "Morgan"}),))
    con.commit()
    con.close()
    validation_db = tmp_path / "validation.db"
    _make_validation_db(validation_db, ["valSess"])
    return dict(corpus_dir=corpus_dir, global_db=global_db, validation_db=validation_db)


def _dataset(corpus, **kwargs):
    kwargs.setdefault("window_length", 3)
    kwargs.setdefault("num_classes", 20000)
    return PlaicraftDataset(
        corpus["corpus_dir"], global_database_path=corpus["global_db"],
        validation_db_path=corpus["validation_db"], **kwargs,
    )


def test_split_uses_validation_db_membership(corpus):
    ds = _dataset(corpus)
    train_sids = {p.parent.parent.name for _fs, _fe, p in ds.file_boundaries}
    assert train_sids == {"trainA", "trainB"}
    ds.set_test()
    test_sids = {p.parent.parent.name for _fs, _fe, p in ds.file_boundaries}
    assert test_sids == {"valSess"}


def test_fps_read_from_hdf5_attr_not_session_fps(corpus):
    ds = _dataset(corpus, action_encoding="raw_fused")  # strict fps==12.5 check
    session_dir = corpus["corpus_dir"] / "trainA"
    assert ds._read_fps(session_dir) == 12.5  # hdf5 attr, not session.fps's 30.0


def test_player_index_matches_hash_to_bucket_from_global_db(corpus):
    ds = _dataset(corpus)
    session_dir = corpus["corpus_dir"] / "trainA"
    assert ds._session_players[session_dir] == _hash_to_bucket("Morgan", 20000)


def test_player_falls_back_to_other_metadata_when_absent_from_global_db(corpus):
    ds = _dataset(corpus)
    ds.set_test()
    session_dir = corpus["corpus_dir"] / "valSess"
    assert ds._session_players[session_dir] == _hash_to_bucket("Morgan", 20000)


def test_num_classes_zero_skips_player_lookup_entirely(corpus, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("should never query player identity when num_classes==0")
    monkeypatch.setattr(PlaicraftDataset, "_player_name", _boom)
    ds = _dataset(corpus, num_classes=0)
    assert all(v == 0 for v in ds._session_players.values())


def test_missing_player_name_raises(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    _make_session(corpus_dir, "orphan")
    global_db = tmp_path / "global.db"
    _make_global_db(global_db, [])
    validation_db = tmp_path / "validation.db"
    _make_validation_db(validation_db, [])
    with pytest.raises(ValueError, match="orphan"):
        PlaicraftDataset(
            corpus_dir, window_length=3, num_classes=20000,
            global_database_path=global_db, validation_db_path=validation_db,
        )


def test_hash_to_bucket_formula():
    import hashlib
    name = "Morgan"
    assert _hash_to_bucket(name, 20000) == int(hashlib.md5(name.encode()).hexdigest(), 16) % 20000
