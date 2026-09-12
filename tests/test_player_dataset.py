"""Player id from the session DB through to the training batch (issue #85).

The load-bearing behaviour is the refusal: pointing a num_classes=2 run at a corpus with no
player field must raise at dataset construction, not train a 'player-conditioned' model on a
single-player corpus whose only symptom would be a flat val/player/margin weeks later.
"""
import json
import sqlite3

import h5py
import numpy as np
import pytest
import torch

from improved_diffusion.debug_actions import read_session_player
from improved_diffusion.debug_dataset import ContinuousDebugDataset


def _session(root, sid, n_ticks=40, player_index=None, H=24, W=40, fps=12.5):
    """A minimal debug_toy session: DB with a session row, plus the video HDF5."""
    d = root / sid
    (d / "encoded_video_hdf5").mkdir(parents=True)
    con = sqlite3.connect(d / f"{sid}.db")
    con.execute("CREATE TABLE session (session_id TEXT, fps REAL, other_metadata TEXT)")
    con.execute("CREATE TABLE keyboard (key_id TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_click (mouse_key_type TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_movement (timestamp INTEGER, mouseDX INTEGER, mouseDY INTEGER)")
    meta = {"name": "Debug_Agent", "version": "debug_v1"}
    if player_index is not None:
        meta = {"name": ["Bubsy", "Mario"][player_index], "player_index": player_index,
                "version": "debug_v1"}
    con.execute("INSERT INTO session VALUES (?, ?, ?)", (sid, fps, json.dumps(meta)))
    con.commit()
    con.close()
    with h5py.File(d / "encoded_video_hdf5" / f"{sid}_encoded_video.hdf5", "w") as f:
        f.create_dataset("frames", data=np.zeros((n_ticks, 3, H, W), dtype=np.float32))
    return d


def _corpus(tmp_path, players):
    for i, p in enumerate(players):
        _session(tmp_path, f"s{i:02d}", player_index=p)
    return tmp_path


# ── read_session_player ────────────────────────────────────────────────────────────────────

def test_read_session_player_reads_the_index(tmp_path):
    assert read_session_player(_session(tmp_path, "a", player_index=1)) == 1


def test_read_session_player_returns_none_on_a_pre_issue85_corpus(tmp_path):
    assert read_session_player(_session(tmp_path, "b", player_index=None)) is None


def test_read_session_player_tolerates_a_session_table_without_the_column(tmp_path):
    """Older corpora (and several test fixtures) have a session table with no other_metadata
    column at all. read_session_fps survives that because it only selects fps; this must too,
    and report 'no player' rather than raising OperationalError out of dataset construction."""
    d = tmp_path / "noschema"
    (d / "encoded_video_hdf5").mkdir(parents=True)
    con = sqlite3.connect(d / "noschema.db")
    con.execute("CREATE TABLE session (session_id TEXT, fps REAL)")
    con.execute("INSERT INTO session VALUES ('noschema', 12.5)")
    con.commit()
    con.close()
    assert read_session_player(d) is None


def test_read_session_player_tolerates_a_missing_session_table(tmp_path):
    d = tmp_path / "notable"
    (d / "encoded_video_hdf5").mkdir(parents=True)
    sqlite3.connect(d / "notable.db").close()
    assert read_session_player(d) is None


# ── the refusal ────────────────────────────────────────────────────────────────────────────

def test_missing_player_field_raises_when_labels_are_requested(tmp_path):
    _corpus(tmp_path, [None, None, 0])
    with pytest.raises(ValueError, match="no player_index"):
        ContinuousDebugDataset(tmp_path, window_length=20, num_classes=2)


def test_missing_player_field_is_fine_when_labels_are_not_requested(tmp_path):
    """Every pre-issue-85 run script must keep working untouched."""
    ds = ContinuousDebugDataset(_corpus(tmp_path, [None, None]), window_length=20)
    assert len(ds) > 0


def test_player_index_outside_num_classes_raises(tmp_path):
    _corpus(tmp_path, [0, 1])
    with pytest.raises(ValueError, match="outside range"):
        ContinuousDebugDataset(tmp_path, window_length=20, num_classes=1)


# ── the batch ──────────────────────────────────────────────────────────────────────────────

def test_getitem_returns_the_sessions_player(tmp_path):
    ds = ContinuousDebugDataset(_corpus(tmp_path, [0, 1]), window_length=20, num_classes=2)
    seen = {}
    for i in range(len(ds)):
        frames, idx_map, keypress, mouse, player = ds[i]
        assert player.dtype == torch.long and player.shape == ()
        for (fs, fe, path) in ds.file_boundaries:
            if fs <= ds._get_start_frame_index(i) < fe:
                seen.setdefault(path.parent.parent.name, set()).add(int(player))
    assert seen == {"s00": {0}, "s01": {1}}, seen


def test_batch_is_a_five_tuple_only_for_the_player_path(tmp_path):
    ds_off = ContinuousDebugDataset(_corpus(tmp_path / "off", [None]), window_length=20)
    (tmp_path / "on").mkdir()
    ds_on = ContinuousDebugDataset(_corpus(tmp_path / "on", [1]), window_length=20, num_classes=2)
    # Both return 5 items: the dataset always carries the label, num_classes only controls
    # whether a MISSING one is fatal. The trainer keys on tuple length, so this must be stable.
    assert len(ds_off[0]) == 5 and len(ds_on[0]) == 5
    assert int(ds_off[0][4]) == 0 and int(ds_on[0][4]) == 1


def test_set_test_rebuilds_the_player_map(tmp_path):
    """set_test swaps the session list; a stale player map would KeyError in __getitem__."""
    ds = ContinuousDebugDataset(_corpus(tmp_path, [0, 1]), window_length=20, num_classes=2)
    ds.set_test()
    _f, _i, _k, _m, player = ds[0]
    assert int(player) in (0, 1)
