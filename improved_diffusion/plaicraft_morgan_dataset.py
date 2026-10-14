"""plaicraft-debug#93: real PLAICraft (Morgan) sessions through the debug-world loader."""
import hashlib
import json
import sqlite3
from pathlib import Path

import h5py

from .debug_dataset import ContinuousDebugDataset

DEFAULT_GLOBAL_DATABASE_PATH = (
    "/ubc/cs/research/plai-scratch/plaicraft-dataset/versioning/global_databases/"
    "version_all_modalities_h5/high-quality/1.5k_hours_1140_players/global_database_training.db"
)
DEFAULT_VALIDATION_DB_PATH = "/ubc/cs/research/ubc_ml/plaicraft/validation.db"


def _hash_to_bucket(name, num_classes):
    """int(md5(name)) % num_classes -- identical to pi0's multimodal_io._hash_to_bucket."""
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % num_classes


def _ro_connect(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _name_from_global_db(session_id, global_database_path):
    con = _ro_connect(global_database_path)
    try:
        row = con.execute(
            "SELECT player_name FROM session_metadata WHERE session_id = ?", (session_id,)
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else None


def _name_from_other_metadata(session_dir):
    sid = session_dir.name
    con = _ro_connect(session_dir / f"{sid}.db")
    try:
        row = con.execute("SELECT other_metadata FROM session").fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        con.close()
    if row is None or not row[0]:
        return None
    try:
        return json.loads(row[0]).get("name")
    except (json.JSONDecodeError, TypeError):
        return None


def session_player_index(session_dir, num_classes, global_database_path=DEFAULT_GLOBAL_DATABASE_PATH):
    """md5(player_name) % num_classes; name from the global DB, else the session DB's other_metadata."""
    session_dir = Path(session_dir)
    name = _name_from_global_db(session_dir.name, global_database_path)
    if name is None:
        name = _name_from_other_metadata(session_dir)
    if name is None:
        raise ValueError(
            f"plaicraft_morgan: no player name for session {session_dir.name} in the "
            f"global DB ({global_database_path}) or its session.other_metadata."
        )
    return _hash_to_bucket(name, num_classes)


class PlaicraftDataset(ContinuousDebugDataset):
    """ContinuousDebugDataset, with four overrides for the real (non-synthetic) corpus:
    - split: sessions listed in validation.db are test, the rest are train, unless
      train_session_ids names the train sessions;
    - fps: the HDF5's own `fps` attr, falling back to session.fps;
    - player: md5(name) % num_classes, name from the global DB or, failing that, the
      session DB's other_metadata (the val session predates the global DB);
    - action builder: B1's 79-key PLAIOmni builder.
    """

    def __init__(self, dataset_path, *args, global_database_path=DEFAULT_GLOBAL_DATABASE_PATH,
                 validation_db_path=DEFAULT_VALIDATION_DB_PATH, action_builder="raw79",
                 train_session_ids=None, **kwargs):
        self.global_database_path = Path(global_database_path)
        self.validation_db_path = Path(validation_db_path)
        self.train_session_ids = train_session_ids
        super().__init__(dataset_path, *args, action_builder=action_builder, **kwargs)

    # ------------------------------------------------------------------ #
    # split
    # ------------------------------------------------------------------ #

    def _validation_session_ids(self):
        con = _ro_connect(self.validation_db_path)
        try:
            rows = con.execute("SELECT DISTINCT Session_ID FROM validation_rows").fetchall()
        finally:
            con.close()
        return {r[0] for r in rows}

    def _discover_sessions(self):
        files = sorted(
            self.dataset_path.glob("*/encoded_video_hdf5/*.hdf5"),
            key=lambda p: p.parent.parent.name,
        )
        if not files:
            raise ValueError(f"No hdf5 files found under {self.dataset_path}")
        if self.train_session_ids and not self.is_test:
            missing = set(self.train_session_ids) - {f.parent.parent.name for f in files}
            if missing:
                raise ValueError(f"plaicraft_morgan: train_session_ids not in corpus: {sorted(missing)}")
            return [f for f in files if f.parent.parent.name in self.train_session_ids]
        test_sids = self._validation_session_ids()
        test_files = [f for f in files if f.parent.parent.name in test_sids]
        train_files = [f for f in files if f.parent.parent.name not in test_sids]
        return test_files if self.is_test else train_files

    # ------------------------------------------------------------------ #
    # fps
    # ------------------------------------------------------------------ #

    def _read_fps(self, session_dir):
        session_dir = Path(session_dir)
        hdf5_dir = session_dir / "encoded_video_hdf5"
        matches = sorted(hdf5_dir.glob("*.hdf5"))
        if matches:
            with h5py.File(matches[0], "r") as f:
                attrs = f["frames"].attrs if "fps" in f["frames"].attrs else f.attrs  # encoder stamps the dataset
                if "fps" in attrs:
                    return float(attrs["fps"])
        # Real PLAICraft DB, read-only fallback -- the fps column, never other_metadata.
        con = _ro_connect(session_dir / f"{session_dir.name}.db")
        try:
            row = con.execute("SELECT fps FROM session").fetchone()
        finally:
            con.close()
        return float(row[0]) if row is not None else None

    def _validate_action_encoding(self):
        from .debug_actions import validate_action_encoding

        for _fs, _fe, path in self.file_boundaries:
            fps = self._read_fps(path.parent.parent)
            validate_action_encoding(self.action_encoding, fps=fps)

    # ------------------------------------------------------------------ #
    # player
    # ------------------------------------------------------------------ #

    def _resolve_players(self):
        self._session_players = {}
        for _fs, _fe, path in self.file_boundaries:
            session_dir = path.parent.parent
            if self.num_classes > 0:
                idx = session_player_index(session_dir, self.num_classes, self.global_database_path)
            else:
                idx = 0
            self._session_players[session_dir] = idx
