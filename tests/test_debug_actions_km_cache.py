"""plaicraft-debug#80: the actions_km_codes.npy cache (load_or_build's km_fsq path)."""
import json
import sqlite3
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from improved_diffusion import debug_actions as da
from improved_diffusion.km_tokenizer.model import DEFAULT_CHECKPOINT, _sha256


def _make_session(tmp_path, n_ticks=3):
    session_dir = tmp_path / "sess"
    (session_dir / "encoded_video_hdf5").mkdir(parents=True)
    with h5py.File(session_dir / "encoded_video_hdf5" / "sess_encoded_video.hdf5", "w") as f:
        f.create_dataset("frames", data=np.zeros((n_ticks, 3, 2, 2), dtype=np.float32))
    con = sqlite3.connect(str(session_dir / "sess.db"))
    con.execute("CREATE TABLE keyboard (key_id TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_click (mouse_key_type TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_movement (timestamp INTEGER, mouseDX INTEGER, mouseDY INTEGER)")
    con.execute("INSERT INTO keyboard VALUES ('87', 0, 80)")
    for t in range(n_ticks):
        for b in range(da.SUBBINS_PER_TICK):
            con.execute("INSERT INTO mouse_movement VALUES (?, ?, ?)", (t * 80 + b * 10, b, -b))
    con.commit()
    con.close()
    return session_dir


def test_load_or_build_km_fsq_shape_and_cache_files(tmp_path):
    session_dir = _make_session(tmp_path, n_ticks=3)
    codes, mouse = da.load_or_build(session_dir, action_encoding="km_fsq")
    assert codes.shape == (3, da.KM_CODE_DIM)
    assert mouse.shape == (3, 0)
    codes_path, sidecar_path = da._km_cache_paths(session_dir)
    assert codes_path.exists() and sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["tokenizer_sha256"] == _sha256(DEFAULT_CHECKPOINT)
    assert sidecar["subbin_rule_version"] == da.SUBBIN_RULE_VERSION


def test_load_or_build_km_fsq_matches_direct_tokenizer_call(tmp_path):
    session_dir = _make_session(tmp_path, n_ticks=2)
    codes, _ = da.load_or_build(session_dir, action_encoding="km_fsq")

    from improved_diffusion.km_tokenizer.keypress_scatter import scatter_keypress
    from improved_diffusion.km_tokenizer.model import load_tokenizer

    key_sub, mouse_sub = da.build_action_array(session_dir / "sess.db", 2)
    key_press = scatter_keypress(torch.from_numpy(key_sub).float()).unsqueeze(0)
    mouse_movement = torch.from_numpy(mouse_sub).float().unsqueeze(0)
    tokenizer = load_tokenizer()
    with torch.no_grad():
        prequantized, _, _ = tokenizer._encode_prequantized(key_press, mouse_movement)
        _, quantized_codes = tokenizer._quantize(prequantized)
    expected = quantized_codes[0].reshape(2, da.KM_CODE_DIM).numpy()
    assert np.allclose(codes, expected, atol=1e-5)


def test_sidecar_mismatch_rebuilds_not_fails(tmp_path):
    session_dir = _make_session(tmp_path, n_ticks=2)
    codes, _ = da.load_or_build(session_dir, action_encoding="km_fsq")
    codes_path, sidecar_path = da._km_cache_paths(session_dir)

    sidecar_path.write_text(json.dumps({"tokenizer_sha256": "stale", "subbin_rule_version": "0"}))
    codes2, _ = da.load_or_build(session_dir, action_encoding="km_fsq")  # must rebuild, not raise
    assert np.allclose(codes, codes2, atol=1e-5)
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["subbin_rule_version"] == da.SUBBIN_RULE_VERSION


def test_missing_codes_file_rebuilds(tmp_path):
    session_dir = _make_session(tmp_path, n_ticks=2)
    da.load_or_build(session_dir, action_encoding="km_fsq")
    codes_path, sidecar_path = da._km_cache_paths(session_dir)
    codes_path.unlink()
    codes, _ = da.load_or_build(session_dir, action_encoding="km_fsq")  # rebuilds cleanly
    assert codes.shape == (2, da.KM_CODE_DIM)


def test_unknown_action_encoding_raises(tmp_path):
    session_dir = _make_session(tmp_path, n_ticks=2)
    with pytest.raises(ValueError):
        da.load_or_build(session_dir, action_encoding="bogus")
