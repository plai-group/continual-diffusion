"""plaicraft-debug#96: km_ctx14's 10Hz sub-bin derivation, lattice snap, and codes cache."""
import json
import sqlite3

import h5py
import numpy as np
import pytest
import torch

from improved_diffusion import debug_actions as da
from improved_diffusion.km_tokenizer.ctx14_10hz import CHECKPOINT_SHA256, DEFAULT_CHECKPOINT_DIR


def _make_session_79(tmp_path, fps, n_ticks=3, key_events=(), mouse_rows=()):
    session_dir = tmp_path / "sess"
    (session_dir / "encoded_video_hdf5").mkdir(parents=True)
    with h5py.File(session_dir / "encoded_video_hdf5" / "sess.hdf5", "w") as f:
        f.create_dataset("frames", data=np.zeros((n_ticks, 3, 2, 2), dtype=np.float32))
    con = sqlite3.connect(str(session_dir / "sess.db"))
    con.execute("CREATE TABLE session (fps REAL)")
    con.execute("INSERT INTO session VALUES (?)", (fps,))
    con.execute("CREATE TABLE keyboard (key_id TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_click (mouse_key_type TEXT, start_timestamp INTEGER, end_timestamp INTEGER)")
    con.execute("CREATE TABLE mouse_movement (timestamp INTEGER, mouseDX INTEGER, mouseDY INTEGER)")
    for key_id, s, e in key_events:
        con.execute("INSERT INTO keyboard VALUES (?, ?, ?)", (key_id, s, e))
    for ts, dx, dy in mouse_rows:
        con.execute("INSERT INTO mouse_movement VALUES (?, ?, ?)", (ts, dx, dy))
    con.commit()
    con.close()
    return session_dir


def test_subbins_per_tick_for_fps():
    assert da.subbins_per_tick_for_fps(12.5) == 8
    assert da.subbins_per_tick_for_fps(10.0) == 10


def test_build_action_array_79_bins_at_10hz_sub_rate(tmp_path):
    session_dir = _make_session_79(tmp_path, fps=10.0, n_ticks=3, key_events=[("87", 0, 100)])
    key_sub, mouse_sub = da.build_action_array_79(
        session_dir / "sess.db", n_ticks=3, subbins_per_tick=da.subbins_per_tick_for_fps(10.0))
    assert key_sub.shape == (30, da.KEYPRESS_DIM_79)
    assert mouse_sub.shape == (30, da.MOUSE_DIM)


def test_quantize_km_ctx14_matches_fsq_round_trip():
    torch.manual_seed(0)
    from improved_diffusion.km_tokenizer.ctx14_10hz import FSQ

    fsq = FSQ(list(da._CTX14_FSQ_LEVELS))
    raw = torch.rand(5, 8, 4) * 2 - 1
    quantized, indices = fsq(raw)
    expected = fsq.indices_to_codes(indices).reshape(5, 32)
    got = da.quantize_km_ctx14(quantized.reshape(5, 32))
    assert torch.allclose(got, expected, atol=1e-5)


def test_quantize_km_ctx14_is_idempotent_on_lattice_points():
    torch.manual_seed(1)
    from improved_diffusion.km_tokenizer.ctx14_10hz import FSQ

    fsq = FSQ(list(da._CTX14_FSQ_LEVELS))
    raw = torch.rand(4, 8, 4) * 2 - 1
    lattice = fsq(raw)[0].reshape(4, 32)
    once = da.quantize_km_ctx14(lattice)
    twice = da.quantize_km_ctx14(once)
    assert torch.allclose(once, twice, atol=1e-6)


def test_quantize_km_ctx14_wrong_last_dim_raises():
    with pytest.raises(ValueError, match="32"):
        da.quantize_km_ctx14(torch.zeros(2, 3, 16))


@pytest.mark.skipif(not DEFAULT_CHECKPOINT_DIR.exists(), reason="staged km_ctx14 checkpoint not present")
def test_load_or_build_km_ctx14_shape_and_cache_files(tmp_path):
    session_dir = _make_session_79(tmp_path, fps=10.0, n_ticks=3, key_events=[("87", 0, 100)])
    codes, mouse = da.load_or_build(session_dir, action_encoding="km_ctx14", action_builder="raw79")
    assert codes.shape == (3, da.KM_CTX14_CODE_DIM)
    assert mouse.shape == (3, 0)
    codes_path, sidecar_path = da._km_ctx14_cache_paths(session_dir)
    assert codes_path.exists() and sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar == {
        "tokenizer": "km_ctx14_10hz", "tokenizer_sha256": CHECKPOINT_SHA256,
        "subbin_rule_version": da.SUBBIN_RULE_VERSION, "action_builder": "raw79",
    }


@pytest.mark.skipif(not DEFAULT_CHECKPOINT_DIR.exists(), reason="staged km_ctx14 checkpoint not present")
def test_km_ctx14_cache_round_trips_through_disk(tmp_path):
    session_dir = _make_session_79(
        tmp_path, fps=10.0, n_ticks=10,
        key_events=[("87", 0, 500)], mouse_rows=[(50, 5, -5)],
    )
    first, _ = da.load_or_build(session_dir, action_encoding="km_ctx14", action_builder="raw79")
    second, _ = da.load_or_build(session_dir, action_encoding="km_ctx14", action_builder="raw79")
    assert np.array_equal(np.asarray(first), np.asarray(second))


@pytest.mark.skipif(not DEFAULT_CHECKPOINT_DIR.exists(), reason="staged km_ctx14 checkpoint not present")
def test_km_ctx14_codes_are_identical_whether_or_not_the_session_is_chunked(tmp_path, monkeypatch):
    events = [("87", 0, 500), ("32", 300, 900)]
    whole, _ = da.load_or_build(
        _make_session_79(tmp_path / "a", fps=10.0, n_ticks=10, key_events=events),
        action_encoding="km_ctx14", action_builder="raw79")
    monkeypatch.setattr(da, "KM_ENCODE_CHUNK_TICKS", 3)
    chunked, _ = da.load_or_build(
        _make_session_79(tmp_path / "b", fps=10.0, n_ticks=10, key_events=events),
        action_encoding="km_ctx14", action_builder="raw79")
    assert np.allclose(np.asarray(whole), np.asarray(chunked), atol=1e-5)


def test_encode_km_ctx14_context_ignores_actions_after_boundary(tmp_path):
    """plaicraft-debug#96 B4: the bidirectional encoder must never see past `boundary` --
    two sessions identical up to it, differing only after, must encode identically."""
    from improved_diffusion.km_tokenizer.ctx14_10hz import KmTokenizerCtx14_10Hz, KmTokenizerCtx14_10HzConfig

    torch.manual_seed(0)
    tokenizer = KmTokenizerCtx14_10Hz(KmTokenizerCtx14_10HzConfig())
    tokenizer.eval()

    window_start, boundary = 2, 5
    shared = [("87", 0, 300)]  # held through the shared prefix
    a = _make_session_79(tmp_path / "a", fps=10.0, n_ticks=20, key_events=shared + [("65", 600, 900)])
    b = _make_session_79(tmp_path / "b", fps=10.0, n_ticks=20, key_events=shared + [("83", 600, 1900)])

    codes_a = da.encode_km_ctx14_context(a, window_start, boundary, tokenizer, "cpu", "raw79")
    codes_b = da.encode_km_ctx14_context(b, window_start, boundary, tokenizer, "cpu", "raw79")
    assert codes_a.shape == (boundary - window_start, da.KM_CTX14_CODE_DIM)
    assert np.array_equal(codes_a, codes_b)


def test_load_ticks_79_bins_a_10hz_session_at_its_own_tick_rate(tmp_path):
    """plaicraft-debug#96: a key held at t=1000ms must land in tick 10 of a 10Hz session
    (100ms/tick), not tick 12 -- the GT drift a fixed 8-subbin assumption would cause."""
    session_dir = _make_session_79(tmp_path, fps=10.0, n_ticks=20, key_events=[("87", 1000, 1100)])
    ticks = da.load_ticks_79(session_dir)
    from improved_diffusion.km_tokenizer.keypress_lookup import KEYPRESS_LOOKUP
    w_idx = KEYPRESS_LOOKUP["87"]
    hit_ticks = np.nonzero(ticks[:, w_idx])[0]
    assert list(hit_ticks) == [11]  # raw tick 10 (1000-1100ms), shifted +1 by the causal shift


def test_load_ticks_79_on_a_12_5hz_session_is_unchanged(tmp_path):
    """A 12.5Hz (80ms/tick) session: a key held through [960,1040)ms (exactly tick 12's
    window) lands in tick 12, not 10 -- the pre-#96 8-subbin behaviour, byte-identical."""
    session_dir = _make_session_79(tmp_path, fps=12.5, n_ticks=20, key_events=[("87", 960, 1040)])
    ticks = da.load_ticks_79(session_dir)
    from improved_diffusion.km_tokenizer.keypress_lookup import KEYPRESS_LOOKUP
    w_idx = KEYPRESS_LOOKUP["87"]
    hit_ticks = np.nonzero(ticks[:, w_idx])[0]
    assert list(hit_ticks) == [13]  # raw tick 12 (960-1040ms), shifted +1 by the causal shift


def test_load_or_build_raw_raw79_bins_a_10hz_session_at_its_own_tick_rate(tmp_path):
    session_dir = _make_session_79(tmp_path, fps=10.0, n_ticks=20, key_events=[("87", 1000, 1100)])
    keypress, _ = da.load_or_build_raw(session_dir, action_builder="raw79")
    assert keypress.shape == (20, da.KEYPRESS_DIM)
    assert np.nonzero(keypress[:, 0])[0].tolist() == [11]


def test_raw79_cache_sidecar_invalidates_between_10hz_and_12_5hz(tmp_path):
    session_dir = _make_session_79(tmp_path, fps=10.0, n_ticks=5, key_events=[("87", 0, 100)])
    da.load_or_build_raw(session_dir, action_builder="raw79")
    _, _, sidecar_path = da._raw_cache_paths(session_dir)
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar == {
        "subbin_rule_version": da.SUBBIN_RULE_VERSION, "action_builder": "raw79", "subbins_per_tick": 10,
    }
    # Flip the on-disk session fps to 12.5Hz without touching the cache -- it must rebuild, not reuse a stale 10-subbin cache.
    con = sqlite3.connect(str(session_dir / "sess.db"))
    con.execute("UPDATE session SET fps = 12.5")
    con.commit()
    con.close()
    da.load_or_build_raw(session_dir, action_builder="raw79")
    sidecar2 = json.loads(sidecar_path.read_text())
    assert sidecar2 == {"subbin_rule_version": da.SUBBIN_RULE_VERSION, "action_builder": "raw79"}


def test_km_fsq_sub_binning_still_defaults_to_legacy_constant(tmp_path):
    """build_action_array_79 with no subbins_per_tick override must stay byte-identical to
    pre-#96 (km_fsq/raw's path, always 8)."""
    session_dir = _make_session_79(tmp_path, fps=12.5, n_ticks=3, key_events=[("87", 0, 80)])
    key_sub, mouse_sub = da.build_action_array_79(session_dir / "sess.db", n_ticks=3)
    assert key_sub.shape == (3 * da.SUBBINS_PER_TICK, da.KEYPRESS_DIM_79)
    assert mouse_sub.shape == (3 * da.SUBBINS_PER_TICK, da.MOUSE_DIM)
