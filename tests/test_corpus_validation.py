"""Issue #81: CorpusValidationSet reads the frozen validation.npz + manifest.json
contract plaicraft-debug writes. Fixture below fabricates a tiny 3-exercise package."""
import json

import numpy as np
import pytest
import torch

from improved_diffusion.corpus_validation import CorpusValidationSet

NAMES = ["walk-forward", "strafe-left", "walk-backward"]
T, N_OBS, H, W = 6, 3, 4, 4


def _make_fixture(tmp_path):
    vdir = tmp_path / "validation"
    vdir.mkdir()
    n = len(NAMES)
    rng = np.random.RandomState(0)
    frames = rng.uniform(-1, 1, size=(n, T, 3, H, W)).astype(np.float32)
    keypress = rng.rand(n, T, 8).astype(np.float32)
    mouse = rng.randn(n, T, 2).astype(np.float32)
    window_start_ticks = np.array([100, 200, 300], dtype=np.int64)
    boundary_ticks = window_start_ticks + N_OBS
    np.savez(
        vdir / "validation.npz", frames=frames, keypress=keypress, mouse=mouse,
        names=np.array(NAMES), session_ids=np.array([f"sess{i}" for i in range(n)]),
        window_start_ticks=window_start_ticks, boundary_ticks=boundary_ticks,
    )
    manifest = {
        "schema_version": "1", "corpus_dir": str(tmp_path), "tick_ms": 80, "subbin_ms": 10,
        "subbin_rule_version": "1", "T": T, "n_observed": N_OBS, "n_held_out_sessions": 20,
        "built_at": "2026-01-01T00:00:00Z",
        "exercises": [
            {"index": 0, "name": "walk-forward", "session_id": "sess0",
             "swap_kind": "keypress", "swap_dim": 0, "swap_counterpart_dim": 2},
            {"index": 1, "name": "strafe-left", "session_id": "sess1",
             "swap_kind": "keypress", "swap_dim": 1, "swap_counterpart_dim": 3},
            {"index": 2, "name": "walk-backward", "session_id": "sess2", "swap_kind": "mouse_dx"},
        ],
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest))
    return vdir, frames, keypress, mouse


def test_load_all_and_actions_round_trip(tmp_path):
    vdir, frames, keypress, mouse = _make_fixture(tmp_path)
    vs = CorpusValidationSet(vdir)
    assert vs.T == T and vs.n_observed == N_OBS
    got_frames = vs.load_all()
    assert got_frames.shape == (3, T, 3, H, W)
    assert torch.allclose(got_frames, torch.from_numpy(frames))
    got_k, got_m = vs.load_all_actions()
    assert got_k.shape == (3, T, 8) and got_m.shape == (3, T, 2)
    assert torch.allclose(got_k, torch.from_numpy(keypress))
    assert torch.allclose(got_m, torch.from_numpy(mouse))
    raw_k, raw_m = vs.load_all_actions_raw()
    assert torch.allclose(raw_k, got_k) and torch.allclose(raw_m, got_m)


def test_slug_format(tmp_path):
    vdir, *_ = _make_fixture(tmp_path)
    vs = CorpusValidationSet(vdir)
    assert vs.slug(vs.rows[0]) == "00_walk-forward"
    assert vs.slug(vs.rows[1]) == "01_strafe-left"
    assert vs.slug(vs.rows[2]) == "02_walk-backward"


def test_rows_carry_swap_metadata(tmp_path):
    vdir, *_ = _make_fixture(tmp_path)
    vs = CorpusValidationSet(vdir)
    assert vs.rows[0]["swap_kind"] == "keypress"
    assert vs.rows[0]["swap_dim"] == 0 and vs.rows[0]["swap_counterpart_dim"] == 2
    assert vs.rows[2]["swap_kind"] == "mouse_dx"
    assert vs.rows[2]["swap_dim"] is None
    assert vs.rows[0]["window_start"] == 100
    assert vs.rows[0]["boundary_tick"] == 100 + N_OBS


def test_mismatched_T_raises(tmp_path):
    vdir, *_ = _make_fixture(tmp_path)
    with pytest.raises(ValueError) as exc:
        CorpusValidationSet(vdir, T=99)
    assert "6" in str(exc.value) and "99" in str(exc.value)


def test_mismatched_n_observed_raises(tmp_path):
    vdir, *_ = _make_fixture(tmp_path)
    with pytest.raises(ValueError) as exc:
        CorpusValidationSet(vdir, n_observed=99)
    assert str(N_OBS) in str(exc.value) and "99" in str(exc.value)


def test_missing_npz_raises(tmp_path):
    vdir = tmp_path / "empty"
    vdir.mkdir()
    with pytest.raises(FileNotFoundError):
        CorpusValidationSet(vdir)
