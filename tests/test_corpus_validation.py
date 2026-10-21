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


def _make_fixture_no_swap(tmp_path, n_channels=4):
    """plaicraft-debug#93: a manifest with no swap package at all (plaicraft_morgan's), and
    C=4 latent frames instead of C=3 pixels."""
    vdir = tmp_path / "validation"
    vdir.mkdir()
    n = len(NAMES)
    rng = np.random.RandomState(0)
    frames = rng.uniform(-1, 1, size=(n, T, n_channels, H, W)).astype(np.float32)
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
        "exercises": [{"index": i, "name": name, "session_id": f"sess{i}"}
                     for i, name in enumerate(NAMES)],  # no swap_kind key at all
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest))
    return vdir


def test_manifest_without_swap_kind_loads_with_swap_kind_none(tmp_path):
    vdir = _make_fixture_no_swap(tmp_path)
    vs = CorpusValidationSet(vdir)
    assert all(row["swap_kind"] is None for row in vs.rows)


def test_c4_latent_frames_load(tmp_path):
    vdir = _make_fixture_no_swap(tmp_path, n_channels=4)
    vs = CorpusValidationSet(vdir)
    assert vs.load_all().shape == (3, T, 4, H, W)


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


def test_wrong_boundary_offset_raises(tmp_path):
    # Corrupt boundary tick away from n_observed offset; must raise ValueError.
    vdir, *_ = _make_fixture(tmp_path)
    npz = dict(np.load(vdir / "validation.npz"))
    npz["boundary_ticks"] = npz["boundary_ticks"].copy()
    npz["boundary_ticks"][1] += 1
    np.savez(vdir / "validation.npz", **npz)
    with pytest.raises(ValueError) as exc:
        CorpusValidationSet(vdir)
    assert "strafe-left" in str(exc.value) and str(N_OBS) in str(exc.value)


def test_row_count_mismatch_raises(tmp_path):
    vdir, *_ = _make_fixture(tmp_path)
    npz = dict(np.load(vdir / "validation.npz"))
    npz["session_ids"] = npz["session_ids"][:-1]  # drop one row, others keep 3
    np.savez(vdir / "validation.npz", **npz)
    with pytest.raises(ValueError) as exc:
        CorpusValidationSet(vdir)
    assert "row-count mismatch" in str(exc.value)


def test_row_count_vs_manifest_mismatch_raises(tmp_path):
    vdir, *_ = _make_fixture(tmp_path)
    manifest = json.loads((vdir / "manifest.json").read_text())
    manifest["exercises"] = manifest["exercises"][:-1]  # manifest now lists 2, npz still has 3
    (vdir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError) as exc:
        CorpusValidationSet(vdir)
    assert "3" in str(exc.value) and "2" in str(exc.value)


def test_frames_time_dim_mismatch_raises(tmp_path):
    vdir, *_ = _make_fixture(tmp_path)
    npz = dict(np.load(vdir / "validation.npz"))
    npz["frames"] = npz["frames"][:, :-1]  # T-1 instead of T
    np.savez(vdir / "validation.npz", **npz)
    with pytest.raises(ValueError) as exc:
        CorpusValidationSet(vdir)
    assert "frames" in str(exc.value)


def test_keypress_wrong_column_count_raises(tmp_path):
    vdir, *_ = _make_fixture(tmp_path)
    npz = dict(np.load(vdir / "validation.npz"))
    npz["keypress"] = npz["keypress"][..., :-1]  # 7 columns instead of 8
    np.savez(vdir / "validation.npz", **npz)
    with pytest.raises(ValueError) as exc:
        CorpusValidationSet(vdir)
    assert "keypress" in str(exc.value)


def test_mouse_wrong_column_count_raises(tmp_path):
    vdir, *_ = _make_fixture(tmp_path)
    npz = dict(np.load(vdir / "validation.npz"))
    npz["mouse"] = np.concatenate([npz["mouse"], npz["mouse"][..., :1]], axis=-1)  # 3 columns instead of 2
    np.savez(vdir / "validation.npz", **npz)
    with pytest.raises(ValueError) as exc:
        CorpusValidationSet(vdir)
    assert "mouse" in str(exc.value)


def test_session_root_slices_actions_from_the_sessions_own_raw79_caches(tmp_path, monkeypatch):
    """plaicraft-debug#93: validation must condition on the same 79-key cache training reads, not the npz."""
    from improved_diffusion import debug_actions
    vdir = _make_fixture_no_swap(tmp_path)
    caches = {f"sess{i}": (np.random.RandomState(i).rand(400, 8).astype(np.float32),
                           np.random.RandomState(10 + i).randn(400, 2).astype(np.float32)) for i in range(len(NAMES))}
    seen = []
    monkeypatch.setattr(debug_actions, "load_or_build_raw",
                        lambda d, action_builder: (seen.append(action_builder), caches[d.name])[1])
    vs = CorpusValidationSet(vdir, T=T, n_observed=N_OBS, session_root=tmp_path / "corpus", action_builder="raw79")
    keypress, mouse = vs.load_all_actions_raw()
    for i, w in enumerate([100, 200, 300]):
        assert np.array_equal(keypress[i].numpy(), caches[f"sess{i}"][0][w:w + T])
        assert np.array_equal(mouse[i].numpy(), caches[f"sess{i}"][1][w:w + T])
    assert set(seen) == {"raw79"}


def test_session_root_km_codes_come_from_the_session_km_cache(tmp_path, monkeypatch):
    from improved_diffusion import debug_actions
    vdir = _make_fixture_no_swap(tmp_path)
    monkeypatch.setattr(debug_actions, "load_or_build_raw", lambda d, action_builder: (np.zeros((400, 8)), np.zeros((400, 2))))
    codes = {f"sess{i}": np.random.RandomState(i).rand(400, 36).astype(np.float32) for i in range(len(NAMES))}
    monkeypatch.setattr(debug_actions, "load_or_build",
                        lambda d, action_encoding, tokenizer_checkpoint, device, action_builder: (codes[d.name], None))
    vs = CorpusValidationSet(vdir, T=T, n_observed=N_OBS, action_encoding="km_fsq",
                             session_root=tmp_path / "corpus", action_builder="raw79")
    km, mouse = vs.load_all_actions()
    assert np.array_equal(km[1].numpy(), codes["sess1"][200:200 + T]) and mouse.shape[-1] == 0


def test_player_index_fn_labels_each_row_with_its_session_player(tmp_path, monkeypatch):
    from improved_diffusion import debug_actions
    monkeypatch.setattr(debug_actions, "load_or_build_raw", lambda d, action_builder: (np.zeros((400, 8)), np.zeros((400, 2))))
    vdir = _make_fixture_no_swap(tmp_path)
    vs = CorpusValidationSet(vdir, T=T, n_observed=N_OBS, player_index_fn=lambda d: int(d.name[-1]) + 7,
                             session_root=tmp_path / "corpus")
    assert vs.player_labels.tolist() == [7, 8, 9]
    assert CorpusValidationSet(vdir, T=T, n_observed=N_OBS).player_labels is None


def test_load_all_keys79_slices_each_rows_window_from_the_79key_ticks(tmp_path, monkeypatch):
    from improved_diffusion import debug_actions
    vdir = _make_fixture_no_swap(tmp_path)
    monkeypatch.setattr(debug_actions, "load_or_build_raw", lambda d, action_builder: (np.zeros((400, 8)), np.zeros((400, 2))))
    ticks = {f"sess{i}": np.random.RandomState(i).rand(400, 79).astype(np.float32) for i in range(len(NAMES))}
    monkeypatch.setattr(debug_actions, "load_ticks_79", lambda d: ticks[d.name])
    vs = CorpusValidationSet(vdir, T=T, n_observed=N_OBS, session_root=tmp_path / "corpus", action_builder="raw79")
    keys79 = vs.load_all_keys79()
    assert keys79.shape == (len(NAMES), T, 79)
    assert np.array_equal(keys79[2].numpy(), ticks["sess2"][300:300 + T])
    assert CorpusValidationSet(vdir, T=T, n_observed=N_OBS).load_all_keys79() is None  # debug world: 8 keys only


def test_wan_prefix_loaded_when_present_and_none_when_absent(tmp_path):
    vdir, *_ = _make_fixture(tmp_path)
    assert CorpusValidationSet(vdir).load_all_wan_prefix() is None
    npz = dict(np.load(vdir / "validation.npz"))
    prefix = np.random.RandomState(1).randn(3, 48, 2, 2).astype(np.float16)
    np.savez(vdir / "validation.npz", wan_prefix=prefix, **npz)
    got = CorpusValidationSet(vdir).load_all_wan_prefix()
    assert got.shape == (3, 1, 48, 2, 2) and got.dtype == torch.float32
    assert torch.allclose(got[:, 0], torch.from_numpy(prefix.astype(np.float32)))
