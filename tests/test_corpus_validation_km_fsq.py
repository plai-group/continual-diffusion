"""Issue #81: CorpusValidationSet's lazy km_fsq code-cache build. The km tokenizer is
mocked throughout (no GPU/real checkpoint needed) -- only build_km_codes's own
cache/sidecar logic is exercised for real."""
import json

import numpy as np
import pytest
import torch

import scripts.build_debug_validation_km_codes as kb
from improved_diffusion import debug_actions as da
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
    return vdir, keypress, mouse


def _stub_tokenizer(monkeypatch, sha="fakehash"):
    calls = []
    monkeypatch.setattr(kb, "_sha256", lambda path: sha)
    monkeypatch.setattr(kb, "load_tokenizer", lambda **kw: object())

    def _fake_encode(tokenizer, keys_raw, mouse_raw):
        calls.append(1)
        return torch.zeros(keys_raw.shape[0], keys_raw.shape[1], da.KM_CODE_DIM)

    monkeypatch.setattr(kb, "_encode_km_actions", _fake_encode)
    return calls


def test_km_fsq_cache_built_when_absent(tmp_path, monkeypatch):
    vdir, *_ = _make_fixture(tmp_path)
    calls = _stub_tokenizer(monkeypatch)
    vs = CorpusValidationSet(vdir, action_encoding="km_fsq", tokenizer_checkpoint=tmp_path / "ckpt.bin", device="cpu")
    k, m = vs.load_all_actions()
    assert k.shape == (3, T, da.KM_CODE_DIM) and m.shape == (3, T, 0)
    assert len(calls) == 1
    assert (vdir / "km_codes.npz").exists() and (vdir / "km_codes_manifest.json").exists()


def test_km_fsq_matching_cache_not_rebuilt(tmp_path, monkeypatch):
    vdir, *_ = _make_fixture(tmp_path)
    calls = _stub_tokenizer(monkeypatch)
    ckpt = tmp_path / "ckpt.bin"
    CorpusValidationSet(vdir, action_encoding="km_fsq", tokenizer_checkpoint=ckpt, device="cpu").load_all_actions()
    assert len(calls) == 1

    vs2 = CorpusValidationSet(vdir, action_encoding="km_fsq", tokenizer_checkpoint=ckpt, device="cpu")
    vs2.load_all_actions()
    assert len(calls) == 1  # sidecar matches -- builder must not re-encode


def test_km_fsq_stale_sidecar_triggers_rebuild(tmp_path, monkeypatch):
    vdir, *_ = _make_fixture(tmp_path)
    calls = _stub_tokenizer(monkeypatch, sha="hash-a")
    ckpt = tmp_path / "ckpt.bin"
    CorpusValidationSet(vdir, action_encoding="km_fsq", tokenizer_checkpoint=ckpt, device="cpu").load_all_actions()
    assert len(calls) == 1

    _stub_tokenizer(monkeypatch, sha="hash-b")  # simulate a different tokenizer checkpoint
    vs2 = CorpusValidationSet(vdir, action_encoding="km_fsq", tokenizer_checkpoint=ckpt, device="cpu")
    vs2.load_all_actions()
    sidecar = json.loads((vdir / "km_codes_manifest.json").read_text())
    assert sidecar["tokenizer_sha256"] == "hash-b"


def test_load_all_actions_raw_unaffected_by_action_encoding(tmp_path, monkeypatch):
    vdir, keypress, mouse = _make_fixture(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("load_all_actions_raw must never touch the km tokenizer")
    monkeypatch.setattr(kb, "_encode_km_actions", _boom)

    vs = CorpusValidationSet(vdir, action_encoding="km_fsq", tokenizer_checkpoint=tmp_path / "ckpt.bin", device="cpu")
    raw_k, raw_m = vs.load_all_actions_raw()
    assert raw_k.shape == (3, T, 8) and raw_m.shape == (3, T, 2)
    assert torch.allclose(raw_k, torch.from_numpy(keypress))
    assert torch.allclose(raw_m, torch.from_numpy(mouse))
    assert not (vdir / "km_codes.npz").exists()
