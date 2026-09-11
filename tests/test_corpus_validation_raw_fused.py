"""plaicraft-debug#81: CorpusValidationSet's raw_fused action_encoding.

No tokenizer involved -- this is a pure symlog fold of the frozen npz's raw-pixel
mouse array, so (unlike km_fsq) it must never touch build_km_codes or the tokenizer.
"""
import json

import numpy as np
import torch

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
    mouse = (rng.randn(n, T, 2) * 50).astype(np.float32)
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


def test_raw_fused_load_all_actions_shape_and_values(tmp_path):
    vdir, keypress, mouse = _make_fixture(tmp_path)
    vs = CorpusValidationSet(vdir, action_encoding="raw_fused")
    fused, empty_mouse = vs.load_all_actions()
    assert fused.shape == (3, T, da.RAW_FUSED_DIM)
    assert empty_mouse.shape == (3, T, 0)
    assert torch.allclose(fused[..., :8], torch.from_numpy(keypress))
    assert torch.allclose(fused[..., 8:], da._symlog(torch.from_numpy(mouse)), atol=1e-5)


def test_raw_fused_load_all_actions_never_touches_km_tokenizer(tmp_path, monkeypatch):
    vdir, *_ = _make_fixture(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("raw_fused must never build km codes")
    import scripts.build_debug_validation_km_codes as kb
    monkeypatch.setattr(kb, "build_km_codes", _boom)

    vs = CorpusValidationSet(vdir, action_encoding="raw_fused")
    vs.load_all_actions()
    assert not (vdir / "km_codes.npz").exists()


def test_raw_fused_load_all_actions_raw_unaffected(tmp_path):
    vdir, keypress, mouse = _make_fixture(tmp_path)
    vs = CorpusValidationSet(vdir, action_encoding="raw_fused")
    raw_k, raw_m = vs.load_all_actions_raw()
    assert raw_k.shape == (3, T, 8) and raw_m.shape == (3, T, 2)
    assert torch.allclose(raw_k, torch.from_numpy(keypress))
    assert torch.allclose(raw_m, torch.from_numpy(mouse))
