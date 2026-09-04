"""plaicraft-debug#81: raw_fused wiring in debug_validation.py.

The decode/swap-encode/swap-decode arms all split a 10-dim fused token into
8-dim keypress + symlog(mouse), inverting the symlog back to raw pixels for
metrics/overlays. This exercises that exact split/inverse formula, and its
round trip through CorpusValidationSet's native-encoding loader.
"""
import json

import numpy as np
import torch

from improved_diffusion import debug_actions as da
from improved_diffusion import debug_validation as dv
from improved_diffusion.corpus_validation import CorpusValidationSet

NAMES = ["walk-forward", "strafe-left"]
T, N_OBS, H, W = 6, 3, 4, 4


def _make_fixture(tmp_path):
    vdir = tmp_path / "validation"
    vdir.mkdir()
    n = len(NAMES)
    rng = np.random.RandomState(1)
    frames = rng.uniform(-1, 1, size=(n, T, 3, H, W)).astype(np.float32)
    keypress = rng.rand(n, T, 8).astype(np.float32)
    mouse = (rng.randn(n, T, 2) * 60).astype(np.float32)  # raw pixels, wide range
    window_start_ticks = np.array([100, 200], dtype=np.int64)
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
            {"index": 1, "name": "strafe-left", "session_id": "sess1", "swap_kind": "mouse_dx"},
        ],
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest))
    return vdir, keypress, mouse


def test_decode_arm_split_round_trips_to_pixel_mouse(tmp_path):
    """Mirrors the ~683 decode arm: p_key = samples_act[...,:8], p_mouse = inv_symlog(rest)."""
    vdir, keypress, mouse = _make_fixture(tmp_path)
    vs = CorpusValidationSet(vdir, action_encoding="raw_fused")
    fused, _ = vs.load_all_actions()  # what run_debug_validation calls keypress_chunk

    g_key = fused[..., :8]
    g_mouse = dv.debug_actions._inv_symlog(fused[..., 8:])

    assert torch.allclose(g_key, torch.from_numpy(keypress))
    assert torch.allclose(g_mouse, torch.from_numpy(mouse), atol=1e-3)


def test_swap_encode_arm_matches_corpus_validation_fusion(tmp_path):
    """Mirrors the ~801 swap-pass encode arm: cat([key_raw, symlog(mouse_raw)], -1)."""
    vdir, keypress, mouse = _make_fixture(tmp_path)
    vs = CorpusValidationSet(vdir, action_encoding="raw_fused")
    key_raw, mouse_raw = vs.load_all_actions_raw()

    actions_in = torch.cat([key_raw, dv.debug_actions._symlog(mouse_raw)], dim=-1)
    fused_native, _ = vs.load_all_actions()
    assert torch.allclose(actions_in, fused_native, atol=1e-5)


def test_symlog_and_inv_symlog_used_by_debug_validation_are_da_module_functions():
    # Ensures debug_validation dispatches to debug_actions's dispatching pair, not a
    # torch-only local shadow (the module's own metric-time `_symlog` is a separate fn).
    assert dv.debug_actions._symlog is da._symlog
    assert dv.debug_actions._inv_symlog is da._inv_symlog
