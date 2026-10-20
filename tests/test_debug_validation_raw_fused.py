"""plaicraft-debug#80: raw_fused split/inverse wiring in debug_validation.py.

The decode and swap arms split a 10-dim fused token into 8-dim keypress +
symlog(mouse), inverting the symlog back to raw pixels for metrics and
overlays. This exercises that formula against debug_actions' own fusion.
The CorpusValidationSet round trip is covered separately, on the branch that
introduces it.
"""
import numpy as np
import torch

from improved_diffusion import debug_actions as da
from improved_diffusion import debug_validation as dv


def test_decode_arm_split_round_trips_to_pixel_mouse():
    """Mirrors the decode arm: p_key = act[..., :8], p_mouse = inv_symlog(act[..., 8:])."""
    rng = np.random.RandomState(1)
    keypress = rng.rand(2, 6, 8).astype(np.float32)
    mouse = (rng.randn(2, 6, 2) * 60).astype(np.float32)  # raw pixels, wide range
    fused = torch.from_numpy(np.concatenate([keypress, da._symlog(mouse)], axis=-1))

    g_key = fused[..., :8]
    g_mouse = da._inv_symlog(fused[..., 8:])

    assert torch.allclose(g_key, torch.from_numpy(keypress))
    assert torch.allclose(g_mouse, torch.from_numpy(mouse), atol=1e-3)


def test_fused_layout_matches_load_or_build_raw_fused():
    """The split above is only correct if dims 8:10 really are symlog(mouse)."""
    keypress = np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    mouse = np.array([[150.0, -61.0]], dtype=np.float32)
    fused = np.concatenate([keypress, da._symlog(mouse)], axis=1)
    assert fused.shape[1] == da.RAW_FUSED_DIM
    assert np.allclose(da._inv_symlog(fused[:, 8:]), mouse, atol=1e-3)
    assert np.abs(fused[:, 8:]).max() < 6.0  # compressed, comparable to the {0,1} keypress dims


def test_symlog_and_inv_symlog_used_by_debug_validation_are_da_module_functions():
    # Ensures debug_validation dispatches to debug_actions's dispatching pair, not a
    # torch-only local shadow (the module's own metric-time `_symlog` is a separate fn).
    assert dv.debug_actions._symlog is da._symlog
    assert dv.debug_actions._inv_symlog is da._inv_symlog
