"""PlayerValidationSet + the cross-arm player metric and overlay (issue #85)."""
import json

import numpy as np
import pytest
import torch as th

from improved_diffusion.corpus_validation import CorpusValidationSet, PlayerValidationSet
from improved_diffusion.debug_validation import _player_metrics, _render_player_overlay

T, N_OBS, N_ROWS, H, W = 20, 10, 4, 24, 40


def _package(tmp_path, with_p2=True, n_rows=N_ROWS):
    d = tmp_path / "validation_player"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    frames = rng.uniform(-1, 1, (n_rows, T, 3, H, W)).astype(np.float32)
    keypress = np.zeros((n_rows, T, 8), dtype=np.float32)
    keypress[:, N_OBS:, 6] = 1.0  # a left-click held across the whole generated region
    arrays = dict(
        frames=frames,
        keypress=keypress,
        mouse=np.zeros((n_rows, T, 2), dtype=np.float32),
        names=np.array([f"player-click-left-{i}" for i in range(n_rows)]),
        session_ids=np.array([f"s{i:02d}" for i in range(n_rows)]),
        window_start_ticks=np.array([100] * n_rows, dtype=np.int64),
        boundary_ticks=np.array([100 + N_OBS] * n_rows, dtype=np.int64),
        player_indices=np.array([0, 1], dtype=np.int64),
    )
    if with_p2:
        p2 = frames.copy()
        p2[:, :, :, -2:, 18:22] = 1.0  # stand-in cue block, present on every frame
        arrays["frames_p2"] = p2
    np.savez(d / "validation.npz", **arrays)
    (d / "manifest.json").write_text(json.dumps({
        "schema_version": "1", "T": T, "n_observed": N_OBS, "tick_ms": 80,
        "subbin_rule_version": "1", "player_indices": [0, 1],
        "exercises": [{"index": i, "name": f"player-click-left-{i}", "session_id": f"s{i:02d}",
                       "swap_kind": "keypress", "swap_dim": 6, "swap_counterpart_dim": 7}
                      for i in range(n_rows)],
    }))
    return d


# ── PlayerValidationSet ────────────────────────────────────────────────────────────────────

def test_loads_both_arms(tmp_path):
    vs = PlayerValidationSet(_package(tmp_path), T=T, n_observed=N_OBS)
    assert len(vs.rows) == N_ROWS
    assert vs.load_all().shape == (N_ROWS, T, 3, H, W)
    assert vs.load_all_p2().shape == vs.load_all().shape
    assert vs.player_indices == [0, 1]


def test_actions_are_shared_between_arms(tmp_path):
    """One trace, two renders -- so there is exactly one keypress/mouse array, inherited
    verbatim from CorpusValidationSet along with its encoding branches."""
    vs = PlayerValidationSet(_package(tmp_path), T=T, n_observed=N_OBS)
    keypress, mouse = vs.load_all_actions_raw()
    assert keypress.shape == (N_ROWS, T, 8) and mouse.shape == (N_ROWS, T, 2)


def test_rejects_a_plain_issue81_package(tmp_path):
    with pytest.raises(ValueError, match="no frames_p2"):
        PlayerValidationSet(_package(tmp_path, with_p2=False), T=T, n_observed=N_OBS)


def test_the_package_still_loads_as_a_plain_corpus_set(tmp_path):
    """The extra arrays must not break the base reader -- that compatibility is the whole
    reason the package lives in its own directory instead of gaining a new filename."""
    assert len(CorpusValidationSet(_package(tmp_path), T=T, n_observed=N_OBS).rows) == N_ROWS


# ── the cross-arm metric ───────────────────────────────────────────────────────────────────

def _arms():
    rng = th.Generator().manual_seed(0)
    gt_a = th.rand(10, 3, H, W, generator=rng) * 2 - 1
    gt_b = gt_a.clone()
    gt_b[:, 0] = 1.0   # the two players' ground truth differs in the red channel
    return gt_a, gt_b


def test_margin_is_positive_when_each_arm_matches_its_own_player():
    gt_a, gt_b = _arms()
    out = _player_metrics(gt_a, gt_b, gt_a, gt_b)
    assert out["l2_matched"] == pytest.approx(0.0, abs=1e-6)
    assert out["margin"] > 0


def test_margin_is_negative_when_the_arms_are_swapped():
    gt_a, gt_b = _arms()
    out = _player_metrics(gt_b, gt_a, gt_a, gt_b)
    assert out["margin"] < 0, "generating the other player's frames must score below zero"


def test_margin_is_zero_when_the_model_ignores_the_player():
    """Both arms identical -- the model produced the same thing regardless of y."""
    gt_a, gt_b = _arms()
    same = (gt_a + gt_b) / 2
    out = _player_metrics(same, same, gt_a, gt_b)
    assert out["margin"] == pytest.approx(0.0, abs=1e-6)


def test_click_variants_are_restricted_to_click_frames():
    gt_a, gt_b = _arms()
    mask = th.zeros(10, dtype=th.bool)
    mask[5:] = True
    out = _player_metrics(gt_a, gt_b, gt_a, gt_b, click_mask=mask)
    assert "margin_click" in out and "margin" in out


def test_click_variants_are_absent_when_nothing_clicks():
    gt_a, gt_b = _arms()
    out = _player_metrics(gt_a, gt_b, gt_a, gt_b, click_mask=th.zeros(10, dtype=th.bool))
    assert not any(k.endswith("_click") for k in out)


# ── the overlay ────────────────────────────────────────────────────────────────────────────

class _NullWriter:
    """Captures appended frames instead of encoding them, as test_debug_validation_overlays
    does: which imageio plugin handles .mp4 varies by install, and the geometry is what these
    assertions are about anyway."""

    def __init__(self):
        self.frames = []

    def append_data(self, im):
        self.frames.append(im)

    def close(self):
        pass


def test_render_player_overlay_draws_three_panels_in_one_row(tmp_path):
    import imageio
    rng = np.random.default_rng(1)
    def f():
        return rng.uniform(-1, 1, (T, 3, H, W)).astype(np.float32)

    writer = _NullWriter()
    orig = imageio.get_writer
    imageio.get_writer = lambda *a, **k: writer
    try:
        _render_player_overlay(f(), f(), f(),
                               actions=(np.zeros((T, 8), np.float32), np.zeros((T, 2), np.float32)),
                               n_observed=N_OBS, out_path=str(tmp_path / "p.mp4"))
    finally:
        imageio.get_writer = orig

    assert len(writer.frames) == T
    frame = writer.frames[0]
    # One row of three panels: 3x wider than tall relative to a single labelled panel.
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.shape[1] > 3 * W, f"expected a 3-wide strip, got {frame.shape}"
