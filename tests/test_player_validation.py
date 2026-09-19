"""PlayerValidationSet + the cross-arm player metric and overlay (issue #85)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch as th

from improved_diffusion.corpus_validation import CorpusValidationSet, PlayerValidationSet
from improved_diffusion.debug_validation import (
    _arm_bars, _player_metrics, _render_player_arm_overlay)

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


def _render(tmp_path, gt, gen, actions_gt, actions_gen, label="P0 red"):
    """Render one arm through a stubbed writer and hand back the first composed frame."""
    import imageio
    writer = _NullWriter()
    orig = imageio.get_writer
    imageio.get_writer = lambda *a, **k: writer
    try:
        _render_player_arm_overlay(frames_gt=gt, frames_gen=gen,
                                   actions_gt=actions_gt, actions_gen=actions_gen,
                                   n_observed=N_OBS, out_path=str(tmp_path / "p.mp4"),
                                   label=label)
    finally:
        imageio.get_writer = orig
    return writer.frames


def test_render_player_arm_overlay_draws_one_row_of_two_panels(tmp_path):
    """One file per player, GT beside that player's own generation -- not a 2x2 of both."""
    rng = np.random.default_rng(1)
    def f():
        return rng.uniform(-1, 1, (T, 3, H, W)).astype(np.float32)

    zeros = (np.zeros((T, 8), np.float32), np.zeros((T, 2), np.float32))
    frames = _render(tmp_path, f(), f(), zeros, zeros)

    assert len(frames) == T
    frame = frames[0]
    assert frame.ndim == 3 and frame.shape[2] == 3
    # A wide strip of two panels, so the grid is 1x2 and never stacks a second row.
    assert frame.shape[1] > 2 * W, f"got {frame.shape}"
    assert frame.shape[1] > 2 * frame.shape[0], f"not a single row: {frame.shape}"
    w = frame.shape[1] // 2
    assert not np.array_equal(frame[:, :w], frame[:, w:]), "GT and generation match"


def test_each_arm_is_written_to_its_own_file(tmp_path):
    """The split is the point: two arms must produce two paths, named for their player."""
    frames = np.zeros((T, 3, H, W), np.float32)
    zeros = (np.zeros((T, 8), np.float32), np.zeros((T, 2), np.float32))
    paths = set()
    for label in ("P0 red", "P1 blue"):
        import imageio
        writer = _NullWriter()
        orig = imageio.get_writer
        imageio.get_writer = lambda *a, **k: writer
        try:
            out = _render_player_arm_overlay(
                frames_gt=frames, frames_gen=frames, actions_gt=zeros, actions_gen=zeros,
                n_observed=N_OBS, out_path=str(tmp_path / f"step0_00_x_{label.split()[-1]}.mp4"),
                label=label)
        finally:
            imageio.get_writer = orig
        paths.add(Path(out).name)
    assert paths == {"step0_00_x_red.mp4", "step0_00_x_blue.mp4"}


def test_the_generated_panel_draws_its_own_actions(tmp_path):
    """The teacher-forcing illusion: if the generated panel reuses the GT bars, a free rollout
    reads as prescribed. Give the two panels different actions and assert they differ."""
    frames = np.zeros((T, 3, H, W), np.float32)
    def bars(dim):
        k = np.zeros((T, 8), np.float32); k[:, dim] = 1.0
        return (k, np.zeros((T, 2), np.float32))

    grid = _render(tmp_path, frames, frames, bars(0), bars(6))[0]
    w = grid.shape[1] // 2
    assert not np.array_equal(grid[:, :w], grid[:, w:]), "the generation reused the GT bars"
