"""Overlay action bars: each row must draw the actions it was actually
generated with, so a model that generates its own actions can be read off the
video instead of having the recorded actions painted over every row."""

import numpy as np

from improved_diffusion import debug_validation as dv
from improved_diffusion import decode_debug as dd


T, H, W = 6, 24, 40


def _frames():
    return np.zeros((T, 3, H, W), dtype=np.float32)


def _actions(seed):
    rng = np.random.RandomState(seed)
    a = np.zeros((T, 10), dtype=np.float32)
    a[:, :8] = (rng.rand(T, 8) > 0.5).astype(np.float32)
    a[:, 8:] = rng.randn(T, 2)
    return a


class _BarRecorder:
    """Stands in for _overlay_frame and records the bar dict each call draws."""

    def __init__(self):
        self.bars = []

    def __call__(self, frame, bar, border=False):
        self.bars.append(bar)
        return np.zeros((10, 10, 3), dtype=np.uint8)


def test_display_shift():
    a = np.arange(T * 10, dtype=np.float32).reshape(T, 10)
    out = dv._to_display_actions(a)
    assert np.allclose(out[:-1], a[1:]), "causal cache must shift back by one frame"
    assert np.allclose(out[-1], 0), "last frame has no successor, so its bar is blank"
    print("  [PASS] _to_display_actions shifts causal -> display convention")


def test_action_bars_from_tensor():
    import torch

    a = _actions(0)
    bars = dv._action_bars(torch.from_numpy(a))
    assert len(bars) == T
    assert set(bars[0]) == {"keys", "clicks", "mouseDX", "mouseDY"}
    print("  [PASS] _action_bars accepts a tensor and returns one bar per frame")


def test_swap_overlay_draws_gt_and_generated_separately(monkeypatched=None):
    """The reported bug: the GT and TRUE panels were fed the same array, so a
    generated action could never differ from ground truth on screen."""
    rec = _BarRecorder()
    orig_overlay, orig_label = dv._overlay_frame, dv._label_panel
    orig_writer = dv_get_writer()
    dv._overlay_frame = rec
    dv._label_panel = lambda panel, text: panel
    try:
        act_gt, act_gen = _actions(1), _actions(2)
        dv._render_triple_overlay(
            frames_gt=_frames(), frames_true=_frames(),
            frames_swap=_frames(), frames_zero=_frames(),
            actions_gt=act_gt, actions_true=act_gen,
            actions_swap=_actions(3), actions_zero=_actions(4),
            n_observed=3, out_path="/dev/null", true_label="GEN",
        )
    finally:
        dv._overlay_frame, dv._label_panel = orig_overlay, orig_label
        dv_restore_writer(orig_writer)

    # 4 panels per frame, in order GT, GEN, SWAP, ZERO.
    assert len(rec.bars) == 4 * T, f"expected {4*T} panel draws, got {len(rec.bars)}"
    gt_bars = rec.bars[0::4]
    gen_bars = rec.bars[1::4]
    expect_gt = dv._action_bars(act_gt)
    expect_gen = dv._action_bars(act_gen)
    assert gt_bars == expect_gt, "GT panel must draw the recorded actions"
    assert gen_bars == expect_gen, "GEN panel must draw the GENERATED actions"
    assert gt_bars != gen_bars, "GT and GEN panels drew identical bars -- regression"
    print("  [PASS] swap overlay draws GT and generated actions from separate arrays")


def test_val_overlay_pred_row_uses_generated_actions():
    rec = _BarRecorder()
    orig_overlay = dd._overlay_frame
    orig_get = dd.get_frame_actions
    orig_writer = dd_get_writer()
    dd._overlay_frame = rec
    dd.get_frame_actions = lambda *a, **k: dv._action_bars(_actions(11))
    try:
        pred_bars = dv._action_bars(_actions(12))
        dd.render_overlay(
            gt_frames=_frames(), pred_frames=_frames(),
            session_db_path="unused", start_frame_idx=0,
            out_path="/dev/null", n_observed=3, pred_actions=pred_bars,
        )
    finally:
        dd._overlay_frame, dd.get_frame_actions = orig_overlay, orig_get
        dd_restore_writer(orig_writer)

    assert len(rec.bars) == 2 * T
    row_gt, row_pred = rec.bars[0::2], rec.bars[1::2]
    assert row_pred == pred_bars, "predicted row must draw the generated actions"
    assert row_gt != row_pred, "both rows drew the same bars -- regression"
    print("  [PASS] val/overlay predicted row draws generated actions")


def test_val_overlay_defaults_to_recorded_actions():
    """Without pred_actions (the action-conditioned runs) behaviour is unchanged."""
    rec = _BarRecorder()
    orig_overlay, orig_get = dd._overlay_frame, dd.get_frame_actions
    orig_writer = dd_get_writer()
    dd._overlay_frame = rec
    dd.get_frame_actions = lambda *a, **k: dv._action_bars(_actions(11))
    try:
        dd.render_overlay(
            gt_frames=_frames(), pred_frames=_frames(),
            session_db_path="unused", start_frame_idx=0,
            out_path="/dev/null", n_observed=3,
        )
    finally:
        dd._overlay_frame, dd.get_frame_actions = orig_overlay, orig_get
        dd_restore_writer(orig_writer)
    assert rec.bars[0::2] == rec.bars[1::2], "both rows should show recorded actions"
    print("  [PASS] val/overlay without pred_actions is unchanged")


# ---- stub out the mp4 writer; these tests are about wiring, not encoding ----
class _NullWriter:
    def append_data(self, frame): pass
    def close(self): pass


def dv_get_writer():
    import imageio
    orig = imageio.get_writer
    imageio.get_writer = lambda *a, **k: _NullWriter()
    return orig


def dv_restore_writer(orig):
    import imageio
    imageio.get_writer = orig


dd_get_writer, dd_restore_writer = dv_get_writer, dv_restore_writer


if __name__ == "__main__":
    print("Running overlay action-bar tests...")
    test_display_shift()
    test_action_bars_from_tensor()
    test_swap_overlay_draws_gt_and_generated_separately()
    test_val_overlay_pred_row_uses_generated_actions()
    test_val_overlay_defaults_to_recorded_actions()
    print("ALL OVERLAY TESTS PASSED")
