"""issue #76 slice 2: keypress_cross_entropy / keypress_ce_baserate, positive-dims-only,
pressed-frames-only, hand-checkable nats."""

import math

import torch

from improved_diffusion.debug_actions import keypress_cross_entropy, keypress_ce_baserate


def test_matches_hand_computed_logsigmoid_for_moderate_logits():
    # Cross-checks logsigmoid against the textbook -log(sigmoid(x)) form, on held dims only.
    decoded = torch.tensor([[2.0, -1.5, 0.0, 3.0, -3.0, 0.5, -0.5, 1.0]])
    y = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    p = torch.sigmoid(decoded)
    expected = -(y * torch.log(p)).sum(dim=-1).mean()  # single (pressed) frame -> mean is a no-op
    assert torch.allclose(keypress_cross_entropy(decoded, y), expected, atol=1e-5)


def test_single_key_hand_computed():
    decoded = torch.zeros(1, 8)
    y = torch.zeros(1, 8)
    y[0, 0] = 1.0
    # logsigmoid(0) = -log(2), one held key -> CE = log(2).
    assert torch.allclose(keypress_cross_entropy(decoded, y), torch.tensor(math.log(2)), atol=1e-6)


def test_chorded_frame_sums_every_held_key():
    # Two keys held in the same frame, at different logits -- CE must sum both terms, not
    # average them, so a chorded frame costs everything it holds (provisioning for #77-style
    # multi-key data even though this corpus never chords).
    decoded = torch.zeros(1, 8)
    decoded[0, 0] = 0.0
    decoded[0, 1] = 1.0
    y = torch.zeros(1, 8)
    y[0, 0] = 1.0
    y[0, 1] = 1.0
    expected = -(math.log(torch.sigmoid(torch.tensor(0.0))) + math.log(torch.sigmoid(torch.tensor(1.0))))
    assert torch.allclose(keypress_cross_entropy(decoded, y), torch.tensor(float(expected)), atol=1e-5)


def test_all_zero_frame_excluded_not_zero_contribution():
    # Frame 0 holds nothing -- a positive-only CE scores it a hard 0 regardless of the model, so
    # it must be dropped from the mean entirely rather than diluting it with a 0 term.
    decoded = torch.tensor([[5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # all-zero-y frame
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])   # one held key, logit 0
    y = torch.zeros(2, 8)
    y[1, 0] = 1.0
    ce = keypress_cross_entropy(decoded, y)
    # If the zero frame wrongly contributed 0 to a mean over both frames, this would be half.
    assert torch.allclose(ce, torch.tensor(math.log(2)), atol=1e-6)


def test_finite_and_large_when_confidently_wrong():
    decoded = torch.tensor([[-50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    y = torch.zeros(1, 8)
    y[0, 0] = 1.0  # confidently wrong on the only held key
    ce = keypress_cross_entropy(decoded, y)
    assert torch.isfinite(ce)
    assert ce.item() > 49


def test_near_zero_when_confidently_correct():
    decoded = torch.zeros(1, 8)
    decoded[0, 0] = 50.0
    y = torch.zeros(1, 8)
    y[0, 0] = 1.0
    assert keypress_cross_entropy(decoded, y).item() < 1e-10


def test_baserate_matches_closed_form():
    y = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],  # all-zero frame, must be excluded from the mean
    ])
    q = y.mean(dim=0)
    per_frame = -(y * torch.log(q)).sum(dim=-1)
    pressed = y.sum(dim=-1) > 0
    expected = per_frame[pressed].mean()
    assert torch.allclose(keypress_ce_baserate(y), expected, atol=1e-5)


def test_baserate_chorded_frame_sums_both_dims():
    # Row 0 holds both keys -- its per-frame term must be the sum of both dims' surprise,
    # not just one of them, matching keypress_cross_entropy's chorded-frame convention.
    y = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    q = y.mean(dim=0)  # [2/3, 2/3]
    per_frame = -(y * torch.log(q)).sum(dim=-1)
    pressed = y.sum(dim=-1) > 0
    expected = per_frame[pressed].mean()
    assert torch.allclose(keypress_ce_baserate(y), expected, atol=1e-6)
    assert abs(float(per_frame[0]) - 2 * (-math.log(2 / 3))) < 1e-6


def test_baserate_collapses_leading_dims():
    y = torch.zeros(5, 20, 8)
    y[:, :, 0] = 1.0  # key 0 always held, always at base rate 1 -> zero surprise
    ce = keypress_ce_baserate(y)
    assert torch.allclose(ce, torch.tensor(0.0), atol=1e-6)


def test_no_pressed_frames_is_zero_not_nan():
    """An empty pressed-frame selection means 0 nats, not an empty .mean() -> nan."""
    y_true = torch.zeros(4, 8)
    decoded = torch.randn(4, 8)
    assert float(keypress_cross_entropy(decoded, y_true)) == 0.0
    assert float(keypress_ce_baserate(y_true)) == 0.0
