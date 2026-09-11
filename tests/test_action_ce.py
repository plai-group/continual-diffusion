"""action_ce must stay encoding-agnostic: probabilities in, full Bernoulli CE out.

Issue #76's CE applied logsigmoid (treating its input as a logit) and summed only
the positive-key term over pressed frames, which made it blind to false presses --
a key predicted "pressed" when it wasn't cost nothing. These tests pin down the
replacement: CE over probabilities, both Bernoulli terms, meaned over all frames.
"""
import torch

from improved_diffusion.action_ce import EPS, keypress_cross_entropy, keypress_ce_baserate


def test_perfect_probability_is_near_zero():
    y = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]])
    p = y.clone()
    ce = keypress_cross_entropy(p, y)
    assert 0 <= ce.item() < 1e-3  # bounded by the EPS clamp, not exactly 0


def test_inverted_probability_is_large():
    y = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]])
    p = 1 - y
    ce = keypress_cross_entropy(p, y)
    assert ce.item() > 10.0


def test_wrong_prediction_scores_worse_than_correct():
    y = torch.tensor([[1., 0., 1., 0., 0., 0., 0., 0.]])
    p_correct = torch.full_like(y, 0.5)
    p_correct[y == 1] = 0.9
    p_correct[y == 0] = 0.1
    p_wrong = 1 - p_correct
    assert keypress_cross_entropy(p_correct, y) < keypress_cross_entropy(p_wrong, y)


def test_false_press_is_penalised():
    # #76's positive-only CE scores this at 0 since y has no held keys here.
    y = torch.zeros(1, 8)
    p = torch.zeros(1, 8)
    p[0, 3] = 0.99  # predicts key 3 pressed; ground truth says unheld
    ce = keypress_cross_entropy(p, y)
    assert ce.item() > 1.0


def test_baserate_matches_hand_computed_two_key_entropy():
    # 4 frames, 2 keys: key0 held in frames 0,1 (rate 0.5), key1 held in frame 2 (rate 0.25).
    y = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 0.]])
    # Per-frame CE = -(y*log(q) + (1-y)*log(1-q)) summed over keys, q=[0.5, 0.25]:
    # frame0=frame1=frame3: -log(0.5) - log(0.75) = 0.693147 + 0.287682 = 0.980829
    # frame2: -log(0.5) - log(0.25) = 0.693147 + 1.386294 = 2.079442
    expected = (3 * 0.980829 + 2.079442) / 4
    assert abs(keypress_ce_baserate(y).item() - expected) < 1e-4


def test_baserate_finite_for_never_pressed_key():
    y = torch.zeros(4, 3)
    ce = keypress_ce_baserate(y)
    assert torch.isfinite(ce)
    assert ce.item() == 0.0


def test_baserate_finite_for_always_pressed_key():
    y = torch.ones(4, 3)
    ce = keypress_ce_baserate(y)
    assert torch.isfinite(ce)
    assert ce.item() == 0.0
