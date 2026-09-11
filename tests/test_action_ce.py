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


def test_probabilities_outside_unit_interval_stay_finite():
    # raw/raw_fused are MSE regression heads, not squashed ones -- they overshoot past
    # 1 and undershoot below 0 routinely, and only the clamp keeps log() finite.
    y = torch.tensor([[1., 0., 1., 0.]])
    p = torch.tensor([[1.3, -0.2, 1.0, 0.0]])
    ce = keypress_cross_entropy(p, y)
    assert torch.isfinite(ce)
    assert 0 <= ce.item() < 1e-3  # every dim is correct once clamped


def test_confident_wrong_prediction_outside_unit_interval_is_bounded():
    # The clamp also caps the worst case: an overshooting false press must not be inf.
    y = torch.zeros(1, 4)
    p = torch.full((1, 4), 1.4)
    ce = keypress_cross_entropy(p, y)
    assert torch.isfinite(ce)
    assert ce.item() < 4 * -torch.log(torch.tensor(EPS)).item() + 1e-3


def test_explicit_baserate_is_nonzero_on_a_single_frame():
    # The regression guard: q derived from one frame equals that frame, so the
    # entropy collapses to exactly 0 and the anchor becomes unbeatable.
    y = torch.tensor([[1., 0., 1., 0., 0., 0., 0., 0.]])
    q = torch.tensor([0.19, 0.03, 0.05, 0.03, 0.02, 0.03, 0.05, 0.045])
    assert keypress_ce_baserate(y).item() == 0.0
    assert keypress_ce_baserate(y, q).item() > 1.0


def test_baserate_equals_cross_entropy_of_a_constant_marginal_predictor():
    # Pins what the anchor means: the CE a head that ignores the frames and always
    # emits q would have paid on this window.
    g = torch.Generator().manual_seed(7)
    y = (torch.rand(9, 8, generator=g) > 0.7).float()
    q = torch.full((8,), 0.3)
    assert abs(keypress_ce_baserate(y, q).item()
               - keypress_cross_entropy(q.expand_as(y), y).item()) < 1e-5


def test_explicit_baserate_with_a_never_pressed_key_stays_finite():
    # q_k == 0 can only pair with y_k == 0 (nothing in the pool pressed it), so xlogy
    # takes the term to 0 rather than -inf.
    y = torch.tensor([[1., 0., 0.], [0., 0., 0.]])
    q = torch.tensor([0.5, 0.25, 0.0])
    ce = keypress_ce_baserate(y, q)
    assert torch.isfinite(ce)
