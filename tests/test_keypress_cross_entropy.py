"""issue #76 slice 2: keypress_cross_entropy / keypress_ce_baserate, hand-checkable nats."""

import math

import torch

from improved_diffusion.debug_actions import keypress_cross_entropy, keypress_ce_baserate


def test_matches_hand_computed_bce_for_moderate_logits():
    # Cross-checks BCE-with-logits against the textbook -[y log p + (1-y) log(1-p)] form.
    decoded = torch.tensor([[2.0, -1.5, 0.0, 3.0, -3.0, 0.5, -0.5, 1.0]])
    y = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    p = torch.sigmoid(decoded)
    expected = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).sum(dim=-1)
    assert torch.allclose(keypress_cross_entropy(decoded, y), expected, atol=1e-5)


def test_all_zero_logits_give_eight_log_two():
    decoded = torch.zeros(1, 8)
    for y_val in (0.0, 1.0):
        y = torch.full((1, 8), y_val)
        assert torch.allclose(keypress_cross_entropy(decoded, y), torch.tensor([8 * math.log(2)]), atol=1e-5)


def test_finite_and_large_when_confidently_wrong():
    decoded = torch.tensor([[50.0, -50.0] * 4])
    y = torch.tensor([[1.0, 1.0] * 4])  # wrong on every "-50" dim, right on every "50" dim
    ce = keypress_cross_entropy(decoded, y)
    assert torch.isfinite(ce).all()
    assert ce.item() > 199  # 4 confidently-wrong dims contribute ~50 nats each


def test_near_zero_when_confidently_correct():
    decoded = torch.full((1, 8), 50.0)
    y = torch.ones(1, 8)
    assert keypress_cross_entropy(decoded, y).item() < 1e-10


def test_baserate_matches_closed_form_entropy():
    y = torch.tensor([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0],
    ])  # no dim is all-0 or all-1, so the plain log form has no singularity
    p = y.mean(dim=0)
    expected = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).sum()
    assert torch.allclose(keypress_ce_baserate(y), expected, atol=1e-5)


def test_baserate_degenerate_dims_contribute_zero():
    y = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])  # col0 always 0, col1 always 1
    assert keypress_ce_baserate(y).item() == 0.0


def test_baserate_collapses_leading_dims():
    # (N, T, 8) windows should pool over N*T when computing the per-key base rate.
    y = torch.zeros(5, 20, 8)
    y[:, :, 0] = 1.0  # key 0 always held, others never
    ce = keypress_ce_baserate(y)
    assert ce.item() == 0.0
